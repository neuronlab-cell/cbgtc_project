# optuna_driver.py
# Normal-regime Optuna tuning for STN–GPe–GPi network
#
# Philosophy:
#   • Optimize ONLY the normal physiological regime (δ = 0)
#   • Target: firing rates, CV, beta power fraction (13–30 Hz)
#   • Parameters searched:
#       - tonic drives (ISTN_mean, I_baseline_GPe, I_baseline_GPi)
#       - intrinsic scalers (stn_intrinsic_scale, gpe_adapt_scale)
#       - OU noise levels
#       - population sizes
#   • DO NOT modify synapse weights or connection probabilities
#   • DO NOT use pathological (δ = 1) scoring here
#
# Objective:
#   J = − [  w_rate * rate_error
#          + w_cv   * cv_error
#          + w_beta * beta_error ]
#
#   The lower the combined error, the higher the score we return to Optuna.
#
# Usage:
#   cd ~/cbgtc_project
#   source .venv/bin/activate
#   python -m stn_gp.optuna.optuna_driver
#
# The best parameters are saved in best_theta.json

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import optuna

from stn_gp.optuna.sim_api import run_simulation


# ============================================================
# Target biological ranges (NORMAL)
# ============================================================

TARGETS = {
    "stn_rate": (15.0, 25.0),   # Hz
    "gpe_rate": (60.0, 80.0),   # Hz
    "gpi_rate": (60.0, 90.0),   # Hz

    "cv": (0.6, 1.0),           # realistic irregularity

    # Beta power fraction: beta(13–30Hz)/broad(1–80Hz)
    "beta_frac": (0.03, 0.10),
}

# ============================================================
# Utility functions
# ============================================================

def _compute_firing_rates(spikes: np.ndarray, dt_ms: float, burn_steps: int):
    sp = spikes[burn_steps:]
    T = sp.shape[0]
    dur = T * dt_ms / 1000.0
    return sp.sum(axis=0) / dur


def _compute_cv(spikes: np.ndarray, dt_ms: float, burn_steps: int):
    sp = spikes[burn_steps:]
    N = sp.shape[1]
    cvs = np.zeros(N)
    for i in range(N):
        t_idx = np.where(sp[:, i] > 0)[0]
        if len(t_idx) < 5:
            cvs[i] = np.nan
            continue
        isi = np.diff(t_idx) * dt_ms
        cvs[i] = np.std(isi) / (np.mean(isi) + 1e-9)
    return cvs


def _compute_beta_fraction(rate_t: np.ndarray, dt_ms: float):
    # FFT beta fraction = power(13–30Hz) / power(1–80Hz)
    x = rate_t - np.mean(rate_t)
    n = len(x)
    fs = 1000.0 / dt_ms
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    X = np.fft.rfft(x)
    psd = np.abs(X) ** 2

    def band(f1, f2):
        m = (freqs >= f1) & (freqs <= f2)
        if np.any(m):
            return psd[m].sum()
        return 0.0

    beta = band(13, 30)
    broad = band(1, 80)
    return beta / (broad + 1e-12)


def _bounded_error(value, lo, hi):
    if lo <= value <= hi:
        return 0.0
    # distance outside the band (normalized)
    if value < lo:
        return (lo - value) / lo
    else:
        return (value - hi) / hi


# ============================================================
# PARAMETER SEARCH SPACE
# ============================================================

def sample_theta(trial: optuna.Trial) -> Dict:
    theta = {}

    # ------------------------------------------------------------------
    # DRIVES — PRIMARY CONTROL FOR RATES
    # ------------------------------------------------------------------
    theta["ISTN_mean"] = trial.suggest_float("ISTN_mean", 15.0, 35.0)
    theta["I_baseline_GPe"] = trial.suggest_float("I_baseline_GPe", 100.0, 220.0)
    theta["I_baseline_GPi"] = trial.suggest_float("I_baseline_GPi", 120.0, 260.0)

    # ------------------------------------------------------------------
    # INTRINSIC SCALERS — CONTROL CV/IRREGULARITY
    # ------------------------------------------------------------------
    theta["stn_intrinsic_scale"] = trial.suggest_float("stn_intrinsic_scale", 0.6, 1.4)
    theta["gpe_adapt_scale"] = trial.suggest_float("gpe_adapt_scale", 0.6, 1.6)
    theta["gpi_adapt_scale"] = trial.suggest_float("gpi_adapt_scale", 0.6, 1.6)

    # ------------------------------------------------------------------
    # NOISE — CONTROLS RATE VARIABILITY & CV
    # ------------------------------------------------------------------
    theta["noise_sigma_stn"] = trial.suggest_float("noise_sigma_stn", 0.1, 2.0)
    theta["noise_sigma_gpe"] = trial.suggest_float("noise_sigma_gpe", 0.1, 2.0)
    theta["noise_sigma_gpi"] = trial.suggest_float("noise_sigma_gpi", 0.1, 2.0)

    # ------------------------------------------------------------------
    # POPULATION SIZES — affects PSD stability
    # ------------------------------------------------------------------
    theta["n_stn"] = trial.suggest_int("n_stn", 40, 80)
    theta["n_gpe"] = trial.suggest_int("n_gpe", 80, 180)
    theta["n_gpi"] = trial.suggest_int("n_gpi", 60, 140)

    return theta


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(trial: optuna.Trial) -> float:

    theta = sample_theta(trial)

    # ---------- run simulation ----------
    sim = run_simulation(
        theta,
        delta=0.0,          # NORMAL ONLY
        t_total_s=4.0,
        burn_in_s=1.0,
        dt_ms=0.025,
    )

    dt = sim["dt_ms"]
    burn = sim["burn_steps"]

    # ---------- FIRING RATES ----------
    stn_rates = _compute_firing_rates(sim["spikes_stn"], dt, burn)
    gpe_rates = _compute_firing_rates(sim["spikes_gpe"], dt, burn)
    gpi_rates = _compute_firing_rates(sim["spikes_gpi"], dt, burn)

    stn_rate = float(np.nanmean(stn_rates))
    gpe_rate = float(np.nanmean(gpe_rates))
    gpi_rate = float(np.nanmean(gpi_rates))

    # ---------- CV ----------
    stn_cv = np.nanmean(_compute_cv(sim["spikes_stn"], dt, burn))
    gpe_cv = np.nanmean(_compute_cv(sim["spikes_gpe"], dt, burn))
    gpi_cv = np.nanmean(_compute_cv(sim["spikes_gpi"], dt, burn))

    mean_cv = float(np.nanmean([stn_cv, gpe_cv, gpi_cv]))

    # ---------- BETA FRACTION ----------
    rate_stn_t = sim["spikes_stn"][burn:].sum(axis=1)
    beta_frac = _compute_beta_fraction(rate_stn_t, dt)

    # ====================================================
    # BUILD ERROR TERMS
    # ====================================================
    # Rate error
    e_rate = (
        _bounded_error(stn_rate, *TARGETS["stn_rate"]) +
        _bounded_error(gpe_rate, *TARGETS["gpe_rate"]) +
        _bounded_error(gpi_rate, *TARGETS["gpi_rate"])
    ) / 3.0

    # CV error
    e_cv = _bounded_error(mean_cv, *TARGETS["cv"])

    # Beta error
    e_beta = _bounded_error(beta_frac, *TARGETS["beta_frac"])

    # Weighted combined error
    w_rate = 3.0
    w_cv   = 2.0
    w_beta = 1.0

    total_error = (w_rate * e_rate + w_cv * e_cv + w_beta * e_beta)

    # Optuna maximizes → return negative error
    score = -total_error

    # Store for inspection
    trial.set_user_attr("stn_rate", stn_rate)
    trial.set_user_attr("gpe_rate", gpe_rate)
    trial.set_user_attr("gpi_rate", gpi_rate)
    trial.set_user_attr("mean_cv", mean_cv)
    trial.set_user_attr("beta_frac", beta_frac)

    return score


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    print("🔧 Starting NORMAL-REGIME Optuna optimization...")

    db_path = Path("normal_stn_gpe_gpi_optuna.db").absolute()
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="normal_stn_gpe_gpi",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    print(f"📁 Study DB = {db_path}")

    # Set desired trial count
    n_trials = 2

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n=== BEST TRIAL ===")
    best = study.best_trial
    print("Score:", best.value)
    print("Params:", best.params)
    print("Metrics:", best.user_attrs)

    with open("best_theta.json", "w") as f:
        json.dump(best.params, f, indent=2)

    print("💾 Saved best parameters to best_theta.json")


if __name__ == "__main__":
    main()
