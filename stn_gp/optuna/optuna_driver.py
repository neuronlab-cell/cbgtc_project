# optuna_driver.py
# NORMAL-regime Optuna tuning for STN–GPe–GPi network.
#
# Optimizes the STN–GPe–GPi microcircuit to match:
#   • Firing rates
#   • CV (per nucleus, separate ranges)
#   • STN beta fraction
#
# No pathology (δ=0 only).
# No synapse or connectivity optimization.
#
# Best parameters saved to best_theta.json.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import numpy as np
import optuna

from stn_gp.optuna.sim_api import run_simulation


# ============================================================
# TARGET BIOLOGICAL RANGES (NORMAL STATE)
# ============================================================

TARGETS = {
    # Firing rate (Hz)
    "stn_rate": (15.0, 25.0),
    "gpe_rate": (60.0, 80.0),
    "gpi_rate": (60.0, 90.0),

    # ISI CV — *per nucleus* bands
    "stn_cv": (0.7, 1.0),
    "gpe_cv": (1.0, 1.4),
    "gpi_cv": (0.6, 0.9),

    # STN beta fraction
    "beta_frac": (0.03, 0.10),
}

# weights for multi-objective combination
WEIGHTS = {
    "rate": 3.0,
    "cv": 2.0,
    "beta": 1.0,
}


# ============================================================
# Utility functions
# ============================================================

def _bounded_error(value, lo, hi):
    """Return normalized distance outside [lo, hi]."""
    if lo <= value <= hi:
        return 0.0
    if value < lo:
        return (lo - value) / (hi - lo)
    else:
        return (value - hi) / (hi - lo)


def _compute_rates(spikes, dt, burn):
    sp = spikes[burn:]
    T = sp.shape[0]
    dur_s = T * dt / 1000.0
    return sp.sum(axis=0) / dur_s


def _compute_cv(spikes, dt, burn):
    sp = spikes[burn:]
    N = sp.shape[1]
    cvs = np.zeros(N)
    for i in range(N):
        idx = np.where(sp[:, i] > 0)[0]
        if idx.size < 5:
            cvs[i] = np.nan
            continue
        isis = np.diff(idx) * dt
        cvs[i] = np.std(isis) / (np.mean(isis) + 1e-9)
    return cvs


def _compute_beta_fraction(rate_t, dt):
    x = rate_t - np.mean(rate_t)
    n = len(x)
    fs = 1000.0 / dt
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    X = np.fft.rfft(x)
    psd = np.abs(X) ** 2

    def band(f1, f2):
        mask = (freqs >= f1) & (freqs <= f2)
        return float(psd[mask].sum()) if np.any(mask) else 0.0

    beta = band(13, 30)
    broad = band(1, 80)
    return beta / (broad + 1e-12)


# ============================================================
# PARAMETER SEARCH SPACE
# ============================================================

def sample_theta(trial: optuna.Trial) -> Dict:
    θ = {}

    # --- Tonic drives controlling rates ---
    θ["ISTN_mean"] = trial.suggest_float("ISTN_mean", 15.0, 35.0)
    θ["I_baseline_GPe"] = trial.suggest_float("I_baseline_GPe", 120.0, 220.0)
    θ["I_baseline_GPi"] = trial.suggest_float("I_baseline_GPi", 150.0, 300.0)

    # --- Intrinsic scaling (affects CV/irregularity) ---
    θ["stn_intrinsic_scale"] = trial.suggest_float("stn_intrinsic_scale", 0.5, 1.5)
    θ["gpe_adapt_scale"] = trial.suggest_float("gpe_adapt_scale", 0.5, 1.8)
    θ["gpi_adapt_scale"] = trial.suggest_float("gpi_adapt_scale", 0.5, 1.8)

    # --- Noise ---
    θ["noise_sigma_stn"] = trial.suggest_float("noise_sigma_stn", 0.05, 2.0)
    θ["noise_sigma_gpe"] = trial.suggest_float("noise_sigma_gpe", 0.05, 2.0)
    θ["noise_sigma_gpi"] = trial.suggest_float("noise_sigma_gpi", 0.05, 2.0)

    # --- Population sizes ---
    θ["n_stn"] = trial.suggest_int("n_stn", 40, 80)
    θ["n_gpe"] = trial.suggest_int("n_gpe", 80, 180)
    θ["n_gpi"] = trial.suggest_int("n_gpi", 60, 150)

    return θ


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(trial: optuna.Trial) -> float:
    θ = sample_theta(trial)

    sim = run_simulation(
        θ,
        delta=0.0,          # NORMAL ONLY
        t_total_s=4.0,
        burn_in_s=1.0,
        dt_ms=0.025,
    )

    dt = sim["dt_ms"]
    burn = sim["burn_steps"]

    # --- firing rates ---
    stn_r = np.nanmean(_compute_rates(sim["spikes_stn"], dt, burn))
    gpe_r = np.nanmean(_compute_rates(sim["spikes_gpe"], dt, burn))
    gpi_r = np.nanmean(_compute_rates(sim["spikes_gpi"], dt, burn))

    # --- CV (per nucleus) ---
    stn_cv = np.nanmean(_compute_cv(sim["spikes_stn"], dt, burn))
    gpe_cv = np.nanmean(_compute_cv(sim["spikes_gpe"], dt, burn))
    gpi_cv = np.nanmean(_compute_cv(sim["spikes_gpi"], dt, burn))

    # --- STN beta fraction ---
    rate_stn_t = sim["spikes_stn"][burn:].sum(axis=1)
    beta_frac = _compute_beta_fraction(rate_stn_t, dt)

    # --- error terms ---
    e_rate = (
        _bounded_error(stn_r, *TARGETS["stn_rate"]) +
        _bounded_error(gpe_r, *TARGETS["gpe_rate"]) +
        _bounded_error(gpi_r, *TARGETS["gpi_rate"])
    ) / 3.0

    e_cv = (
        _bounded_error(stn_cv, *TARGETS["stn_cv"]) +
        _bounded_error(gpe_cv, *TARGETS["gpe_cv"]) +
        _bounded_error(gpi_cv, *TARGETS["gpi_cv"])
    ) / 3.0

    e_beta = _bounded_error(beta_frac, *TARGETS["beta_frac"])

    # weighted combined error
    total_error = (
        WEIGHTS["rate"] * e_rate +
        WEIGHTS["cv"]   * e_cv +
        WEIGHTS["beta"] * e_beta
    )

    score = -total_error  # Maximize

    trial.set_user_attr("stn_rate", stn_r)
    trial.set_user_attr("gpe_rate", gpe_r)
    trial.set_user_attr("gpi_rate", gpi_r)
    trial.set_user_attr("stn_cv", stn_cv)
    trial.set_user_attr("gpe_cv", gpe_cv)
    trial.set_user_attr("gpi_cv", gpi_cv)
    trial.set_user_attr("beta_frac", beta_frac)

    return score


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    print("🔧 Starting NORMAL-state Optuna optimization…")

    db_path = Path("normal_stn_gpe_gpi_optuna.db").absolute()
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="normal_stn_gpe_gpi",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    print(f"📁 Study DB = {db_path}")

    # change to 50+ later
    n_trials = 50

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
