# objectives.py
# Objective functions for **normal-only** optimization of STN–GPe–GPi.
#
# Computes:
#   • Firing rates
#   • ISI CV
#   • Beta fraction (13–30 Hz) of population spike-rate PSD
#
# Returns:
#   score = - (weighted sum of normalized errors)
#
# This is used by optuna_driver.py (normal-only version).

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


# ============================================================
# Biological normal targets
# ============================================================

TARGETS = {
    "stn_rate": (15.0, 25.0),
    "gpe_rate": (60.0, 80.0),
    "gpi_rate": (60.0, 90.0),

    "cv": (0.6, 1.0),

    # Beta fraction = beta(13–30) / broad(1–80)
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
        if t_idx.size < 5:
            cvs[i] = np.nan
            continue
        isi = np.diff(t_idx) * dt_ms
        cvs[i] = np.std(isi) / (np.mean(isi) + 1e-9)
    return cvs


def _compute_beta_fraction(rate_t: np.ndarray, dt_ms: float):
    x = rate_t - np.mean(rate_t)
    n = len(x)
    fs = 1000.0 / dt_ms
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    X = np.fft.rfft(x)
    psd = np.abs(X) ** 2

    def band(f1, f2):
        m = (freqs >= f1) & (freqs <= f2)
        return psd[m].sum() if np.any(m) else 0.0

    beta = band(13, 30)
    broad = band(1, 80)
    return beta / (broad + 1e-12)


def _bounded_error(value: float, lo: float, hi: float) -> float:
    if lo <= value <= hi:
        return 0.0
    if value < lo:
        return (lo - value) / lo
    else:
        return (value - hi) / hi


# ============================================================
# Core function
# ============================================================

def compute_normal_objective(sim_out: Dict) -> Tuple[float, Dict]:
    """
    Compute normal-regime objective score and metrics dict.

    Returns:
        score (float): Optuna maximizes this (so we return -error)
        metrics (dict): human-readable metrics for logging
    """

    dt = float(sim_out["dt_ms"])
    burn = int(sim_out["burn_steps"])

    # -------- firing rates --------
    stn_rates = _compute_firing_rates(sim_out["spikes_stn"], dt, burn)
    gpe_rates = _compute_firing_rates(sim_out["spikes_gpe"], dt, burn)
    gpi_rates = _compute_firing_rates(sim_out["spikes_gpi"], dt, burn)

    stn_rate = float(np.nanmean(stn_rates))
    gpe_rate = float(np.nanmean(gpe_rates))
    gpi_rate = float(np.nanmean(gpi_rates))

    # -------- cv --------
    stn_cv = np.nanmean(_compute_cv(sim_out["spikes_stn"], dt, burn))
    gpe_cv = np.nanmean(_compute_cv(sim_out["spikes_gpe"], dt, burn))
    gpi_cv = np.nanmean(_compute_cv(sim_out["spikes_gpi"], dt, burn))
    mean_cv = float(np.nanmean([stn_cv, gpe_cv, gpi_cv]))

    # -------- beta fraction --------
    rate_stn_t = sim_out["spikes_stn"][burn:].sum(axis=1)
    beta_frac = float(_compute_beta_fraction(rate_stn_t, dt))

    # ====================================================
    # ERROR TERMS
    # ====================================================
    e_rate = (
        _bounded_error(stn_rate, *TARGETS["stn_rate"]) +
        _bounded_error(gpe_rate, *TARGETS["gpe_rate"]) +
        _bounded_error(gpi_rate, *TARGETS["gpi_rate"])
    ) / 3.0

    e_cv = _bounded_error(mean_cv, *TARGETS["cv"])
    e_beta = _bounded_error(beta_frac, *TARGETS["beta_frac"])

    # weights
    w_rate = 3.0
    w_cv   = 2.0
    w_beta = 1.0

    total_error = w_rate * e_rate + w_cv * e_cv + w_beta * e_beta

    score = -float(total_error)

    metrics = {
        "stn_rate": stn_rate,
        "gpe_rate": gpe_rate,
        "gpi_rate": gpi_rate,
        "mean_cv": mean_cv,
        "beta_frac": beta_frac,
        "total_error": total_error,
    }

    return score, metrics
