# stn_gp/optuna/objectives.py
# Normal-only objective for STN–GPe–GPi network.
#
# Metrics:
#   • Firing rate per nucleus (Hz)
#   • ISI CV per nucleus
#   • STN beta fraction (13–30 Hz / 1–80 Hz) from population spike rate
#
# We define physiological "normal" bands and penalize deviations.
# Optuna should MAXIMIZE the score, which is -loss where:
#   loss = Σ w_k * (normalized_range_error_k)^2

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


# ============================================================
# Normal targets (hand-tuned from literature + your table)
# ============================================================

TARGETS = {
    # Firing rate (Hz)
    "stn_rate": (15.0, 25.0),
    "gpe_rate": (60.0, 80.0),
    "gpi_rate": (60.0, 90.0),

    # CV (ISI coefficient of variation)
    "stn_cv": (0.7, 1.0),
    "gpe_cv": (1.0, 1.4),
    "gpi_cv": (0.6, 0.9),

    # STN beta fraction (13–30 Hz / 1–80 Hz)
    "beta_frac": (0.03, 0.10),
}


# Relative importance of each metric in the loss
WEIGHTS = {
    "stn_rate": 2.0,
    "gpe_rate": 2.0,
    "gpi_rate": 2.0,
    "stn_cv": 1.5,
    "gpe_cv": 1.5,
    "gpi_cv": 1.5,
    "beta_frac": 1.0,
}


# ============================================================
# Helper utilities
# ============================================================

def _range_error(value: float, lo: float, hi: float) -> float:
    """
    Return a normalized distance from [lo, hi]:
      - 0 if value is inside [lo, hi]
      - grows linearly as you move away, normalized by band width.
    """
    if hi <= lo:
        return 0.0
    if lo <= value <= hi:
        return 0.0
    if value < lo:
        d = lo - value
    else:
        d = value - hi
    width = hi - lo
    return float(d / width)


def _compute_firing_rates(spikes: np.ndarray, dt_ms: float, burn_steps: int) -> np.ndarray:
    """
    spikes: (T, N) binary
    returns: per-neuron rates (Hz), shape (N,)
    """
    sp = spikes[burn_steps:]
    T_eff = sp.shape[0]
    if T_eff <= 0:
        return np.zeros(spikes.shape[1], dtype=float)

    dur_s = T_eff * dt_ms / 1000.0
    return sp.sum(axis=0) / dur_s


def _compute_cv(spikes: np.ndarray, dt_ms: float, burn_steps: int) -> np.ndarray:
    """
    spikes: (T, N) binary
    returns: per-neuron ISI CV, shape (N,)
            NaN for cells with too few spikes.
    """
    sp = spikes[burn_steps:]
    T_eff, N = sp.shape
    cvs = np.zeros(N, dtype=float)

    for i in range(N):
        t_idx = np.where(sp[:, i] > 0)[0]
        # Require at least a few spikes to define ISIs
        if t_idx.size < 5:
            cvs[i] = np.nan
            continue
        isi_ms = np.diff(t_idx) * dt_ms
        mean_isi = np.mean(isi_ms)
        if mean_isi <= 0:
            cvs[i] = np.nan
        else:
            cvs[i] = float(np.std(isi_ms) / mean_isi)

    return cvs


def _compute_beta_fraction(rate_t: np.ndarray, dt_ms: float) -> float:
    """
    rate_t: 1D population spike count per bin (post-burn-in), shape (T,)
    returns beta_fraction = P_beta(13–30 Hz) / P_broad(1–80 Hz)
    """
    x = rate_t.astype(float)
    x = x - np.mean(x)
    n = x.shape[0]
    if n <= 1:
        return 0.0

    fs = 1000.0 / dt_ms
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x)
    psd = np.abs(X) ** 2

    def band_power(f1: float, f2: float) -> float:
        mask = (freqs >= f1) & (freqs <= f2)
        if not np.any(mask):
            return 0.0
        # simple sum is OK for relative fractions
        return float(np.sum(psd[mask]))

    beta_pow = band_power(13.0, 30.0)
    broad_pow = band_power(1.0, 80.0)

    if broad_pow <= 0.0:
        return 0.0
    return float(beta_pow / broad_pow)


# ============================================================
# Main evaluation function for Optuna
# ============================================================

def evaluate_normal_model(sim_out: Dict) -> Tuple[float, Dict]:
    """
    Evaluate a single normal (δ=0) simulation.

    sim_out must contain:
        "spikes_stn": (T, N_stn) binary
        "spikes_gpe": (T, N_gpe) binary
        "spikes_gpi": (T, N_gpi) binary
        "dt_ms": float
        "burn_steps": int

    Returns:
        score (float)  : higher is better (Optuna maximizes)
        metrics (dict) : actual measured values for logging
    """
    spikes_stn = sim_out["spikes_stn"]
    spikes_gpe = sim_out["spikes_gpe"]
    spikes_gpi = sim_out["spikes_gpi"]
    dt_ms = float(sim_out["dt_ms"])
    burn_steps = int(sim_out["burn_steps"])

    # -------------------------
    # Firing rates
    # -------------------------
    stn_rates = _compute_firing_rates(spikes_stn, dt_ms, burn_steps)
    gpe_rates = _compute_firing_rates(spikes_gpe, dt_ms, burn_steps)
    gpi_rates = _compute_firing_rates(spikes_gpi, dt_ms, burn_steps)

    stn_rate_mean = float(np.mean(stn_rates)) if stn_rates.size else 0.0
    gpe_rate_mean = float(np.mean(gpe_rates)) if gpe_rates.size else 0.0
    gpi_rate_mean = float(np.mean(gpi_rates)) if gpi_rates.size else 0.0

    # -------------------------
    # CV (ISI irregularity)
    # -------------------------
    stn_cvs = _compute_cv(spikes_stn, dt_ms, burn_steps)
    gpe_cvs = _compute_cv(spikes_gpe, dt_ms, burn_steps)
    gpi_cvs = _compute_cv(spikes_gpi, dt_ms, burn_steps)

    # nanmean ignores silent / low-spike cells
    stn_cv_mean = float(np.nanmean(stn_cvs)) if np.isfinite(stn_cvs).any() else 0.0
    gpe_cv_mean = float(np.nanmean(gpe_cvs)) if np.isfinite(gpe_cvs).any() else 0.0
    gpi_cv_mean = float(np.nanmean(gpi_cvs)) if np.isfinite(gpi_cvs).any() else 0.0

    # -------------------------
    # STN beta fraction
    # -------------------------
    # Population spike count over time, post-burnin
    stn_pop_rate_t = spikes_stn[burn_steps:].sum(axis=1)
    beta_frac = _compute_beta_fraction(stn_pop_rate_t, dt_ms)

    # -------------------------
    # Build metrics dict
    # -------------------------
    metrics = {
        "stn_rate": stn_rate_mean,
        "gpe_rate": gpe_rate_mean,
        "gpi_rate": gpi_rate_mean,
        "stn_cv": stn_cv_mean,
        "gpe_cv": gpe_cv_mean,
        "gpi_cv": gpi_cv_mean,
        "beta_frac": beta_frac,
    }

    # -------------------------
    # Loss = Σ w * (normalized range error)^2
    # -------------------------
    loss = 0.0

    for key, value in metrics.items():
        lo, hi = TARGETS[key]
        err = _range_error(value, lo, hi)
        w = WEIGHTS.get(key, 1.0)
        loss += w * (err ** 2)

    # Score = -loss (higher is better)
    score = -float(loss)
    return score, metrics
