# objectives.py
# Paired δ-objective system for STN–GPe modeling.
#
# Implements:
#   • compute_normal_score(sim_out, cfg)
#   • compute_pathological_score(sim_out, cfg)
#   • compute_combined_score(...)
#
# Key design features:
#   1) Strong firing-rate band penalties (normal vs PD ranges)
#   2) Infrastructure for explicit δ-contrast (normal vs PD metrics)
#   3) Tight beta logic (ratio + sharpness, different in normal vs PD)
#   5) Rebalanced weights to make the objective surface less flat
#
# sim_api.py must provide sim_out dicts with:
#   {
#       "spikes_stn": (T, N_stn) binary,
#       "spikes_gpe": (T, N_gpe) binary,
#       "V_stn": (T, N_stn) voltages,
#       "V_gpe": (T, N_gpe) voltages,
#       "dt_ms": float,
#       "burn_steps": int,
#   }

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np


# ============================================================
# Helper utilities
# ============================================================

def _slice_after_burn(arr: np.ndarray, burn_steps: int) -> np.ndarray:
    """Return arr[burn_steps:] safely."""
    if burn_steps <= 0:
        return arr
    if burn_steps >= arr.shape[0]:
        return arr[0:0]
    return arr[burn_steps:]


def compute_population_rate(spikes: np.ndarray, dt_ms: float) -> float:
    """
    spikes: (T, N) binary
    returns mean rate per neuron (Hz)
    """
    if spikes.size == 0:
        return 0.0
    T = spikes.shape[0]
    dur_s = T * (dt_ms / 1000.0)
    total_spikes = float(spikes.sum())
    n_neurons = spikes.shape[1]
    return (total_spikes / n_neurons) / dur_s


def compute_psd(signal: np.ndarray, dt_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-sided FFT PSD.
    """
    x = signal - np.mean(signal)
    n = x.shape[0]
    if n <= 1:
        return np.array([0.0]), np.array([0.0])
    n_fft = int(2 ** np.ceil(np.log2(n)))
    x_padded = np.zeros(n_fft)
    x_padded[:n] = x
    fs = 1000.0 / dt_ms
    X = np.fft.rfft(x_padded)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    psd = (np.abs(X) ** 2) / (fs * n_fft)
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, f1: float, f2: float) -> float:
    mask = (freqs >= f1) & (freqs <= f2)
    if not np.any(mask):
        return 0.0
    df = float(np.mean(np.diff(freqs)))
    return float(np.sum(psd[mask]) * df)


def beta_sharpness(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Peak/mean ratio inside beta band."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0
    band = psd[mask]
    return float(np.max(band) / (np.mean(band) + 1e-12))


def compute_beta_synchrony(rate_stn: np.ndarray, rate_gpe: np.ndarray) -> float:
    """
    Simple beta envelope synchrony:
    Pearson correlation between beta-band filtered STN and GPe population rate signals.
    """
    def filt_beta(vec: np.ndarray) -> np.ndarray:
        n = len(vec)
        if n <= 1:
            return np.zeros_like(vec)
        fs = 1000.0  # arbitrary; correlation is scale-invariant
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        X = np.fft.rfft(vec - np.mean(vec))
        mask = (freqs >= 13.0) & (freqs <= 30.0)
        X_filt = np.zeros_like(X)
        X_filt[mask] = X[mask]
        recon = np.fft.irfft(X_filt, n=n)
        return np.abs(recon)

    eb = filt_beta(rate_stn)
    eg = filt_beta(rate_gpe)

    if eb.std() < 1e-6 or eg.std() < 1e-6:
        return 0.0

    return float(np.corrcoef(eb, eg)[0, 1])


def _rate_band_score(rate: float, lo: float, hi: float, curvature: float = 2.0) -> float:
    """
    Band-style reward: 1 if rate in [lo, hi], then smoothly decays outside.
    curvature controls how fast it drops (higher = harsher).
    """
    if hi <= lo:
        return 0.0
    if lo <= rate <= hi:
        return 1.0
    # distance from nearest boundary
    if rate < lo:
        d = (lo - rate) / max(lo, 1.0)
    else:
        d = (rate - hi) / max(hi, 1.0)
    return float(max(0.0, 1.0 - (d ** curvature)))


# ============================================================
# Objective configuration dataclass
# ============================================================

@dataclass
class PairedObjectiveConfig:
    # Basic frequency bands
    beta_lo: float = 13.0
    beta_hi: float = 30.0
    neigh_lo: float = 5.0
    neigh_hi: float = 60.0

    # Normal-state firing ranges
    stn_rate_norm_min: float = 10.0
    stn_rate_norm_max: float = 25.0
    gpe_rate_norm_min: float = 40.0
    gpe_rate_norm_max: float = 80.0

    # Pathological-state firing ranges
    stn_rate_pd_min: float = 20.0
    stn_rate_pd_max: float = 45.0
    gpe_rate_pd_min: float = 10.0
    gpe_rate_pd_max: float = 50.0

    # Normal: beta modest, broad
    normal_beta_ratio_low: float = 0.3
    normal_beta_ratio_high: float = 1.2
    normal_sharpness_max: float = 2.0

    # PD: strong, sharp beta
    pd_beta_ratio_min: float = 2.0
    pd_sharpness_min: float = 3.0

    # Synchrony targets
    sync_norm_lo: float = 0.2
    sync_norm_hi: float = 0.5
    sync_pd_min: float = 0.6

    # Weight terms (rebalanced)
    w_rate: float = 2.0
    w_beta_ratio: float = 2.0
    w_beta_sharp: float = 2.0
    w_sync: float = 1.0

    # Explicit δ-contrast weight (used in compute_combined_score if metrics are provided)
    w_delta_contrast: float = 3.0


# ============================================================
# Core analysis for one condition (normal or PD)
# ============================================================

def _analyze_condition(sim_out: Dict, cfg: PairedObjectiveConfig, mode: str) -> Tuple[float, Dict]:
    """
    Analyze a single condition (normal or PD) and return:
      - score (higher is better)
      - metrics dict used for later δ-contrast
    mode: 'normal' or 'pd'
    """
    assert mode in ("normal", "pd")

    spikes_stn = sim_out["spikes_stn"]
    spikes_gpe = sim_out["spikes_gpe"]
    V_stn = sim_out["V_stn"]
    dt = float(sim_out["dt_ms"])
    burn = int(sim_out["burn_steps"])

    # Slice post-burnin
    stn_sp = _slice_after_burn(spikes_stn, burn)
    gpe_sp = _slice_after_burn(spikes_gpe, burn)
    stn_V = _slice_after_burn(V_stn, burn)
    if stn_V.size == 0:
        # Degenerate case: no data after burn-in
        return 0.0, {
            "stn_rate": 0.0,
            "gpe_rate": 0.0,
            "beta_ratio": 0.0,
            "beta_sharp": 0.0,
            "sync": 0.0,
        }

    # LFP-like signal from STN
    lfp = np.mean(stn_V, axis=1)

    # ---------- firing rates ----------
    stn_rate = compute_population_rate(stn_sp, dt)
    gpe_rate = compute_population_rate(gpe_sp, dt)

    if mode == "normal":
        s_rate_stn = _rate_band_score(
            stn_rate,
            cfg.stn_rate_norm_min,
            cfg.stn_rate_norm_max,
            curvature=2.5,
        )
        s_rate_gpe = _rate_band_score(
            gpe_rate,
            cfg.gpe_rate_norm_min,
            cfg.gpe_rate_norm_max,
            curvature=2.5,
        )
    else:  # PD
        s_rate_stn = _rate_band_score(
            stn_rate,
            cfg.stn_rate_pd_min,
            cfg.stn_rate_pd_max,
            curvature=2.5,
        )
        s_rate_gpe = _rate_band_score(
            gpe_rate,
            cfg.gpe_rate_pd_min,
            cfg.gpe_rate_pd_max,
            curvature=2.5,
        )

    s_rate = cfg.w_rate * (s_rate_stn + s_rate_gpe) / 2.0

    # ---------- PSD and beta metrics ----------
    freqs, psd = compute_psd(lfp, dt)
    beta_power = band_power(freqs, psd, cfg.beta_lo, cfg.beta_hi)
    neigh_power = band_power(freqs, psd, cfg.neigh_lo, cfg.neigh_hi)
    beta_ratio = beta_power / (neigh_power + 1e-12)
    sharp = beta_sharpness(freqs, psd, cfg.beta_lo, cfg.beta_hi)

    if mode == "normal":
        # Ratio: reward being inside [low, high]; penalize both too low & too high
        if cfg.normal_beta_ratio_low <= beta_ratio <= cfg.normal_beta_ratio_high:
            s_ratio = 1.0
        else:
            # distance to closest bound, scaled
            d = min(
                abs(beta_ratio - cfg.normal_beta_ratio_low),
                abs(beta_ratio - cfg.normal_beta_ratio_high),
            )
            s_ratio = max(0.0, 1.0 - d)
        s_ratio *= cfg.w_beta_ratio

        # Sharpness: penalize sharp peaks (we want broad beta / 1/f-ish)
        if sharp <= cfg.normal_sharpness_max:
            s_sharp = 1.0
        else:
            d = sharp - cfg.normal_sharpness_max
            s_sharp = max(0.0, 1.0 - 0.5 * d)
        s_sharp *= cfg.w_beta_sharp

    else:  # PD
        # Ratio: want strong beta dominance
        if beta_ratio >= cfg.pd_beta_ratio_min:
            s_ratio = 1.0
        else:
            d = cfg.pd_beta_ratio_min - beta_ratio
            s_ratio = max(0.0, 1.0 - d)
        s_ratio *= cfg.w_beta_ratio

        # Sharpness: want sharp peak
        if sharp >= cfg.pd_sharpness_min:
            s_sharp = 1.0
        else:
            d = cfg.pd_sharpness_min - sharp
            s_sharp = max(0.0, 1.0 - d)
        s_sharp *= cfg.w_beta_sharp

    # ---------- synchrony ----------
    rate_stn_t = stn_sp.sum(axis=1)
    rate_gpe_t = gpe_sp.sum(axis=1)
    sync = compute_beta_synchrony(rate_stn_t, rate_gpe_t)

    if mode == "normal":
        # Reward moderate synchrony (not too low, not rigid)
        if cfg.sync_norm_lo <= sync <= cfg.sync_norm_hi:
            s_sync = 1.0
        else:
            d = min(abs(sync - cfg.sync_norm_lo), abs(sync - cfg.sync_norm_hi))
            s_sync = max(0.0, 1.0 - 2.0 * d)
    else:
        # PD: reward strong synchrony
        if sync >= cfg.sync_pd_min:
            s_sync = 1.0
        else:
            d = cfg.sync_pd_min - sync
            s_sync = max(0.0, 1.0 - 2.0 * d)
    s_sync *= cfg.w_sync

    # ---------- total score for this condition ----------
    score = float(s_rate + s_ratio + s_sharp + s_sync)

    metrics = {
        "stn_rate": float(stn_rate),
        "gpe_rate": float(gpe_rate),
        "beta_ratio": float(beta_ratio),
        "beta_sharp": float(sharp),
        "sync": float(sync),
    }

    return score, metrics


# ============================================================
# Public scoring functions (per condition)
# ============================================================

def compute_normal_score(sim_out: Dict, cfg: PairedObjectiveConfig) -> float:
    """
    Compute normal-state score (δ = 0).
    Higher is better.
    """
    score, _ = _analyze_condition(sim_out, cfg, mode="normal")
    return score


def compute_pathological_score(sim_out: Dict, cfg: PairedObjectiveConfig) -> float:
    """
    Compute pathological-state score (δ = 1).
    Higher is better (more PD-like).
    """
    score, _ = _analyze_condition(sim_out, cfg, mode="pd")
    return score


# ============================================================
# Combined score for Optuna (with optional δ-contrast)
# ============================================================

def compute_combined_score(
    normal_score: float,
    path_score: float,
    gamma_normal: float = 1.0,
    gamma_path: float = 1.0,
    cfg: Optional[PairedObjectiveConfig] = None,
    normal_metrics: Optional[Dict] = None,
    path_metrics: Optional[Dict] = None,
) -> float:
    """
    Weighted sum of normal and pathological objectives, with optional δ-contrast term.

    If normal_metrics and path_metrics are provided (from _analyze_condition),
    and cfg.w_delta_contrast > 0, we add a contrast reward that prefers:
        beta_ratio_pd  >> beta_ratio_normal
        beta_sharp_pd  >> beta_sharp_normal
    """
    base = float(gamma_normal * normal_score + gamma_path * path_score)

    if cfg is None or cfg.w_delta_contrast <= 0.0:
        return base

    if normal_metrics is None or path_metrics is None:
        return base

    # Δ-beta ratio: reward when PD beta ratio exceeds normal beta ratio
    delta_beta = max(0.0, path_metrics["beta_ratio"] - normal_metrics["beta_ratio"])

    # Δ-sharpness: reward when PD beta sharpness exceeds normal
    delta_sharp = max(0.0, path_metrics["beta_sharp"] - normal_metrics["beta_sharp"])

    contrast = delta_beta + 0.5 * delta_sharp
    return float(base + cfg.w_delta_contrast * contrast)
