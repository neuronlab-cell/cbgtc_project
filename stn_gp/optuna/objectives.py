# objectives.py
# Full paired δ-objective system for STN–GPe modeling.
#
# Implements:
#   • compute_normal_score(theta, sim_out_normal, config)
#   • compute_pathological_score(theta, sim_out_path, config)
#   • compute_combined_score(...)
#
# This file contains: 
#   – No simulation code
#   – No Optuna logic
#   – Pure math + scoring
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
    total_spikes = spikes.sum()
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
    df = np.mean(np.diff(freqs))
    return float(np.sum(psd[mask]) * df)


def beta_sharpness(freqs, psd, f_lo, f_hi):
    """Peak/mean ratio inside beta band."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0
    band = psd[mask]
    return float(np.max(band) / (np.mean(band) + 1e-12))


def compute_beta_synchrony(rate_stn, rate_gpe):
    """
    Simple beta envelope synchrony:
    Pearson correlation between beta-band filtered STN and GPe rates.
    (Here we approximate envelope by absolute of beta-filtered FFT band.)
    """
    # FFT both, isolate beta 13–30 Hz
    def filt_beta(vec):
        n = len(vec)
        fs = 1000.0  # arbitrary; correlation is scale-invariant
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        X = np.fft.rfft(vec - np.mean(vec))
        mask = (freqs >= 13) & (freqs <= 30)
        X_filt = np.zeros_like(X)
        X_filt[mask] = X[mask]
        recon = np.fft.irfft(X_filt, n=n)
        return np.abs(recon)

    eb = filt_beta(rate_stn)
    eg = filt_beta(rate_gpe)

    if eb.std() < 1e-6 or eg.std() < 1e-6:
        return 0.0

    return float(np.corrcoef(eb, eg)[0, 1])


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

    # Desired normal-state firing ranges
    stn_rate_norm_min: float = 5.0
    stn_rate_norm_max: float = 25.0
    gpe_rate_norm_min: float = 20.0
    gpe_rate_norm_max: float = 80.0

    # Desired pathological firing ranges
    stn_rate_pd_min: float = 20.0
    stn_rate_pd_max: float = 45.0
    gpe_rate_pd_min: float = 10.0
    gpe_rate_pd_max: float = 50.0

    # Normal: keep beta ratio modest
    normal_beta_ratio_low: float = 0.3
    normal_beta_ratio_high: float = 1.5
    normal_sharpness_max: float = 2.0

    # PD: reward strong, sharp beta
    pd_beta_ratio_min: float = 2.0
    pd_sharpness_min: float = 3.0

    # Target PSD for pathological regime (normalized)
    target_pd_freqs: Optional[np.ndarray] = None
    target_pd_psd: Optional[np.ndarray] = None

    # Weight terms
    w_rate: float = 1.0
    w_beta_ratio: float = 1.0
    w_beta_sharp: float = 1.0
    w_psd_match: float = 1.0
    w_sync: float = 1.0


# ============================================================
# Normal-state score (δ = 0)
# ============================================================

def compute_normal_score(sim_out: Dict, cfg: PairedObjectiveConfig) -> float:
    """
    sim_out: dict from sim_api for δ=0
    returns: scalar normal-state score (higher is better)
    """

    spikes_stn = sim_out["spikes_stn"]
    spikes_gpe = sim_out["spikes_gpe"]
    V_stn = sim_out["V_stn"]
    dt = sim_out["dt_ms"]
    burn = sim_out["burn_steps"]

    # Slice post-burnin
    stn_sp = _slice_after_burn(spikes_stn, burn)
    gpe_sp = _slice_after_burn(spikes_gpe, burn)
    stn_V = _slice_after_burn(V_stn[:, :], burn)  # (T, N) → (T, N)

    # Make a 1D "LFP" as mean voltage
    lfp = np.mean(stn_V, axis=1)

    # ---------- firing rates ----------
    stn_rate = compute_population_rate(stn_sp, dt)
    gpe_rate = compute_population_rate(gpe_sp, dt)

    # rate score: 1.0 = inside range, otherwise penalize
    def rate_score(rate, lo, hi):
        if lo <= rate <= hi:
            return 1.0
        # Quadratic penalty
        if rate < lo:
            return max(0.0, 1.0 - ((lo - rate) ** 2) / (lo ** 2))
        else:
            return max(0.0, 1.0 - ((rate - hi) ** 2) / (hi ** 2))

    s_rate = (
        rate_score(stn_rate, cfg.stn_rate_norm_min, cfg.stn_rate_norm_max)
        + rate_score(gpe_rate, cfg.gpe_rate_norm_min, cfg.gpe_rate_norm_max)
    ) * cfg.w_rate

    # ---------- PSD ----------
    freqs, psd = compute_psd(lfp, dt)
    beta_power = band_power(freqs, psd, cfg.beta_lo, cfg.beta_hi)
    neigh_power = band_power(freqs, psd, cfg.neigh_lo, cfg.neigh_hi)
    beta_ratio = beta_power / (neigh_power + 1e-12)

    # Penalty if too strong or too weak
    if cfg.normal_beta_ratio_low <= beta_ratio <= cfg.normal_beta_ratio_high:
        s_ratio = cfg.w_beta_ratio * 1.0
    else:
        # decrease score as ratio deviates
        diff = min(
            abs(beta_ratio - cfg.normal_beta_ratio_low),
            abs(beta_ratio - cfg.normal_beta_ratio_high),
        )
        s_ratio = cfg.w_beta_ratio * max(0.0, 1.0 - diff)

    # ---------- sharpness ----------
    sharp = beta_sharpness(freqs, psd, cfg.beta_lo, cfg.beta_hi)
    if sharp <= cfg.normal_sharpness_max:
        s_sharp = cfg.w_beta_sharp * 1.0
    else:
        # Penalize high sharpness (pathological)
        diff = sharp - cfg.normal_sharpness_max
        s_sharp = cfg.w_beta_sharp * max(0.0, 1.0 - diff * 0.5)

    # ---------- synchrony: want moderate (not too high) ----------
    # Build population rates over time
    rate_stn_t = stn_sp.sum(axis=1)
    rate_gpe_t = gpe_sp.sum(axis=1)
    sync = compute_beta_synchrony(rate_stn_t, rate_gpe_t)
    # Reward moderate correlation (0.2–0.5)
    if 0.2 <= sync <= 0.5:
        s_sync = cfg.w_sync * 1.0
    else:
        diff = min(abs(sync - 0.2), abs(sync - 0.5))
        s_sync = cfg.w_sync * max(0.0, 1.0 - diff * 2.0)

    # ---------- total normal score ----------
    return float(s_rate + s_ratio + s_sharp + s_sync)


# ============================================================
# Pathological-state score (δ = 1)
# ============================================================

def compute_pathological_score(sim_out: Dict, cfg: PairedObjectiveConfig) -> float:
    """
    sim_out: dict from sim_api for δ=1
    returns: scalar pathological-state score (higher = more PD-like)
    """

    spikes_stn = sim_out["spikes_stn"]
    spikes_gpe = sim_out["spikes_gpe"]
    V_stn = sim_out["V_stn"]
    dt = sim_out["dt_ms"]
    burn = sim_out["burn_steps"]

    # Slice post-burnin
    stn_sp = _slice_after_burn(spikes_stn, burn)
    gpe_sp = _slice_after_burn(spikes_gpe, burn)
    stn_V = _slice_after_burn(V_stn, burn)
    lfp = np.mean(stn_V, axis=1)

    # ---------- firing rates (PD ranges) ----------
    stn_rate = compute_population_rate(stn_sp, dt)
    gpe_rate = compute_population_rate(gpe_sp, dt)

    def pd_rate_score(rate, lo, hi):
        if lo <= rate <= hi:
            return 1.0
        diff = min(abs(rate - lo), abs(rate - hi))
        return max(0.0, 1.0 - diff / max(1.0, hi))

    s_rate = cfg.w_rate * (
        pd_rate_score(stn_rate, cfg.stn_rate_pd_min, cfg.stn_rate_pd_max)
        + pd_rate_score(gpe_rate, cfg.gpe_rate_pd_min, cfg.gpe_rate_pd_max)
    )

    # ---------- PSD ----------
    freqs, psd = compute_psd(lfp, dt)
    beta_power = band_power(freqs, psd, cfg.beta_lo, cfg.beta_hi)
    neigh_power = band_power(freqs, psd, cfg.neigh_lo, cfg.neigh_hi)
    beta_ratio = beta_power / (neigh_power + 1e-12)

    # PD: want strong beta dominance
    if beta_ratio >= cfg.pd_beta_ratio_min:
        s_ratio = cfg.w_beta_ratio * 1.0
    else:
        diff = cfg.pd_beta_ratio_min - beta_ratio
        s_ratio = cfg.w_beta_ratio * max(0.0, 1.0 - diff)

    # ---------- sharpness ----------
    sharp = beta_sharpness(freqs, psd, cfg.beta_lo, cfg.beta_hi)
    if sharp >= cfg.pd_sharpness_min:
        s_sharp = cfg.w_beta_sharp * 1.0
    else:
        diff = cfg.pd_sharpness_min - sharp
        s_sharp = cfg.w_beta_sharp * max(0.0, 1.0 - diff)

    # ---------- synchrony ----------
    rate_stn_t = stn_sp.sum(axis=1)
    rate_gpe_t = gpe_sp.sum(axis=1)
    sync = compute_beta_synchrony(rate_stn_t, rate_gpe_t)
    # PD: want strong synchrony (>0.6)
    if sync >= 0.6:
        s_sync = cfg.w_sync * 1.0
    else:
        diff = 0.6 - sync
        s_sync = cfg.w_sync * max(0.0, 1.0 - diff * 2.0)

    # ---------- PSD similarity ----------
    if cfg.target_pd_freqs is not None and cfg.target_pd_psd is not None:
        # Normalize both
        t_freqs = cfg.target_pd_freqs
        t_psd = cfg.target_pd_psd / (np.trapz(cfg.target_pd_psd, t_freqs) + 1e-12)
        psd_i = psd / (np.trapz(psd, freqs) + 1e-12)

        # Interpolate target PSD onto sim freqs
        t_interp = np.interp(freqs, t_freqs, t_psd)
        mse = np.mean((psd_i - t_interp) ** 2)
        # Score ~ exp(-mse)
        s_match = cfg.w_psd_match * float(np.exp(-3.0 * mse))
    else:
        s_match = 0.0

    # ---------- total pathological score ----------
    return float(s_rate + s_ratio + s_sharp + s_sync + s_match)


# ============================================================
# Combined score for Optuna
# ============================================================

def compute_combined_score(
    normal_score: float,
    path_score: float,
    gamma_normal: float = 1.0,
    gamma_path: float = 1.0,
) -> float:
    """
    Weighted sum of normal and pathological objectives.
    Optuna will maximize this.
    """
    return float(gamma_normal * normal_score + gamma_path * path_score)
