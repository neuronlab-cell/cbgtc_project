# STUB: compute PSD, beta peak, phase lag

# metrics.py
# Lightweight analysis utilities for the STN–GPi loop:
# - population rates from spike matrices
# - Welch PSD + beta-peak finder
# - phase lag between STN and GPi population activity (via cross-correlation)

from __future__ import annotations
import numpy as np
from scipy.signal import welch, correlate
from typing import Tuple, Optional

def population_rate(spike_mat: np.ndarray, fs_hz: float) -> np.ndarray:
    """
    Convert a binary spike matrix (T, N) to a population rate trace (spikes/s).
    fs_hz: sampling frequency (steps per second) = 1000/dt_ms
    """
    assert spike_mat.ndim == 2, "spike_mat must be (T, N)"
    spikes_per_step = spike_mat.sum(axis=1).astype(np.float32)   # spikes per time step
    return spikes_per_step * fs_hz  # spikes/s

def psd_welch(x: np.ndarray, fs_hz: float, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD with sensible defaults for our time series.
    """
    if nperseg is None:
        nperseg = min(4096, len(x))
    f, Pxx = welch(x, fs=fs_hz, nperseg=nperseg)
    return f, Pxx

def find_beta_peak(f: np.ndarray, Pxx: np.ndarray, beta_band: Tuple[float, float] = (13.0, 30.0)) -> Tuple[float, float]:
    """
    Return (f_peak, power_at_peak) within the specified beta band.
    If no clear peak, returns (np.nan, 0.0).
    """
    lo, hi = beta_band
    band = (f >= lo) & (f <= hi)
    if not np.any(band):
        return np.nan, 0.0
    idx = np.argmax(Pxx[band])
    f_band = f[band]
    P_band = Pxx[band]
    return float(f_band[idx]), float(P_band[idx])

def phase_lag_ms(x: np.ndarray, y: np.ndarray, fs_hz: float, max_lag_ms: float = 100.0) -> float:
    """
    Estimate phase/timing lag between two population signals using cross-correlation.
    Positive lag (ms) means x LEADS y by that many milliseconds.
    """
    assert x.ndim == y.ndim == 1 and len(x) == len(y), "signals must be 1D and equal length"
    x = (x - x.mean()).astype(np.float32)
    y = (y - y.mean()).astype(np.float32)
    max_lag_steps = int(round(max_lag_ms * fs_hz / 1000.0))
    corr = correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x))
    # restrict search to +/- max_lag
    mask = (lags >= -max_lag_steps) & (lags <= max_lag_steps)
    corr_m = corr[mask]
    lags_m = lags[mask]
    best_idx = int(np.argmax(corr_m))
    best_lag_steps = lags_m[best_idx]
    # convert to ms; sign convention: positive means x leads y
    return float(best_lag_steps * 1000.0 / fs_hz)

def beta_metrics_from_spikes(spk_stn: np.ndarray, spk_gpi: np.ndarray, fs_hz: float) -> dict:
    """
    Convenience: compute PSD and beta peak from STN & GPi population rates,
    plus cross-correlation-based phase lag (STN vs GPi).
    """
    r_stn = population_rate(spk_stn, fs_hz)
    r_gpi = population_rate(spk_gpi, fs_hz)
    f1, Pstn = psd_welch(r_stn, fs_hz)
    f2, Pgpi = psd_welch(r_gpi, fs_hz)
    fpk_stn, Ppk_stn = find_beta_peak(f1, Pstn)
    fpk_gpi, Ppk_gpi = find_beta_peak(f2, Pgpi)
    lag_ms = phase_lag_ms(r_stn, r_gpi, fs_hz, max_lag_ms=100.0)
    return dict(
        fs_hz=float(fs_hz),
        stn=dict(beta_peak_hz=float(fpk_stn), beta_power=float(Ppk_stn)),
        gpi=dict(beta_peak_hz=float(fpk_gpi), beta_power=float(Ppk_gpi)),
        stn_leads_gpi_ms=float(lag_ms)
    )
