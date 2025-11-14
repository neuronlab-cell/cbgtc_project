# psd.py
# Centralized PSD helpers (Welch) + beta-band utilities for STN–GPi analyses.

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.signal import welch

def psd_welch(x: np.ndarray, fs_hz: float, nperseg: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch power spectral density.
    x: 1D time series (e.g., population rate)
    fs_hz: sampling frequency in Hz
    nperseg: segment length (defaults to min(4096, len(x)))
    Returns (freqs, power)
    """
    if nperseg is None:
        nperseg = min(4096, len(x))
    f, Pxx = welch(x.astype(np.float32), fs=fs_hz, nperseg=nperseg)
    return f, Pxx

def band_mask(f: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Boolean mask for frequencies lo_hz..hi_hz inclusive."""
    lo, hi = float(lo_hz), float(hi_hz)
    return (f >= lo) & (f <= hi)

def bandpower(f: np.ndarray, Pxx: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    """
    Integrate PSD over a band using the rectangle rule.
    Returns power in the same units as Pxx * Hz.
    """
    m = band_mask(f, lo_hz, hi_hz)
    if not np.any(m):
        return 0.0
    df = np.diff(f[m]).mean() if np.sum(m) > 1 else (f[m][0] if f[m][0] > 0 else 0.0)
    return float(np.sum(Pxx[m]) * (df if df > 0 else 1.0))

def beta_peak(f: np.ndarray, Pxx: np.ndarray, beta_band: Tuple[float, float] = (13.0, 30.0)) -> Tuple[float, float]:
    """
    Find the frequency of the maximum within beta band and its power value.
    Returns (f_peak_hz, power_at_peak). If band is empty, returns (nan, 0.0).
    """
    lo, hi = beta_band
    m = band_mask(f, lo, hi)
    if not np.any(m):
        return float("nan"), 0.0
    idx = np.argmax(Pxx[m])
    f_band = f[m]
    P_band = Pxx[m]
    return float(f_band[idx]), float(P_band[idx])

def psd_and_beta(x: np.ndarray, fs_hz: float,
                 beta_band: Tuple[float, float] = (13.0, 30.0),
                 nperseg: int | None = None) -> dict:
    """
    Convenience: compute PSD, beta peak and beta band power in one call.
    Returns a dict with freqs, power, beta_peak_hz, beta_peak_power, beta_band_power.
    """
    f, P = psd_welch(x, fs_hz, nperseg=nperseg)
    fpk, Ppk = beta_peak(f, P, beta_band=beta_band)
    Pband = bandpower(f, P, beta_band[0], beta_band[1])
    return dict(
        freqs=f,
        power=P,
        beta_peak_hz=fpk,
        beta_peak_power=Ppk,
        beta_band_power=Pband,
        fs_hz=float(fs_hz),
        beta_band=tuple(beta_band),
    )
