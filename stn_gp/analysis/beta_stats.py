# stn_gp/analysis/beta_stats.py
#
# Usage (from repo root, venv active):
#   python -m stn_gp.analysis.beta_stats /path/to/run_dir
#
# Example:
#   python -m stn_gp.analysis.beta_stats \
#       /home/ubuntu/cbgtc_project/runs/run_20251119_024724
#
# Computes:
#   - STN, GPe, GPi beta-band power (13–30 Hz) and beta/total(1–80 Hz) ratios
#   - STN–GPe beta-band coherence (mean + max in 13–30 Hz)
#   - STN→GPe phase lag at the peak beta coherence frequency

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch, coherence, csd


# ---------------------------
# Helpers
# ---------------------------

def _load_lfp(pop_name: str, arrays_dir: Path, burn_steps: int) -> np.ndarray:
    """
    Load V_<pop>.npy, drop burn-in, compute mean across neurons.

    Returns
    -------
    lfp : np.ndarray, shape (T_eff,)
        LFP-like signal (mean membrane voltage).
    """
    path = arrays_dir / f"V_{pop_name}.npy"
    if not path.exists():
        print(f"[WARN] {path.name} not found in {arrays_dir}")
        return np.array([], dtype=float)

    V = np.load(path)  # shape (T, N)
    if V.ndim != 2:
        raise ValueError(f"{path.name} must be 2D (T, N), got shape {V.shape}.")

    if burn_steps >= V.shape[0]:
        print(f"[WARN] burn_steps ({burn_steps}) >= T ({V.shape[0]}), no data left.")
        return np.array([], dtype=float)

    V_win = V[burn_steps:, :]  # drop burn-in
    lfp = np.mean(V_win, axis=1)  # (T_eff,)
    return lfp.astype(float)


def _band_power(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    """
    Integrate PSD over [f_lo, f_hi] using a simple Riemann sum.
    """
    f = np.asarray(freqs, float).reshape(-1)
    p = np.asarray(psd, float).reshape(-1)
    if f.size < 2 or p.size != f.size:
        return 0.0

    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return 0.0

    df = np.mean(np.diff(f))
    return float(np.sum(p[mask]) * df)


def _psd_basic(signal: np.ndarray, fs_hz: float):
    """
    Basic PSD wrapper around scipy.signal.welch.
    """
    x = np.asarray(signal, float).reshape(-1)
    if x.size <= 1:
        return np.array([0.0]), np.array([0.0])

    # Detrend by mean
    x = x - np.mean(x)

    nperseg = min(4096, x.size)
    f, Pxx = welch(x, fs=fs_hz, nperseg=nperseg)
    return f, Pxx


def _coherence_and_phase(x: np.ndarray, y: np.ndarray, fs_hz: float):
    """
    Compute:
      - magnitude-squared coherence Cxy(f)
      - cross-spectrum phase angle phi(f) (radians)

    Returns
    -------
    f : np.ndarray
        Frequency axis (Hz).
    Cxy : np.ndarray
        Coherence (0..1).
    phi : np.ndarray
        Phase of cross-spectrum (radians), same shape as f.
    """
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    n = min(x.size, y.size)

    if n <= 1:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])

    nperseg = min(4096, n)

    # Coherence
    f_coh, Cxy = coherence(x, y, fs=fs_hz, nperseg=nperseg)

    # Cross-spectrum for phase
    f_csd, Sxy = csd(x, y, fs=fs_hz, nperseg=nperseg)
    if f_csd.shape != f_coh.shape:
        raise RuntimeError("Frequency axes from coherence and csd differ unexpectedly.")

    phi = np.angle(Sxy)  # radians
    return f_coh, Cxy, phi


# ---------------------------
# Main analysis
# ---------------------------

def summarize_beta_power(pop_name: str, lfp: np.ndarray, fs_hz: float) -> None:
    """
    Print beta-band and total-band power for a given population LFP.
    """
    if lfp.size <= 1:
        print(f"\n=== {pop_name}: insufficient LFP samples ===")
        return

    freqs, Pxx = _psd_basic(lfp, fs_hz)

    # Define bands
    beta_lo, beta_hi = 13.0, 30.0
    total_lo, total_hi = 1.0, 80.0

    beta_power = _band_power(freqs, Pxx, beta_lo, beta_hi)
    total_power = _band_power(freqs, Pxx, total_lo, total_hi)

    ratio = beta_power / total_power if total_power > 0.0 else 0.0

    print(f"\n=== {pop_name} beta power ===")
    print(f"Band (Hz)       : {beta_lo:.1f}–{beta_hi:.1f}")
    print(f"Beta power      : {beta_power:.4e}")
    print(f"Total 1–80 Hz   : {total_power:.4e}")
    print(f"Beta / total    : {ratio:.4f}")


def summarize_stn_gpe_coupling(stn_lfp: np.ndarray, gpe_lfp: np.ndarray, fs_hz: float) -> None:
    """
    Compute and print STN–GPe coherence and phase lag in the beta band.
    """
    if stn_lfp.size <= 1 or gpe_lfp.size <= 1:
        print("\n=== STN–GPe coupling: insufficient data ===")
        return

    f, Cxy, phi = _coherence_and_phase(stn_lfp, gpe_lfp, fs_hz)

    beta_lo, beta_hi = 13.0, 30.0
    beta_mask = (f >= beta_lo) & (f <= beta_hi)

    if not np.any(beta_mask):
        print("\n=== STN–GPe coupling ===")
        print("No frequency bins in 13–30 Hz; cannot compute beta coherence.")
        return

    f_beta = f[beta_mask]
    C_beta = Cxy[beta_mask]
    phi_beta = phi[beta_mask]

    # Mean and max coherence in beta
    C_beta_mean = float(np.mean(C_beta))
    max_idx = int(np.argmax(C_beta))
    C_beta_max = float(C_beta[max_idx])
    f_peak = float(f_beta[max_idx])
    phi_peak = float(phi_beta[max_idx])  # radians

    # Convert to degrees and time lag (ms)
    phi_deg = float(np.degrees(phi_peak))
    if f_peak > 0.0:
        tau_sec = phi_peak / (2.0 * np.pi * f_peak)
        tau_ms = float(tau_sec * 1000.0)
    else:
        tau_ms = 0.0

    print("\n=== STN–GPe beta-band coupling (13–30 Hz) ===")
    print(f"Mean coherence          : {C_beta_mean:.3f}")
    print(f"Max coherence           : {C_beta_max:.3f}")
    print(f"Freq @ max coherence    : {f_peak:.2f} Hz")
    print(f"Phase (STN→GPe) @ peak  : {phi_deg:.1f} deg")
    print(f"Approx time lag         : {tau_ms:.2f} ms")
    print("Note: sign convention is STN (x) vs GPe (y) in csd(x, y).")


def main(run_dir_str: str) -> None:
    run_dir = Path(run_dir_str).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    arrays_dir = run_dir / "arrays"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")
    if not arrays_dir.exists():
        raise FileNotFoundError(f"arrays/ directory not found in {run_dir}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    dt_ms = float(manifest["dt_ms"])
    burn_steps = int(manifest.get("burn_steps", 0))
    fs_hz = 1000.0 / dt_ms

    print(f"Run directory      : {run_dir}")
    print(f"dt (ms)            : {dt_ms}")
    print(f"fs (Hz)            : {fs_hz:.2f}")
    print(f"Burn-in steps      : {burn_steps}")

    # Load LFP signals
    stn_lfp = _load_lfp("stn", arrays_dir, burn_steps)
    gpe_lfp = _load_lfp("gpe", arrays_dir, burn_steps)
    gpi_lfp = _load_lfp("gpi", arrays_dir, burn_steps)

    # Beta power summaries
    summarize_beta_power("STN", stn_lfp, fs_hz)
    summarize_beta_power("GPe", gpe_lfp, fs_hz)
    summarize_beta_power("GPi", gpi_lfp, fs_hz)

    # STN–GPe coupling
    summarize_stn_gpe_coupling(stn_lfp, gpe_lfp, fs_hz)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m stn_gp.analysis.beta_stats /path/to/run_dir")
        sys.exit(1)
    main(sys.argv[1])
