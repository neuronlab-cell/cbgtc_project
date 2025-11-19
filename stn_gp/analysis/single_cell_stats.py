# stn_gp/analysis/single_cell_stats.py
#
# Single-cell statistics for STN, GPe, GPi:
#   • CV and CV2 of ISIs
#   • Burstiness (fraction of spikes in bursts)
#   • % oscillatory neurons in beta band (13–30 Hz)
#
# Usage (from repo root, venv active):
#   python -m stn_gp.analysis.single_cell_stats /path/to/run_dir
#
# Example:
#   python -m stn_gp.analysis.single_cell_stats runs/run_20251119_024724

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.signal import welch


# -----------------------------
# Basic spike-time utilities
# -----------------------------

def spikes_to_times(spikes: np.ndarray, dt_ms: float, burn_steps: int) -> np.ndarray:
    """
    Convert a (T, N) binary spike matrix into a list of 1D arrays of spike times (ms).
    Spikes in the first 'burn_steps' are ignored.

    Returns
    -------
    times_per_neuron : np.ndarray of dtype=object, shape (N,)
        Each entry is a 1D float array of spike times in ms.
    """
    T, N = spikes.shape
    # Restrict to post-burn-in window
    spikes_win = spikes[burn_steps:, :]
    t0_ms = burn_steps * dt_ms
    t_vec = np.arange(spikes_win.shape[0], dtype=float) * dt_ms + t0_ms

    times_per_neuron = np.empty(N, dtype=object)
    for j in range(N):
        idx = np.nonzero(spikes_win[:, j])[0]
        times_per_neuron[j] = t_vec[idx]
    return times_per_neuron


def isi_from_times(spike_times_ms: np.ndarray) -> np.ndarray:
    """Return inter-spike intervals (same units as spike_times_ms)."""
    t = np.asarray(spike_times_ms, dtype=float).reshape(-1)
    if t.size < 2:
        return np.zeros(0, dtype=float)
    t_sorted = np.sort(t)
    return np.diff(t_sorted)


# -----------------------------
# CV and CV2
# -----------------------------

def cv_from_isi(isi: np.ndarray) -> float:
    """Coefficient of variation of ISI."""
    isi = np.asarray(isi, dtype=float)
    if isi.size < 2:
        return np.nan
    mu = np.mean(isi)
    if mu <= 0.0:
        return np.nan
    return float(np.std(isi) / mu)


def cv2_from_isi(isi: np.ndarray) -> float:
    """
    Local irregularity measure CV2, defined on consecutive ISIs:
      CV2_i = 2 * |Δ_{i+1} - Δ_i| / (Δ_{i+1} + Δ_i)
    Returns mean CV2 over all valid pairs.
    """
    isi = np.asarray(isi, dtype=float)
    if isi.size < 3:  # need at least 3 spikes → 2 ISIs → 1 pair
        return np.nan
    isi1 = isi[:-1]
    isi2 = isi[1:]
    denom = isi1 + isi2
    valid = denom > 0.0
    if not np.any(valid):
        return np.nan
    cv2_vals = 2.0 * np.abs(isi2[valid] - isi1[valid]) / denom[valid]
    return float(np.mean(cv2_vals))


# -----------------------------
# Burstiness
# -----------------------------

def burst_metrics_from_times(
    spike_times_ms: np.ndarray,
    short_isi_ms: float = 15.0,
) -> Tuple[float, int, int]:
    """
    Simple burst detector based on short ISI threshold.

    A burst is defined as a run of consecutive ISIs < short_isi_ms.
    We count spikes belonging to such runs as "burst spikes".

    Returns
    -------
    burstiness : float
        Fraction of spikes that are part of bursts (0..1). NaN if <2 spikes.
    n_bursts : int
        Number of detected bursts.
    spikes_in_bursts : int
        Total number of spikes that belong to bursts.
    """
    t = np.asarray(spike_times_ms, dtype=float).reshape(-1)
    n_spikes = t.size
    if n_spikes < 2:
        return np.nan, 0, 0

    isi = isi_from_times(t)
    is_short = isi < short_isi_ms

    in_burst = False
    burst_sizes = []
    current_burst_size = 0

    # We think in terms of spike indices. ISI[i] is between spike i and i+1.
    # When we see a short ISI, we join spike i and i+1 into a burst.
    for i, short in enumerate(is_short):
        if short:
            if not in_burst:
                # Start a new burst including spikes i and i+1
                in_burst = True
                current_burst_size = 2
            else:
                # Extend existing burst by including spike i+1
                current_burst_size += 1
        else:
            if in_burst:
                burst_sizes.append(current_burst_size)
                in_burst = False
                current_burst_size = 0

    # If we're still in a burst at the end, close it
    if in_burst and current_burst_size > 0:
        burst_sizes.append(current_burst_size)

    if len(burst_sizes) == 0:
        return 0.0, 0, 0

    spikes_in_bursts = int(np.sum(burst_sizes))
    burstiness = spikes_in_bursts / float(n_spikes)
    return float(burstiness), len(burst_sizes), spikes_in_bursts


# -----------------------------
# Oscillatory classification
# -----------------------------

def per_neuron_rate_signal(
    spikes: np.ndarray,
    dt_ms: float,
    burn_steps: int,
    neuron_index: int,
    smooth_ms: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """
    Build a simple rate-like time series for a single neuron.

    Returns
    -------
    rate : np.ndarray, shape (T_eff,)
        Rate estimate (arbitrary units, proportional to Hz).
    fs_hz : float
        Sampling frequency in Hz.
    """
    spikes_win = spikes[burn_steps:, neuron_index].astype(float)
    fs_hz = 1000.0 / dt_ms
    # Scale to something ~Hz (optional; any consistent scale is okay for PSD ratios)
    rate = spikes_win * fs_hz

    # Optional: smooth with a short boxcar to reduce noise
    if smooth_ms > 0.0:
        k = int(round(smooth_ms / dt_ms))
        k = max(k, 1)
        kernel = np.ones(k, dtype=float) / k
        rate = np.convolve(rate, kernel, mode="same")

    return rate, fs_hz


def classify_oscillatory_beta(
    spikes: np.ndarray,
    dt_ms: float,
    burn_steps: int,
    beta_band: Tuple[float, float] = (13.0, 30.0),
    ref_band: Tuple[float, float] = (5.0, 80.0),
    power_ratio_thresh: float = 3.0,
) -> Tuple[bool, float]:
    """
    Classify a neuron as beta-oscillatory based on its rate PSD.

    Criteria (simple):
      • Find peak power in beta band
      • Estimate baseline as median PSD in reference band (excluding beta)
      • If peak_beta / baseline >= power_ratio_thresh → oscillatory

    Returns
    -------
    is_osc : bool
        True if neuron classified as beta oscillatory.
    peak_freq_beta : float
        Frequency of the beta peak in Hz (0.0 if not defined).
    """
    rate, fs_hz = per_neuron_rate_signal(spikes, dt_ms, burn_steps, neuron_index=0)  # dummy; will override

    # The function is meant to be called after per-neuron extraction; we’ll override this in a wrapper.
    raise RuntimeError("classify_oscillatory_beta is intended for internal use via classify_population_oscillatory.")


def classify_population_oscillatory(
    spikes: np.ndarray,
    dt_ms: float,
    burn_steps: int,
    beta_band: Tuple[float, float] = (13.0, 30.0),
    ref_band: Tuple[float, float] = (5.0, 80.0),
    power_ratio_thresh: float = 3.0,
    smooth_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each neuron as beta-oscillatory or not.

    Parameters
    ----------
    spikes : np.ndarray, shape (T, N)
        Binary spike matrix.
    dt_ms : float
        Time step in ms.
    burn_steps : int
        Number of initial steps to discard.
    beta_band : (float, float)
        Beta band (Hz), e.g. (13, 30).
    ref_band : (float, float)
        Reference band for baseline power, e.g. (5, 80).
    power_ratio_thresh : float
        Threshold on peak_beta / baseline to call a neuron oscillatory.
    smooth_ms : float
        Smoothing window (ms) for rate estimate.

    Returns
    -------
    is_osc : np.ndarray of bool, shape (N,)
        True for neurons classified as beta-oscillatory.
    peak_freqs : np.ndarray of float, shape (N,)
        Peak beta frequency for each neuron (0.0 for non-oscillatory).
    """
    T, N = spikes.shape
    is_osc = np.zeros(N, dtype=bool)
    peak_freqs = np.zeros(N, dtype=float)

    spikes_win = spikes[burn_steps:, :]
    if spikes_win.shape[0] < 10:
        # Too short to say anything meaningful
        return is_osc, peak_freqs

    fs_hz = 1000.0 / dt_ms

    for j in range(N):
        # Skip very silent neurons
        if spikes_win[:, j].sum() < 5:
            continue

        rate = spikes_win[:, j].astype(float) * fs_hz
        if smooth_ms > 0.0:
            k = int(round(smooth_ms / dt_ms))
            k = max(k, 1)
            kernel = np.ones(k, dtype=float) / k
            rate = np.convolve(rate, kernel, mode="same")

        # PSD via Welch
        nperseg = min(4096, len(rate))
        freqs, psd = welch(rate, fs=fs_hz, nperseg=nperseg)

        beta_lo, beta_hi = beta_band
        ref_lo, ref_hi = ref_band

        beta_mask = (freqs >= beta_lo) & (freqs <= beta_hi)
        ref_mask = (freqs >= ref_lo) & (freqs <= ref_hi)

        # Exclude beta band from reference to avoid double-counting
        ref_mask = ref_mask & (~beta_mask)

        if not np.any(beta_mask) or not np.any(ref_mask):
            continue

        beta_psd = psd[beta_mask]
        ref_psd = psd[ref_mask]

        if beta_psd.size == 0 or ref_psd.size == 0:
            continue

        # Peak in beta
        idx_peak = int(np.argmax(beta_psd))
        peak_power = float(beta_psd[idx_peak])
        peak_freq = float(freqs[beta_mask][idx_peak])

        baseline = float(np.median(ref_psd))
        if baseline <= 0.0:
            continue

        ratio = peak_power / baseline
        if ratio >= power_ratio_thresh:
            is_osc[j] = True
            peak_freqs[j] = peak_freq

    return is_osc, peak_freqs


# -----------------------------
# Population-level summary
# -----------------------------

def summarize_population(
    name: str,
    spikes: np.ndarray,
    dt_ms: float,
    burn_steps: int,
    short_isi_ms: float = 15.0,
) -> None:
    """
    Compute and print CV, CV2, burstiness, and % beta-oscillatory cells for a population.
    """
    if spikes.size == 0:
        print(f"\n{name}: no spikes array.")
        return

    T, N = spikes.shape
    print(f"\n===== {name} population (N={N}) =====")

    times_per_neuron = spikes_to_times(spikes, dt_ms, burn_steps)

    cv_list = []
    cv2_list = []
    burstiness_list = []

    for j in range(N):
        t = times_per_neuron[j]
        isi = isi_from_times(t)
        cv = cv_from_isi(isi)
        cv2 = cv2_from_isi(isi)
        b, n_bursts, spikes_in_bursts = burst_metrics_from_times(t, short_isi_ms=short_isi_ms)

        cv_list.append(cv)
        cv2_list.append(cv2)
        burstiness_list.append(b)

    cv_arr = np.array(cv_list, dtype=float)
    cv2_arr = np.array(cv2_list, dtype=float)
    burst_arr = np.array(burstiness_list, dtype=float)

    def nanstats(x: np.ndarray) -> Dict[str, float]:
        return dict(
            mean=float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan"),
            std=float(np.nanstd(x)) if np.any(~np.isnan(x)) else float("nan"),
            min=float(np.nanmin(x)) if np.any(~np.isnan(x)) else float("nan"),
            max=float(np.nanmax(x)) if np.any(~np.isnan(x)) else float("nan"),
        )

    stats_cv = nanstats(cv_arr)
    stats_cv2 = nanstats(cv2_arr)
    stats_b = nanstats(burst_arr)

    print("\n-- CV (ISI) --")
    print(f"Mean : {stats_cv['mean']:.3f}")
    print(f"Std  : {stats_cv['std']:.3f}")
    print(f"Min  : {stats_cv['min']:.3f}")
    print(f"Max  : {stats_cv['max']:.3f}")

    print("\n-- CV2 (local ISI variability) --")
    print(f"Mean : {stats_cv2['mean']:.3f}")
    print(f"Std  : {stats_cv2['std']:.3f}")
    print(f"Min  : {stats_cv2['min']:.3f}")
    print(f"Max  : {stats_cv2['max']:.3f}")

    print("\n-- Burstiness (fraction of spikes in bursts) --")
    print(f"Mean : {stats_b['mean']:.3f}")
    print(f"Std  : {stats_b['std']:.3f}")
    print(f"Min  : {stats_b['min']:.3f}")
    print(f"Max  : {stats_b['max']:.3f}")

    # Oscillatory classification
    is_osc, peak_freqs = classify_population_oscillatory(
        spikes, dt_ms, burn_steps,
        beta_band=(13.0, 30.0),
        ref_band=(5.0, 80.0),
        power_ratio_thresh=3.0,
        smooth_ms=10.0,
    )

    n_osc = int(is_osc.sum())
    frac_osc = n_osc / float(N) if N > 0 else 0.0
    mean_peak = float(np.mean(peak_freqs[is_osc])) if n_osc > 0 else float("nan")

    print("\n-- Beta-oscillatory cells (13–30 Hz) --")
    print(f"Count         : {n_osc} / {N}")
    print(f"Fraction      : {frac_osc*100.0:5.1f} %")
    print(f"Mean peak fβ  : {mean_peak:5.2f} Hz")


# -----------------------------
# Main CLI entry point
# -----------------------------

def main(run_dir_str: str) -> None:
    run_dir = Path(run_dir_str).expanduser().resolve()

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    dt_ms = float(manifest["dt_ms"])
    burn_steps = int(manifest.get("burn_steps", 0))

    arrays_dir = run_dir / "arrays"

    # STN
    stn_path = arrays_dir / "spikes_stn.npy"
    if stn_path.exists():
        stn_spikes = np.load(stn_path)
        summarize_population("STN", stn_spikes, dt_ms, burn_steps, short_isi_ms=15.0)
    else:
        print("\nNo STN spikes array found.")

    # GPe
    gpe_path = arrays_dir / "spikes_gpe.npy"
    if gpe_path.exists():
        gpe_spikes = np.load(gpe_path)
        summarize_population("GPe", gpe_spikes, dt_ms, burn_steps, short_isi_ms=15.0)
    else:
        print("\nNo GPe spikes array found.")

    # GPi
    gpi_path = arrays_dir / "spikes_gpi.npy"
    if gpi_path.exists():
        gpi_spikes = np.load(gpi_path)
        summarize_population("GPi", gpi_spikes, dt_ms, burn_steps, short_isi_ms=15.0)
    else:
        print("\nNo GPi spikes array found.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m stn_gp.analysis.single_cell_stats /path/to/run_dir")
        sys.exit(1)
    main(sys.argv[1])
