# objectives.py
# Math-only objective functions for STN–GP network simulations.
#
# Responsibilities:
#   • Take simulation outputs (spikes, voltages, dt) and compute:
#       - Firing rates (Hz)
#       - Simple ISI-based irregularity metrics (CV of ISI)
#       - PSD of a 1D signal (e.g., STN population "LFP")
#       - Band-limited power and power ratios (e.g., beta band)
#   • Combine these into scalar scores suitable for optimization.
#
# Non-responsibilities:
#   • Running simulations (handled by sim_api.py)
#   • Optuna study setup / trial orchestration (handled by optuna_driver.py)
#
# Expected usage pattern (pseudo-code in optuna_driver):
#
#   from stn_gp.optuna import sim_api, objectives
#
#   sim_out = sim_api.run_simulation(params)
#   score = objectives.stn_gpe_beta_objective(
#       stn_pop_spikes = sim_out["stn_pop_spikes"],   # 1D population spike train
#       gpe_pop_spikes = sim_out["gpe_pop_spikes"],   # 1D population spike train
#       stn_lfp        = sim_out["stn_lfp"],          # 1D LFP-like signal (e.g. mean V)
#       dt_ms          = sim_out["dt_ms"],
#       cfg            = objective_cfg,
#   )
#
# sim_api.py is free to define its own structure as long as it can
# provide the above arrays to these objective functions.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------


def _safe_slice_by_time(
    arr: np.ndarray,
    dt_ms: float,
    t_start_ms: float = 0.0,
    t_end_ms: Optional[float] = None,
) -> np.ndarray:
    """
    Slice a 1D time series [0..T) by time in ms.
    Assumes arr is uniformly sampled with step dt_ms.

    Parameters
    ----------
    arr : np.ndarray, shape (T,)
        Time series.
    dt_ms : float
        Time step in ms.
    t_start_ms : float
        Start time (inclusive).
    t_end_ms : float or None
        End time (exclusive). If None, use full length.

    Returns
    -------
    sliced : np.ndarray
        arr restricted to [t_start_ms, t_end_ms).
    """
    n = arr.shape[0]
    if t_end_ms is None:
        t_end_ms = n * dt_ms

    i0 = max(int(np.floor(t_start_ms / dt_ms)), 0)
    i1 = min(int(np.floor(t_end_ms / dt_ms)), n)
    if i1 <= i0:
        return arr[0:0]
    return arr[i0:i1]


# ---------------------------------------------------------------------
# Firing rate and spike statistics
# ---------------------------------------------------------------------


def population_rate_from_spike_train(
    pop_spike_train: np.ndarray,
    dt_ms: float,
    t_start_ms: float = 0.0,
    t_end_ms: Optional[float] = None,
    n_neurons: Optional[int] = None,
) -> float:
    """
    Compute mean population firing rate (Hz) from a summed population spike train.

    Parameters
    ----------
    pop_spike_train : np.ndarray, shape (T,)
        Population spike train: at each time bin, number of spikes across neurons.
        Typically this is an integer or float-valued array, but it can also be
        a binary array if each time step is "any spike yes/no".
    dt_ms : float
        Time step in ms.
    t_start_ms : float
        Start of analysis window (ms).
    t_end_ms : float or None
        End of analysis window (ms). None -> full simulation.
    n_neurons : int or None
        Number of neurons in the population. If None, we assume that
        pop_spike_train already represents the *sum* of spikes across all neurons
        and we compute a rate for this population as a whole (i.e. total spikes / time).

    Returns
    -------
    rate_hz : float
        Mean firing rate in Hz (spikes/s per neuron if n_neurons provided,
        otherwise spikes/s for the whole population).
    """
    s = np.asarray(pop_spike_train, dtype=float).reshape(-1)
    s_win = _safe_slice_by_time(s, dt_ms, t_start_ms, t_end_ms)
    T_sec = s_win.shape[0] * (dt_ms / 1000.0)
    if T_sec <= 0.0:
        return 0.0

    total_spikes = float(np.sum(s_win))
    if n_neurons is not None and n_neurons > 0:
        rate_hz = (total_spikes / n_neurons) / T_sec
    else:
        rate_hz = total_spikes / T_sec
    return rate_hz


def isi_cv_from_spike_times(spike_times_ms: np.ndarray) -> float:
    """
    Compute coefficient of variation (CV) of inter-spike intervals.

    Parameters
    ----------
    spike_times_ms : np.ndarray, shape (n_spikes,)
        Spike times (ms) for a single neuron or a population.

    Returns
    -------
    cv : float
        CV of ISI. Returns 0.0 if fewer than 2 spikes.
    """
    times = np.asarray(spike_times_ms, dtype=float).reshape(-1)
    if times.shape[0] < 2:
        return 0.0
    isi = np.diff(np.sort(times))
    mean_isi = np.mean(isi)
    if mean_isi <= 0.0:
        return 0.0
    return float(np.std(isi) / mean_isi)


# ---------------------------------------------------------------------
# PSD and band power
# ---------------------------------------------------------------------


def psd_simple(
    signal: np.ndarray,
    dt_ms: float,
    n_fft: Optional[int] = None,
    detrend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a simple one-sided PSD using an FFT-based estimator.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        1D signal (e.g., STN "LFP" from mean membrane voltage).
    dt_ms : float
        Time step in ms.
    n_fft : int or None
        FFT length. If None, use next power of 2 >= len(signal).
    detrend : bool
        If True, subtract mean before FFT.

    Returns
    -------
    freqs_hz : np.ndarray, shape (K,)
        Frequency axis (Hz).
    psd : np.ndarray, shape (K,)
        Power spectral density (arbitrary units; consistent across runs).
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    if detrend:
        x = x - np.mean(x)

    n = x.shape[0]
    if n <= 1:
        return np.array([0.0]), np.array([0.0])

    if n_fft is None:
        # next power of 2 for reasonable FFT performance
        n_fft = int(2 ** np.ceil(np.log2(n)))

    # Zero-pad if needed
    if n_fft > n:
        x_padded = np.zeros(n_fft, dtype=float)
        x_padded[:n] = x
        x = x_padded
    elif n_fft < n:
        x = x[:n_fft]

    # Sampling frequency in Hz
    fs = 1000.0 / dt_ms

    # One-sided FFT
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    psd = (np.abs(X) ** 2) / (fs * n_fft)

    return freqs, psd


def band_power(
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    f_lo: float,
    f_hi: float,
) -> float:
    """
    Integrate PSD over a frequency band [f_lo, f_hi].

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frequency axis (Hz).
    psd : np.ndarray
        Power spectral density.
    f_lo : float
        Lower bound of band (Hz).
    f_hi : float
        Upper bound of band (Hz).

    Returns
    -------
    power : float
        Band-limited power (arbitrary units).
    """
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    p = np.asarray(psd, dtype=float).reshape(-1)
    if p.shape[0] != f.shape[0]:
        raise ValueError("freqs_hz and psd must have same length.")

    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return 0.0
    # Simple Riemann sum; fine for relative comparisons
    df = np.mean(np.diff(f))
    return float(np.sum(p[mask]) * df)


# ---------------------------------------------------------------------
# Objective configuration
# ---------------------------------------------------------------------


@dataclass
class BandSpec:
    f_lo: float
    f_hi: float


@dataclass
class STNGPObjectiveConfig:
    """
    Configuration for a simple STN–GPe beta objective.

    This is a *suggested* structure — sim_api / optuna_driver can adapt
    or extend it as needed.

    Attributes
    ----------
    dt_ms : float
        Time step in ms (for validation; optional but convenient).
    burn_in_ms : float
        Initial period to ignore when computing metrics (transient).
    t_end_ms : float or None
        End of analysis window. If None, analyze full post-burn-in run.
    target_stn_rate_hz : float
        Desired STN firing rate (Hz).
    target_gpe_rate_hz : float
        Desired GPe firing rate (Hz).
    beta_band : BandSpec
        Frequency band for beta (e.g. 13–30 Hz).
    ref_band : BandSpec
        Reference band for ratio (e.g. 5–45 Hz or 1–80 Hz).
    w_rate_stn : float
        Weight for STN rate error term.
    w_rate_gpe : float
        Weight for GPe rate error term.
    w_beta_ratio : float
        Weight for beta power ratio term.
    beta_ratio_target : float
        Target ratio: (beta_power / ref_power). Can be > 1 if you want
        strong beta, or e.g. 0.3, etc.
    """
    dt_ms: float
    burn_in_ms: float = 500.0
    t_end_ms: Optional[float] = None

    target_stn_rate_hz: float = 15.0
    target_gpe_rate_hz: float = 40.0

    beta_band: BandSpec = BandSpec(13.0, 30.0)
    ref_band: BandSpec = BandSpec(5.0, 45.0)

    w_rate_stn: float = 1.0
    w_rate_gpe: float = 1.0
    w_beta_ratio: float = 1.0

    beta_ratio_target: float = 0.3  # Example target; to be tuned from data


# ---------------------------------------------------------------------
# High-level objective for Optuna (single-condition)
# ---------------------------------------------------------------------


def stn_gpe_beta_objective(
    stn_pop_spikes: np.ndarray,
    gpe_pop_spikes: np.ndarray,
    stn_lfp: np.ndarray,
    dt_ms: float,
    cfg: STNGPObjectiveConfig,
    n_stn: Optional[int] = None,
    n_gpe: Optional[int] = None,
) -> float:
    """
    Compute a scalar objective summarizing:
      • STN mean firing rate
      • GPe mean firing rate
      • STN beta-band power ratio

    The returned value is a *loss* (lower is better).

    Parameters
    ----------
    stn_pop_spikes : np.ndarray, shape (T,)
        Population spike train for STN (sum across neurons at each time step).
    gpe_pop_spikes : np.ndarray, shape (T,)
        Population spike train for GPe.
    stn_lfp : np.ndarray, shape (T,)
        LFP-like signal from STN (e.g., mean membrane voltage over STN cells).
    dt_ms : float
        Time step in ms (should match cfg.dt_ms; used here directly).
    cfg : STNGPObjectiveConfig
        Configuration specifying targets and weights.
    n_stn : int or None
        Number of STN neurons (if you want per-neuron rate).
    n_gpe : int or None
        Number of GPe neurons (if you want per-neuron rate).

    Returns
    -------
    loss : float
        Scalar loss suitable for Optuna minimization.
    """
    # ------------------------------
    # 1) Define analysis window
    # ------------------------------
    t_start = cfg.burn_in_ms
    t_end = cfg.t_end_ms

    # Slice spike trains and LFP into the analysis window
    stn_spk = _safe_slice_by_time(stn_pop_spikes, dt_ms, t_start, t_end)
    gpe_spk = _safe_slice_by_time(gpe_pop_spikes, dt_ms, t_start, t_end)
    stn_lfp_win = _safe_slice_by_time(stn_lfp, dt_ms, t_start, t_end)

    # ------------------------------
    # 2) Firing rates
    # ------------------------------
    stn_rate = population_rate_from_spike_train(
        stn_spk, dt_ms, t_start_ms=0.0, t_end_ms=None, n_neurons=n_stn
    )
    gpe_rate = population_rate_from_spike_train(
        gpe_spk, dt_ms, t_start_ms=0.0, t_end_ms=None, n_neurons=n_gpe
    )

    rate_err_stn = (stn_rate - cfg.target_stn_rate_hz) ** 2
    rate_err_gpe = (gpe_rate - cfg.target_gpe_rate_hz) ** 2

    # ------------------------------
    # 3) Beta power ratio
    # ------------------------------
    freqs, psd = psd_simple(stn_lfp_win, dt_ms)

    beta_power = band_power(freqs, psd, cfg.beta_band.f_lo, cfg.beta_band.f_hi)
    ref_power = band_power(freqs, psd, cfg.ref_band.f_lo, cfg.ref_band.f_hi)
    if ref_power <= 0.0:
        # If there's essentially no power, penalize heavily
        beta_ratio = 0.0
    else:
        beta_ratio = beta_power / ref_power

    beta_err = (beta_ratio - cfg.beta_ratio_target) ** 2

    # ------------------------------
    # 4) Weighted loss
    # ------------------------------
    loss = (
        cfg.w_rate_stn * rate_err_stn
        + cfg.w_rate_gpe * rate_err_gpe
        + cfg.w_beta_ratio * beta_err
    )

    return float(loss)
