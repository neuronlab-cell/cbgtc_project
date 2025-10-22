# plots.py
# Reusable plotting utilities for STN–GPi outputs:
# - spike rasters
# - population-rate PSD overlays
# Returns matplotlib Figure objects (caller decides where to save).

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from typing import Tuple

def make_raster_figure(spikes_stn: np.ndarray,
                       spikes_gpi: np.ndarray,
                       dt_ms: float,
                       burn_steps: int = 0) -> plt.Figure:
    """
    Build a 2-panel raster (STN, GPi). Inputs are binary matrices (T, N).
    burn_steps: number of initial steps to skip (burn-in).
    """
    s_stn = spikes_stn[burn_steps:]
    s_gpi = spikes_gpi[burn_steps:]

    def _draw(ax, spk, title, color):
        T, N = spk.shape
        t_ms = np.arange(T, dtype=np.float32) * dt_ms
        for j in range(N):
            idx = np.nonzero(spk[:, j])[0]
            if idx.size:
                ax.vlines(t_ms[idx] / 1000.0, j + 0.5, j + 1.5, linewidth=0.5, color=color)
        ax.set_title(title)
        ax.set_ylabel("Neuron index")
        ax.set_xlim(0.0, t_ms[-1] / 1000.0)
        ax.set_ylim(0.5, N + 0.5)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _draw(axes[0], s_stn, "STN spikes (raster)", "tab:blue")
    _draw(axes[1], s_gpi, "GPi spikes (raster)", "tab:orange")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig

def make_psd_figure(spikes_stn: np.ndarray,
                    spikes_gpi: np.ndarray,
                    dt_ms: float,
                    burn_steps: int = 0,
                    fmax_hz: float = 100.0) -> plt.Figure:
    """
    Compute population-rate PSDs for STN and GPi and plot them on the same axes.
    """
    fs_hz = 1000.0 / dt_ms
    stn_rate = spikes_stn.sum(axis=1).astype(np.float32) * fs_hz  # spikes/s
    gpi_rate = spikes_gpi.sum(axis=1).astype(np.float32) * fs_hz

    r1 = stn_rate[burn_steps:]
    r2 = gpi_rate[burn_steps:]

    nperseg = min(4096, len(r1), len(r2)) if min(len(r1), len(r2)) > 0 else 256
    f1, P1 = welch(r1, fs=fs_hz, nperseg=nperseg)
    f2, P2 = welch(r2, fs=fs_hz, nperseg=nperseg)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogy(f1, P1 + 1e-12, label="STN")
    ax.semilogy(f2, P2 + 1e-12, label="GPi")
    ax.set_xlim(0, fmax_hz)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Population rate PSD")
    ax.legend()
    fig.tight_layout()
    return fig
