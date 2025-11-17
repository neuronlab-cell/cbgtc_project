# inspect_best.py
# Run one normal (δ=0) and one PD (δ=1) simulation with best_theta.json
# and save raster + PSD plots (similar style to run_loop.py).

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from stn_gp.optuna.sim_api import run_simulation


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def figure_raster(ax, spike_mat: np.ndarray, dt_ms: float, title: str, color="k"):
    """
    Simple raster plot.

    spike_mat: shape (T, N), binary or int spike counts.
    """
    T, N = spike_mat.shape
    t_ms = np.arange(T) * dt_ms
    for j in range(N):
        s_idx = np.nonzero(spike_mat[:, j])[0]
        if s_idx.size:
            ax.vlines(
                t_ms[s_idx] / 1000.0,
                j + 0.5,
                j + 1.5,
                linewidth=0.5,
                color=color,
            )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index")
    ax.set_xlim(0, t_ms[-1] / 1000.0)
    ax.set_ylim(0.5, N + 0.5)


def compute_psd(signal: np.ndarray, fs_hz: float):
    f, Pxx = welch(signal, fs=fs_hz, nperseg=min(4096, len(signal)))
    return f, Pxx


def run_and_plot(theta_path: Path, out_dir: Path):
    # Load best_theta.json
    with open(theta_path, "r") as f:
        theta = json.load(f)

    ensure_dir(out_dir)

    # Simulation settings
    t_total_s = 8.0
    burn_in_s = 1.0
    dt_ms = 0.025

    conditions = [
        ("normal", 0.0),
        ("pd", 1.0),
    ]

    for label, delta in conditions:
        print(f"=== Running condition: {label} (delta={delta}) ===")
        sim = run_simulation(
            theta,
            delta=delta,
            t_total_s=t_total_s,
            burn_in_s=burn_in_s,
            dt_ms=dt_ms,
        )

        spikes_stn = sim["spikes_stn"]
        spikes_gpe = sim["spikes_gpe"]
        dt = sim["dt_ms"]
        burn_steps = sim["burn_steps"]

        # ------------------ Raster ------------------
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        figure_raster(
            axes[0],
            spikes_stn[burn_steps:],
            dt,
            title=f"STN spikes (raster) – {label}",
            color="tab:blue",
        )
        figure_raster(
            axes[1],
            spikes_gpe[burn_steps:],
            dt,
            title=f"GPe spikes (raster) – {label}",
            color="tab:orange",
        )
        plt.tight_layout()
        fig.savefig(out_dir / f"raster_{label}.png", dpi=160)
        plt.close(fig)

        # ------------------ PSD of population spike counts ------------------
        fs_hz = 1000.0 / dt

        stn_rate = spikes_stn.sum(axis=1).astype(np.float32)
        gpe_rate = spikes_gpe.sum(axis=1).astype(np.float32)

        f_stn, P_stn = compute_psd(stn_rate[burn_steps:], fs_hz)
        f_gpe, P_gpe = compute_psd(gpe_rate[burn_steps:], fs_hz)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
        ax2.semilogy(f_stn, P_stn + 1e-12, label="STN")
        ax2.semilogy(f_gpe, P_gpe + 1e-12, label="GPe")
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Power")
        ax2.set_title(f"Population rate PSD – {label}")
        ax2.legend()
        plt.tight_layout()
        fig2.savefig(out_dir / f"psd_{label}.png", dpi=160)
        plt.close(fig2)

        print(f"Saved raster_{label}.png and psd_{label}.png to {out_dir}")


def main():
    # This file lives in stn_gp/optuna/, so repo root is two levels up
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]

    theta_path = repo_root / "best_theta.json"
    out_dir = repo_root / "optuna_inspect"

    if not theta_path.exists():
        raise FileNotFoundError(f"best_theta.json not found at {theta_path}")

    print(f"Using best_theta.json from: {theta_path}")
    print(f"Saving figures to: {out_dir}")

    run_and_plot(theta_path, out_dir)


if __name__ == "__main__":
    main()
