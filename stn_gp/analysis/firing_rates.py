# stn_gp/analysis/firing_rates.py
#
# Usage (from repo root, venv active):
#   python -m stn_gp.analysis.firing_rates /home/ubuntu/cbgtc_project/runs/run_20251119_024724
#
# Prints mean / std / min / max firing rates (Hz) for STN, GPe, GPi.

from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np


def compute_rates(spikes: np.ndarray, dt_ms: float, burn_steps: int) -> np.ndarray:
    """
    spikes: (T, N) binary matrix
    dt_ms:  timestep in ms
    burn_steps: number of initial steps to ignore
    returns: per-neuron firing rates in Hz, shape (N,)
    """
    # drop burn-in
    spikes_win = spikes[burn_steps:, :]
    T_eff = spikes_win.shape[0]
    sim_dur_s = T_eff * (dt_ms / 1000.0)

    if sim_dur_s <= 0.0:
        return np.zeros(spikes.shape[1], dtype=float)

    spikes_per_neuron = spikes_win.sum(axis=0)  # shape (N,)
    rates_hz = spikes_per_neuron / sim_dur_s
    return rates_hz


def summarize(pop_name: str, rates: np.ndarray) -> None:
    if rates.size == 0:
        print(f"{pop_name}: no neurons?")
        return

    mean = float(np.mean(rates))
    std = float(np.std(rates))
    min_r = float(np.min(rates))
    max_r = float(np.max(rates))

    print(f"\n=== {pop_name} firing rates (Hz) ===")
    print(f"Mean : {mean:7.3f}")
    print(f"Std  : {std:7.3f}")
    print(f"Min  : {min_r:7.3f}")
    print(f"Max  : {max_r:7.3f}")


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

    # ---- STN ----
    stn_path = arrays_dir / "spikes_stn.npy"
    if stn_path.exists():
        stn_spikes = np.load(stn_path)
        stn_rates = compute_rates(stn_spikes, dt_ms, burn_steps)
        summarize("STN", stn_rates)
    else:
        print("No STN spikes array found.")

    # ---- GPe ----
    gpe_path = arrays_dir / "spikes_gpe.npy"
    if gpe_path.exists():
        gpe_spikes = np.load(gpe_path)
        gpe_rates = compute_rates(gpe_spikes, dt_ms, burn_steps)
        summarize("GPe", gpe_rates)
    else:
        print("No GPe spikes array found.")

    # ---- GPi ----
    gpi_path = arrays_dir / "spikes_gpi.npy"
    if gpi_path.exists():
        gpi_spikes = np.load(gpi_path)
        gpi_rates = compute_rates(gpi_spikes, dt_ms, burn_steps)
        summarize("GPi", gpi_rates)
    else:
        print("No GPi spikes array found.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m stn_gp.analysis.firing_rates /path/to/run_dir")
        sys.exit(1)
    main(sys.argv[1])
