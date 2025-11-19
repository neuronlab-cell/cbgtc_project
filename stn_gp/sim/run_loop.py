# run_loop.py
# Main entry point: build STN–GPe–GPi network, run simulation, save outputs (arrays + quick figures).

from __future__ import annotations
import os, sys, json, time, argparse, datetime
import numpy as np
from pathlib import Path

# Optional YAML config
try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

from .build_network import build_network
from .integrators import step_once

# --- plotting helpers (kept local for quick validation) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch


def load_yaml_or_default(path: str | Path, default: dict) -> dict:
    p = Path(path).expanduser()
    if HAVE_YAML and p.exists():
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        out = dict(default)
        for k, v in data.items():
            out[k] = v
        return out
    return default


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def figure_raster(ax, spike_mat: np.ndarray, dt_ms: float, title: str, color="k"):
    """
    spike_mat: shape (T, N), binary
    """
    T, N = spike_mat.shape
    t_ms = np.arange(T) * dt_ms
    for j in range(N):
        s_idx = np.nonzero(spike_mat[:, j])[0]
        if s_idx.size:
            ax.vlines(t_ms[s_idx] / 1000.0, j + 0.5, j + 1.5, linewidth=0.5, color=color)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index")
    ax.set_xlim(0, t_ms[-1] / 1000.0)
    ax.set_ylim(0.5, N + 0.5)


def compute_psd(signal: np.ndarray, fs_hz: float):
    f, Pxx = welch(signal, fs=fs_hz, nperseg=min(4096, len(signal)))
    return f, Pxx


def main():
    parser = argparse.ArgumentParser(description="Run STN–GPe–GPi loop simulation")
    parser.add_argument("--config", type=str, default="../configs/params_stn_gpe.yaml",
                        help="YAML config for simulation/network (optional)")
    parser.add_argument("--paths", type=str, default="../configs/paths.yaml",
                        help="YAML for project paths (optional)")
    parser.add_argument("--seconds", type=float, default=None, help="Override total sim time in seconds")
    parser.add_argument("--burnin", type=float, default=None, help="Override burn-in seconds")
    parser.add_argument("--out", type=str, default=None, help="Override output base directory")
    args = parser.parse_args()

    # ------- defaults (only used if YAML missing) -------
    default_cfg = dict(
        seed=42,
        dt_ms=0.025,
        # populations
        n_stn=50,
        n_gpe=100,
        n_gpi=80,
        # connectivity
        p_stn_to_gpe=0.25,
        p_gpe_to_stn=0.35,
        p_stn_to_gpi=0.30,
        p_gpe_to_gpi=0.50,
        # delays (ms)
        delay_stn_to_gpe_ms=5.0,
        delay_gpe_to_stn_ms=8.0,
        delay_stn_to_gpi_ms=5.0,
        delay_gpe_to_gpi_ms=5.0,
        # synapse kinetics
        ampa_decay_ms=3.0,
        gaba_decay_ms=8.0,
        # E_rev (mV)
        E_AMPA_mV=0.0,
        E_GABA_mV=-70.0,
        # weights
        w_stn_to_gpe_mean_pA=22.0,
        w_stn_to_gpe_cv=0.20,
        w_gpe_to_stn_mean_uAcm2=0.09,
        w_gpe_to_stn_cv=0.20,
        w_stn_to_gpi_mean_pA=18.0,
        w_stn_to_gpi_cv=0.20,
        w_gpe_to_gpi_mean_pA=25.0,
        w_gpe_to_gpi_cv=0.20,
        # background OU (unused here—builder owns it)
        stn_params={},
        gpe_params={},
        gpi_params={},
        # sim window
        t_total_s=8.0,
        burn_in_s=1.0,
        # recording
        record_subset_stn=10,
        record_subset_gpe=10,
        record_subset_gpi=10,
    )

    cfg = load_yaml_or_default(args.config, default_cfg)
    if args.seconds is not None:
        cfg["t_total_s"] = float(args.seconds)
    if args.burnin is not None:
        cfg["burn_in_s"] = float(args.burnin)

    # optional paths yaml
    paths = {}
    if HAVE_YAML and Path(args.paths).expanduser().exists():
        with open(Path(args.paths).expanduser(), "r") as f:
            paths = yaml.safe_load(f) or {}
    project_root = Path(paths.get("project_root", "~/cbgtc_project")).expanduser()
    local_runs = project_root / "runs"

    # ------- build network -------
    net, used = build_network(cfg)

    dt_ms = used["dt_ms"]
    fs_hz = 1000.0 / dt_ms
    T_steps = int(round(cfg["t_total_s"] * 1000.0 / dt_ms))
    burn_steps = int(round(cfg["burn_in_s"] * 1000.0 / dt_ms))

    # ------- run folder -------
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_base = Path(args.out).expanduser() if args.out else local_runs / run_id
    ensure_dir(out_base)
    (out_base / "figs").mkdir(exist_ok=True, parents=True)
    (out_base / "arrays").mkdir(exist_ok=True, parents=True)

    # manifest
    manifest = {
        "module": "stn_gp",
        "run_id": run_id,
        "dt_ms": dt_ms,
        "fs_hz": fs_hz,
        "T_steps": T_steps,
        "burn_steps": burn_steps,
        "n_stn": net.n_stn,
        "n_gpe": net.n_gpe,
        "n_gpi": net.n_gpi,
        "record_subset_stn": cfg.get("record_subset_stn", 10),
        "record_subset_gpe": cfg.get("record_subset_gpe", 10),
        "record_subset_gpi": cfg.get("record_subset_gpi", 10),
        "config": cfg,
        "start_time": datetime.datetime.now().isoformat(),
    }
    with open(out_base / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ------- recording buffers -------
    rec_stn = cfg.get("record_subset_stn", 10)
    rec_gpe = cfg.get("record_subset_gpe", 10)
    rec_gpi = cfg.get("record_subset_gpi", 10)

    idx_stn = np.arange(min(rec_stn, net.n_stn))
    idx_gpe = np.arange(min(rec_gpe, net.n_gpe))
    idx_gpi = np.arange(min(rec_gpi, net.n_gpi))

    V_stn_rec = np.zeros((T_steps, idx_stn.size), dtype=np.float32)
    V_gpe_rec = np.zeros((T_steps, idx_gpe.size), dtype=np.float32)
    V_gpi_rec = np.zeros((T_steps, idx_gpi.size), dtype=np.float32)

    spk_stn_mat = np.zeros((T_steps, net.n_stn), dtype=np.uint8)
    spk_gpe_mat = np.zeros((T_steps, net.n_gpe), dtype=np.uint8)
    spk_gpi_mat = np.zeros((T_steps, net.n_gpi), dtype=np.uint8)

    # ------- simulation loop -------
    t_ms = 0.0
    t0 = time.time()
    for t in range(T_steps):
        # step_once now returns STN, GPe, GPi
        V_stn, spk_stn, V_gpe, spk_gpe, V_gpi, spk_gpi = step_once(net, t_ms=t_ms)

        if idx_stn.size:
            V_stn_rec[t, :] = V_stn[idx_stn]
        if idx_gpe.size:
            V_gpe_rec[t, :] = V_gpe[idx_gpe]
        if idx_gpi.size:
            V_gpi_rec[t, :] = V_gpi[idx_gpi]

        spk_stn_mat[t, :] = spk_stn
        spk_gpe_mat[t, :] = spk_gpe
        spk_gpi_mat[t, :] = spk_gpi

        t_ms += dt_ms

    elapsed = time.time() - t0
    manifest["elapsed_s"] = elapsed
    with open(out_base / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ------- save arrays -------
    np.save(out_base / "arrays" / "V_stn.npy", V_stn_rec)
    np.save(out_base / "arrays" / "V_gpe.npy", V_gpe_rec)
    np.save(out_base / "arrays" / "V_gpi.npy", V_gpi_rec)

    np.save(out_base / "arrays" / "spikes_stn.npy", spk_stn_mat)
    np.save(out_base / "arrays" / "spikes_gpe.npy", spk_gpe_mat)
    np.save(out_base / "arrays" / "spikes_gpi.npy", spk_gpi_mat)

    # ------- quick figs: raster + PSD -------
    # Raster (skip burn-in)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    figure_raster(
        axes[0],
        spk_stn_mat[burn_steps:],
        dt_ms,
        title="STN spikes (raster)",
        color="tab:blue",
    )
    figure_raster(
        axes[1],
        spk_gpe_mat[burn_steps:],
        dt_ms,
        title="GPe spikes (raster)",
        color="tab:orange",
    )
    figure_raster(
        axes[2],
        spk_gpi_mat[burn_steps:],
        dt_ms,
        title="GPi spikes (raster)",
        color="tab:green",
    )

    plt.tight_layout()
    fig.savefig(out_base / "figs" / "raster.png", dpi=160)
    plt.close(fig)

    # PSD of population spike count per step
    stn_rate = spk_stn_mat.sum(axis=1).astype(np.float32)
    gpe_rate = spk_gpe_mat.sum(axis=1).astype(np.float32)
    gpi_rate = spk_gpi_mat.sum(axis=1).astype(np.float32)

    f1, Pstn = compute_psd(stn_rate[burn_steps:], fs_hz)
    f2, Pgpe = compute_psd(gpe_rate[burn_steps:], fs_hz)
    f3, Pgpi = compute_psd(gpi_rate[burn_steps:], fs_hz)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    ax2.semilogy(f1, Pstn + 1e-12, label="STN")
    ax2.semilogy(f2, Pgpe + 1e-12, label="GPe")
    ax2.semilogy(f3, Pgpi + 1e-12, label="GPi")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.set_title("Population rate PSD")
    ax2.legend()
    fig2.savefig(out_base / "figs" / "psd.png", dpi=160)
    plt.close(fig2)

    print(f"✅ Run complete: {run_id}")
    print(f"   Saved to: {out_base}")
    print("   Figures: raster.png, psd.png")
    print("   Arrays:  V_stn.npy, V_gpe.npy, V_gpi.npy, spikes_stn.npy, spikes_gpe.npy, spikes_gpi.npy")


if __name__ == "__main__":
    main()
