# sim_api.py
# Clean simulation API for Optuna.
#
# Responsibilities:
#   • Accept (theta, delta) and simulation settings
#   • Apply dopamine-modulated parameter transforms (on Optuna-facing names)
#   • Map those into the config keys that build_network() expects
#   • Build STN–GPe network using existing build_network()
#   • Run simulation using step_once()
#   • Return clean arrays ready for scoring in objectives.py

from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from stn_gp.sim.build_network import build_network
from stn_gp.sim.integrators import step_once
from stn_gp.models.gp_adex import AdExParams_GPe


# ============================================================
# δ-MODULATED PARAMETER TRANSFORM (Optuna-facing names)
# ============================================================

def apply_delta_parameters(theta: Dict, delta: float) -> Dict:
    """
    Given base parameters θ and dopamine level δ ∈ [0,1],
    return a dict with *Optuna-facing names* after δ-modulation.
    """
    cfg = dict(theta)
    delta = max(0.0, min(1.0, float(delta)))

    # GPe → STN inhibition increases in PD
    if "gpe_to_stn_mean" in cfg:
        base = float(cfg["gpe_to_stn_mean"])
        cfg["gpe_to_stn_mean"] = base * (1.0 + 1.0 * delta)

    # GPe baseline decreases in PD (weaker pacemaker)
    if "gpe_I_baseline" in cfg:
        base = float(cfg["gpe_I_baseline"])
        cfg["gpe_I_baseline"] = max(0.0, base * (1.0 - 0.5 * delta))

    # STN tonic drive increases in PD
    if "stn_ISTN" in cfg:
        base = float(cfg["stn_ISTN"])
        cfg["stn_ISTN"] = base * (1.0 + 0.5 * delta)

    # STN → GPe AMPA gain slightly increases in PD
    if "stn_to_gpe_mean" in cfg:
        base = float(cfg["stn_to_gpe_mean"])
        cfg["stn_to_gpe_mean"] = base * (1.0 + 0.3 * delta)

    return cfg


# ============================================================
# MAP Optuna-level KEYS → build_network() KEYS
# ============================================================

def _build_network_cfg(
    model_cfg: Dict,
    t_total_s: float,
    burn_in_s: float,
    dt_ms: float,
    seed: Optional[int] = None,
) -> Dict:
    """
    Convert the δ-modulated Optuna-level config into the
    internal config expected by build_network().
    """
    cfg: Dict = {}

    cfg["dt_ms"] = float(dt_ms)
    if seed is not None:
        cfg["seed"] = int(seed)

    # Population sizes
    if "n_stn" in model_cfg:
        cfg["n_stn"] = int(model_cfg["n_stn"])
    if "n_gpe" in model_cfg:
        cfg["n_gpe"] = int(model_cfg["n_gpe"])

    # Delays
    if "delay_stn_to_gpe_ms" in model_cfg:
        cfg["delay_stn_to_gpe_ms"] = float(model_cfg["delay_stn_to_gpe_ms"])
    if "delay_gpe_to_stn_ms" in model_cfg:
        cfg["delay_gpe_to_stn_ms"] = float(model_cfg["delay_gpe_to_stn_ms"])

    # Synapse weights
    if "stn_to_gpe_mean" in model_cfg:
        cfg["w_stn_to_gpe_mean_pA"] = float(model_cfg["stn_to_gpe_mean"])
    if "gpe_to_stn_mean" in model_cfg:
        cfg["w_gpe_to_stn_mean_uAcm2"] = float(model_cfg["gpe_to_stn_mean"])

    # Noise (σ only)
    if "noise_sigma_stn" in model_cfg:
        cfg["stn_ou_sigma"] = float(model_cfg["noise_sigma_stn"])
    if "noise_sigma_gpe" in model_cfg:
        cfg["gpe_ou_sigma"] = float(model_cfg["noise_sigma_gpe"])

    # Intrinsic parameters: only STN goes through build_network
    stn_params = {}
    if "stn_ISTN" in model_cfg:
        stn_params["ISTN"] = float(model_cfg["stn_ISTN"])
    cfg["stn_params"] = stn_params

    # GPe params left empty to avoid AdExParams_GPe(**kwargs) errors
    cfg["gpe_params"] = {}

    # record times for logging/debug
    cfg["t_total_s"] = t_total_s
    cfg["burn_in_s"] = burn_in_s

    return cfg


# ============================================================
# SIMULATION API
# ============================================================

def run_simulation(
    theta: Dict,
    delta: float,
    t_total_s: float = 5.0,
    burn_in_s: float = 1.0,
    dt_ms: float = 0.025,
    seed: Optional[int] = None,
) -> Dict:
    """
    Run STN–GPe simulation for given (θ, δ).
    """
    # 1) Dopamine modulation
    model_cfg = apply_delta_parameters(theta, delta)

    # 2) Convert to build_network cfg
    cfg_for_net = _build_network_cfg(
        model_cfg=model_cfg,
        t_total_s=t_total_s,
        burn_in_s=burn_in_s,
        dt_ms=dt_ms,
        seed=seed,
    )

    # 3) Build the network
    net, used_cfg = build_network(cfg_for_net)

    dt = float(used_cfg["dt_ms"])
    T = int(round(t_total_s * 1000.0 / dt))
    burn_steps = int(round(burn_in_s * 1000.0 / dt))

    # 3b) Apply GPe I_baseline AFTER building (since build_network hardcodes AdExParams_GPe)
    if "gpe_I_baseline" in model_cfg:
        target_I = float(model_cfg["gpe_I_baseline"])
        default = AdExParams_GPe()
        base_I = float(default.I_baseline) if default.I_baseline != 0 else 1.0
        ratio = target_I / base_I

        for cell in net.gpe:
            cell.p.I_baseline *= ratio

    # 4) Buffers
    N_stn = net.n_stn
    N_gpe = net.n_gpe

    V_stn = np.zeros((T, N_stn), dtype=np.float32)
    V_gpe = np.zeros((T, N_gpe), dtype=np.float32)
    spikes_stn = np.zeros((T, N_stn), dtype=np.uint8)
    spikes_gpe = np.zeros((T, N_gpe), dtype=np.uint8)

    # 5) Simulation
    t_ms = 0.0
    for t in range(T):
        V_STN_t, spk_STN_t, V_GPE_t, spk_GPE_t = step_once(net, t_ms=t_ms)

        V_stn[t, :] = V_STN_t
        V_gpe[t, :] = V_GPE_t
        spikes_stn[t, :] = spk_STN_t
        spikes_gpe[t, :] = spk_GPE_t

        t_ms += dt

    # 6) Return
    return {
        "spikes_stn": spikes_stn,
        "spikes_gpe": spikes_gpe,
        "V_stn": V_stn,
        "V_gpe": V_gpe,
        "dt_ms": dt,
        "burn_steps": burn_steps,
    }
