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
#
# Non-responsibilities:
#   • No scoring
#   • No Optuna logic
#   • No plotting
#   • No file system writes

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
    return a *model-level* parameter dict using Optuna-facing names.

    θ contains (as sampled in optuna_driver):
        - stn_to_gpe_mean      (baseline AMPA jump, pA)
        - gpe_to_stn_mean      (baseline GABA jump, µA/cm²)
        - stn_ISTN             (baseline STN tonic drive, µA/cm²)
        - gpe_I_baseline       (baseline GPe AdEx drive, pA)
        - delay_stn_to_gpe_ms
        - delay_gpe_to_stn_ms
        - noise_sigma_stn
        - noise_sigma_gpe
        - n_stn, n_gpe

    δ:
        0 → normal
        1 → PD (dopamine-depleted)

    This function does NOT know about build_network's internal keys.
    It just returns a dict with the same high-level names, but δ-modulated.
    """
    cfg = dict(theta)

    # Clamp delta to [0, 1]
    delta = max(0.0, min(1.0, float(delta)))

    # 1) GPe → STN inhibitory strength increases with δ (PD)
    if "gpe_to_stn_mean" in cfg:
        base = float(cfg["gpe_to_stn_mean"])
        cfg["gpe_to_stn_mean"] = base * (1.0 + 1.0 * delta)  # up to 2×

    # 2) GPe baseline decreases with δ (weaker pacemaker in PD)
    if "gpe_I_baseline" in cfg:
        base = float(cfg["gpe_I_baseline"])
        cfg["gpe_I_baseline"] = max(0.0, base * (1.0 - 0.5 * delta))  # down to 50%

    # 3) STN tonic drive increases with δ (hyperactive STN in PD)
    if "stn_ISTN" in cfg:
        base = float(cfg["stn_ISTN"])
        cfg["stn_ISTN"] = base * (1.0 + 0.5 * delta)  # up to 1.5×

    # 4) STN → GPe excitatory gain (moderate effect in PD)
    if "stn_to_gpe_mean" in cfg:
        base = float(cfg["stn_to_gpe_mean"])
        cfg["stn_to_gpe_mean"] = base * (1.0 + 0.3 * delta)

    # Noise left for Optuna to control directly

    return cfg


# ============================================================
# MAP MODEL-LEVEL NAMES → build_network CONFIG KEYS
# ============================================================

def _build_network_cfg(
    model_cfg: Dict,
    t_total_s: float,
    burn_in_s: float,
    dt_ms: float,
    seed: Optional[int] = None,
) -> Dict:
    """
    Take the δ-modulated model-level config (Optuna-facing names)
    and produce a cfg dict compatible with build_network().

    Mapping:
        stn_to_gpe_mean      → w_stn_to_gpe_mean_pA
        gpe_to_stn_mean      → w_gpe_to_stn_mean_uAcm2
        stn_ISTN             → stn_params["ISTN"]
        noise_sigma_stn      → stn_ou_sigma
        noise_sigma_gpe      → gpe_ou_sigma
        n_stn, n_gpe         → n_stn, n_gpe
        delay_*              → delay_*
    """
    cfg: Dict = {}

    # ----- time + seed -----
    cfg["dt_ms"] = float(dt_ms)
    if seed is not None:
        cfg["seed"] = int(seed)

    # ----- population sizes -----
    if "n_stn" in model_cfg:
        cfg["n_stn"] = int(model_cfg["n_stn"])
    if "n_gpe" in model_cfg:
        cfg["n_gpe"] = int(model_cfg["n_gpe"])

    # ----- delays -----
    if "delay_stn_to_gpe_ms" in model_cfg:
        cfg["delay_stn_to_gpe_ms"] = float(model_cfg["delay_stn_to_gpe_ms"])
    if "delay_gpe_to_stn_ms" in model_cfg:
        cfg["delay_gpe_to_stn_ms"] = float(model_cfg["delay_gpe_to_stn_ms"])

    # ----- synaptic weights -----
    if "stn_to_gpe_mean" in model_cfg:
        cfg["w_stn_to_gpe_mean_pA"] = float(model_cfg["stn_to_gpe_mean"])
    if "gpe_to_stn_mean" in model_cfg:
        cfg["w_gpe_to_stn_mean_uAcm2"] = float(model_cfg["gpe_to_stn_mean"])

    # ----- OU noise sigmas -----
    if "noise_sigma_stn" in model_cfg:
        cfg["stn_ou_sigma"] = float(model_cfg["noise_sigma_stn"])
    if "noise_sigma_gpe" in model_cfg:
        cfg["gpe_ou_sigma"] = float(model_cfg["noise_sigma_gpe"])

    # ----- STN intrinsic parameters via stn_params -----
    stn_params = {}
    if "stn_ISTN" in model_cfg:
        stn_params["ISTN"] = float(model_cfg["stn_ISTN"])
    cfg["stn_params"] = stn_params

    # GPe intrinsic params are *not* passed via gpe_params to avoid
    # AdExParams_GPe(**gpe_params) errors. We will rescale I_baseline
    # after the network is built, at the neuron level.

    cfg["gpe_params"] = {}

    # Carry sim times in case we want them logged
    cfg["t_total_s"] = float(t_total_s)
    cfg["burn_in_s"] = float(burn_in_s)

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
    Run a single STN–GPe simulation under (theta, delta).

    Returns:
        {
            "spikes_stn": (T, N_stn),
            "spikes_gpe": (T, N_gpe),
            "V_stn":      (T, N_stn),
            "V_gpe":      (T, N_gpe),
            "dt_ms":      float,
            "burn_steps": int,
        }
    """

    # 1. Apply dopamine modulation at model level (θ → θ(δ))
    model_cfg = apply_delta_parameters(theta, delta)

    # 2. Map model-level cfg → build_network-compatible cfg
    cfg_for_net = _build_network_cfg(
        model_cfg=model_cfg,
        t_total_s=t_total_s,
        burn_in_s=burn_in_s,
        dt_ms=dt_ms,
        seed=seed,
    )

    # 3. Build network
    net, used_cfg = build_network(cfg_for_net)

    dt = float(used_cfg["dt_ms"])
    T = int(round(t_total_s * 1000.0 / dt))
    burn_steps = int(round(burn_in_s * 1000.0 / dt))

    # 3b. Rescale GPe I_baseline using model_cfg["gpe_I_baseline"], if present
    if "gpe_I_baseline" in model_cfg:
        target_I = float(model_cfg["gpe_I_baseline"])
        # Use the canonical GPe preset as reference for scaling
        default_params = AdExParams_GPe()
        base_I = float(default_params.I_baseline) if default_params.I_baseline != 0 else 1.0
        ratio = target_I / base_I

        for cell in net.gpe:
            cell.p.I_baseline = cell.p.I_baseline * ratio

    # 4. Allocate recording buffers
    N_stn = net.n_stn
    N_gpe = net.n_gpe

    V_stn = np.zeros((T, N_stn), dtype=np.float32)
    V_gpe = np.zeros((T, N_gpe), dtype=np.float32)
    spikes_stn = np.zeros((T, N_stn), dtype=np.uint8)
    spikes_gpe = np.zeros((T, N_g
