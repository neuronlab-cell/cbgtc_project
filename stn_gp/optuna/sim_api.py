# sim_api.py
# Clean simulation API for Optuna.
#
# Responsibilities:
#   • Accept (theta, delta) and simulation settings
#   • Apply dopamine-modulated parameter transforms
#   • Build STN–GPe network using your existing build_network()
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


# ============================================================
# δ-MODULATED PARAMETER TRANSFORM
# ============================================================

def apply_delta_parameters(theta: Dict, delta: float) -> Dict:
    """
    Given base parameters θ and dopamine level δ ∈ [0,1],
    return a parameter dict compatible with build_network().
    
    θ contains:
        - stn_to_gpe_mean
        - gpe_to_stn_mean
        - stn_ISTN
        - gpe_I_baseline
        - delays
        - noise levels
        - anything else you add later

    δ:
        0 → normal
        1 → PD (dopamine-depleted)
    """
    # Copy so we don’t mutate original
    cfg = dict(theta)

    # ======================================================
    # Dopamine-sensitive parameters (the δ group)
    # ======================================================

    # 1) GPe → STN inhibitory strength increases with δ
    if "gpe_to_stn_mean" in cfg:
        base = cfg["gpe_to_stn_mean"]
        cfg["gpe_to_stn_mean"] = base * (1.0 + 1.0 * delta)   # scale factor chosen conservatively

    # 2) GPe baseline decreases with δ (weak pacemaking → irregular)
    if "gpe_I_baseline" in cfg:
        base = cfg["gpe_I_baseline"]
        cfg["gpe_I_baseline"] = base * (1.0 - 0.5 * delta)

    # 3) STN tonic drive increases with δ (hyperactive in PD)
    if "stn_ISTN" in cfg:
        base = cfg["stn_ISTN"]
        cfg["stn_ISTN"] = base * (1.0 + 0.5 * delta)

    # 4) Optional: STN → GPe excitatory gain (moderate effect)
    if "stn_to_gpe_mean" in cfg:
        base = cfg["stn_to_gpe_mean"]
        cfg["stn_to_gpe_mean"] = base * (1.0 + 0.3 * delta)

    # Everything else stays unchanged (delays, noise, etc.)
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
            "V_stn": (T, N_stn),
            "V_gpe": (T, N_gpe),
            "dt_ms": float,
            "burn_steps": int,
        }
    """

    # --------------------------------------------------------
    # 1. Apply dopamine modulation (θ → θ(δ))
    # --------------------------------------------------------
    cfg = apply_delta_parameters(theta, delta)

    # Insert dt and sim times if user didn't include them
    cfg["dt_ms"] = dt_ms
    cfg["t_total_s"] = t_total_s
    cfg["burn_in_s"] = burn_in_s
    if seed is not None:
        cfg["seed"] = seed

    # --------------------------------------------------------
    # 2. Build network (existing code)
    # --------------------------------------------------------
    net, used_cfg = build_network(cfg)

    dt = used_cfg["dt_ms"]
    T = int(round(t_total_s * 1000.0 / dt))
    burn_steps = int(round(burn_in_s * 1000.0 / dt))

    # --------------------------------------------------------
    # 3. Allocate recording buffers
    # --------------------------------------------------------
    N_stn = net.n_stn
    N_gpe = net.n_gpe

    V_stn = np.zeros((T, N_stn), dtype=np.float32)
    V_gpe = np.zeros((T, N_gpe), dtype=np.float32)
    spikes_stn = np.zeros((T, N_stn), dtype=np.uint8)
    spikes_gpe = np.zeros((T, N_gpe), dtype=np.uint8)

    # --------------------------------------------------------
    # 4. Simulation loop
    # --------------------------------------------------------
    t_ms = 0.0
    for t in range(T):
        # step_once returns:
        #   V_stn, spk_stn, V_gpe, spk_gpe
        V_STN_t, spk_STN_t, V_GPE_t, spk_GPE_t = step_once(net, t_ms=t_ms)

        V_stn[t, :] = V_STN_t
        V_gpe[t, :] = V_GPE_t
        spikes_stn[t, :] = spk_STN_t
        spikes_gpe[t, :] = spk_GPE_t

        t_ms += dt

    # --------------------------------------------------------
    # 5. Return in a clean standardized format
    # --------------------------------------------------------
    return {
        "spikes_stn": spikes_stn,
        "spikes_gpe": spikes_gpe,
        "V_stn": V_stn,
        "V_gpe": V_gpe,
        "dt_ms": dt,
        "burn_steps": burn_steps,
    }
