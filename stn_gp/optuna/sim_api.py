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
        (+ anything else you later add for Optuna)

    δ:
        0 → normal
        1 → PD (dopamine-depleted)

    This function *does not* know about build_network's internal keys.
    It just returns a dict with the same high-level names, but δ-modulated.
    """
    # Copy so we don’t mutate original
    cfg = dict(theta)

    # Clamp delta to [0, 1] just in case
    delta = max(0.0, min(1.0, float(delta)))

    # ======================================================
    # Dopamine-sensitive parameters (the δ group)
    # ======================================================

    # 1) GPe → STN inhibitory strength increases with δ
    #    PD: stronger GPe inhibition onto STN
    if "gpe_to_stn_mean" in cfg:
        base = float(cfg["gpe_to_stn_mean"])
        cfg["gpe_to_stn_mean"] = base * (1.0 + 1.0 * delta)  # up to 2× at δ=1

    # 2) GPe baseline decreases with δ (weak pacemaking → irregular)
    if "gpe_I_baseline" in cfg:
        base = float(cfg["gpe_I_baseline"])
        # At δ=1, drop to ~50% of baseline
        cfg["gpe_I_baseline"] = max(0.0, base * (1.0 - 0.5 * delta))

    # 3) STN tonic drive increases with δ (hyperactive STN in PD)
    if "stn_ISTN" in cfg:
        base = float(cfg["stn_ISTN"])
        # At δ=1, ~1.5× baseline
        cfg["stn_ISTN"] = base * (1.0 + 0.5 * delta)

    # 4) STN → GPe excitatory gain (moderate effect)
    if "stn_to_gpe_mean" in cfg:
        base = float(cfg["stn_to_gpe_mean"])
        # Small increase in PD
        cfg["stn_to_gpe_mean"] = base * (1.0 + 0.3 * delta)

    # 5) Optional: noise modulation (keep simple for now; can adjust later)
    #    For now, we leave noise_sigma_* as is so that Optuna controls them.
    #    If you want, you can later make STN noise slightly lower in PD
    #    to sharpen beta peaks.

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

    This is where we map:

        stn_to_gpe_mean      → w_stn_to_gpe_mean_pA
        gpe_to_stn_mean      → w_gpe_to_stn_mean_uAcm2
        stn_ISTN             → stn_params["ISTN"]
        gpe_I_baseline       → gpe_params["I_baseline"]
        noise_sigma_stn      → stn_ou_sigma
        noise_sigma_gpe      → gpe_ou_sigma
        n_stn, n_gpe         → n_stn, n_gpe
        delay_*              → delay_*
    """
    cfg: Dict = {}

    # ----- time + seed (build_network only uses dt_ms + seed; t_total/burn used by sim_api) -----
    cfg["dt_ms"] = float(dt_ms)
    if seed is not None:
        cfg["seed"] = int(seed)

    # ----- population sizes -----
    if "n_stn" in model_cfg:
        cfg["n_stn"] = int(model_cfg["n_stn"])
    if "n_gpe" in model_cfg:
        cfg["n_gpe"] = int(model_cfg["n_gpe"])

    # ----- delays -----
    # (build_network already expects these exact keys)
    if "delay_stn_to_gpe_ms" in model_cfg:
        cfg["delay_stn_to_gpe_ms"] = float(model_cfg["delay_stn_to_gpe_ms"])
    if "delay_gpe_to_stn_ms" in model_cfg:
        cfg["delay_gpe_to_stn_ms"] = float(model_cfg["delay_gpe_to_stn_ms"])

    # ----- synaptic weights -----
    # STN → GPe: pA jumps, build_network key = w_stn_to_gpe_mean_pA
    if "stn_to_gpe_mean" in model_cfg:
        cfg["w_stn_to_gpe_mean_pA"] = float(model_cfg["stn_to_gpe_mean"])

    # GPe → STN: µA/cm² jumps, build_network key = w_gpe_to_stn_mean_uAcm2
    if "gpe_to_stn_mean" in model_cfg:
        cfg["w_gpe_to_stn_mean_uAcm2"] = float(model_cfg["gpe_to_stn_mean"])

    # ----- OU noise -----
    # We only override sigma; mu and tau_ms stay as defaults from default_build_config().
    if "noise_sigma_stn" in model_cfg:
        cfg["stn_ou_sigma"] = float(model_cfg["noise_sigma_stn"])
    if "noise_sigma_gpe" in model_cfg:
        cfg["gpe_ou_sigma"] = float(model_cfg["noise_sigma_gpe"])

    # ----- intrinsic parameters via stn_params / gpe_params -----
    stn_params = {}
    gpe_params = {}

    # STN tonic drive ISTN (µA/cm²)
    if "stn_ISTN" in model_cfg:
        stn_params["ISTN"] = float(model_cfg["stn_ISTN"])

    # GPe AdEx baseline current (pA)
    if "gpe_I_baseline" in model_cfg:
        gpe_params["I_baseline"] = float(model_cfg["gpe_I_baseline"])

    # If user wants to pass other overrides in future, we can merge them here.

    cfg["stn_params"] = stn_params
    cfg["gpe_params"] = gpe_params

    # We do NOT touch:
    #   - p_stn_to_gpe, p_gpe_to_stn
    #   - ampa_decay_ms, gaba_decay_ms
    #   - E_AMPA_mV, E_GABA_mV
    #   - OU mu, tau_ms
    # so those remain as in default_build_config().

    # We also carry t_total_s and burn_in_s back to caller; build_network ignores these,
    # but it's convenient to keep everything together if we ever log this cfg.
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

    Parameters
    ----------
    theta : dict
        Optuna-facing base parameters (see optuna_driver.sample_theta).
    delta : float
        Dopamine level in [0, 1]: 0 = normal, 1 = PD-like.
    t_total_s : float
        Total simulation time (seconds).
    burn_in_s : float
        Burn-in time to discard from analysis (seconds).
    dt_ms : float
        Time step (ms).
    seed : int or None
        Optional RNG seed override.

    Returns
    -------
    dict with:
        "spikes_stn": (T, N_stn),
        "spikes_gpe": (T, N_gpe),
        "V_stn":      (T, N_stn),
        "V_gpe":      (T, N_gpe),
        "dt_ms":      float,
        "burn_steps": int,
    """

    # --------------------------------------------------------
    # 1. Apply dopamine modulation (θ → θ(δ)) at model-level
    # --------------------------------------------------------
    model_cfg = apply_delta_parameters(theta, delta)

    # --------------------------------------------------------
    # 2. Map model-level config → build_network-compatible cfg
    # --------------------------------------------------------
    cfg_for_net = _build_network_cfg(
        model_cfg=model_cfg,
        t_total_s=t_total_s,
        burn_in_s=burn_in_s,
        dt_ms=dt_ms,
        seed=seed,
    )

    # --------------------------------------------------------
    # 3. Build network (existing code)
    # --------------------------------------------------------
    net, used_cfg = build_network(cfg_for_net)

    dt = float(used_cfg["dt_ms"])
    T = int(round(t_total_s * 1000.0 / dt))
    burn_steps = int(round(burn_in_s * 1000.0 / dt))

    # --------------------------------------------------------
    # 4. Allocate recording buffers
    # --------------------------------------------------------
    N_stn = net.n_stn
    N_gpe = net.n_gpe

    V_stn = np.zeros((T, N_stn), dtype=np.float32)
    V_gpe = np.zeros((T, N_gpe), dtype=np.float32)
    spikes_stn = np.zeros((T, N_stn), dtype=np.uint8)
    spikes_gpe = np.zeros((T, N_gpe), dtype=np.uint8)

    # --------------------------------------------------------
    # 5. Simulation loop
    # --------------------------------------------------------
    t_ms = 0.0
    for t in range(T):
        # step_once returns:
        #   V_stn_t, spk_stn_t, V_gpe_t, spk_gpe_t
        V_STN_t, spk_STN_t, V_GPE_t, spk_GPE_t = step_once(net, t_ms=t_ms)

        V_stn[t, :] = V_STN_t
        V_gpe[t, :] = V_GPE_t
        spikes_stn[t, :] = spk_STN_t
        spikes_gpe[t, :] = spk_GPE_t

        t_ms += dt

    # --------------------------------------------------------
    # 6. Return in a clean standardized format
    # --------------------------------------------------------
    return {
        "spikes_stn": spikes_stn,
        "spikes_gpe": spikes_gpe,
        "V_stn": V_stn,
        "V_gpe": V_gpe,
        "dt_ms": dt,
        "burn_steps": burn_steps,
    }
