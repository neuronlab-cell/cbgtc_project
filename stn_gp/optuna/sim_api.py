# sim_api.py
# Clean simulation API for (normal) Optuna tuning.
#
# Responsibilities:
#   • Accept a parameter dict theta (no longer focused on synaptic weights)
#   • (Optionally) apply a dopamine modulation delta, but for now δ=0 is identity
#   • Map theta → build_network() config (drive / noise / sizes)
#   • Build STN–GPe–GPi network using existing build_network()
#   • Apply post-build scaling for:
#       - STN mean ISTN
#       - GPe / GPi mean I_baseline
#       - Intrinsic scalers (e.g., STN H/AHP, GPe adaptation)
#   • Run simulation using step_once()
#   • Return arrays ready for scoring in objectives.py
#
# Expected theta keys (you can subset/extend in optuna_driver):
#
#   Drives:
#     - "ISTN_mean"         : target mean tonic current for STN (µA/cm²)
#     - "I_baseline_GPe"    : target mean I_baseline (pA) for GPe
#     - "I_baseline_GPi"    : target mean I_baseline (pA) for GPi
#
#   Intrinsic scalers (optional):
#     - "stn_intrinsic_scale" : scales gH and gAHP in STN cells
#     - "gpe_adapt_scale"     : scales a and b in GPe AdEx
#     - "gpi_adapt_scale"     : (reserved, currently unused unless set)
#
#   Noise (OU sigmas, optional):
#     - "noise_sigma_stn"   : overrides stn_ou_sigma
#     - "noise_sigma_gpe"   : overrides gpe_ou_sigma
#     - "noise_sigma_gpi"   : overrides gpi_ou_sigma (if used in build_network)
#
#   Population sizes (optional, otherwise defaults in build_network):
#     - "n_stn", "n_gpe", "n_gpi"
#
# NOTE:
#   • delta is currently not used to change parameters (normal-only mode),
#     but is kept in the API so we can later re-introduce PD tuning.
#   • We deliberately do NOT touch synaptic weights or connection probabilities here.

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from stn_gp.sim.build_network import build_network
from stn_gp.sim.integrators import step_once


# ============================================================
# δ-MODULATED PARAMETER TRANSFORM (currently identity for δ=0)
# ============================================================

def apply_delta_parameters(theta: Dict, delta: float) -> Dict:
    """
    Given base parameters θ and dopamine level δ ∈ [0,1],
    return a dict after δ-modulation.

    Current mode: normal-only.
    So δ is ignored and we simply clamp it and return θ as-is.

    This wrapper is kept so that when you later want to
    introduce PD (δ=1), you can add those transforms here.
    """
    cfg = dict(theta)
    _ = max(0.0, min(1.0, float(delta)))  # clamp, but unused
    return cfg


# ============================================================
# MAP theta → build_network() CONFIG
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

    We do not set synaptic weights or connection probabilities here.
    We only:
      • set dt, seed
      • optionally set population sizes
      • set OU noise sigmas
      • pass empty {stn,gpe,gpi}_params (post-build scaling is applied later)
      • record t_total_s / burn_in_s for bookkeeping
    """
    cfg: Dict = {}

    # Time / seed
    cfg["dt_ms"] = float(dt_ms)
    if seed is not None:
        cfg["seed"] = int(seed)

    # Population sizes (if present)
    if "n_stn" in model_cfg:
        cfg["n_stn"] = int(model_cfg["n_stn"])
    if "n_gpe" in model_cfg:
        cfg["n_gpe"] = int(model_cfg["n_gpe"])
    if "n_gpi" in model_cfg:
        cfg["n_gpi"] = int(model_cfg["n_gpi"])

    # Noise sigmas (OU)
    if "noise_sigma_stn" in model_cfg:
        cfg["stn_ou_sigma"] = float(model_cfg["noise_sigma_stn"])
    if "noise_sigma_gpe" in model_cfg:
        cfg["gpe_ou_sigma"] = float(model_cfg["noise_sigma_gpe"])
    if "noise_sigma_gpi" in model_cfg:
        cfg["gpi_ou_sigma"] = float(model_cfg["noise_sigma_gpi"])

    # Intrinsic parameter dicts left empty:
    # we will scale post-build to avoid constructor signature issues.
    cfg["stn_params"] = {}
    cfg["gpe_params"] = {}
    cfg["gpi_params"] = {}

    # For logging/debug
    cfg["t_total_s"] = float(t_total_s)
    cfg["burn_in_s"] = float(burn_in_s)

    return cfg


# ============================================================
# POST-BUILD SCALING HELPERS
# ============================================================

def _safe_mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _rescale_mean_attribute(cells, attr_name: str, target_mean: float) -> None:
    """
    Given a list of neuron objects whose .p has an attribute attr_name,
    rescale that attribute multiplicatively so that its mean equals target_mean.
    """
    if target_mean <= 0.0:
        return

    current_vals = [getattr(cell.p, attr_name) for cell in cells]
    current_mean = _safe_mean(current_vals)
    if current_mean <= 0.0:
        return

    ratio = target_mean / current_mean
    for cell in cells:
        setattr(cell.p, attr_name, getattr(cell.p, attr_name) * ratio)


def _apply_intrinsic_scalers(model_cfg: Dict, net) -> None:
    """
    Apply optional small multiplicative scalers to intrinsic parameters,
    after the network has been built.

    Expected keys in model_cfg:
      - "stn_intrinsic_scale": scale gH and gAHP in STN cells
      - "gpe_adapt_scale": scale a and b in GPe cells
      - "gpi_adapt_scale": (optional) scale a and b in GPi cells
    """
    # STN: scale H-current and AHP conductances
    stn_scale = float(model_cfg.get("stn_intrinsic_scale", 1.0))
    if stn_scale != 1.0 and hasattr(net, "stn"):
        for cell in net.stn:
            if hasattr(cell.p, "gH"):
                cell.p.gH *= stn_scale
            if hasattr(cell.p, "gAHP"):
                cell.p.gAHP *= stn_scale

    # GPe: scale adaptation parameters a and b
    gpe_scale = float(model_cfg.get("gpe_adapt_scale", 1.0))
    if gpe_scale != 1.0 and hasattr(net, "gpe"):
        for cell in net.gpe:
            if hasattr(cell.p, "a"):
                cell.p.a *= gpe_scale
            if hasattr(cell.p, "b"):
                cell.p.b *= gpe_scale

    # GPi: optional adaptation scaling (if you decide to use it)
    gpi_scale = float(model_cfg.get("gpi_adapt_scale", 1.0))
    if gpi_scale != 1.0 and hasattr(net, "gpi"):
        for cell in net.gpi:
            if hasattr(cell.p, "a"):
                cell.p.a *= gpi_scale
            if hasattr(cell.p, "b"):
                cell.p.b *= gpi_scale


def _apply_drive_targets(model_cfg: Dict, net) -> None:
    """
    Apply mean-drive targets (ISTN_mean, I_baseline_GPe, I_baseline_GPi)
    via multiplicative rescaling of the existing per-cell parameters.
    """
    # STN ISTN
    if "ISTN_mean" in model_cfg and hasattr(net, "stn"):
        target = float(model_cfg["ISTN_mean"])
        # Each STN cell has cell.p.ISTN; we rescale to match the desired mean.
        _rescale_mean_attribute(net.stn, "ISTN", target)

    # GPe I_baseline
    if "I_baseline_GPe" in model_cfg and hasattr(net, "gpe"):
        target = float(model_cfg["I_baseline_GPe"])
        _rescale_mean_attribute(net.gpe, "I_baseline", target)

    # GPi I_baseline
    if "I_baseline_GPi" in model_cfg and hasattr(net, "gpi"):
        target = float(model_cfg["I_baseline_GPi"])
        _rescale_mean_attribute(net.gpi, "I_baseline", target)


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
    Run a STN–GPe–GPi simulation for given (θ, δ).

    Current mode: "normal-only" — δ is passed through apply_delta_parameters
    but does not yet change anything. That function is kept so you can
    later reintroduce PD transforms.

    Returns:
        {
            "spikes_stn": (T, N_stn) uint8,
            "spikes_gpe": (T, N_gpe) uint8,
            "spikes_gpi": (T, N_gpi) uint8,
            "V_stn":      (T, N_stn) float32,
            "V_gpe":      (T, N_gpe) float32,
            "V_gpi":      (T, N_gpi) float32,
            "dt_ms":      float,
            "burn_steps": int,
        }
    """
    # 1) Dopamine modulation (currently identity for δ=0 normal regime)
    model_cfg = apply_delta_parameters(theta, delta)

    # 2) Convert to build_network cfg (time, sizes, noise, empty param dicts)
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

    # 4) Apply post-build drive targets and intrinsic scalers
    _apply_drive_targets(model_cfg, net)
    _apply_intrinsic_scalers(model_cfg, net)

    # 5) Allocate recording buffers
    N_stn = net.n_stn
    N_gpe = net.n_gpe
    N_gpi = getattr(net, "n_gpi", len(getattr(net, "gpi", [])))

    V_stn = np.zeros((T, N_stn), dtype=np.float32)
    V_gpe = np.zeros((T, N_gpe), dtype=np.float32)
    V_gpi = np.zeros((T, N_gpi), dtype=np.float32)

    spikes_stn = np.zeros((T, N_stn), dtype=np.uint8)
    spikes_gpe = np.zeros((T, N_gpe), dtype=np.uint8)
    spikes_gpi = np.zeros((T, N_gpi), dtype=np.uint8)

    # 6) Time stepping
    t_ms = 0.0
    for t in range(T):
        # step_once(net, t_ms) now returns STN, GPe, GPi
        V_STN_t, spk_STN_t, V_GPE_t, spk_GPE_t, V_GPI_t, spk_GPI_t = step_once(
            net,
            t_ms=t_ms,
        )

        V_stn[t, :] = V_STN_t
        V_gpe[t, :] = V_GPE_t
        V_gpi[t, :] = V_GPI_t

        spikes_stn[t, :] = spk_STN_t
        spikes_gpe[t, :] = spk_GPE_t
        spikes_gpi[t, :] = spk_GPI_t

        t_ms += dt

    # 7) Package results
    return {
        "spikes_stn": spikes_stn,
        "spikes_gpe": spikes_gpe,
        "spikes_gpi": spikes_gpi,
        "V_stn": V_stn,
        "V_gpe": V_gpe,
        "V_gpi": V_gpi,
        "dt_ms": dt,
        "burn_steps": burn_steps,
    }
