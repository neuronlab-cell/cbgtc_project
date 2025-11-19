# build_network.py
# Build the STN/GPe/GPi network: populations, connectivity, synapses, delays, and background drive.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from ..models.stn_light_hh import STNLightHH, STNLightHHParams
from ..models.gp_adex import GPAdEx, AdExParams_GPe, AdExParams_GPi
from ..models.synapses import (
    SynapseConfig,
    ExponentialSynapsesCurrent,   # pA (for AdEx targets: GPe, GPi)
    ExponentialSynapsesDensity,   # µA/cm² (for HH targets: STN)
)
from ..models.noise import OUProcess, OUConfig


# ------------------------------------------------------------
# Utility: random connectivity and weights
# ------------------------------------------------------------
def sample_connectivity(
    n_pre: int,
    n_post: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a boolean matrix (n_post, n_pre) where True means a connection exists."""
    return rng.random((n_post, n_pre)) < p


def weight_matrix(
    mask: np.ndarray,
    w_mean: float,
    w_cv: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a weight matrix with log-normal distributed weights on existing connections.

    Parameters
    ----------
    mask : (n_post, n_pre) bool
        Connectivity mask; True entries receive weights.
    w_mean : float
        Mean weight on existing connections.
    w_cv : float
        Coefficient of variation of weights on existing connections.
    """
    n_post, n_pre = mask.shape
    W = np.zeros((n_post, n_pre), dtype=np.float32)
    if np.any(mask) and w_mean > 0.0:
        # Log-normal parameters from mean + CV
        sigma = np.sqrt(np.log(1.0 + w_cv**2)) if w_cv > 0.0 else 0.0
        mu = np.log(w_mean) - 0.5 * sigma**2
        samples = np.exp(rng.normal(mu, sigma, size=mask.sum())).astype(np.float32)
        W[mask] = samples
    return W


# ------------------------------------------------------------
# Lightweight default config
# ------------------------------------------------------------
def default_build_config() -> Dict:
    """
    Defaults used if a field is not supplied in the incoming cfg dict.
    These match the values assumed in run_loop.py and sim_api.py.
    """
    return dict(
        seed=42,
        dt_ms=0.025,

        # population sizes
        n_stn=50,
        n_gpe=100,
        n_gpi=80,

        # connectivity probabilities
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

        # synapse reversal (mV)
        E_AMPA_mV=0.0,
        E_GABA_mV=-70.0,

        # synapse weights (jump sizes upon spike)
        # units:
        #   STN→GPe, STN→GPi, GPe→GPi : pA (ExponentialSynapsesCurrent)
        #   GPe→STN                   : µA/cm² (ExponentialSynapsesDensity)
        w_stn_to_gpe_mean_pA=22.0,
        w_stn_to_gpe_cv=0.20,
        w_gpe_to_stn_mean_uAcm2=0.09,
        w_gpe_to_stn_cv=0.20,
        w_stn_to_gpi_mean_pA=18.0,
        w_stn_to_gpi_cv=0.20,
        w_gpe_to_gpi_mean_pA=25.0,
        w_gpe_to_gpi_cv=0.20,

        # background drive (OU) per population
        stn_ou_mu=1.8,       # µA/cm²
        stn_ou_sigma=0.15,   # µA/cm²
        stn_ou_tau_ms=8.0,

        gpe_ou_mu=0.0,       # pA
        gpe_ou_sigma=30.0,   # pA
        gpe_ou_tau_ms=8.0,

        gpi_ou_mu=0.0,       # pA
        gpi_ou_sigma=30.0,   # pA
        gpi_ou_tau_ms=8.0,

        # model param overrides for STN only (HH dataclass accepts kwargs)
        stn_params={},

        # GPe / GPi param overrides are currently unused; we rely on AdExParams_GPe/GPi presets.
        gpe_params={},
        gpi_params={},

        # intrinsic heterogeneity (coefficient of variation)
        stn_hetero_ISTN_cv=0.10,
        gpe_hetero_Ibaseline_cv=0.10,
        gpi_hetero_Ibaseline_cv=0.10,
    )


# ------------------------------------------------------------
# Simple container for the network
# ------------------------------------------------------------
@dataclass
class Network:
    rng: np.random.Generator
    dt_ms: float

    # populations
    stn: list[STNLightHH]
    gpe: list[GPAdEx]
    gpi: list[GPAdEx]

    # synapses
    syn_stn_to_gpe: ExponentialSynapsesCurrent
    syn_gpe_to_stn: ExponentialSynapsesDensity
    syn_stn_to_gpi: ExponentialSynapsesCurrent
    syn_gpe_to_gpi: ExponentialSynapsesCurrent

    # background noise
    stn_ou: OUProcess
    gpe_ou: OUProcess
    gpi_ou: OUProcess

    # counts
    n_stn: int
    n_gpe: int
    n_gpi: int


# ------------------------------------------------------------
# Builder function
# ------------------------------------------------------------
def _clone_dataclass(obj, **updates):
    """Utility: shallow clone of a dataclass with some fields updated."""
    d = obj.__dict__.copy()
    d.update(updates)
    return obj.__class__(**d)


def build_network(cfg: Optional[Dict] = None) -> Tuple[Network, Dict]:
    """
    Create the STN/GPe/GPi network and return (network, cfg_used).

    Parameters
    ----------
    cfg : dict or None
        Overrides for the default_build_config() dictionary.

    Returns
    -------
    net : Network
        Network object consumed by integrators.step_once().
    cfg_used : dict
        The final configuration dict after applying defaults + overrides.
    """
    base = default_build_config()
    if cfg:
        base.update(cfg)

    rng = np.random.default_rng(base["seed"])
    dt = float(base["dt_ms"])

    n_stn = int(base["n_stn"])
    n_gpe = int(base["n_gpe"])
    n_gpi = int(base["n_gpi"])

    # ---------------- 1. Populations ----------------

    # STN: HH with optional overrides
    base_stn_params = STNLightHHParams(**base["stn_params"]) if base["stn_params"] else STNLightHHParams()

    # GPe / GPi: AdEx presets (we ignore gpe_params/gpi_params for now to avoid mismatched kwargs)
    base_gpe_params = AdExParams_GPe()
    base_gpi_params = AdExParams_GPi()

    cv_ISTN = float(base.get("stn_hetero_ISTN_cv", 0.0))
    cv_Ibase_gpe = float(base.get("gpe_hetero_Ibaseline_cv", 0.0))
    cv_Ibase_gpi = float(base.get("gpi_hetero_Ibaseline_cv", 0.0))

    # STN population
    stn: list[STNLightHH] = []
    for _ in range(n_stn):
        if cv_ISTN > 0.0:
            scale = 1.0 + rng.normal(0.0, cv_ISTN)
            ISTN_i = max(0.0, base_stn_params.ISTN * scale)
            p_i = _clone_dataclass(base_stn_params, ISTN=ISTN_i)
        else:
            p_i = base_stn_params
        stn.append(STNLightHH(params=p_i, rng=rng))

    # GPe population
    gpe: list[GPAdEx] = []
    for _ in range(n_gpe):
        if cv_Ibase_gpe > 0.0:
            scale = 1.0 + rng.normal(0.0, cv_Ibase_gpe)
            I_base_i = max(0.0, base_gpe_params.I_baseline * scale)
            p_i = _clone_dataclass(base_gpe_params, I_baseline=I_base_i)
        else:
            p_i = base_gpe_params
        gpe.append(GPAdEx(params=p_i, rng=rng))

    # GPi population
    gpi: list[GPAdEx] = []
    for _ in range(n_gpi):
        if cv_Ibase_gpi > 0.0:
            scale = 1.0 + rng.normal(0.0, cv_Ibase_gpi)
            I_base_i = max(0.0, base_gpi_params.I_baseline * scale)
            p_i = _clone_dataclass(base_gpi_params, I_baseline=I_base_i)
        else:
            p_i = base_gpi_params
        gpi.append(GPAdEx(params=p_i, rng=rng))

    # ---------------- 2. Connectivity ----------------
    # STN → GPe
    mask_s2g = sample_connectivity(
        n_pre=n_stn,
        n_post=n_gpe,
        p=float(base["p_stn_to_gpe"]),
        rng=rng,
    )
    # GPe → STN
    mask_g2s = sample_connectivity(
        n_pre=n_gpe,
        n_post=n_stn,
        p=float(base["p_gpe_to_stn"]),
        rng=rng,
    )
    # STN → GPi
    mask_s2i = sample_connectivity(
        n_pre=n_stn,
        n_post=n_gpi,
        p=float(base["p_stn_to_gpi"]),
        rng=rng,
    )
    # GPe → GPi
    mask_g2i = sample_connectivity(
        n_pre=n_gpe,
        n_post=n_gpi,
        p=float(base["p_gpe_to_gpi"]),
        rng=rng,
    )

    # ---------------- 3. Weights ----------------
    W_s2g = weight_matrix(
        mask_s2g,
        float(base["w_stn_to_gpe_mean_pA"]),
        float(base["w_stn_to_gpe_cv"]),
        rng,
    )
    W_g2s = weight_matrix(
        mask_g2s,
        float(base["w_gpe_to_stn_mean_uAcm2"]),
        float(base["w_gpe_to_stn_cv"]),
        rng,
    )
    W_s2i = weight_matrix(
        mask_s2i,
        float(base["w_stn_to_gpi_mean_pA"]),
        float(base["w_stn_to_gpi_cv"]),
        rng,
    )
    W_g2i = weight_matrix(
        mask_g2i,
        float(base["w_gpe_to_gpi_mean_pA"]),
        float(base["w_gpe_to_gpi_cv"]),
        rng,
    )

    # ---------------- 4. Synapses ----------------
    syn_stn_to_gpe = ExponentialSynapsesCurrent(
        SynapseConfig(
            n_pre=n_stn,
            n_post=n_gpe,
            dt_ms=dt,
            tau_decay_ms=float(base["ampa_decay_ms"]),
            E_rev_mV=float(base["E_AMPA_mV"]),
            delay_ms=float(base["delay_stn_to_gpe_ms"]),
            W=W_s2g,
        )
    )

    syn_gpe_to_stn = ExponentialSynapsesDensity(
        SynapseConfig(
            n_pre=n_gpe,
            n_post=n_stn,
            dt_ms=dt,
            tau_decay_ms=float(base["gaba_decay_ms"]),
            E_rev_mV=float(base["E_GABA_mV"]),
            delay_ms=float(base["delay_gpe_to_stn_ms"]),
            W=W_g2s,
        )
    )

    syn_stn_to_gpi = ExponentialSynapsesCurrent(
        SynapseConfig(
            n_pre=n_stn,
            n_post=n_gpi,
            dt_ms=dt,
            tau_decay_ms=float(base["ampa_decay_ms"]),
            E_rev_mV=float(base["E_AMPA_mV"]),
            delay_ms=float(base["delay_stn_to_gpi_ms"]),
            W=W_s2i,
        )
    )

    syn_gpe_to_gpi = ExponentialSynapsesCurrent(
        SynapseConfig(
            n_pre=n_gpe,
            n_post=n_gpi,
            dt_ms=dt,
            tau_decay_ms=float(base["gaba_decay_ms"]),
            E_rev_mV=float(base["E_GABA_mV"]),
            delay_ms=float(base["delay_gpe_to_gpi_ms"]),
            W=W_g2i,
        )
    )

    # ---------------- 5. Background drive (OU noise) ----------------
    stn_ou = OUProcess(
        OUConfig(
            n=n_stn,
            dt_ms=dt,
            tau_ms=float(base["stn_ou_tau_ms"]),
            mu=float(base["stn_ou_mu"]),
            sigma=float(base["stn_ou_sigma"]),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
    )

    gpe_ou = OUProcess(
        OUConfig(
            n=n_gpe,
            dt_ms=dt,
            tau_ms=float(base["gpe_ou_tau_ms"]),
            mu=float(base["gpe_ou_mu"]),
            sigma=float(base["gpe_ou_sigma"]),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
    )

    gpi_ou = OUProcess(
        OUConfig(
            n=n_gpi,
            dt_ms=dt,
            tau_ms=float(base["gpi_ou_tau_ms"]),
            mu=float(base["gpi_ou_mu"]),
            sigma=float(base["gpi_ou_sigma"]),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
    )

    # ---------------- 6. Package ----------------
    net = Network(
        rng=rng,
        dt_ms=dt,
        stn=stn,
        gpe=gpe,
        gpi=gpi,
        syn_stn_to_gpe=syn_stn_to_gpe,
        syn_gpe_to_stn=syn_gpe_to_stn,
        syn_stn_to_gpi=syn_stn_to_gpi,
        syn_gpe_to_gpi=syn_gpe_to_gpi,
        stn_ou=stn_ou,
        gpe_ou=gpe_ou,
        gpi_ou=gpi_ou,
        n_stn=n_stn,
        n_gpe=n_gpe,
        n_gpi=n_gpi,
    )
    return net, base
