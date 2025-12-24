# gp_adex.py
# Unified Adaptive Exponential (AdEx) neuron for pallidal cells (GPe & GPi)
#
# One backbone (GPAdEx), two parameter presets:
#   - AdExParams_GPi(): tonic, regular pacemaker (validated)
#   - AdExParams_GPe(): more adaptive/rebound-prone, irregular (to validate)
#
# Biologically grounded/stable features:
#   • Capped exponential term to prevent numerical runaway (Na+ activation saturates)
#   • Explicit absolute refractory (≈2 ms) to reflect Na+ inactivation
#
# Units: mV, ms, nS, pA, pF
# Dynamics:
#   C dV/dt = -gL(V-EL) + gL*ΔT*exp((V-VT)/ΔT) - w + I_total
#   τ_w dw/dt = a*(V-EL) - w
#   If V >= V_peak: spike → V←V_reset, w←w+b, refractory countdown

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import math

@dataclass
class AdExParams:
    # Membrane / leak
    C: float = 200.0        # pF
    gL: float = 10.0        # nS
    EL: float = -60.0       # mV
    # Exponential spike initiation
    VT: float = -52.0       # mV
    dT: float = 2.5         # mV  (ΔT; named dT to be ASCII-safe)
    # Adaptation
    a: float = 0.5          # nS
    tau_w: float = 120.0    # ms
    b: float = 0.0          # pA
    # Spike / reset / refractory
    V_reset: float = -55.0  # mV
    V_peak: float = 20.0    # mV
    t_ref_ms: float = 2.0   # ms
    # Drive / init
    I_baseline: float = 200.0  # pA
    V_init: float = -60.0      # mV
    # Numerical guard on exponential argument ( (V-VT)/dT )
    exp_arg_min: float = -20.0
    exp_arg_max: float = +10.0

@dataclass
class AdExState:
    V: float
    w: float
    ref_remaining_ms: float = 0.0  # absolute refractory timer

class GPAdEx:
    """
    Adaptive Exponential IF neuron for pallidal cells (GPe/GPi).

    Methods
    -------
    step(dt_ms, I_ext=0.0, I_syn=0.0, t_ms=None) -> (V, spiked)
    reset(V=None)
    get_observables() -> dict
    """

    def __init__(self, params: Optional[AdExParams | dict] = None,
                 rng: Optional[np.random.Generator] = None):
        if isinstance(params, dict):
            self.p = AdExParams(**params)
        elif isinstance(params, AdExParams) or params is None:
            self.p = params if isinstance(params, AdExParams) else AdExParams()
        else:
            raise TypeError("params must be dict, AdExParams, or None")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.state = AdExState(V=self.p.V_init, w=0.0, ref_remaining_ms=0.0)

    def reset(self, V: Optional[float] = None) -> None:
        self.state = AdExState(V=self.p.V_init if V is None else V, w=0.0, ref_remaining_ms=0.0)

    def step(self, dt_ms: float, I_ext: float = 0.0, I_syn: float = 0.0,
             t_ms: Optional[float] = None) -> Tuple[float, bool]:
        """
        Advance one time step.
        I_ext, I_syn, I_baseline in pA (positive depolarizes). Returns (V, spiked).
        """
        p = self.p
        s = self.state
        spiked = False

        # Absolute refractory: hold V at reset; w evolves at V_reset
        if s.ref_remaining_ms > 0.0:
            dw = (p.a * (p.V_reset - p.EL) - s.w) / p.tau_w
            s.w += dt_ms * dw
            s.ref_remaining_ms = max(0.0, s.ref_remaining_ms - dt_ms)
            s.V = p.V_reset
            return s.V, False

        # Total current
        I_total = p.I_baseline + I_ext - I_syn  # pA

        # Exponential term (clamped)
        exp_arg = (s.V - p.VT) / p.dT
        exp_arg = min(max(exp_arg, p.exp_arg_min), p.exp_arg_max)
        I_exp = p.gL * p.dT * math.exp(exp_arg)  # nS*mV = pA

        # Voltage update
        dV = ( -p.gL*(s.V - p.EL) + I_exp - s.w + I_total ) / p.C
        s.V += dt_ms * dV

        # Adaptation update
        dw = ( p.a*(s.V - p.EL) - s.w ) / p.tau_w
        s.w += dt_ms * dw

        # Spike/reset
        if s.V >= p.V_peak:
            spiked = True
            s.V = p.V_reset
            s.w += p.b
            s.ref_remaining_ms = p.t_ref_ms

        return s.V, spiked

    def get_observables(self) -> dict:
        return {"V": self.state.V, "w": self.state.w, "ref_ms": self.state.ref_remaining_ms}

# ----------- Parameter presets -----------

def AdExParams_GPi() -> AdExParams:
    """
    Validated GPi preset (tonic, regular ~60–80 Hz with I_baseline≈200 pA).
    Matches your single-cell tuning: EL=-60, VT=-52, ΔT=2.5, a=0.5, tau_w=120, b=0, V_reset=-55, tref=2ms.
    """
    return AdExParams(
        C=200.0, gL=10.0,
        EL=-60.0, VT=-52.0, dT=2.5,
        a=0.5, tau_w=120.0, b=0.0,
        V_reset=-55.0, V_peak=20.0, t_ref_ms=2.0,
        I_baseline=201.5,  # ~70 Hz with burn-in in your diagnostics
        V_init=-60.0,
        exp_arg_min=-20.0, exp_arg_max=+10.0,
    )

def AdExParams_GPe() -> AdExParams:
    """
    Biologically plausible GPe preset (more adaptive/irregular; to be validated):
      - Lower leak reversal (EL ~ -65 mV), slightly easier threshold (VT ~ -50 mV)
      - Softer spike initiation (ΔT ~ 3.5 mV)
      - Stronger & slower adaptation (a ~ 3 nS, tau_w ~ 250 ms, b ~ 30 pA)
      - Slightly more negative reset (-60 mV)
      - Baseline current chosen to land ~40–60 Hz once validated
    Expect higher CV than GPi and better propensity to participate in STN–GPe beta loops.
    """
    return AdExParams(
        C=200.0, gL=10.0,
        EL=-60.0, VT=-50.0, dT=3.5,
        a=2.5, tau_w=250.0, b=27.0,
        V_reset=-60.0, V_peak=20.0, t_ref_ms=2.0,
        I_baseline=200.5,  # initial guess; we will tune to ~45–55 Hz
        V_init=-65.0,
        exp_arg_min=-20.0, exp_arg_max=+10.0,
    )
