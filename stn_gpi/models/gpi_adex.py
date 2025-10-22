# STUB: AdEx GPi model — UPDATED (stabilized & GPi-like defaults)
# Expose: init(params), state dict, step(dt), get_observables()

# gpi_adex.py
# Adaptive Exponential Integrate-and-Fire (AdEx) neuron model for GPi
#
# Biophysically grounded tweaks:
#   • Capped exponential term to prevent numerical runaway (Na+ activation is finite).
#   • Explicit absolute refractory (≈2 ms) to reflect Na+ inactivation.
#   • GPi-like defaults: EL ~ -60 mV, VT ~ -52 mV, ΔT ~ 2.5 mV, light adaptation (a ~ 0.5 nS),
#     modest V_reset (-55 mV), baseline tonic drive via I_baseline (pA).
#
# Equations (units: mV, ms, nS, pA, pF):
#   C dV/dt = -gL(V - EL) + gL*ΔT*exp((V - VT)/ΔT) - w + I_total
#   τ_w dw/dt = a*(V - EL) - w
#   If V >= V_peak: spike → V ← V_reset, w ← w + b, enter refractory (t_ref_ms)
#
# I_total = I_baseline + I_ext - I_syn  (all in pA)

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
    EL: float = -60.0       # mV  (GPi-like: slightly depolarized)
    # Exponential spike initiation
    VT: float = -52.0       # mV  (threshold)
    ΔT: float = 2.5         # mV  (slope factor)
    # Adaptation (weak for tonic GPi)
    a: float = 0.5          # nS  (subthreshold adaptation)
    τ_w: float = 120.0      # ms  (adaptation time constant)
    b: float = 0.0          # pA  (spike-triggered increment; keep small or zero)
    # Spike shape / reset
    V_reset: float = -55.0  # mV
    V_peak: float = 20.0    # mV (spike cutoff for detection)
    t_ref_ms: float = 2.0   # ms absolute refractory (physiological)
    # Drive / init
    I_baseline: float = 600.0  # pA, tonic depolarizing drive (tune to hit ~70 Hz)
    V_init: float = -60.0      # mV (start near EL)

    # Numerical guard on exponential
    exp_arg_min: float = -20.0   # clamp range for (V - VT)/ΔT
    exp_arg_max: float = +10.0

@dataclass
class AdExState:
    V: float
    w: float
    ref_remaining_ms: float = 0.0  # absolute refractory counter

class GPiAdEx:
    """
    Single-compartment AdEx neuron tuned for GPi tonic pacemaking.
    Includes capped exponential, explicit refractory, and light adaptation.
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

    # ---------------- Public API ----------------

    def reset(self, V: Optional[float] = None) -> None:
        """Reset to resting potential and clear refractory."""
        self.state = AdExState(V=self.p.V_init if V is None else V, w=0.0, ref_remaining_ms=0.0)

    def step(self, dt_ms: float, I_ext: float = 0.0, I_syn: float = 0.0,
             t_ms: Optional[float] = None) -> Tuple[float, bool]:
        """
        Advance dynamics by one time step (Euler).
        I_ext, I_syn, I_baseline are in pA (positive depolarizes).
        Returns (V, spiked).
        """
        p = self.p
        s = self.state
        spiked = False

        # If in absolute refractory: hold V at reset, let adaptation relax, count down, return.
        if s.ref_remaining_ms > 0.0:
            # Adaptation still evolves during refractory at V_reset
            dw = (p.a * (p.V_reset - p.EL) - s.w) / p.τ_w
            s.w += dt_ms * dw
            s.ref_remaining_ms = max(0.0, s.ref_remaining_ms - dt_ms)
            s.V = p.V_reset
            return s.V, False

        # Total input (pA)
        I_total = p.I_baseline + I_ext - I_syn

        # Exponential current with clamped argument (prevents numerical blow-up)
        exp_arg = (s.V - p.VT) / p.ΔT
        exp_arg = min(max(exp_arg, p.exp_arg_min), p.exp_arg_max)
        I_exp = p.gL * p.ΔT * math.exp(exp_arg)  # nS*mV = pA

        # Voltage update
        dV = (-p.gL * (s.V - p.EL) + I_exp - s.w + I_total) / p.C
        s.V += dt_ms * dV

        # Adaptation update
        dw = (p.a * (s.V - p.EL) - s.w) / p.τ_w
        s.w += dt_ms * dw

        # Spike detection and reset
        if s.V >= p.V_peak:
            spiked = True
            s.V = p.V_reset
            s.w += p.b
            s.ref_remaining_ms = p.t_ref_ms

        return s.V, spiked

    # Optional helper to expose observables
    def get_observables(self) -> dict:
        return {"V": self.state.V, "w": self.state.w, "ref_ms": self.state.ref_remaining_ms}
