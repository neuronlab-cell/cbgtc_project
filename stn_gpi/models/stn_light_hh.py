# STUB: 4-current STN model (Na, K, T, Leak) + SK-type AHP — NO CODE YET
# Expose: init(params), state dict, advance(dt), get_observables()

# stn_light_hh.py
# Single-compartment STN neuron:
#   I_Na (fast sodium), I_K (delayed rectifier potassium),
#   I_T (low-threshold T-type calcium), I_L (leak),
#   I_AHP (SK-type Ca2+-activated K+; biologically grounded)
#
# Membrane equation (mV, ms, mS/cm^2, µA/cm^2):
#   C_m dV/dt = - (I_Na + I_K + I_T + I_L + I_AHP + I_syn) + (I_ext + I_bias)
#
# Spike detection: upward crossing of V_spike_thresh (no reset; HH dynamics generate spikes)

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from typing import Optional, Dict, Tuple


def _safe_div(num: float, den: float, eps: float = 1e-9) -> float:
    return num / (den if abs(den) > eps else (eps if den >= 0 else -eps))


def _exp(x: float) -> float:
    # guard extreme values for numerical stability
    if x > 50:
        return math.exp(50)
    if x < -50:
        return math.exp(-50)
    return math.exp(x)


@dataclass
class STNLightHHParams:
    # Conductances (mS/cm^2)
    gNa: float = 30.0
    gK: float = 40.0
    gT: float = 2.5
    gL: float = 0.02
    gAHP: float = 0.20   # SK-like AHP conductance (small, slows & stabilizes)

    # Reversal potentials (mV)
    ENa: float = 55.0
    EK: float = -80.0
    ECa: float = 120.0
    EL: float = -72.0    # locked-in operating point (−72 mV)

    # Membrane capacitance (µF/cm^2)
    Cm: float = 1.0

    # T-type Ca gating (sigmoids + constant taus)
    # p_inf(V) = 1 / (1 + exp(-(V - Vp_half)/kp))
    # q_inf(V) = 1 / (1 + exp((V - Vq_half)/abs(kq)))
    Vp_half: float = -52.0   # ΔVp = +5 mV from −57
    kp: float = 6.2
    Vq_half: float = -81.0
    kq: float = -4.0
    tau_p_ms: float = 3.0
    tau_q_ms: float = 20.0

    # SK/AHP calcium-activation dynamics
    # d[Ca]/dt = -alpha_Ca * I_T  - Ca/tau_Ca   (I_T < 0 is inward; term raises Ca)
    alpha_Ca: float = 0.005    # µM per (µA/cm^2·ms) — small conversion factor
    tau_Ca_ms: float = 120.0   # Ca clearance (ms)
    Kd_w_uM: float = 0.2       # half-activation Ca (µM)
    n_w: float = 4.0           # Hill coefficient for SK activation
    tau_w_ms: float = 80.0     # SK gating decay (ms)

    # Spike detection
    V_spike_thresh: float = 0.0
    min_isi_ms: float = 2.0

    # Initialization
    V_init: float = -70.0
    I_bias: float = -1.0       # small tonic hyperpolarizing bias


@dataclass
class STNLightHHState:
    V: float
    m: float
    h: float
    n: float
    p: float
    q: float
    Ca: float    # intracellular Ca proxy (µM)
    w: float     # SK/AHP activation [0..1]
    last_spike_ms: float = -1e9


class STNLightHH:
    """
    4-current STN model + biologically grounded SK-type AHP.
    Currents:
      I_Na = gNa * m^3 * h * (V - ENa)
      I_K  = gK  * n^4       * (V - EK)
      I_T  = gT  * p^2 * q   * (V - ECa)
      I_L  = gL             * (V - EL)
      I_AHP = gAHP * w      * (V - EK)   (SK: w ~ Hill(Ca))
    """

    def __init__(self, params: Optional[Dict] = None, rng: Optional[np.random.Generator] = None):
        if isinstance(params, dict):
            self.p = STNLightHHParams(**params)
        elif isinstance(params, STNLightHHParams) or params is None:
            self.p = params if isinstance(params, STNLightHHParams) else STNLightHHParams()
        else:
            raise TypeError("params must be dict, STNLightHHParams, or None")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.state = self._init_state(self.p.V_init)

    # ---------- Public API ----------

    def reset(self, V: Optional[float] = None) -> None:
        Vm = self.p.V_init if V is None else V
        self.state = self._init_state(Vm)

    def get_state(self) -> STNLightHHState:
        return self.state

    def set_state(self, new_state: STNLightHHState) -> None:
        self.state = new_state

    def steady_state_gates(self, Vm: float) -> Dict[str, float]:
        m_inf, h_inf, n_inf = self._mh_n_inf(Vm)
        p_inf = 1.0 / (1.0 + _exp(-(Vm - self.p.Vp_half) / self.p.kp))
        q_inf = 1.0 / (1.0 + _exp((Vm - self.p.Vq_half) / abs(self.p.kq)))
        # Ca and w steady-state at rest (low Ca, minimal SK)
        Ca_inf = 0.02  # tiny baseline (µM)
        w_inf = self._w_inf(Ca_inf)
        return dict(m_inf=m_inf, h_inf=h_inf, n_inf=n_inf, p_inf=p_inf, q_inf=q_inf, Ca_inf=Ca_inf, w_inf=w_inf)

    def step(
        self,
        dt_ms: float,
        I_ext: float = 0.0,
        I_syn: float = 0.0,
        t_ms: Optional[float] = None,
    ) -> Tuple[float, bool]:
        s = self.state
        p = self.p

        # 1) Fast gates (Na/K alpha-beta; T-type fixed taus)
        m_inf, h_inf, n_inf, tau_m, tau_h, tau_n = self._mh_n_inf_tau(s.V)
        s.m += dt_ms * (m_inf - s.m) / tau_m
        s.h += dt_ms * (h_inf - s.h) / tau_h
        s.n += dt_ms * (n_inf - s.n) / tau_n

        p_inf = 1.0 / (1.0 + _exp(-(s.V - p.Vp_half) / p.kp))
        q_inf = 1.0 / (1.0 + _exp((s.V - p.Vq_half) / abs(p.kq)))
        s.p += dt_ms * (p_inf - s.p) / max(p.tau_p_ms, 1e-3)
        s.q += dt_ms * (q_inf - s.q) / max(p.tau_q_ms, 1e-3)

        # 2) Currents
        INa = p.gNa * (s.m ** 3) * s.h * (s.V - p.ENa)
        IK  = p.gK  * (s.n ** 4)       * (s.V - p.EK)
        IT  = p.gT  * (s.p ** 2) * s.q * (s.V - p.ECa)    # inward (negative) at typical Vm
        IL  = p.gL                     * (s.V - p.EL)
        IAHP = p.gAHP * s.w            * (s.V - p.EK)

        # 3) Calcium and SK/AHP updates
        # Ca increases with inward T-type current magnitude (−IT), decays with tau_Ca
        s.Ca += dt_ms * ( - p.alpha_Ca * IT - (s.Ca / max(p.tau_Ca_ms, 1e-3)) )
        s.Ca = max(s.Ca, 0.0)  # no negative concentrations
        w_inf = self._w_inf(s.Ca)
        s.w += dt_ms * (w_inf - s.w) / max(p.tau_w_ms, 1e-3)
        s.w = min(max(s.w, 0.0), 1.0)

        # 4) Voltage update
        I_ion = INa + IK + IT + IL + IAHP
        I_drive = I_ext + p.I_bias
        dVdt = (-I_ion - I_syn + I_drive) / p.Cm
        s.V += dt_ms * dVdt

        # 5) Spike detection
        spiked = False
        if t_ms is not None:
            if s.V >= p.V_spike_thresh and self._prev_V < p.V_spike_thresh:
                if (t_ms - s.last_spike_ms) >= p.min_isi_ms:
                    spiked = True
                    s.last_spike_ms = t_ms

        self._prev_V = s.V
        return s.V, spiked

    # ---------- Internal helpers ----------

    def _init_state(self, V0: float) -> STNLightHHState:
        g = self.steady_state_gates(V0)
        st = STNLightHHState(
            V=V0,
            m=g["m_inf"],
            h=g["h_inf"],
            n=g["n_inf"],
            p=g["p_inf"],
            q=g["q_inf"],
            Ca=g["Ca_inf"],
            w=g["w_inf"],
            last_spike_ms=-1e9,
        )
        self._prev_V = V0
        return st

    def _mh_n_inf(self, V: float) -> Tuple[float, float, float]:
        am = 0.32 * _safe_div((V + 54.0), (1.0 - math.exp(-(V + 54.0) / 4.0)))
        bm = 0.28 * _safe_div((V + 27.0), (math.exp((V + 27.0) / 5.0) - 1.0))
        m_inf = am / (am + bm)

        ah = 0.128 * math.exp(-(V + 50.0) / 18.0)
        bh = 4.0 / (1.0 + math.exp(-(V + 27.0) / 5.0))
        h_inf = ah / (ah + bh)

        an = 0.032 * _safe_div((V + 52.0), (1.0 - math.exp(-(V + 52.0) / 5.0)))
        bn = 0.5 * math.exp(-(V + 57.0) / 40.0)
        n_inf = an / (an + bn)

        return m_inf, h_inf, n_inf

    def _mh_n_inf_tau(self, V: float) -> Tuple[float, float, float, float, float, float]:
        am = 0.32 * _safe_div((V + 54.0), (1.0 - math.exp(-(V + 54.0) / 4.0)))
        bm = 0.28 * _safe_div((V + 27.0), (math.exp((V + 27.0) / 5.0) - 1.0))
        m_inf = am / (am + bm)
        tau_m = 1.0 / max(am + bm, 1e-6)

        ah = 0.128 * math.exp(-(V + 50.0) / 18.0)
        bh = 4.0 / (1.0 + math.exp(-(V + 27.0) / 5.0))
        h_inf = ah / (ah + bh)
        tau_h = 1.0 / max(ah + bh, 1e-6)

        an = 0.032 * _safe_div((V + 52.0), (1.0 - math.exp(-(V + 52.0) / 5.0)))
        bn = 0.5 * math.exp(-(V + 57.0) / 40.0)
        n_inf = an / (an + bn)
        tau_n = 1.0 / max(an + bn, 1e-6)

        tau_m = min(max(tau_m, 0.05), 10.0)
        tau_h = min(max(tau_h, 0.2), 200.0)
        tau_n = min(max(tau_n, 0.2), 200.0)
        return m_inf, h_inf, n_inf, tau_m, tau_h, tau_n

    def _w_inf(self, Ca_uM: float) -> float:
        # Hill(Ca) activation for SK channels
        K = max(self.p.Kd_w_uM, 1e-6)
        n = max(self.p.n_w, 1.0)
        x = (Ca_uM / K) ** n
        return x / (1.0 + x)
