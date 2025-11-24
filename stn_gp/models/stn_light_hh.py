from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from typing import Optional, Dict, Tuple


def _safe_div(num: float, den: float, eps: float = 1e-9) -> float:
    return num / (den if abs(den) > eps else (eps if den >= 0 else -eps))


def _exp(x: float) -> float:
    if x > 50:
        return math.exp(50)
    if x < -50:
        return math.exp(-50)
    return math.exp(x)


@dataclass
class STNLightHHParams:
    # Conductances (mS/cm^2) - MATCHING C CODE CONSTANTS
    gNa: float = 37.5
    gK: float = 45.0
    gT: float = 0.5
    gL: float = 2.25
    gCa: float = 0.5
    gAHP: float = 9.0
    gH: float = 0.5

    # Reversal potentials (mV)
    ENa: float = 55.0
    EK: float = -80.0
    ECa: float = 140.0
    EL: float = -58.0

    # Membrane capacitance (µF/cm^2)
    Cm: float = 1.0

    # Initialization / Tonic Drive
    V_init: float = -58.0
    ISTN: float = 30.6

    # --- C-CODE KINETICS PARAMETERS (Last Working Adjustment) ---
    Vp_half: float = -52.0
    kp: float = 6.2
    Vq_half: float = -81.0
    kq: float = -4.0
    tau_p_ms: float = 3.0
    tau_q_ms: float = 20.0

    # I_H (H-current) - PACEMAKER FLOOR ADJUSTMENT
    Vr_half: float = -74.0
    kr: float = 9.0
    tau_r_ms: float = 200.0

    # I_AHP (Ca-activated K)
    alpha_Ca: float = 0.005
    tau_Ca_ms: float = 120.0
    Kd_w_uM: float = 0.2
    n_w: float = 4.0
    tau_w_ms: float = 80.0
    k1: float = 15.0

    # Spike detection
    V_spike_thresh: float = 0.0
    min_isi_ms: float = 2.0


@dataclass
class STNLightHHState:
    V: float
    n: float
    h: float
    r: float
    s: float
    Ca: float
    w: float
    last_spike_ms: float = -1e9


class STNLightHH:
    def __init__(self, params: Optional[Dict] = None, rng: Optional[np.random.Generator] = None):
        if isinstance(params, dict):
            self.p = STNLightHHParams(**params)
        elif isinstance(params, STNLightHHParams) or params is None:
            self.p = params if isinstance(params, STNLightHHParams) else STNLightHHParams()
        else:
            raise TypeError("params must be dict, STNLightHHParams, or None")

        self.rng = rng if rng is not None else np.random.default_rng()
        self.state = self._init_state(self.p.V_init)

    def reset(self, V: Optional[float] = None) -> None:
        Vm = self.p.V_init if V is None else V
        self.state = self._init_state(Vm)

    # --- HH rate functions ---

    def _alpha_m(self, V: float) -> float:
        return 0.32 * _safe_div((V + 54.0), (1.0 - math.exp(-(V + 54.0) / 4.0)))

    def _beta_m(self, V: float) -> float:
        return 0.28 * _safe_div((V + 27.0), (math.exp((V + 27.0) / 5.0) - 1.0))

    def _alpha_h(self, V: float) -> float:
        return 0.128 * math.exp(-(V + 50.0) / 18.0)

    def _beta_h(self, V: float) -> float:
        return 4.0 / (1.0 + math.exp(-(V + 27.0) / 5.0))

    def _alpha_n(self, V: float) -> float:
        return 0.032 * _safe_div((V + 52.0), (1.0 - math.exp(-(V + 52.0) / 5.0)))

    def _beta_n(self, V: float) -> float:
        return 0.5 * math.exp(-(V + 57.0) / 40.0)

    def _m_inf(self, V: float) -> float:
        am = self._alpha_m(V)
        bm = self._beta_m(V)
        return am / (am + bm)

    def _h_inf(self, V: float) -> float:
        ah = self._alpha_h(V)
        bh = self._beta_h(V)
        return ah / (ah + bh)

    def _n_inf(self, V: float) -> float:
        an = self._alpha_n(V)
        bn = self._beta_n(V)
        return an / (an + bn)

    def _taun_c(self, V: float) -> float:
        P_taun0 = 1.0
        P_taun1 = 100.0
        P_thn = -80.0
        P_sigmant = -26.0
        # KINETIC SPEED: Factor is 0.75
        return (P_taun0 + P_taun1 / (1.0 + _exp(-(V - P_thn) / P_sigmant))) * 0.75

    def _tauh_c(self, V: float) -> float:
        P_tauh0 = 1.0
        P_tauh1 = 500.0
        P_thh = -57.0
        P_sigmant = -3.0
        # KINETIC SPEED: Factor is 0.75
        return (P_tauh0 + P_tauh1 / (1.0 + _exp(-(V - P_thh) / P_sigmant))) * 0.75

    # T-type Ca activation
    def _a_inf(self, V: float) -> float:
        return 1.0 / (1.0 + _exp(-(V - self.p.Vp_half) / self.p.kp))

    # High-voltage Ca activation
    def _s_inf(self, V: float) -> float:
        return 1.0 / (1.0 + _exp(-(V + 39.0) / 8.0))

    # AHP activation (Ca-dependent)
    def _w_inf(self, Ca_uM: float) -> float:
        K = self.p.Kd_w_uM
        n = self.p.n_w
        x = (Ca_uM / K) ** n
        return x / (1.0 + x)

    def steady_state_gates(self, Vm: float) -> Dict[str, float]:
        g = {}
        g.update(dict(n=self._n_inf(Vm), h=self._h_inf(Vm)))
        g.update(
            dict(
                r=1.0 / (1.0 + _exp((Vm - self.p.Vr_half) / self.p.kr)),
                s=self._s_inf(Vm),
                Ca=0.02,
                w=self._w_inf(0.02),
            )
        )
        return g

    def _init_state(self, V0: float) -> STNLightHHState:
        g = self.steady_state_gates(V0)
        st = STNLightHHState(
            V=V0,
            n=g["n"],
            h=g["h"],
            r=g["r"],
            s=g["s"],
            Ca=g["Ca"],
            w=g["w"],
            last_spike_ms=-1e9,
        )
        self._prev_V = V0
        return st

    # --- CURRENTS ---

    def Il(self, v: float) -> float:
        return self.p.gL * (v - self.p.EL)

    def Ik(self, v: float, n: float) -> float:
        return self.p.gK * (n**4) * (v - self.p.EK)

    def Ina(self, v: float, h: float) -> float:
        return self.p.gNa * (self._m_inf(v) ** 3) * h * (v - self.p.ENa)

    def It(self, v: float, r: float) -> float:
        return self.p.gT * (self._a_inf(v) ** 3) * (r**2) * (v - self.p.ECa)

    def Ica(self, v: float, s: float) -> float:
        return self.p.gCa * (self._s_inf(v) ** 2) * s * (v - self.p.ECa)

    def Iahp(self, v: float, ca: float) -> float:
        return self.p.gAHP * (v - self.p.EK) * ca / (ca + self.p.k1)

    def IH(self, v: float, r: float) -> float:
        # H-current depolarizing but with a more negative reversal
        E_H = -50.0
        return self.p.gH * r * (v - E_H)

    # --- STEP FUNCTION ---

    def step(
        self,
        dt_ms: float,
        I_ext: float = 0.0,
        I_syn: float = 0.0,
        t_ms: Optional[float] = None,
    ) -> Tuple[float, bool]:
        s = self.state
        p = self.p

        # 1) Gate updates using explicit tau functions with kinetic speedup factor
        s.n += dt_ms * (self._n_inf(s.V) - s.n) / self._taun_c(s.V)
        s.h += dt_ms * (self._h_inf(s.V) - s.h) / self._tauh_c(s.V)

        s.r += dt_ms * (
            (1.0 / (1.0 + _exp((s.V - p.Vr_half) / p.kr))) - s.r
        ) / max(p.tau_r_ms, 1e-3)

        # High-voltage Ca activation treated as instantaneous
        s.s = self._s_inf(s.V)

        # 2) Currents
        I_Na = self.Ina(s.V, s.h)
        I_K = self.Ik(s.V, s.n)
        I_L = self.Il(s.V)
        I_T = self.It(s.V, s.r)
        I_CaH = self.Ica(s.V, s.s)
        I_AHP = self.Iahp(s.V, s.Ca)
        I_H = self.IH(s.V, s.r)

        # 3) Calcium and AHP updates
        I_Ca_total = I_T + I_CaH
        s.Ca += dt_ms * (-p.alpha_Ca * I_Ca_total - (s.Ca / max(p.tau_Ca_ms, 1e-3)))
        s.Ca = max(s.Ca, 0.0)
        s.w = self._w_inf(s.Ca)

        # 4) Voltage update
        I_ion = I_Na + I_K + I_L + I_T + I_CaH + I_AHP + I_H
        I_drive = I_ext + p.ISTN

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
