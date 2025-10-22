# integrators.py
# Advance the STN–GPe network by one timestep (ms) in a clear, correct order.

from __future__ import annotations
import numpy as np
from typing import Tuple
from .build_network import Network

def _collect_V_stn(net: Network) -> np.ndarray:
    return np.array([cell.state.V for cell in net.stn], dtype=np.float32)

def _collect_V_gpe(net: Network) -> np.ndarray:
    return np.array([cell.state.V for cell in net.gpe], dtype=np.float32)

def step_once(net: Network, t_ms: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Advance the network by one dt (net.dt_ms).

    Update order (per timestep):
      1) Compute synaptic currents from existing conductances (includes past arrivals).
      2) Draw background drives (OU).
      3) Step neurons (update V, gates, adaptation) and detect spikes.
      4) Enqueue this step's spikes into outgoing synapses (arrive after fixed delay).

    Returns
    -------
    V_stn  : (n_stn,)  membrane voltages after update (mV)
    spk_stn: (n_stn,)  boolean spike vector (1 if spiked this step)
    V_gpe  : (n_gpe,)  membrane voltages after update (mV)
    spk_gpe: (n_gpe,)  boolean spike vector
    """
    dt = net.dt_ms

    # (1) Synaptic currents *for this step* (use current V to compute I from existing conductances)
    V_stn = _collect_V_stn(net)
    V_gpe = _collect_V_gpe(net)

    # GPe receives AMPA (pA); STN receives GABA (µA/cm²)
    I_gpe_syn_pA    = net.syn_stn_to_gpe.step(V_gpe)   # shape (n_gpe,)
    I_stn_syn_uAcm2 = net.syn_gpe_to_stn.step(V_stn)   # shape (n_stn,)

    # (2) Background drive (OU)
    I_stn_bg_uAcm2 = net.stn_ou.step()  # µA/cm²
    I_gpe_bg_pA    = net.gpe_ou.step()  # pA

    # (3) Step neurons and record spikes
    spk_stn = np.zeros(net.n_stn, dtype=np.int8)
    spk_gpe = np.zeros(net.n_gpe, dtype=np.int8)

    # STN: expects (µA/cm²) inputs
    for i, cell in enumerate(net.stn):
        _, spk = cell.step(
            dt_ms=dt,
            I_ext=float(I_stn_bg_uAcm2[i]),
            I_syn=float(I_stn_syn_uAcm2[i]),
            t_ms=t_ms
        )
        spk_stn[i] = 1 if spk else 0

    # GPe: expects (pA) inputs
    for j, cell in enumerate(net.gpe):
        _, spk = cell.step(
            dt_ms=dt,
            I_ext=float(I_gpe_bg_pA[j]),
            I_syn=float(I_gpe_syn_pA[j]),
            t_ms=t_ms
        )
        spk_gpe[j] = 1 if spk else 0

    # Update V arrays after stepping
    V_stn = _collect_V_stn(net)
    V_gpe = _collect_V_gpe(net)

    # (4) Enqueue outgoing spikes into synapse delay buffers (arrive in future steps)
    if spk_stn.any():
        net.syn_stn_to_gpe.push_spikes(spk_stn)
    if spk_gpe.any():
        net.syn_gpe_to_stn.push_spikes(spk_gpe)

    return V_stn, spk_stn, V_gpe, spk_gpe
