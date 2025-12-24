# integrators.py
# Advance the STN–GPe–GPi network by one timestep (ms) in a clear, correct order.

from __future__ import annotations
import numpy as np
from typing import Tuple
from .build_network import Network


def _collect_V_stn(net: Network) -> np.ndarray:
    return np.array([cell.state.V for cell in net.stn], dtype=np.float32)


def _collect_V_gpe(net: Network) -> np.ndarray:
    return np.array([cell.state.V for cell in net.gpe], dtype=np.float32)


def _collect_V_gpi(net: Network) -> np.ndarray:
    return np.array([cell.state.V for cell in net.gpi], dtype=np.float32)


def step_once(
    net: Network,
    t_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Advance the STN–GPe–GPi network by one timestep.

    Parameters
    ----------
    net : Network
        Network object built by build_network().
    t_ms : float
        Current simulation time in ms (used for spike timing / refractory).

    Returns
    -------
    V_stn : np.ndarray, shape (n_stn,)
        Updated STN membrane voltages (mV).
    spk_stn : np.ndarray, shape (n_stn,)
        STN spike indicators (0/1) for this timestep.
    V_gpe : np.ndarray, shape (n_gpe,)
        Updated GPe membrane voltages (mV).
    spk_gpe : np.ndarray, shape (n_gpe,)
        GPe spike indicators (0/1) for this timestep.
    V_gpi : np.ndarray, shape (n_gpi,)
        Updated GPi membrane voltages (mV).
    spk_gpi : np.ndarray, shape (n_gpi,)
        GPi spike indicators (0/1) for this timestep.
    """
    dt = float(net.dt_ms)

    # ------------------------------------------------------------
    # (1) Collect current voltages (before stepping) for synapses
    # ------------------------------------------------------------
    V_stn = _collect_V_stn(net)
    V_gpe = _collect_V_gpe(net)
    V_gpi = _collect_V_gpi(net)

    # ------------------------------------------------------------
    # (2) Synaptic currents at this step
    #
    # Synapse objects maintain their own conductance state and delay
    # ring buffers. Calling .step(V_post) both decays g and applies
    # any arrivals due at this timestep.
    # ------------------------------------------------------------

    # STN → GPe (excitatory, current mode)
    I_syn_gpe_from_stn = net.syn_stn_to_gpe.step(V_gpe)  # shape (n_gpe,), pA

    # GPe → STN (inhibitory, density mode)
    I_syn_stn_from_gpe = net.syn_gpe_to_stn.step(V_stn)  # shape (n_stn,), µA/cm²

    # STN → GPi (excitatory, current mode)
    I_syn_gpi_from_stn = net.syn_stn_to_gpi.step(V_gpi)  # shape (n_gpi,), pA

    # GPe → GPi (inhibitory, current mode)
    I_syn_gpi_from_gpe = net.syn_gpe_to_gpi.step(V_gpi)  # shape (n_gpi,), pA

    # Total synaptic current onto GPi (signs already baked into synapses)
    I_syn_gpi = I_syn_gpi_from_stn + I_syn_gpi_from_gpe  # pA

    # ------------------------------------------------------------
    # (3) Background OU noise for each population
    # ------------------------------------------------------------
    I_ou_stn = net.stn_ou.step()  # µA/cm², shape (n_stn,)
    I_ou_gpe = net.gpe_ou.step()  # pA, shape (n_gpe,)
    I_ou_gpi = net.gpi_ou.step()  # pA, shape (n_gpi,)

    # ------------------------------------------------------------
    # (4) Step neuron dynamics and record spikes
    # ------------------------------------------------------------
    spk_stn = np.zeros(net.n_stn, dtype=np.uint8)
    spk_gpe = np.zeros(net.n_gpe, dtype=np.uint8)
    spk_gpi = np.zeros(net.n_gpi, dtype=np.uint8)

    # STN: HH in µA/cm²
    for i, cell in enumerate(net.stn):
        V, spk = cell.step(
            dt_ms=dt,
            I_ext=float(I_ou_stn[i]),          # background drive
            I_syn=float(I_syn_stn_from_gpe[i]),  # GPe → STN inhibition (density units)
            t_ms=t_ms,
        )
        spk_stn[i] = 1 if spk else 0

    # GPe: AdEx in pA
    for j, cell in enumerate(net.gpe):
        V, spk = cell.step(
            dt_ms=dt,
            I_ext=float(I_ou_gpe[j]),          # OU current (pA)
            I_syn=float(I_syn_gpe_from_stn[j]),  # STN → GPe excitation (pA)
            t_ms=t_ms,
        )
        spk_gpe[j] = 1 if spk else 0

    # GPi: AdEx in pA
    for k, cell in enumerate(net.gpi):
        V, spk = cell.step(
            dt_ms=dt,
            I_ext=float(I_ou_gpi[k]),          # OU current (pA)
            I_syn=float(I_syn_gpi[k]),         # STN + GPe inputs (pA)
            t_ms=t_ms,
        )
        spk_gpi[k] = 1 if spk else 0

    # ------------------------------------------------------------
    # (5) Update V arrays after stepping
    # ------------------------------------------------------------
    V_stn = _collect_V_stn(net)
    V_gpe = _collect_V_gpe(net)
    V_gpi = _collect_V_gpi(net)

    # ------------------------------------------------------------
    # (6) Enqueue outgoing spikes into synapse delay buffers
    #     (these will arrive in future timesteps)
    # ------------------------------------------------------------
    if spk_stn.any():
        net.syn_stn_to_gpe.push_spikes(spk_stn)
        net.syn_stn_to_gpi.push_spikes(spk_stn)

    if spk_gpe.any():
        net.syn_gpe_to_stn.push_spikes(spk_gpe)
        net.syn_gpe_to_gpi.push_spikes(spk_gpe)

    # ------------------------------------------------------------
    # (7) Return updated voltages and spikes for this timestep
    # ------------------------------------------------------------
    return V_stn, spk_stn, V_gpe, spk_gpe, V_gpi, spk_gpi
