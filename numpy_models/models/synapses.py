# synapses.py
# Exponential-decay conductance synapses with a fixed population delay.
# Two variants:
#  1) ExponentialSynapsesCurrent  -> returns current in pA   (nS * mV = pA), for AdEx targets (GPi)
#  2) ExponentialSynapsesDensity  -> returns current in µA/cm^2 (mS/cm^2 * mV), for HH targets (STN)
#
# Usage pattern each timestep:
#   syn.push_spikes(spikes_pre_vec)   # spikes_pre_vec: shape (N_pre,), 0/1
#   I_post = syn.step(V_post_vec)     # returns vector current for each postsynaptic neuron

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def _exp_decay_factor(dt_ms: float, tau_ms: float) -> float:
    tau = max(tau_ms, 1e-6)
    return np.exp(-dt_ms / tau)

@dataclass
class SynapseConfig:
    n_pre: int
    n_post: int
    dt_ms: float
    tau_decay_ms: float
    E_rev_mV: float
    delay_ms: float
    # Weight matrix W has shape (n_post, n_pre):
    #   - For ExponentialSynapsesCurrent: units = nS conductance jump per presynaptic spike
    #   - For ExponentialSynapsesDensity: units = mS/cm^2 conductance jump per presynaptic spike
    W: np.ndarray

class _BaseExponentialSynapses:
    """Common machinery: decay, fixed delay ring-buffer, conductance update."""
    def __init__(self, cfg: SynapseConfig):
        assert cfg.W.shape == (cfg.n_post, cfg.n_pre), "W must be (n_post, n_pre)"
        self.cfg = cfg
        self.decay = _exp_decay_factor(cfg.dt_ms, cfg.tau_decay_ms)
        # Fixed delay in integer steps
        self.delay_steps = max(int(round(cfg.delay_ms / cfg.dt_ms)), 0)
        self._ring_len = max(self.delay_steps, 1)  # avoid zero-length buffer
        # Ring buffer holds pending conductance increments per future step
        self._ring = np.zeros((self._ring_len, cfg.n_post), dtype=np.float32)
        self._ridx = 0  # current index in ring buffer
        # Current conductance per postsynaptic neuron
        self.g = np.zeros(cfg.n_post, dtype=np.float32)

    def push_spikes(self, spikes_pre: np.ndarray) -> None:
        """Queue presynaptic spikes to arrive after the fixed delay.
        spikes_pre: shape (n_pre,), values 0/1 or bool."""
        s = np.asarray(spikes_pre, dtype=np.float32).reshape(-1)
        assert s.shape[0] == self.cfg.n_pre, "spikes_pre has wrong length"
        if self.delay_steps == 0:
            # Arrive immediately: add W @ spikes to conductance now
            inc = self.cfg.W @ s  # shape (n_post,)
            self.g += inc
        else:
            # Sum weighted spikes for each postsynaptic neuron and enqueue in ring
            inc = self.cfg.W @ s  # (n_post,)
            enqueue_idx = (self._ridx + self.delay_steps) % self._ring_len
            self._ring[enqueue_idx] += inc

    def _advance_conductance(self) -> None:
        """Decay conductance and add any arrivals scheduled for this step."""
        # Decay existing conductance
        self.g *= self.decay
        # Add arrivals due now
        arrivals = self._ring[self._ridx]
        if np.any(arrivals):
            self.g += arrivals
            self._ring[self._ridx].fill(0.0)
        # Advance ring pointer
        self._ridx = (self._ridx + 1) % self._ring_len

class ExponentialSynapsesCurrent(_BaseExponentialSynapses):
    """
    Exponential synapses returning current in pA (nS * mV = pA).
    Use for AdEx targets (e.g., STN->GPi AMPA).
    """
    def step(self, V_post_mV: np.ndarray) -> np.ndarray:
        V = np.asarray(V_post_mV, dtype=np.float32).reshape(-1)
        assert V.shape[0] == self.cfg.n_post, "V_post has wrong length"
        self._advance_conductance()
        # I (pA) = g (nS) * (V - E_rev) (mV)
        return self.g * (V - self.cfg.E_rev_mV)

class ExponentialSynapsesDensity(_BaseExponentialSynapses):
    """
    Exponential synapses returning current density in µA/cm^2 (mS/cm^2 * mV).
    Use for HH targets (e.g., GPi->STN GABA_A).
    """
    def step(self, V_post_mV: np.ndarray) -> np.ndarray:
        V = np.asarray(V_post_mV, dtype=np.float32).reshape(-1)
        assert V.shape[0] == self.cfg.n_post, "V_post has wrong length"
        self._advance_conductance()
        # I_density (µA/cm^2) = g_density (mS/cm^2) * (V - E_rev) (mV)
        return self.g * (V - self.cfg.E_rev_mV)
