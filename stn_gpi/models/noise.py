# noise.py
# Simple background input sources for network simulations.

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional

# ---------- Ornstein–Uhlenbeck (OU) current ----------
@dataclass
class OUConfig:
    n: int                 # number of target neurons
    dt_ms: float           # ms
    tau_ms: float = 5.0    # correlation time (ms)
    mu: float = 0.0        # mean (same units as output current)
    sigma: float = 1.0     # stationary std dev
    seed: Optional[int] = None

class OUProcess:
    """
    Discrete-time OU process: x_{t+dt} = x_t + (dt/tau)*(mu - x_t) + sqrt(2*dt/tau)*sigma*xi
    Returns a vector of size n each step.
    """
    def __init__(self, cfg: OUConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.x = np.full(cfg.n, cfg.mu, dtype=np.float32)
        # Precompute coefficients
        self.alpha = (cfg.dt_ms / cfg.tau_ms)
        self.noise_scale = np.sqrt(2.0 * cfg.dt_ms / cfg.tau_ms) * cfg.sigma

    def step(self) -> np.ndarray:
        # Euler–Maruyama update
        dx = self.alpha * (self.cfg.mu - self.x) + self.noise_scale * self.rng.standard_normal(self.x.shape).astype(np.float32)
        self.x = (self.x + dx).astype(np.float32)
        return self.x

# ---------- Poisson shot noise (synaptic-like kicks) ----------
@dataclass
class ShotConfig:
    n: int                   # number of target neurons
    dt_ms: float
    rate_hz: float = 200.0   # presynaptic Poisson rate per neuron
    jump: float = 0.05       # amplitude added per event (same units as output)
    tau_decay_ms: float = 5.0
    seed: Optional[int] = None

class PoissonShotNoise:
    """
    Each neuron receives an independent Poisson event train (rate_hz).
    Each event adds 'jump', which then decays exponentially with tau_decay_ms.
    """
    def __init__(self, cfg: ShotConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.level = np.zeros(cfg.n, dtype=np.float32)
        self.decay = np.exp(-cfg.dt_ms / max(cfg.tau_decay_ms, 1e-6))
        self.p_event = min(cfg.rate_hz * (cfg.dt_ms / 1000.0), 0.99)  # per-step probability

    def step(self) -> np.ndarray:
        # Decay existing level
        self.level *= self.decay
        # Sample events (Bernoulli per neuron)
        events = self.rng.random(self.level.shape) < self.p_event
        if np.any(events):
            self.level[events] += self.cfg.jump
        return self.level
