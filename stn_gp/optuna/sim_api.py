# sim_api.py
# Lightweight simulation API for STN–GP networks using:
#   - STNLightHH (HH-style, current density units: µA/cm^2)
#   - GPAdEx (AdEx-style, current units: pA)
#   - Exponential synapses (conductance-based, with fixed delay)
#
# Design goals:
#   • Generic "population" abstraction for STN, GPe, GPi
#   • Network class that:
#       - stores populations
#       - manages synapses (current vs density variants)
#       - steps everything forward in time
#   • No plotting / IO — purely core sim logic.
#
# Typical usage:
#   from stn_gp.sim.sim_api import STNGPNetwork
#
#   net = STNGPNetwork(dt_ms=0.05)
#   net.add_stn_population("STN", n=10)
#   net.add_gpe_population("GPe", n=20)
#   # configure synapse weights, then:
#   for step in range(n_steps):
#       t_ms = step * dt
#       V, spikes = net.step(t_ms)
#   # V and spikes are dicts keyed by population name.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from stn_gp.models.stn_light_hh import (
    STNLightHH,
    STNLightHHParams,
    STNLightHHState,
)
from stn_gp.models.gp_adex import (
    GPAdEx,
    AdExParams,
    AdExParams_GPe,
    AdExParams_GPi,
)
from stn_gp.models.synapses import (
    SynapseConfig,
    ExponentialSynapsesCurrent,
    ExponentialSynapsesDensity,
)

NeuronKind = Literal["stn_hh", "gpe_adex", "gpi_adex"]
SynapseKind = Literal["current", "density"]  # AdEx vs HH targets


# ---------------------------------------------------------------------
# Population abstraction
# ---------------------------------------------------------------------

@dataclass
class Population:
    """
    A homogeneous population of neurons (STN HH, GPe AdEx, or GPi AdEx).

    This wraps a list of single-cell objects and provides a vectorized-ish
    step interface returning:
        V:      (N,) array of membrane potentials [mV]
        spikes: (N,) boolean array of spike events
    """
    name: str
    kind: NeuronKind
    neurons: List[Union[STNLightHH, GPAdEx]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.neurons:
            raise ValueError(f"Population '{self.name}' must have at least one neuron.")
        self.N = len(self.neurons)
        self.V = np.array([self._get_V(n) for n in self.neurons], dtype=float)
        self.spikes = np.zeros(self.N, dtype=bool)

    @staticmethod
    def _get_V(neuron: Union[STNLightHH, GPAdEx]) -> float:
        # STNLightHH and GPAdEx both store membrane voltage in .state.V
        return float(neuron.state.V)

    def reset(self) -> None:
        """Reset all neurons to their built-in initial conditions."""
        for n in self.neurons:
            n.reset()
        self.V = np.array([self._get_V(n) for n in self.neurons], dtype=float)
        self.spikes[:] = False

    def step(
        self,
        dt_ms: float,
        I_ext: Optional[np.ndarray] = None,
        I_syn: Optional[np.ndarray] = None,
        t_ms: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance the whole population by one time step.

        Parameters
        ----------
        dt_ms : float
            Time step in ms.
        I_ext : array-like or None
            External drive per neuron. Units:
                • STN HH: µA/cm^2
                • GPe/GPi AdEx: pA
            If None, treated as 0 for all cells.
        I_syn : array-like or None
            Synaptic current into each neuron. Same units as I_ext
            for that neuron type.
        t_ms : float or None
            Current simulation time in ms (used for spike detection
            in HH model).

        Returns
        -------
        V : np.ndarray, shape (N,)
            Updated membrane voltages [mV].
        spikes : np.ndarray, shape (N,), dtype=bool
            Spike events at this step.
        """
        if I_ext is None:
            I_ext_vec = np.zeros(self.N, dtype=float)
        else:
            I_ext_vec = np.asarray(I_ext, dtype=float).reshape(self.N)

        if I_syn is None:
            I_syn_vec = np.zeros(self.N, dtype=float)
        else:
            I_syn_vec = np.asarray(I_syn, dtype=float).reshape(self.N)

        V_out = np.empty(self.N, dtype=float)
        spikes_out = np.zeros(self.N, dtype=bool)

        for i, neuron in enumerate(self.neurons):
            V_i, spk_i = neuron.step(
                dt_ms=dt_ms,
                I_ext=I_ext_vec[i],
                I_syn=I_syn_vec[i],
                t_ms=t_ms,
            )
            V_out[i] = V_i
            spikes_out[i] = bool(spk_i)

        self.V = V_out
        self.spikes = spikes_out
        return self.V, self.spikes


# ---------------------------------------------------------------------
# Synapse connection abstraction
# ---------------------------------------------------------------------

@dataclass
class SynapseConnection:
    """
    A fixed topology synaptic projection between two populations.

    Attributes
    ----------
    name : str
        Identifier for logging/debugging.
    pre : str
        Name of presynaptic population.
    post : str
        Name of postsynaptic population.
    kind : {'current', 'density'}
        'current'  -> ExponentialSynapsesCurrent, pA, for AdEx targets
        'density'  -> ExponentialSynapsesDensity, µA/cm^2, for HH targets
    syn :
        The underlying synapse object.
    """
    name: str
    pre: str
    post: str
    kind: SynapseKind
    syn: Union[ExponentialSynapsesCurrent, ExponentialSynapsesDensity]


# ---------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------

class STNGPNetwork:
    """
    Small network manager for STN–GPe/GPi simulations.

    Features:
      • Add STN, GPe, and GPi populations with reasonable defaults.
      • Connect populations via conductance-based synapses
        (AMPA, GABA_A, etc.) using the synapses.py module.
      • Step everything forward with a single `step()` call.

    Notes on units:
      • STN (HH): currents are in µA/cm^2 (density).
      • GPe/GPi (AdEx): currents are in pA (absolute).
      • Synapses:
          - ExponentialSynapsesDensity -> µA/cm^2 (HH targets)
          - ExponentialSynapsesCurrent -> pA (AdEx targets)
    """

    def __init__(self, dt_ms: float):
        self.dt_ms = float(dt_ms)
        self.populations: Dict[str, Population] = {}
        self.syn_connections: List[SynapseConnection] = []
        # Map from post-pop name to list of indices into self.syn_connections
        self._incoming_syn_idx: Dict[str, List[int]] = {}

    # ------------------------
    # Population constructors
    # ------------------------

    def add_stn_population(
        self,
        name: str,
        n: int,
        params: Optional[STNLightHHParams] = None,
    ) -> None:
        """Create a population of STN HH neurons."""
        if name in self.populations:
            raise ValueError(f"Population '{name}' already exists.")
        p = params if params is not None else STNLightHHParams()
        neurons = [STNLightHH(params=p) for _ in range(n)]
        self.populations[name] = Population(name=name, kind="stn_hh", neurons=neurons)

    def add_gpe_population(
        self,
        name: str,
        n: int,
        params: Optional[AdExParams] = None,
    ) -> None:
        """Create a population of GPe AdEx neurons."""
        if name in self.populations:
            raise ValueError(f"Population '{name}' already exists.")
        p = params if params is not None else AdExParams_GPe()
        neurons = [GPAdEx(params=p) for _ in range(n)]
        self.populations[name] = Population(name=name, kind="gpe_adex", neurons=neurons)

    def add_gpi_population(
        self,
        name: str,
        n: int,
        params: Optional[AdExParams] = None,
    ) -> None:
        """Create a population of GPi AdEx neurons."""
        if name in self.populations:
            raise ValueError(f"Population '{name}' already exists.")
        p = params if params is not None else AdExParams_GPi()
        neurons = [GPAdEx(params=p) for _ in range(n)]
        self.populations[name] = Population(name=name, kind="gpi_adex", neurons=neurons)

    # ------------------------
    # Synapse wiring
    # ------------------------

    def add_synapse_current(
        self,
        name: str,
        pre: str,
        post: str,
        cfg: SynapseConfig,
    ) -> None:
        """
        Add a synapse projection that returns current in pA (AdEx targets).

        Use this when the postsynaptic population is GPe/GPi AdEx.
        """
        if pre not in self.populations:
            raise KeyError(f"Unknown pre population '{pre}'.")
        if post not in self.populations:
            raise KeyError(f"Unknown post population '{post}'.")
        if cfg.n_pre != self.populations[pre].N or cfg.n_post != self.populations[post].N:
            raise ValueError("SynapseConfig n_pre/n_post do not match population sizes.")

        syn = ExponentialSynapsesCurrent(cfg)
        idx = len(self.syn_connections)
        conn = SynapseConnection(name=name, pre=pre, post=post, kind="current", syn=syn)
        self.syn_connections.append(conn)
        self._incoming_syn_idx.setdefault(post, []).append(idx)

    def add_synapse_density(
        self,
        name: str,
        pre: str,
        post: str,
        cfg: SynapseConfig,
    ) -> None:
        """
        Add a synapse projection that returns current density in µA/cm^2 (HH targets).

        Use this when the postsynaptic population is STN HH.
        """
        if pre not in self.populations:
            raise KeyError(f"Unknown pre population '{pre}'.")
        if post not in self.populations:
            raise KeyError(f"Unknown post population '{post}'.")
        if cfg.n_pre != self.populations[pre].N or cfg.n_post != self.populations[post].N:
            raise ValueError("SynapseConfig n_pre/n_post do not match population sizes.")

        syn = ExponentialSynapsesDensity(cfg)
        idx = len(self.syn_connections)
        conn = SynapseConnection(name=name, pre=pre, post=post, kind="density", syn=syn)
        self.syn_connections.append(conn)
        self._incoming_syn_idx.setdefault(post, []).append(idx)

    # ------------------------
    # Reset
    # ------------------------

    def reset(self) -> None:
        """Reset all neuron populations and synaptic conductances."""
        for pop in self.populations.values():
            pop.reset()
        for conn in self.syn_connections:
            # Rebuild synapse to clear internal state (ring buffer, g, etc.)
            cfg = conn.syn.cfg  # type: ignore[attr-defined]
            if conn.kind == "current":
                conn.syn = ExponentialSynapsesCurrent(cfg)
            else:
                conn.syn = ExponentialSynapsesDensity(cfg)

    # ------------------------
    # Step
    # ------------------------

    def step(
        self,
        t_ms: float,
        I_ext_by_pop: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Advance the entire network by one time step.

        Parameters
        ----------
        t_ms : float
            Current simulation time in ms.
        I_ext_by_pop : dict or None
            Optional mapping from population name to external drive
            vector with shape (N,). Units:
              • STN HH: µA/cm^2
              • GPe/GPi AdEx: pA
            If a population is missing from the dict, zeros are used.

        Returns
        -------
        V_by_pop : dict
            Mapping pop_name -> (N,) membrane voltages [mV].
        spikes_by_pop : dict
            Mapping pop_name -> (N,) boolean spike indicators.
        """
        dt = self.dt_ms
        if I_ext_by_pop is None:
            I_ext_by_pop = {}

        # First, compute all synaptic currents based on spikes from previous step
        I_syn_by_pop: Dict[str, np.ndarray] = {}
        for post_name, post_pop in self.populations.items():
            # Start from zero synaptic current
            I_syn_vec = np.zeros(post_pop.N, dtype=float)
            # Sum contributions from all incoming projections (if any)
            for idx in self._incoming_syn_idx.get(post_name, []):
                conn = self.syn_connections[idx]
                pre_pop = self.populations[conn.pre]

                # 1) Push presynaptic spikes into the synapse
                conn.syn.push_spikes(pre_pop.spikes.astype(np.float32))

                # 2) Compute synaptic current for this postsyn population
                I_syn_proj = conn.syn.step(post_pop.V.astype(np.float32))

                # Units are handled by choosing correct synapse type:
                #   - 'current'  -> pA for AdEx
                #   - 'density'  -> µA/cm^2 for HH
                I_syn_vec += I_syn_proj

            I_syn_by_pop[post_name] = I_syn_vec

        # Then update each population using synaptic + external inputs
        V_by_pop: Dict[str, np.ndarray] = {}
        spikes_by_pop: Dict[str, np.ndarray] = {}

        for name, pop in self.populations.items():
            I_ext_vec = I_ext_by_pop.get(name)
            I_syn_vec = I_syn_by_pop.get(name, None)

            V, spikes = pop.step(
                dt_ms=dt,
                I_ext=I_ext_vec,
                I_syn=I_syn_vec,
                t_ms=t_ms,
            )
            V_by_pop[name] = V.copy()
            spikes_by_pop[name] = spikes.copy()

        return V_by_pop, spikes_by_pop
