"""
JAX implementation of Adaptive Exponential (AdEx) neuron model.

This module provides AdEx neurons for pallidal cells (GPe and GPi) in the
basal ganglia. The AdEx model captures spike-frequency adaptation and
provides realistic firing patterns for these tonic pacemaker neurons.

Key features:
- Pure functional programming (no mutations)
- Vectorizable across populations using jax.vmap
- JIT-compilable for high performance
- Two parameter presets: GPe (irregular) and GPi (regular)

Author: Kavin Nakkeeran, Johns Hopkins
Date: December 2025
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_exp(x: jnp.ndarray, clip_min: float = -20.0, clip_max: float = 10.0) -> jnp.ndarray:
    """
    Safe exponential for AdEx spike initiation term.
    
    The exponential term in AdEx can blow up, so we clip the argument
    to prevent numerical overflow. This corresponds to the biological
    reality that Na+ activation saturates.
    
    Args:
        x: Input value(s)
        clip_min: Minimum exponent (default: -20)
        clip_max: Maximum exponent (default: 10)
        
    Returns:
        exp(x) with x clipped to [clip_min, clip_max]
    """
    return jnp.exp(jnp.clip(x, clip_min, clip_max))


# ============================================================================
# PARAMETER PRESETS
# ============================================================================

def default_adex_params_gpi() -> Dict[str, float]:
    """
    Create default parameters for GPi (Globus Pallidus internal) neurons.
    
    GPi neurons are tonic, regular pacemakers firing at ~60-80 Hz.
    They provide the main inhibitory output of the basal ganglia to thalamus.
    
    Returns:
        Dictionary of AdEx parameters:
        - C: Membrane capacitance (pF)
        - gL: Leak conductance (nS)
        - EL: Leak reversal potential (mV)
        - VT: Spike threshold (mV)
        - dT: Spike slope factor (mV)
        - a: Subthreshold adaptation (nS)
        - tau_w: Adaptation time constant (ms)
        - b: Spike-triggered adaptation (pA)
        - V_reset: Reset voltage after spike (mV)
        - V_peak: Spike detection threshold (mV)
        - t_ref_ms: Absolute refractory period (ms)
        - I_baseline: Tonic drive current (pA)
        - V_init: Initial voltage (mV)
    
    Notes:
        These parameters have been validated to produce ~70 Hz firing
        with realistic CV (coefficient of variation) < 0.2.
    """
    return {
        # Membrane properties
        'C': 200.0,         # pF
        'gL': 10.0,         # nS
        'EL': -60.0,        # mV
        
        # Exponential spike initiation
        'VT': -52.0,        # mV (threshold)
        'dT': 2.5,          # mV (sharpness)
        
        # Adaptation
        'a': 0.5,           # nS (subthreshold)
        'tau_w': 120.0,     # ms (time constant)
        'b': 0.0,           # pA (spike-triggered, minimal for GPi)
        
        # Spike/reset
        'V_reset': -55.0,   # mV
        'V_peak': 20.0,     # mV (spike detection)
        't_ref_ms': 2.0,    # ms (absolute refractory period)
        
        # Drive
        'I_baseline': 201.5,  # pA (~70 Hz tonic firing)
        
        # Initial conditions
        'V_init': -60.0,    # mV
        
        # Numerical safety
        'exp_arg_min': -20.0,
        'exp_arg_max': 10.0
    }


def default_adex_params_gpe() -> Dict[str, float]:
    """
    Create default parameters for GPe (Globus Pallidus external) neurons.
    
    GPe neurons are more irregular pacemakers (40-60 Hz) with stronger
    adaptation. They participate in the STN-GPe beta oscillator that
    generates pathological rhythms in Parkinson's disease.
    
    Returns:
        Dictionary of AdEx parameters (see default_adex_params_gpi for key definitions)
    
    Notes:
        - Stronger adaptation (higher a, b) than GPi
        - Slower adaptation recovery (higher tau_w)
        - Expected CV > 0.3 (more irregular)
    """
    return {
        # Membrane properties
        'C': 200.0,
        'gL': 10.0,
        'EL': -60.0,
        
        # Exponential spike (slightly easier threshold, softer slope)
        'VT': -50.0,        # mV (slightly depolarized)
        'dT': 3.5,          # mV (softer spike initiation)
        
        # Adaptation (stronger and slower than GPi)
        'a': 2.5,           # nS (stronger subthreshold)
        'tau_w': 250.0,     # ms (slower recovery)
        'b': 27.0,          # pA (spike-triggered increment)
        
        # Spike/reset
        'V_reset': -60.0,   # mV (more hyperpolarized)
        'V_peak': 20.0,     # mV
        't_ref_ms': 2.0,    # ms
        
        # Drive
        'I_baseline': 200.5,  # pA (~45-55 Hz)
        
        # Initial
        'V_init': -65.0,    # mV (slightly hyperpolarized)
        
        # Numerical
        'exp_arg_min': -20.0,
        'exp_arg_max': 10.0
    }


def default_adex_state(V_init: float = -60.0) -> Dict[str, float]:
    """
    Create default initial state for AdEx neuron.
    
    Args:
        V_init: Initial membrane voltage (mV)
        
    Returns:
        Dictionary with state variables:
        - V: Membrane voltage (mV)
        - w: Adaptation current (pA)
        - ref_remaining_ms: Remaining refractory time (ms)
        - last_spike_ms: Time of last spike (ms)
    """
    return {
        'V': V_init,
        'w': 0.0,
        'ref_remaining_ms': 0.0,
        'last_spike_ms': -1e9
    }


# ============================================================================
# MAIN STEP FUNCTION
# ============================================================================

def adex_step(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, float],
    dt_ms: float,
    I_ext: jnp.ndarray,
    I_syn: jnp.ndarray,
    t_ms: float
) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Advance AdEx neuron by one timestep.
    
    Implements the adaptive exponential integrate-and-fire model:
        C dV/dt = -gL(V - EL) + gL*ΔT*exp((V - VT)/ΔT) - w + I_total
        τ_w dw/dt = a(V - EL) - w
        
    With spike condition: if V ≥ V_peak, then V ← V_reset, w ← w + b
    
    Args:
        state: Current state dict with keys {V, w, ref_remaining_ms, last_spike_ms}
        params: Parameter dict (see default_adex_params_gpi/gpe())
        dt_ms: Timestep in milliseconds
        I_ext: External current input (pA, positive = depolarizing)
        I_syn: Synaptic current input (pA, positive = depolarizing)
        t_ms: Current simulation time (ms)
        
    Returns:
        Tuple of (new_state, (V_out, spiked)) where:
        - new_state: Updated state dictionary
        - V_out: Output voltage (mV)
        - spiked: Boolean indicating if neuron spiked
        
    Notes:
        - During refractory period, V is clamped at V_reset
        - Adaptation (w) continues to evolve during refractory period
        - Sign convention: positive current = depolarizing (excitatory)
        - This function is designed to be vmapped across populations
    """
    # Check if in refractory period
    in_refractory = state['ref_remaining_ms'] > 0.0
    
    # 1. Decrement refractory timer
    ref_new = jnp.maximum(0.0, state['ref_remaining_ms'] - dt_ms)
    
    # 2. During refractory: clamp V and evolve w at V_reset
    # Not in refractory: normal dynamics
    
    # Choose V for adaptation calculation
    V_for_w = jnp.where(in_refractory, params['V_reset'], state['V'])
    
    # Update adaptation (always evolves)
    dw = (params['a'] * (V_for_w - params['EL']) - state['w']) / params['tau_w']
    w_temp = state['w'] + dt_ms * dw
    
    # If in refractory, output voltage is V_reset and skip normal voltage update
    # If not in refractory, compute normal voltage dynamics
    
    # Total input current (only used if not in refractory)
    I_total = params['I_baseline'] + I_ext - I_syn
    
    # Exponential term (only used if not in refractory)
    exp_arg = (state['V'] - params['VT']) / params['dT']
    exp_arg_clipped = jnp.clip(exp_arg, params['exp_arg_min'], params['exp_arg_max'])
    I_exp = params['gL'] * params['dT'] * jnp.exp(exp_arg_clipped)
    
    # Voltage derivative (only used if not in refractory)
    dV = (-params['gL'] * (state['V'] - params['EL']) + I_exp - state['w'] + I_total) / params['C']
    V_integrated = state['V'] + dt_ms * dV
    
    # Choose voltage: refractory → V_reset, active → integrated
    V_new = jnp.where(in_refractory, params['V_reset'], V_integrated)
    
    # 3. Spike detection (only if not in refractory)
    not_in_refractory = jnp.logical_not(in_refractory)
    threshold_crossed = (V_new >= params['V_peak']) & not_in_refractory
    spiked = threshold_crossed
    
    # 4. Apply spike reset if spiked
    V_new = jnp.where(spiked, params['V_reset'], V_new)
    w_new = jnp.where(spiked, w_temp + params['b'], w_temp)
    ref_new = jnp.where(spiked, params['t_ref_ms'], ref_new)
    last_spike_ms_new = jnp.where(spiked, t_ms, state['last_spike_ms'])
    
    # 7. Construct new state
    new_state = {
        'V': V_new,
        'w': w_new,
        'ref_remaining_ms': ref_new,
        'last_spike_ms': last_spike_ms_new
    }
    
    return new_state, (V_new, spiked)


# ============================================================================
# VECTORIZED VERSIONS
# ============================================================================

def create_vectorized_adex(compile: bool = True):
    """
    Create vectorized version of adex_step for population simulations.
    
    Args:
        compile: If True, JIT-compile the vectorized function
        
    Returns:
        Vectorized adex_step function for population arrays
        
    Example:
        >>> adex_pop = create_vectorized_adex(compile=True)
        >>> states = create_population_state(100, 'gpi')
        >>> params = default_adex_params_gpi()
        >>> I_ext = jnp.zeros(100)
        >>> I_syn = jnp.zeros(100)
        >>> new_states, (V, spikes) = adex_pop(states, params, 0.025, I_ext, I_syn, 0.0)
    """
    adex_vectorized = jax.vmap(
        adex_step,
        in_axes=(
            0,      # state: different for each neuron
            None,   # params: same for all
            None,   # dt_ms: same for all
            0,      # I_ext: different for each neuron
            0,      # I_syn: different for each neuron
            None    # t_ms: same for all
        )
    )
    
    if compile:
        return jax.jit(adex_vectorized)
    else:
        return adex_vectorized


# ============================================================================
# POPULATION INITIALIZATION
# ============================================================================

def create_population_state(
    n_neurons: int,
    cell_type: str = 'gpi',
    heterogeneity: float = 0.1,
    seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """
    Create initial state for a population of AdEx neurons with heterogeneity.
    
    Args:
        n_neurons: Number of neurons in population
        cell_type: 'gpi' or 'gpe' (determines V_init)
        heterogeneity: Amount of random variation in I_baseline (CV)
        seed: Random seed for reproducibility
        
    Returns:
        State dictionary with arrays of shape (n_neurons,)
        
    Example:
        >>> # GPi population
        >>> gpi_states = create_population_state(100, 'gpi', heterogeneity=0.1)
        >>> 
        >>> # GPe population  
        >>> gpe_states = create_population_state(150, 'gpe', heterogeneity=0.15)
    """
    import numpy as np
    
    rng = np.random.default_rng(seed)
    
    # Get base state based on cell type
    if cell_type.lower() == 'gpi':
        base = default_adex_state(V_init=-60.0)
    elif cell_type.lower() == 'gpe':
        base = default_adex_state(V_init=-65.0)
    else:
        raise ValueError(f"Unknown cell_type: {cell_type}. Use 'gpi' or 'gpe'")
    
    # Create state arrays
    states = {}
    for key, base_val in base.items():
        if key in ['last_spike_ms', 'ref_remaining_ms']:
            # No heterogeneity for these
            states[key] = jnp.full(n_neurons, base_val, dtype=jnp.float32)
        else:
            # Add small variation to V and w
            if key == 'V':
                noise = rng.normal(0, 2.0, n_neurons)  # 2 mV std
            elif key == 'w':
                noise = rng.normal(0, 5.0, n_neurons)  # 5 pA std
            else:
                noise = 0.0
            
            vals = base_val + noise
            states[key] = jnp.array(vals, dtype=jnp.float32)
    
    return states


def create_heterogeneous_params(
    base_params: Dict[str, float],
    n_neurons: int,
    param_name: str = 'I_baseline',
    cv: float = 0.1,
    seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """
    Create heterogeneous parameters across a population.
    
    Typically used to create variability in intrinsic excitability
    by varying I_baseline across neurons.
    
    Args:
        base_params: Base parameter dictionary
        n_neurons: Number of neurons
        param_name: Which parameter to make heterogeneous (default: 'I_baseline')
        cv: Coefficient of variation (std/mean)
        seed: Random seed
        
    Returns:
        Parameter dictionary with the specified param as array of shape (n_neurons,)
        
    Example:
        >>> base = default_adex_params_gpi()
        >>> hetero_params = create_heterogeneous_params(base, 100, 'I_baseline', cv=0.1)
        >>> # Now hetero_params['I_baseline'] is an array with mean ~201.5, CV=0.1
    """
    import numpy as np
    
    rng = np.random.default_rng(seed)
    
    # Copy base params
    params = dict(base_params)
    
    # Make one parameter heterogeneous
    base_val = base_params[param_name]
    sigma = cv * base_val
    
    # Log-normal distribution to ensure positive values
    if cv > 0:
        mu = jnp.log(base_val) - 0.5 * jnp.log(1 + cv**2)
        sigma_log = jnp.sqrt(jnp.log(1 + cv**2))
        vals = rng.lognormal(mu, sigma_log, n_neurons)
    else:
        vals = np.full(n_neurons, base_val)
    
    params[param_name] = jnp.array(vals, dtype=jnp.float32)
    
    return params


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating GPi and GPe neuron simulations.
    """
    import time
    
    print("JAX AdEx Neuron Model (GPe/GPi)")
    print("=" * 70)
    
    # ========================================================================
    # Test 1: Single GPi neuron
    # ========================================================================
    print("\n1. Single GPi Neuron:")
    print("-" * 70)
    
    state_gpi = default_adex_state(V_init=-60.0)
    params_gpi = default_adex_params_gpi()
    
    print(f"Initial V: {state_gpi['V']:.2f} mV")
    print(f"Running 100 ms simulation...")
    
    spike_times = []
    for i in range(4000):
        t_ms = i * 0.025
        state_gpi, (V, spiked) = adex_step(state_gpi, params_gpi, 0.025, 0.0, 0.0, t_ms)
        if spiked:
            spike_times.append(t_ms)
    
    if len(spike_times) > 0:
        ISIs = jnp.diff(jnp.array(spike_times))
        firing_rate = len(spike_times) / 0.1  # Hz
        CV = jnp.std(ISIs) / jnp.mean(ISIs) if len(ISIs) > 1 else 0.0
        
        print(f"  Spikes: {len(spike_times)}")
        print(f"  Firing rate: {firing_rate:.1f} Hz")
        print(f"  CV (irregularity): {CV:.3f}")
        print(f"  Mean ISI: {jnp.mean(ISIs):.2f} ms")
    
    # ========================================================================
    # Test 2: Single GPe neuron
    # ========================================================================
    print("\n2. Single GPe Neuron:")
    print("-" * 70)
    
    state_gpe = default_adex_state(V_init=-65.0)
    params_gpe = default_adex_params_gpe()
    
    print(f"Initial V: {state_gpe['V']:.2f} mV")
    print(f"Running 100 ms simulation...")
    
    spike_times_gpe = []
    for i in range(4000):
        t_ms = i * 0.025
        state_gpe, (V, spiked) = adex_step(state_gpe, params_gpe, 0.025, 0.0, 0.0, t_ms)
        if spiked:
            spike_times_gpe.append(t_ms)
    
    if len(spike_times_gpe) > 0:
        ISIs_gpe = jnp.diff(jnp.array(spike_times_gpe))
        firing_rate_gpe = len(spike_times_gpe) / 0.1
        CV_gpe = jnp.std(ISIs_gpe) / jnp.mean(ISIs_gpe) if len(ISIs_gpe) > 1 else 0.0
        
        print(f"  Spikes: {len(spike_times_gpe)}")
        print(f"  Firing rate: {firing_rate_gpe:.1f} Hz")
        print(f"  CV (irregularity): {CV_gpe:.3f}")
        print(f"  Mean ISI: {jnp.mean(ISIs_gpe):.2f} ms")
    
    # ========================================================================
    # Test 3: GPi population (100 neurons)
    # ========================================================================
    print("\n3. GPi Population (100 neurons):")
    print("-" * 70)
    
    n_neurons = 100
    states = create_population_state(n_neurons, 'gpi', heterogeneity=0.1)
    params = default_adex_params_gpi()
    adex_pop = create_vectorized_adex(compile=True)
    
    I_ext = jnp.zeros(n_neurons)
    I_syn = jnp.zeros(n_neurons)
    
    print(f"Initial mean V: {jnp.mean(states['V']):.2f} mV")
    
    # Warm up JIT
    states, _ = adex_pop(states, params, 0.025, I_ext, I_syn, 0.0)
    
    # Simulate
    t0 = time.time()
    total_spikes = 0
    for i in range(4000):
        states, (V, spikes) = adex_pop(states, params, 0.025, I_ext, I_syn, i * 0.025)
        total_spikes += jnp.sum(spikes)
    t1 = time.time()
    
    mean_rate = total_spikes / n_neurons / 0.1
    
    print(f"  Simulation time: {(t1-t0)*1000:.1f} ms")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Mean firing rate: {mean_rate:.1f} Hz")
    print(f"  Final mean V: {jnp.mean(states['V']):.2f} mV")
    print(f"  Final mean w: {jnp.mean(states['w']):.2f} pA")
    
    # ========================================================================
    # Test 4: Speed comparison
    # ========================================================================
    print("\n4. Speed Benchmark (1000 neurons, 1000 steps):")
    print("-" * 70)
    
    n = 1000
    n_steps = 1000
    
    states_bench = create_population_state(n, 'gpi')
    I_ext_bench = jnp.zeros(n)
    I_syn_bench = jnp.zeros(n)
    
    # Warm up
    adex_pop(states_bench, params, 0.025, I_ext_bench, I_syn_bench, 0.0)
    
    # Benchmark
    t0 = time.time()
    for i in range(n_steps):
        states_bench, _ = adex_pop(states_bench, params, 0.025, I_ext_bench, I_syn_bench, i*0.025)
    states_bench['V'].block_until_ready()
    t1 = time.time()
    
    total_time_ms = (t1 - t0) * 1000
    time_per_step_ms = total_time_ms / n_steps
    time_per_neuron_us = (time_per_step_ms * 1000) / n
    
    print(f"  Total time: {total_time_ms:.1f} ms")
    print(f"  Time per step: {time_per_step_ms:.3f} ms")
    print(f"  Time per neuron: {time_per_neuron_us:.3f} µs")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed!")
    print("=" * 70)
