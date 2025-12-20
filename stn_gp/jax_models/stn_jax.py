"""
JAX implementation of STN (Subthalamic Nucleus) neuron model.

This module provides a functional, vectorizable implementation of the STN
neuron using modified Hodgkin-Huxley dynamics. The model is designed for
efficient parallel computation on CPU/GPU using JAX.

Key features:
- Pure functional programming (no mutations)
- Vectorizable across populations using jax.vmap
- JIT-compilable for high performance
- Validated against original Python implementation

Author: Kavin Nakkeeran
Functional Neurosurgery Lab, Johns Hopkins University
Date: December 2025
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_div(num: jnp.ndarray, den: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """
    Safe division that avoids division by zero.
    
    Uses jnp.where to handle the 0/0 indeterminacy that appears in 
    Hodgkin-Huxley alpha/beta rate functions.
    
    Args:
        num: Numerator
        den: Denominator
        eps: Small epsilon to prevent division by zero
        
    Returns:
        num / den with safe handling of zero denominators
    """
    safe_den = jnp.where(
        jnp.abs(den) > eps, 
        den, 
        jnp.where(den >= 0, eps, -eps)
    )
    return num / safe_den


def safe_exp(x: jnp.ndarray) -> jnp.ndarray:
    """
    Safe exponential with clipping to prevent overflow.
    
    Args:
        x: Input value(s)
        
    Returns:
        exp(x) with x clipped to [-50, 50]
    """
    return jnp.exp(jnp.clip(x, -50.0, 50.0))


# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

def default_stn_params() -> Dict[str, float]:
    """
    Create default STN neuron parameters.
    
    These parameters are based on the STNLightHH model and have been
    validated to produce realistic STN firing patterns (~40-80 Hz).
    
    Returns:
        Dictionary of parameters with keys:
        - Conductances (gNa, gK, gT, gL, gCa, gAHP, gH) in mS/cm²
        - Reversal potentials (ENa, EK, ECa, EL, E_H) in mV
        - Membrane capacitance (Cm) in µF/cm²
        - Tonic drive (ISTN) in µA/cm²
        - Gating kinetics parameters
        - Calcium dynamics parameters
        - Spike detection parameters
    """
    return {
        # Conductances (mS/cm²)
        'gNa': 37.5,
        'gK': 45.0,
        'gT': 0.5,      # T-type Ca
        'gL': 2.25,
        'gCa': 0.5,     # High-voltage Ca
        'gAHP': 9.0,
        'gH': 0.5,      # H-current (pacemaker)
        
        # Reversal potentials (mV)
        'ENa': 55.0,
        'EK': -80.0,
        'ECa': 140.0,
        'EL': -58.0,
        'E_H': -50.0,
        
        # Membrane capacitance (µF/cm²)
        'Cm': 1.0,
        
        # Tonic drive (µA/cm²)
        'ISTN': 30.6,
        
        # T-type Ca gating
        'Vp_half': -52.0,
        'kp': 6.2,
        
        # H-current gating
        'Vr_half': -74.0,
        'kr': 9.0,
        'tau_r_ms': 200.0,
        
        # Calcium dynamics
        'alpha_Ca': 0.005,
        'tau_Ca_ms': 120.0,
        'k1': 15.0,
        
        # Spike detection
        'V_spike_thresh': 0.0,
        'min_isi_ms': 2.0,
    }


def default_stn_state(V_init: float = -58.0) -> Dict[str, float]:
    """
    Create default initial state for STN neuron.
    
    Args:
        V_init: Initial membrane voltage (mV)
        
    Returns:
        Dictionary with state variables:
        - V: membrane voltage (mV)
        - n: K+ delayed rectifier activation
        - h: Na+ inactivation
        - r: H-current / T-type Ca inactivation
        - s: High-voltage Ca activation (instantaneous)
        - Ca: Intracellular calcium (µM)
        - last_spike_ms: Time of last spike (ms)
    """
    return {
        'V': V_init,
        'n': 0.317,      # Approximate steady state at rest
        'h': 0.596,
        'r': 0.1,
        's': 0.086,
        'Ca': 0.02,
        'last_spike_ms': -1e9
    }


# ============================================================================
# GATING KINETICS
# ============================================================================

def update_gates(
    state: Dict[str, jnp.ndarray], 
    params: Dict[str, float], 
    dt_ms: float
) -> Dict[str, jnp.ndarray]:
    """
    Update gating variables using Hodgkin-Huxley kinetics.
    
    Updates n (K+ activation), h (Na+ inactivation), and r (H-current).
    The s gate (high-voltage Ca) is treated as instantaneous.
    
    Args:
        state: Current neuron state
        params: Neuron parameters
        dt_ms: Timestep (ms)
        
    Returns:
        Dictionary with updated gating variables {n, h, r}
    """
    V = state['V']
    
    # n-gate (K+ delayed rectifier)
    alpha_n = 0.032 * safe_div((V + 52.0), (1.0 - safe_exp(-(V + 52.0) / 5.0)))
    beta_n = 0.5 * safe_exp(-(V + 57.0) / 40.0)
    n_inf = alpha_n / (alpha_n + beta_n)
    tau_n = (1.0 + 100.0 / (1.0 + safe_exp(-(V + 80.0) / -26.0))) * 0.75
    n_new = state['n'] + dt_ms * (n_inf - state['n']) / tau_n
    
    # h-gate (Na+ inactivation)
    alpha_h = 0.128 * safe_exp(-(V + 50.0) / 18.0)
    beta_h = 4.0 / (1.0 + safe_exp(-(V + 27.0) / 5.0))
    h_inf = alpha_h / (alpha_h + beta_h)
    tau_h = (1.0 + 500.0 / (1.0 + safe_exp(-(V + 57.0) / -3.0))) * 0.75
    h_new = state['h'] + dt_ms * (h_inf - state['h']) / tau_h
    
    # r-gate (H-current / T-type Ca inactivation)
    r_inf = 1.0 / (1.0 + safe_exp((V - params['Vr_half']) / params['kr']))
    tau_r = jnp.maximum(params['tau_r_ms'], 1e-3)
    r_new = state['r'] + dt_ms * (r_inf - state['r']) / tau_r
    
    return {'n': n_new, 'h': h_new, 'r': r_new}


# ============================================================================
# IONIC CURRENTS
# ============================================================================

def compute_currents(
    state: Dict[str, jnp.ndarray], 
    params: Dict[str, float]
) -> Dict[str, jnp.ndarray]:
    """
    Compute all ionic currents.
    
    Calculates the seven ionic currents that govern STN dynamics:
    - I_Na: Fast sodium (action potential upstroke)
    - I_K: Delayed rectifier potassium (repolarization)
    - I_L: Leak current
    - I_T: T-type calcium (burst firing)
    - I_CaH: High-voltage calcium
    - I_AHP: Afterhyperpolarization (Ca-activated K+)
    - I_H: H-current (pacemaking)
    
    Args:
        state: Current neuron state
        params: Neuron parameters
        
    Returns:
        Dictionary of currents in µA/cm²
    """
    V = state['V']
    n, h, r = state['n'], state['h'], state['r']
    Ca = state['Ca']
    
    # Instantaneous activation functions
    # m_inf: Na+ activation
    am = 0.32 * safe_div((V + 54.0), (1.0 - safe_exp(-(V + 54.0) / 4.0)))
    bm = 0.28 * safe_div((V + 27.0), (safe_exp((V + 27.0) / 5.0) - 1.0))
    m_inf = am / (am + bm)
    
    # a_inf: T-type Ca activation
    a_inf = 1.0 / (1.0 + safe_exp(-(V - params['Vp_half']) / params['kp']))
    
    # s_inf: High-voltage Ca activation (instantaneous)
    s_inf = 1.0 / (1.0 + safe_exp(-(V + 39.0) / 8.0))
    
    # Get s from state (updated each step to s_inf)
    s = state.get('s', s_inf)
    
    # Compute currents (all in µA/cm²)
    I_Na = params['gNa'] * (m_inf**3) * h * (V - params['ENa'])
    I_K = params['gK'] * (n**4) * (V - params['EK'])
    I_L = params['gL'] * (V - params['EL'])
    I_T = params['gT'] * (a_inf**3) * (r**2) * (V - params['ECa'])
    I_CaH = params['gCa'] * (s_inf**2) * s * (V - params['ECa'])
    I_AHP = params['gAHP'] * (V - params['EK']) * Ca / jnp.maximum(Ca + params['k1'], 1e-9)
    I_H = params['gH'] * r * (V - params['E_H'])
    
    return {
        'I_Na': I_Na,
        'I_K': I_K,
        'I_L': I_L,
        'I_T': I_T,
        'I_CaH': I_CaH,
        'I_AHP': I_AHP,
        'I_H': I_H,
        's_inf': s_inf  # Return for state update
    }


# ============================================================================
# MAIN STEP FUNCTION
# ============================================================================

def stn_step(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, float],
    dt_ms: float,
    I_ext: jnp.ndarray,
    I_syn: jnp.ndarray,
    t_ms: float
) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Advance STN neuron by one timestep.
    
    This is a pure function with no side effects. It takes the current
    state and inputs, and returns the new state and outputs.
    
    Args:
        state: Current state dict with keys {V, n, h, r, s, Ca, last_spike_ms}
        params: Parameter dict (see default_stn_params())
        dt_ms: Timestep in milliseconds
        I_ext: External current input (µA/cm²)
        I_syn: Synaptic current input (µA/cm²)
        t_ms: Current simulation time (ms)
        
    Returns:
        Tuple of (new_state, (V_out, spiked)) where:
        - new_state: Updated state dictionary
        - V_out: Output voltage (mV)
        - spiked: Boolean indicating if neuron spiked
        
    Notes:
        - This function is designed to be vmapped across populations
        - All arrays can be scalars (single neuron) or 1D arrays (population)
        - Use jax.vmap to vectorize across neurons efficiently
    """
    # 1. Update gating variables (n, h, r)
    new_gates = update_gates(state, params, dt_ms)
    
    # 2. Update s (instantaneous high-voltage Ca gate)
    s_inf = 1.0 / (1.0 + safe_exp(-(state['V'] + 39.0) / 8.0))
    
    # 3. Create temporary state with updated gates for current computation
    temp_state = {**state, **new_gates, 's': s_inf}
    
    # 4. Compute ionic currents
    currents = compute_currents(temp_state, params)
    I_ion = (currents['I_Na'] + currents['I_K'] + currents['I_L'] + 
             currents['I_T'] + currents['I_CaH'] + currents['I_AHP'] + 
             currents['I_H'])
    
    # 5. Update membrane voltage
    I_drive = I_ext + params['ISTN']
    dVdt = (-I_ion - I_syn + I_drive) / params['Cm']
    V_new = state['V'] + dt_ms * dVdt
    
    # 6. Update intracellular calcium
    I_Ca_total = currents['I_T'] + currents['I_CaH']
    dCadt = (-params['alpha_Ca'] * I_Ca_total - 
             state['Ca'] / jnp.maximum(params['tau_Ca_ms'], 1e-3))
    Ca_new = jnp.maximum(state['Ca'] + dt_ms * dCadt, 0.0)
    
    # 7. Spike detection (threshold crossing with refractory check)
    threshold_crossed = ((V_new >= params['V_spike_thresh']) & 
                        (state['V'] < params['V_spike_thresh']))
    isi_met = (t_ms - state['last_spike_ms']) >= params['min_isi_ms']
    spiked = threshold_crossed & isi_met
    
    # Update last spike time if spiked
    last_spike_ms_new = jnp.where(spiked, t_ms, state['last_spike_ms'])
    
    # 8. Construct new state
    new_state = {
        'V': V_new,
        'n': new_gates['n'],
        'h': new_gates['h'],
        'r': new_gates['r'],
        's': s_inf,
        'Ca': Ca_new,
        'last_spike_ms': last_spike_ms_new
    }
    
    return new_state, (V_new, spiked)


# ============================================================================
# VECTORIZED VERSIONS
# ============================================================================

def create_vectorized_stn(compile: bool = True):
    """
    Create vectorized version of stn_step for population simulations.
    
    This function creates a version of stn_step that operates on populations
    of neurons in parallel using jax.vmap.
    
    Args:
        compile: If True, JIT-compile the vectorized function for speed
        
    Returns:
        Vectorized stn_step function that operates on population arrays
        
    Example:
        >>> # Create vectorized function
        >>> stn_pop_step = create_vectorized_stn(compile=True)
        >>> 
        >>> # Initialize population of 100 neurons
        >>> states = {
        ...     'V': jnp.ones(100) * -58.0,
        ...     'n': jnp.ones(100) * 0.317,
        ...     # ... other state variables
        ... }
        >>> params = default_stn_params()
        >>> I_ext = jnp.zeros(100)
        >>> I_syn = jnp.zeros(100)
        >>> 
        >>> # Step all neurons simultaneously
        >>> new_states, (V_out, spikes) = stn_pop_step(
        ...     states, params, 0.025, I_ext, I_syn, 0.0
        ... )
    """
    # Vectorize over neurons (axis 0 of state and input arrays)
    stn_vectorized = jax.vmap(
        stn_step,
        in_axes=(
            0,      # state: different for each neuron
            None,   # params: same for all neurons
            None,   # dt_ms: same for all
            0,      # I_ext: different for each neuron
            0,      # I_syn: different for each neuron
            None    # t_ms: same for all
        )
    )
    
    if compile:
        return jax.jit(stn_vectorized)
    else:
        return stn_vectorized


# ============================================================================
# POPULATION INITIALIZATION HELPERS
# ============================================================================

def create_population_state(
    n_neurons: int,
    heterogeneity: float = 0.05,
    seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """
    Create initial state for a population of STN neurons with heterogeneity.
    
    Args:
        n_neurons: Number of neurons in population
        heterogeneity: Amount of random variation (std dev as fraction of mean)
        seed: Random seed for reproducibility
        
    Returns:
        State dictionary with arrays of shape (n_neurons,)
        
    Example:
        >>> states = create_population_state(100, heterogeneity=0.1)
        >>> print(states['V'].shape)
        (100,)
    """
    import numpy as np
    
    rng = np.random.default_rng(seed)
    
    # Base state
    base = default_stn_state()
    
    # Add heterogeneity
    states = {}
    for key, base_val in base.items():
        if key == 'last_spike_ms':
            # Don't add noise to spike times
            states[key] = jnp.full(n_neurons, base_val, dtype=jnp.float32)
        else:
            noise = rng.normal(0, heterogeneity * abs(base_val), n_neurons)
            vals = base_val + noise
            
            # Clip to valid ranges
            if key in ['n', 'h', 'r', 's']:
                vals = np.clip(vals, 0.0, 1.0)
            elif key == 'Ca':
                vals = np.maximum(vals, 0.0)
            
            states[key] = jnp.array(vals, dtype=jnp.float32)
    
    return states


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating single neuron and population simulation.
    """
    print("JAX STN Neuron Model")
    print("=" * 60)
    
    # Single neuron example
    print("\n1. Single Neuron Simulation:")
    print("-" * 60)
    
    state = default_stn_state()
    params = default_stn_params()
    
    print(f"Initial V: {state['V']:.2f} mV")
    
    # Run 10 steps
    for i in range(10):
        state, (V, spiked) = stn_step(state, params, 0.025, 0.0, 0.0, i * 0.025)
        if spiked:
            print(f"  Step {i}: V = {V:.2f} mV [SPIKE]")
    
    print(f"Final V: {state['V']:.2f} mV")
    
    # Population example
    print("\n2. Population Simulation (100 neurons):")
    print("-" * 60)
    
    n_neurons = 100
    states = create_population_state(n_neurons, heterogeneity=0.05)
    stn_pop = create_vectorized_stn(compile=True)
    
    I_ext = jnp.zeros(n_neurons)
    I_syn = jnp.zeros(n_neurons)
    
    print(f"Initial mean V: {jnp.mean(states['V']):.2f} mV")
    print(f"Initial std V:  {jnp.std(states['V']):.2f} mV")
    
    # Warm up JIT
    states, _ = stn_pop(states, params, 0.025, I_ext, I_syn, 0.0)
    
    # Simulate 100ms
    import time
    t0 = time.time()
    spike_count = 0
    for i in range(4000):
        states, (V, spikes) = stn_pop(states, params, 0.025, I_ext, I_syn, i * 0.025)
        spike_count += jnp.sum(spikes)
    t1 = time.time()
    
    print(f"\nSimulation completed:")
    print(f"  Time: {(t1-t0)*1000:.1f} ms")
    print(f"  Total spikes: {spike_count}")
    print(f"  Mean firing rate: {spike_count / n_neurons / 0.1:.1f} Hz")
    print(f"  Final mean V: {jnp.mean(states['V']):.2f} mV")
