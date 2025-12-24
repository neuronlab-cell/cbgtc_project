"""
Rubin-Terman GPe/GPi Hodgkin-Huxley Model

Based on: Rubin & Terman (2004) J Comput Neurosci
"High frequency stimulation of the subthalamic nucleus eliminates 
pathological thalamic rhythmicity in a computational model"

Key features:
- T-type calcium current (rebound bursting)
- Calcium-activated AHP current (burst termination)
- Enables beta oscillations in STN-GPe loop
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, Tuple
import numpy as np


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

def default_gpe_params() -> Dict:
    """
    Default parameters for GPe neurons (Rubin-Terman 2004).
    """
    return {
        # Capacitance
        'C': 1.0,  # μF/cm²
        
        # Leak
        'g_L': 0.1,      # mS/cm²
        'E_L': -65.0,    # mV
        
        # Sodium
        'g_Na': 120.0,   # mS/cm²
        'E_Na': 55.0,    # mV
        
        # Potassium (delayed rectifier)
        'g_K': 30.0,     # mS/cm²
        'E_K': -80.0,    # mV
        
        # T-type calcium (low threshold) - KEY FOR REBOUND
        'g_T': 0.5,      # mS/cm²
        'E_Ca': 120.0,   # mV
        
        # High-threshold calcium
        'g_Ca': 0.15,    # mS/cm²
        
        # AHP (calcium-activated potassium) - KEY FOR BURSTING
        'g_AHP': 30.0,   # mS/cm²
        
        # Calcium dynamics
        'tau_Ca': 20.0,  # ms - calcium decay time constant
        'k_Ca': 0.002,   # calcium accumulation factor
        
        # Applied current
        'I_app': 1.5,    # μA/cm² (baseline drive)
        
        # Spike detection
        'V_thresh': -20.0,  # mV
        'V_reset': -60.0,   # mV (not used in HH, but for compatibility)
        
        # Initial values
        'V_init': -65.0,
        'h_init': 0.0,
        'n_init': 0.0,
        'r_init': 0.0,
        'Ca_init': 0.0,
    }


def default_gpi_params() -> Dict:
    """
    Default parameters for GPi neurons.
    Similar to GPe but with different baseline and conductances.
    """
    params = default_gpe_params()
    
    # GPi-specific modifications
    params['g_T'] = 0.3       # Slightly less T-current
    params['g_AHP'] = 20.0    # Less AHP
    params['I_app'] = 2.0     # Higher baseline (GPi fires faster)
    
    return params


# =============================================================================
# GATING FUNCTIONS
# =============================================================================

def safe_exp(x, min_val=-20.0, max_val=10.0):
    """Numerically stable exponential."""
    return jnp.exp(jnp.clip(x, min_val, max_val))


# Sodium activation (m) - instantaneous
def m_inf(V):
    alpha = 0.32 * (V + 54.0) / (1.0 - safe_exp(-(V + 54.0) / 4.0) + 1e-6)
    beta = 0.28 * (V + 27.0) / (safe_exp((V + 27.0) / 5.0) - 1.0 + 1e-6)
    return alpha / (alpha + beta + 1e-6)


# Sodium inactivation (h)
def h_inf(V):
    alpha = 0.128 * safe_exp(-(V + 50.0) / 18.0)
    beta = 4.0 / (1.0 + safe_exp(-(V + 27.0) / 5.0))
    return alpha / (alpha + beta + 1e-6)

def tau_h(V):
    alpha = 0.128 * safe_exp(-(V + 50.0) / 18.0)
    beta = 4.0 / (1.0 + safe_exp(-(V + 27.0) / 5.0))
    return 1.0 / (alpha + beta + 1e-6)


# Potassium activation (n)
def n_inf(V):
    alpha = 0.032 * (V + 52.0) / (1.0 - safe_exp(-(V + 52.0) / 5.0) + 1e-6)
    beta = 0.5 * safe_exp(-(V + 57.0) / 40.0)
    return alpha / (alpha + beta + 1e-6)

def tau_n(V):
    alpha = 0.032 * (V + 52.0) / (1.0 - safe_exp(-(V + 52.0) / 5.0) + 1e-6)
    beta = 0.5 * safe_exp(-(V + 57.0) / 40.0)
    return 1.0 / (alpha + beta + 1e-6)


# T-type calcium activation (a) - instantaneous
def a_inf(V):
    return 1.0 / (1.0 + safe_exp(-(V + 63.0) / 7.8))


# T-type calcium inactivation (r) - KEY FOR REBOUND
def r_inf(V):
    return 1.0 / (1.0 + safe_exp((V + 67.0) / 2.0))

def tau_r(V):
    return 30.0 + 150.0 / (1.0 + safe_exp((V + 67.0) / 2.0))


# High-threshold calcium activation (s) - instantaneous
def s_inf(V):
    return 1.0 / (1.0 + safe_exp(-(V + 35.0) / 2.0))


# AHP activation - depends on calcium
def ahp_inf(Ca):
    return Ca / (Ca + 10.0)


# =============================================================================
# SINGLE NEURON STEP
# =============================================================================

def gpe_gpi_step(V, h, n, r, Ca, I_syn, I_noise, dt, params):
    """
    Single Rubin-Terman GPe/GPi neuron step.
    
    Args:
        V: membrane potential (mV)
        h: sodium inactivation
        n: potassium activation
        r: T-type calcium inactivation
        Ca: intracellular calcium concentration
        I_syn: synaptic current (μA/cm²)
        I_noise: noise current (μA/cm²)
        dt: timestep (ms)
        params: parameter dictionary
        
    Returns:
        V_new, h_new, n_new, r_new, Ca_new, spike
    """
    # Unpack parameters
    C = params['C']
    g_L = params['g_L']
    E_L = params['E_L']
    g_Na = params['g_Na']
    E_Na = params['E_Na']
    g_K = params['g_K']
    E_K = params['E_K']
    g_T = params['g_T']
    E_Ca = params['E_Ca']
    g_Ca = params['g_Ca']
    g_AHP = params['g_AHP']
    tau_Ca_param = params['tau_Ca']
    k_Ca = params['k_Ca']
    I_app = params['I_app']
    
    # Gating variables (instantaneous)
    m = m_inf(V)
    a = a_inf(V)
    s = s_inf(V)
    ahp = ahp_inf(Ca)
    
    # Currents
    I_L = g_L * (V - E_L)
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_T = g_T * a**3 * r * (V - E_Ca)      # T-type calcium (rebound)
    I_Ca = g_Ca * s**2 * (V - E_Ca)         # High-threshold calcium
    I_AHP = g_AHP * ahp * (V - E_K)         # Calcium-activated K+
    
    # Total ionic current
    I_ion = I_L + I_Na + I_K + I_T + I_Ca + I_AHP
    
    # Membrane potential update
    dV = (-I_ion + I_app + I_syn + I_noise) / C
    V_new = V + dt * dV
    
    # Gating variable updates
    dh = (h_inf(V) - h) / tau_h(V)
    h_new = h + dt * dh
    
    dn = (n_inf(V) - n) / tau_n(V)
    n_new = n + dt * dn
    
    dr = (r_inf(V) - r) / tau_r(V)
    r_new = r + dt * dr
    
    # Calcium dynamics
    # Ca increases with high-threshold Ca current, decays exponentially
    dCa = -k_Ca * (I_Ca + I_T) - Ca / tau_Ca_param
    Ca_new = jnp.maximum(0.0, Ca + dt * dCa)
    
    # Spike detection (threshold crossing)
    spike = (V < params['V_thresh']) & (V_new >= params['V_thresh'])
    
    # Clamp values
    h_new = jnp.clip(h_new, 0.0, 1.0)
    n_new = jnp.clip(n_new, 0.0, 1.0)
    r_new = jnp.clip(r_new, 0.0, 1.0)
    
    # Clamp voltage
    V_new = jnp.clip(V_new, -100.0, 60.0)
    
    return V_new, h_new, n_new, r_new, Ca_new, spike


# =============================================================================
# VECTORIZED STEP FUNCTION
# =============================================================================

def create_vectorized_gpe_gpi(compile=True):
    """
    Create vectorized GPe/GPi step function.
    
    Returns:
        Function that steps entire population at once
    """
    def step_population(V, h, n, r, Ca, I_syn, I_noise, dt, params):
        """
        Step entire GPe or GPi population.
        
        All inputs are arrays of shape (n_neurons,)
        """
        return gpe_gpi_step(V, h, n, r, Ca, I_syn, I_noise, dt, params)
    
    # Vectorize over neuron dimension
    vectorized = jax.vmap(
        step_population,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None)
    )
    
    if compile:
        vectorized = jax.jit(vectorized)
    
    return vectorized


# =============================================================================
# POPULATION STATE CREATION
# =============================================================================

def create_population_state(n_neurons: int, cell_type: str = 'gpe', 
                           heterogeneity: float = 0.1, seed: int = 0) -> Dict:
    """
    Create initial state for GPe or GPi population.
    
    Args:
        n_neurons: number of neurons
        cell_type: 'gpe' or 'gpi'
        heterogeneity: variation in initial conditions (0-1)
        seed: random seed
        
    Returns:
        Dictionary with V, h, n, r, Ca arrays
    """
    np.random.seed(seed)
    
    if cell_type == 'gpe':
        params = default_gpe_params()
    else:
        params = default_gpi_params()
    
    # Initialize near steady state with some heterogeneity
    V_base = params['V_init']
    V = V_base + heterogeneity * 10.0 * np.random.randn(n_neurons)
    V = np.clip(V, -80.0, -40.0)
    
    # Initialize gating variables at steady state for initial V
    h = np.array([float(h_inf(v)) for v in V])
    n = np.array([float(n_inf(v)) for v in V])
    r = np.array([float(r_inf(v)) for v in V])
    
    # Add heterogeneity
    h = np.clip(h + heterogeneity * 0.1 * np.random.randn(n_neurons), 0, 1)
    n = np.clip(n + heterogeneity * 0.1 * np.random.randn(n_neurons), 0, 1)
    r = np.clip(r + heterogeneity * 0.1 * np.random.randn(n_neurons), 0, 1)
    
    # Calcium starts low
    Ca = np.abs(heterogeneity * 0.1 * np.random.randn(n_neurons))
    
    # Refractory (for compatibility, not used in HH)
    refractory = np.zeros(n_neurons)
    
    return {
        'V': jnp.array(V, dtype=jnp.float32),
        'h': jnp.array(h, dtype=jnp.float32),
        'n': jnp.array(n, dtype=jnp.float32),
        'r': jnp.array(r, dtype=jnp.float32),
        'Ca': jnp.array(Ca, dtype=jnp.float32),
        'refractory': jnp.array(refractory, dtype=jnp.float32),
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Rubin-Terman GPe/GPi model...")
    
    # Create single neuron
    params = default_gpe_params()
    V, h, n, r, Ca = -65.0, 0.05, 0.3, 0.1, 0.0
    
    # Run for 500ms
    dt = 0.025
    n_steps = int(500 / dt)
    
    V_trace = []
    spike_times = []
    
    step_fn = create_vectorized_gpe_gpi(compile=True)
    
    # Convert to arrays
    V = jnp.array([V])
    h = jnp.array([h])
    n = jnp.array([n])
    r = jnp.array([r])
    Ca = jnp.array([Ca])
    
    for i in range(n_steps):
        I_syn = jnp.array([0.0])
        I_noise = jnp.array([0.0])
        
        # Add inhibition pulse to test rebound
        if 100 < i * dt < 200:
            I_syn = jnp.array([-2.0])  # Inhibition
        
        V, h, n, r, Ca, spike = step_fn(V, h, n, r, Ca, I_syn, I_noise, dt, params)
        V_trace.append(float(V[0]))
        
        if spike[0]:
            spike_times.append(i * dt)
    
    print(f"  Spikes: {len(spike_times)}")
    print(f"  V range: [{min(V_trace):.1f}, {max(V_trace):.1f}] mV")
    
    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        print(f"  Mean ISI: {np.mean(isis):.1f} ms")
        print(f"  Firing rate: {1000/np.mean(isis):.1f} Hz")
    
    # Check for rebound after inhibition
    post_inhib_spikes = [t for t in spike_times if 200 < t < 300]
    print(f"  Rebound spikes (200-300ms): {len(post_inhib_spikes)}")
    
    print("✓ Test complete")
