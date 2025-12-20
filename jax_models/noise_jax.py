"""
JAX implementation of background noise sources for network simulations.

Provides stochastic input currents to neuron populations:
- Ornstein-Uhlenbeck (OU) process: correlated Gaussian noise
- Poisson shot noise: synaptic-like kicks (future extension)

Key differences from Python version:
- Functional (state explicitly passed/returned)
- Uses JAX PRNG keys instead of numpy generators
- Vectorizable across populations

Author: Kavin Nakkeeran
Functional Neurosurgery Lab, Johns Hopkins University
Date: December 2025
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Tuple


# ============================================================================
# ORNSTEIN-UHLENBECK PROCESS
# ============================================================================

class OUConfig(NamedTuple):
    """
    Configuration for Ornstein-Uhlenbeck process.
    
    The OU process is given by:
        dx = (dt/tau) * (mu - x) + sqrt(2*dt/tau) * sigma * dW
    
    where dW is Gaussian white noise.
    
    Attributes:
        n: Number of parallel OU processes (one per neuron)
        dt_ms: Simulation timestep (ms)
        tau_ms: Correlation time constant (ms)
        mu: Mean (equilibrium value)
        sigma: Standard deviation at equilibrium
        
    Notes:
        - Units of mu and sigma match output units (µA/cm² for HH, pA for AdEx)
        - tau_ms controls temporal correlation: larger = slower fluctuations
        - sigma controls amplitude of fluctuations
    """
    n: int           # Number of processes
    dt_ms: float     # Timestep (ms)
    tau_ms: float    # Correlation time (ms)
    mu: float        # Mean
    sigma: float     # Standard deviation


class OUState(NamedTuple):
    """
    State of OU process.
    
    Attributes:
        x: Current values (n,)
        key: JAX random key for generating noise
    """
    x: jnp.ndarray      # shape (n,)
    key: jnp.ndarray    # JAX PRNG key


def init_ou_state(config: OUConfig, seed: int = 42) -> OUState:
    """
    Initialize OU process state.
    
    Args:
        config: OU configuration
        seed: Random seed for reproducibility
        
    Returns:
        Initial state with x at mean value
    """
    key = random.PRNGKey(seed)
    x = jnp.full(config.n, config.mu, dtype=jnp.float32)
    return OUState(x=x, key=key)


def ou_step(state: OUState, config: OUConfig) -> Tuple[OUState, jnp.ndarray]:
    """
    Advance OU process by one timestep (Euler-Maruyama method).
    
    Args:
        state: Current OU state
        config: OU configuration
        
    Returns:
        Tuple of (new_state, output) where:
        - new_state: Updated OU state
        - output: Current values (n,), same units as mu/sigma
        
    Example:
        >>> config = OUConfig(n=100, dt_ms=0.025, tau_ms=5.0, mu=0.0, sigma=1.0)
        >>> state = init_ou_state(config)
        >>> 
        >>> # In simulation loop:
        >>> state, I_noise = ou_step(state, config)
    """
    # Split key for this step
    key, subkey = random.split(state.key)
    
    # OU parameters
    alpha = config.dt_ms / config.tau_ms
    noise_scale = jnp.sqrt(2.0 * config.dt_ms / config.tau_ms) * config.sigma
    
    # Generate Gaussian noise
    dW = random.normal(subkey, shape=(config.n,), dtype=jnp.float32)
    
    # Euler-Maruyama step
    dx = alpha * (config.mu - state.x) + noise_scale * dW
    x_new = state.x + dx
    
    # New state
    new_state = OUState(x=x_new, key=key)
    
    return new_state, x_new


def create_vectorized_ou(config: OUConfig, compile: bool = True):
    """
    Create optimized OU step function.
    
    Args:
        config: OU configuration (fixed at compile time)
        compile: If True, JIT-compile the function
        
    Returns:
        Function: (state) -> (new_state, output)
        
    Example:
        >>> config = OUConfig(n=100, dt_ms=0.025, tau_ms=5.0, mu=0.0, sigma=1.0)
        >>> ou_fn = create_vectorized_ou(config)
        >>> state = init_ou_state(config)
        >>> 
        >>> # Fast stepping:
        >>> state, I_noise = ou_fn(state)
    """
    def step_with_config(state):
        return ou_step(state, config)
    
    if compile:
        return jax.jit(step_with_config)
    else:
        return step_with_config


# ============================================================================
# POPULATION HELPER
# ============================================================================

def create_ou_for_population(
    n_neurons: int,
    dt_ms: float,
    tau_ms: float = 5.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int = 42
) -> Tuple[OUConfig, OUState]:
    """
    Convenience function to create OU process for a neuron population.
    
    Args:
        n_neurons: Number of neurons
        dt_ms: Simulation timestep (ms)
        tau_ms: Correlation time (ms), default 5.0
        mu: Mean current (µA/cm² or pA depending on target)
        sigma: Std dev of fluctuations
        seed: Random seed
        
    Returns:
        Tuple of (config, initial_state)
        
    Example:
        >>> # For STN population (HH neurons, µA/cm²)
        >>> config, state = create_ou_for_population(
        ...     n_neurons=10000, dt_ms=0.025,
        ...     tau_ms=8.0, mu=1.8, sigma=0.15
        ... )
        >>> 
        >>> # For GPe population (AdEx neurons, pA)
        >>> config, state = create_ou_for_population(
        ...     n_neurons=20000, dt_ms=0.025,
        ...     tau_ms=8.0, mu=0.0, sigma=30.0
        ... )
    """
    config = OUConfig(
        n=n_neurons,
        dt_ms=dt_ms,
        tau_ms=tau_ms,
        mu=mu,
        sigma=sigma
    )
    state = init_ou_state(config, seed=seed)
    
    return config, state


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating OU process usage.
    """
    import time
    
    print("JAX Ornstein-Uhlenbeck Noise")
    print("=" * 70)
    
    # ========================================================================
    # Test 1: Single OU process
    # ========================================================================
    print("\n1. Single OU Process:")
    print("-" * 70)
    
    config_single = OUConfig(
        n=1,
        dt_ms=0.025,
        tau_ms=5.0,
        mu=0.0,
        sigma=1.0
    )
    
    state = init_ou_state(config_single, seed=42)
    
    print(f"Config: tau={config_single.tau_ms} ms, mu={config_single.mu}, sigma={config_single.sigma}")
    print(f"Running 1000 steps (25 ms)...")
    
    values = []
    for i in range(1000):
        state, x = ou_step(state, config_single)
        values.append(float(x[0]))
    
    values = jnp.array(values)
    print(f"  Mean: {jnp.mean(values):.4f} (target: {config_single.mu})")
    print(f"  Std:  {jnp.std(values):.4f} (target: {config_single.sigma})")
    print(f"  Min:  {jnp.min(values):.4f}")
    print(f"  Max:  {jnp.max(values):.4f}")
    
    # ========================================================================
    # Test 2: Population of OU processes
    # ========================================================================
    print("\n2. Population (10,000 neurons):")
    print("-" * 70)
    
    n_neurons = 10000
    config_pop = OUConfig(
        n=n_neurons,
        dt_ms=0.025,
        tau_ms=8.0,
        mu=1.8,      # µA/cm² (typical for STN)
        sigma=0.15
    )
    
    state_pop = init_ou_state(config_pop, seed=42)
    ou_fn = create_vectorized_ou(config_pop, compile=True)
    
    print(f"Config: {n_neurons} processes, tau={config_pop.tau_ms} ms")
    print(f"Mean: {config_pop.mu} µA/cm², Sigma: {config_pop.sigma} µA/cm²")
    
    # Warm up JIT
    state_pop, _ = ou_fn(state_pop)
    
    # Run simulation
    print(f"Running 4000 steps (100 ms)...")
    t0 = time.time()
    for i in range(4000):
        state_pop, I_noise = ou_fn(state_pop)
    I_noise.block_until_ready()  # Wait for GPU
    t1 = time.time()
    
    print(f"  Simulation time: {(t1-t0)*1000:.1f} ms")
    print(f"  Time per step: {(t1-t0)*1000/4000:.3f} ms")
    print(f"  Final mean: {jnp.mean(I_noise):.4f} µA/cm²")
    print(f"  Final std:  {jnp.std(I_noise):.4f} µA/cm²")
    
    # ========================================================================
    # Test 3: Different parameter regimes
    # ========================================================================
    print("\n3. Different Noise Regimes:")
    print("-" * 70)
    
    # STN (HH neurons)
    config_stn, state_stn = create_ou_for_population(
        n_neurons=10000, dt_ms=0.025,
        tau_ms=8.0, mu=1.8, sigma=0.15, seed=1
    )
    print(f"STN (HH):  tau={config_stn.tau_ms} ms, mu={config_stn.mu} µA/cm², sigma={config_stn.sigma}")
    
    # GPe (AdEx neurons)
    config_gpe, state_gpe = create_ou_for_population(
        n_neurons=20000, dt_ms=0.025,
        tau_ms=8.0, mu=0.0, sigma=30.0, seed=2
    )
    print(f"GPe (AdEx): tau={config_gpe.tau_ms} ms, mu={config_gpe.mu} pA, sigma={config_gpe.sigma}")
    
    # GPi (AdEx neurons)
    config_gpi, state_gpi = create_ou_for_population(
        n_neurons=10000, dt_ms=0.025,
        tau_ms=8.0, mu=0.0, sigma=30.0, seed=3
    )
    print(f"GPi (AdEx): tau={config_gpi.tau_ms} ms, mu={config_gpi.mu} pA, sigma={config_gpi.sigma}")
    
    # ========================================================================
    # Test 4: Correlation time effect
    # ========================================================================
    print("\n4. Effect of Correlation Time:")
    print("-" * 70)
    
    for tau in [1.0, 5.0, 20.0, 100.0]:
        cfg = OUConfig(n=1, dt_ms=0.025, tau_ms=tau, mu=0.0, sigma=1.0)
        st = init_ou_state(cfg, seed=42)
        
        vals = []
        for i in range(4000):
            st, x = ou_step(st, cfg)
            vals.append(float(x[0]))
        
        vals = jnp.array(vals)
        
        # Compute autocorrelation at lag=1
        autocorr = jnp.corrcoef(vals[:-1], vals[1:])[0, 1]
        
        print(f"  tau={tau:>6.1f} ms: mean={jnp.mean(vals):>6.3f}, "
              f"std={jnp.std(vals):>5.3f}, autocorr(1)={autocorr:>5.3f}")
    
    print("\n" + "=" * 70)
    print("✓ OU process implementation complete!")
    print("=" * 70)
    print("\nNote: Poisson shot noise can be added similarly if needed.")
