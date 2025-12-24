"""
JAX implementation of synaptic connectivity with sparse storage.

This module provides efficient synaptic transmission using sparse connectivity.
Scales to millions of synapses by storing only nonzero connections.

Author: Kavin Nakkeeran
Functional Neurosurgery Lab, Johns Hopkins University
Date: December 2025
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple
import numpy as np

# ============================================================================
# SYNAPSE CONFIGURATION  
# ============================================================================

class SynapseConfig(NamedTuple):
    """Sparse synapse configuration using connection lists."""
    n_pre: int                # Number of presynaptic neurons
    n_post: int               # Number of postsynaptic neurons
    n_synapses: int           # Number of actual connections
    pre_ids: jnp.ndarray      # Presynaptic IDs (n_synapses,)
    post_ids: jnp.ndarray     # Postsynaptic IDs (n_synapses,)
    weights: jnp.ndarray      # Synaptic weights (n_synapses,)
    delays_steps: jnp.ndarray # Delays in timesteps (n_synapses,)
    tau_decay_ms: float       # Decay time constant
    E_rev_mV: float           # Reversal potential
    dt_ms: float              # Timestep


class SynapseState(NamedTuple):
    """Synapse state with conductances and delay queue."""
    g_post: jnp.ndarray       # Conductance per postsynaptic neuron (n_post,)
    delay_queue: jnp.ndarray  # Pending arrivals (max_delay, n_post)
    queue_ptr: int            # Current queue position


# ============================================================================
# CONNECTIVITY CREATION
# ============================================================================

def create_sparse_connectivity(
    n_pre: int,
    n_post: int,
    p_connect: float,
    weight_mean: float,
    weight_cv: float = 0.2,
    delay_ms: float = 5.0,
    dt_ms: float = 0.025,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create random sparse connectivity.
    
    Returns: (pre_ids, post_ids, weights, delays_steps)
    """
    rng = np.random.default_rng(seed)
    
    # Sample connections
    n_possible = n_pre * n_post
    n_expected = int(n_possible * p_connect)
    
    if p_connect < 0.5:
        conn_ids = rng.choice(n_possible, size=n_expected, replace=False)
        pre_ids = conn_ids // n_post
        post_ids = conn_ids % n_post
    else:
        mask = rng.random((n_post, n_pre)) < p_connect
        post_ids, pre_ids = np.where(mask)
    
    n_syn = len(pre_ids)
    
    # Weights (log-normal)
    if weight_cv > 0:
        sigma = np.sqrt(np.log(1 + weight_cv**2))
        mu = np.log(weight_mean) - 0.5 * sigma**2
        weights = rng.lognormal(mu, sigma, n_syn)
    else:
        weights = np.full(n_syn, weight_mean)
    
    # Delays
    delay_steps = int(round(delay_ms / dt_ms))
    delays = np.full(n_syn, delay_steps, dtype=np.int32)
    
    return (
        jnp.array(pre_ids, dtype=jnp.int32),
        jnp.array(post_ids, dtype=jnp.int32),
        jnp.array(weights, dtype=jnp.float32),
        jnp.array(delays, dtype=jnp.int32)
    )


def create_synapse_config(
    n_pre: int,
    n_post: int,
    p_connect: float,
    weight_mean: float,
    weight_cv: float,
    delay_ms: float,
    tau_decay_ms: float,
    E_rev_mV: float,
    dt_ms: float,
    seed: int = 42
) -> SynapseConfig:
    """Create complete synapse configuration."""
    pre, post, w, d = create_sparse_connectivity(
        n_pre, n_post, p_connect, weight_mean, weight_cv, delay_ms, dt_ms, seed
    )
    
    return SynapseConfig(
        n_pre=n_pre, n_post=n_post, n_synapses=len(pre),
        pre_ids=pre, post_ids=post, weights=w, delays_steps=d,
        tau_decay_ms=tau_decay_ms, E_rev_mV=E_rev_mV, dt_ms=dt_ms
    )


def init_synapse_state(config: SynapseConfig) -> SynapseState:
    """Initialize synapse state (zero conductances)."""
    max_delay = int(jnp.max(config.delays_steps)) + 1 if config.n_synapses > 0 else 1
    
    return SynapseState(
        g_post=jnp.zeros(config.n_post, dtype=jnp.float32),
        delay_queue=jnp.zeros((max_delay, config.n_post), dtype=jnp.float32),
        queue_ptr=0
    )


# ============================================================================
# SYNAPSE DYNAMICS
# ============================================================================

def synapse_step(
    state: SynapseState,
    config: SynapseConfig,
    spikes_pre: jnp.ndarray,
    V_post: jnp.ndarray
) -> Tuple[SynapseState, jnp.ndarray]:
    """
    Advance synapses by one timestep.
    
    Args:
        state: Current state
        config: Configuration
        spikes_pre: Presynaptic spikes (n_pre,), binary
        V_post: Postsynaptic voltages (n_post,), mV
        
    Returns:
        (new_state, I_syn) where I_syn is synaptic current (n_post,)
    """
    # 1. Decay conductances
    decay = jnp.exp(-config.dt_ms / config.tau_decay_ms)
    g_decayed = state.g_post * decay
    
    # 2. Process arrivals from delay queue
    queue_len = state.delay_queue.shape[0]
    arrivals = state.delay_queue[state.queue_ptr]
    new_ptr = (state.queue_ptr + 1) % queue_len
    
    # Clear current slot and add arrivals
    queue_cleared = state.delay_queue.at[state.queue_ptr].set(0.0)
    g_new = g_decayed + arrivals
    
    # 3. Enqueue new spikes
    queue_updated = queue_cleared
    
    if config.n_synapses > 0:
        # Find active synapses
        active = spikes_pre[config.pre_ids].astype(jnp.float32)
        g_inc = config.weights * active
        
        # Compute delivery slots
        delivery = (new_ptr + config.delays_steps) % queue_len
        
        # Add to queue using scatter_add
        for i in range(config.n_synapses):
            # Only process if spike occurred
            if active[i] > 0:
                slot = delivery[i]
                post_id = config.post_ids[i]
                queue_updated = queue_updated.at[slot, post_id].add(g_inc[i])
    
    # 4. Compute current: I = g * (V - E_rev)
    I_syn = g_new * (V_post - config.E_rev_mV)
    
    # 5. New state
    new_state = SynapseState(
        g_post=g_new,
        delay_queue=queue_updated,
        queue_ptr=new_ptr
    )
    
    return new_state, I_syn


def create_jitted_synapse(config: SynapseConfig):
    """Create JIT-compiled synapse function."""
    def step(state, spikes_pre, V_post):
        return synapse_step(state, config, spikes_pre, V_post)
    return jax.jit(step)


# ============================================================================
# EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("JAX Sparse Synapses")
    print("=" * 60)
    
    # Small test
    cfg = create_synapse_config(
        n_pre=100, n_post=200, p_connect=0.15,
        weight_mean=0.5, weight_cv=0.2, delay_ms=5.0,
        tau_decay_ms=3.0, E_rev_mV=0.0, dt_ms=0.025
    )
    
    print(f"\nSmall network:")
    print(f"  100 -> 200 neurons, {cfg.n_synapses} synapses")
    print(f"  Sparsity: {100*cfg.n_synapses/(100*200):.1f}%")
    
    # Scaling
    print(f"\nScaling test:")
    sizes = [(1000, 2000), (10000, 20000), (100000, 200000)]
    
    for n_pre, n_post in sizes:
        c = create_synapse_config(
            n_pre, n_post, 0.15, 0.5, 0.2, 5.0, 3.0, 0.0, 0.025
        )
        sparse_mb = c.n_synapses * 16 / (1024**2)  # 4 arrays × 4 bytes
        dense_mb = n_pre * n_post * 4 / (1024**2)
        print(f"  {n_pre:>6} -> {n_post:<6}: {c.n_synapses:>10} syn, "
              f"sparse={sparse_mb:>6.1f} MB, dense={dense_mb:>8.1f} MB")
    
    print("\n" + "=" * 60)
    print("✓ Complete!")
