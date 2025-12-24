"""Sparse synaptic connectivity with JAX"""
import jax.numpy as jnp
from jax import random
from typing import NamedTuple

class SynapseConfig(NamedTuple):
    n_pre: int
    n_post: int
    connection_prob: float
    g_max: float
    delay_ms: float
    tau_rise: float
    tau_decay: float
    E_syn: float
    dt_ms: float
    connections: jnp.ndarray  # Shape: (n_connections, 2) - [pre_idx, post_idx]
    weights: jnp.ndarray      # Shape: (n_connections,)

class SynapseState(NamedTuple):
    s: jnp.ndarray           # Gating variable (n_connections,)
    x: jnp.ndarray           # Rising phase (n_connections,)
    delay_buffer: jnp.ndarray # Shape: (n_pre, buffer_size)
    buffer_idx: int

def create_synapse_config(n_pre, n_post, connection_prob, g_max, delay_ms, 
                         tau_rise, tau_decay, E_syn, dt_ms, seed=42):
    """Create sparse connectivity"""
    rng = random.PRNGKey(seed)
    
    # Generate sparse connections
    connections_list = []
    weights_list = []
    
    for pre_idx in range(n_pre):
        rng, subkey = random.split(rng)
        connect = random.uniform(subkey, (n_post,)) < connection_prob
        post_indices = jnp.where(connect, size=n_post, fill_value=-1)[0]
        
        for post_idx in post_indices:
            if post_idx >= 0:  # Valid connection
                connections_list.append([pre_idx, post_idx])
                weights_list.append(g_max)
    
    if len(connections_list) == 0:
        connections_list = [[0, 0]]
        weights_list = [0.0]
    
    connections = jnp.array(connections_list, dtype=jnp.int32)
    weights = jnp.array(weights_list, dtype=jnp.float32)
    
    buffer_size = max(1, int(jnp.ceil(delay_ms / dt_ms)))
    
    return SynapseConfig(
        n_pre=n_pre, n_post=n_post,
        connection_prob=connection_prob, g_max=g_max,
        delay_ms=delay_ms, tau_rise=tau_rise, tau_decay=tau_decay,
        E_syn=E_syn, dt_ms=dt_ms,
        connections=connections, weights=weights
    )

def init_synapse_state(config: SynapseConfig) -> SynapseState:
    """Initialize synapse state"""
    n_connections = config.connections.shape[0]
    buffer_size = max(1, int(jnp.ceil(config.delay_ms / config.dt_ms)))
    
    return SynapseState(
        s=jnp.zeros(n_connections, dtype=jnp.float32),
        x=jnp.zeros(n_connections, dtype=jnp.float32),
        delay_buffer=jnp.zeros((config.n_pre, buffer_size), dtype=jnp.bool_),
        buffer_idx=0
    )

def synapse_step(state: SynapseState, config: SynapseConfig, 
                 spikes_pre: jnp.ndarray, V_post: jnp.ndarray):
    """Update synapses - JAX compatible"""
    dt = config.dt_ms
    
    # Update delay buffer
    new_buffer = state.delay_buffer.at[:, state.buffer_idx].set(spikes_pre)
    next_idx = (state.buffer_idx + 1) % state.delay_buffer.shape[1]
    
    # Get delayed spikes
    delayed_spikes = new_buffer[:, next_idx]
    
    # For each connection, check if pre-synaptic neuron spiked
    pre_indices = config.connections[:, 0]
    post_indices = config.connections[:, 1]
    
    # Get spikes for each connection
    spikes_at_connections = delayed_spikes[pre_indices]
    
    # Update gating variables (vectorized)
    alpha_x = 1.0 / config.tau_rise
    alpha_s = 1.0 / config.tau_decay
    
    # x dynamics
    dx = -alpha_x * state.x
    new_x = state.x + dt * dx + spikes_at_connections.astype(jnp.float32)
    
    # s dynamics  
    ds = -alpha_s * state.s + alpha_x * state.x
    new_s = state.s + dt * ds
    new_s = jnp.clip(new_s, 0.0, 1.0)
    
    # Compute currents for each connection
    V_at_connections = V_post[post_indices]
    I_at_connections = config.weights * new_s * (config.E_syn - V_at_connections)
    
    # Sum currents for each post-synaptic neuron
    I_syn = jnp.zeros(config.n_post, dtype=jnp.float32)
    I_syn = I_syn.at[post_indices].add(I_at_connections)
    
    new_state = SynapseState(
        s=new_s, x=new_x,
        delay_buffer=new_buffer,
        buffer_idx=next_idx
    )
    
    return new_state, I_syn
