"""Network builder with proper synaptic scaling."""
import jax.numpy as jnp
import numpy as np
from .stn_jax import create_population_state as create_stn_population, create_vectorized_stn, default_stn_params
from .adex_jax import create_population_state as create_adex_population, create_vectorized_adex, default_adex_params_gpe, default_adex_params_gpi
from .noise_jax import create_ou_for_population
from .synapses_jax import create_synapse_config, init_synapse_state


# Reference network size (where g_max values were tuned)
REF_N_STN = 10
REF_N_GPE = 20
REF_N_GPI = 15


def build_network_state(n_stn, n_gpe, n_gpi, dt_ms, seed=42):
    """
    Build network with automatic synaptic scaling.
    
    Synaptic weights are scaled inversely with the number of presynaptic
    neurons to maintain consistent total synaptic drive regardless of 
    network size.
    
    g_max_scaled = g_max_reference * (n_pre_reference / n_pre)
    """
    
    # Neurons
    stn_state = create_stn_population(n_stn, heterogeneity=0.05, seed=seed)
    gpe_state = create_adex_population(n_gpe, cell_type='gpe', heterogeneity=0.1, seed=seed+1)
    gpi_state = create_adex_population(n_gpi, cell_type='gpi', heterogeneity=0.1, seed=seed+2)
    
    # ==========================================================================
    # SYNAPTIC SCALING
    # ==========================================================================
    # Scale g_max inversely with presynaptic population size
    # This keeps total synaptic current consistent across network sizes
    
    # Reference g_max values (tuned for small network)
    g_stn_gpe_ref = 2.0
    g_gpe_stn_ref = 9.0
    g_stn_gpi_ref = 2.0
    g_gpe_gpi_ref = 3.0
    
    # Scaled g_max values
    g_stn_gpe = g_stn_gpe_ref * (REF_N_STN / n_stn)
    g_gpe_stn = g_gpe_stn_ref * (REF_N_GPE / n_gpe)
    g_stn_gpi = g_stn_gpi_ref * (REF_N_STN / n_stn)
    g_gpe_gpi = g_gpe_gpi_ref * (REF_N_GPE / n_gpe)
    
    # Synapses with scaled weights
    syn_cfg_stn_gpe = create_synapse_config(n_stn, n_gpe, 0.15, g_stn_gpe, 0.2, 5.0, 3.0, 0.0, dt_ms, seed+10)
    syn_state_stn_gpe = init_synapse_state(syn_cfg_stn_gpe)
    
    syn_cfg_gpe_stn = create_synapse_config(n_gpe, n_stn, 0.07, g_gpe_stn, 0.2, 8.0, 8.0, -70.0, dt_ms, seed+11)
    syn_state_gpe_stn = init_synapse_state(syn_cfg_gpe_stn)
    
    syn_cfg_stn_gpi = create_synapse_config(n_stn, n_gpi, 0.30, g_stn_gpi, 0.2, 5.0, 3.0, 0.0, dt_ms, seed+12)
    syn_state_stn_gpi = init_synapse_state(syn_cfg_stn_gpi)
    
    syn_cfg_gpe_gpi = create_synapse_config(n_gpe, n_gpi, 0.05, g_gpe_gpi, 0.2, 5.0, 8.0, -70.0, dt_ms, seed+13)
    syn_state_gpe_gpi = init_synapse_state(syn_cfg_gpe_gpi)
    
    # Noise
    noise_cfg_stn, noise_state_stn = create_ou_for_population(n_stn, dt_ms, mu=1.8, seed=seed+20)
    noise_cfg_gpe, noise_state_gpe = create_ou_for_population(n_gpe, dt_ms, mu=0.0, seed=seed+21)
    noise_cfg_gpi, noise_state_gpi = create_ou_for_population(n_gpi, dt_ms, mu=0.0, seed=seed+22)
    
    state = {
        'stn': stn_state, 'gpe': gpe_state, 'gpi': gpi_state,
        'spikes_stn': jnp.zeros(n_stn, dtype=jnp.bool_),
        'spikes_gpe': jnp.zeros(n_gpe, dtype=jnp.bool_),
        'spikes_gpi': jnp.zeros(n_gpi, dtype=jnp.bool_),
        'synapses': {
            'stn_to_gpe': syn_state_stn_gpe, 'gpe_to_stn': syn_state_gpe_stn,
            'stn_to_gpi': syn_state_stn_gpi, 'gpe_to_gpi': syn_state_gpe_gpi
        },
        'noise': {
            'stn': noise_state_stn, 'gpe': noise_state_gpe, 'gpi': noise_state_gpi
        }
    }
    
    config = {
        'dt_ms': dt_ms,
        'populations': {'n_stn': n_stn, 'n_gpe': n_gpe, 'n_gpi': n_gpi},
        'synapses': {
            'stn_to_gpe': syn_cfg_stn_gpe, 'gpe_to_stn': syn_cfg_gpe_stn,
            'stn_to_gpi': syn_cfg_stn_gpi, 'gpe_to_gpi': syn_cfg_gpe_gpi
        },
        'noise': {'stn': noise_cfg_stn, 'gpe': noise_cfg_gpe, 'gpi': noise_cfg_gpi},
        'neuron_step_fns': {
            'stn': create_vectorized_stn(compile=True),
            'gpe': create_vectorized_adex(compile=True),
            'gpi': create_vectorized_adex(compile=True)
        },
        'neuron_params': {
            'stn': default_stn_params(),
            'gpe': default_adex_params_gpe(),
            'gpi': default_adex_params_gpi()
        }
    }
    
    return state, config
