import jax.numpy as jnp
from stn_jax import create_population_state as create_stn_population, create_vectorized_stn, default_stn_params
from adex_jax import create_population_state as create_adex_population, create_vectorized_adex, default_adex_params_gpe, default_adex_params_gpi
from noise_jax import create_ou_for_population
from synapses_jax import create_synapse_config, init_synapse_state

def build_network_state(n_stn, n_gpe, n_gpi, dt_ms, seed=42):
    """Initializes state and config PyTrees for the STN-GPe-GPi network."""
    
    # 1. Initialize Neuron Populations
    stn_state = create_stn_population(n_stn, heterogeneity=0.05, seed=seed)
    gpe_state = create_adex_population(n_gpe, cell_type='gpe', heterogeneity=0.1, seed=seed+1)
    gpi_state = create_adex_population(n_gpi, cell_type='gpi', heterogeneity=0.1, seed=seed+2)
    
    # 2. Synapse configs and states
    # STN → GPe
    syn_cfg_stn_gpe = create_synapse_config(
        n_pre=n_stn, n_post=n_gpe, p_connect=0.15,
        weight_mean=22.0, weight_cv=0.2, delay_ms=5.0,
        tau_decay_ms=3.0, E_rev_mV=0.0, dt_ms=dt_ms, seed=seed+10
    )
    syn_state_stn_gpe = init_synapse_state(syn_cfg_stn_gpe)
    
    # GPe → STN
    syn_cfg_gpe_stn = create_synapse_config(
        n_pre=n_gpe, n_post=n_stn, p_connect=0.07,
        weight_mean=0.09, weight_cv=0.2, delay_ms=8.0,
        tau_decay_ms=8.0, E_rev_mV=-70.0, dt_ms=dt_ms, seed=seed+11
    )
    syn_state_gpe_stn = init_synapse_state(syn_cfg_gpe_stn)
    
    # STN → GPi
    syn_cfg_stn_gpi = create_synapse_config(
        n_pre=n_stn, n_post=n_gpi, p_connect=0.30,
        weight_mean=18.0, weight_cv=0.2, delay_ms=5.0,
        tau_decay_ms=3.0, E_rev_mV=0.0, dt_ms=dt_ms, seed=seed+12
    )
    syn_state_stn_gpi = init_synapse_state(syn_cfg_stn_gpi)
    
    # GPe → GPi
    syn_cfg_gpe_gpi = create_synapse_config(
        n_pre=n_gpe, n_post=n_gpi, p_connect=0.05,
        weight_mean=25.0, weight_cv=0.2, delay_ms=5.0,
        tau_decay_ms=8.0, E_rev_mV=-70.0, dt_ms=dt_ms, seed=seed+13
    )
    syn_state_gpe_gpi = init_synapse_state(syn_cfg_gpe_gpi)
    
    # 3. Noise setup
    noise_cfg_stn, noise_state_stn = create_ou_for_population(n_stn, dt_ms, mu=1.8, seed=seed+20)
    noise_cfg_gpe, noise_state_gpe = create_ou_for_population(n_gpe, dt_ms, mu=0.0, seed=seed+21)
    noise_cfg_gpi, noise_state_gpi = create_ou_for_population(n_gpi, dt_ms, mu=0.0, seed=seed+22)
    
    # 4. Final state packing
    state = {
        'stn': stn_state, 'gpe': gpe_state, 'gpi': gpi_state,
        'spikes_stn': jnp.zeros(n_stn),
        'spikes_gpe': jnp.zeros(n_gpe),
        'spikes_gpi': jnp.zeros(n_gpi),
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
