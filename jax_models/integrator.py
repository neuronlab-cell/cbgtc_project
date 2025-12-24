"""Network integration"""
import jax.numpy as jnp
from .noise_jax import ou_step
from .synapses_jax import synapse_step

def network_step(state, config, t_ms):
    """Simplified network step for testing"""
    dt = config['dt_ms']
    
    # Noise
    ns_stn, I_n_stn = ou_step(state['noise']['stn'], config['noise']['stn'])
    ns_gpe, I_n_gpe = ou_step(state['noise']['gpe'], config['noise']['gpe'])
    ns_gpi, I_n_gpi = ou_step(state['noise']['gpi'], config['noise']['gpi'])
    
    # Synapses
    syn_stn_gpe, I_syn_gpe_stn = synapse_step(
        state['synapses']['stn_to_gpe'], config['synapses']['stn_to_gpe'],
        state['spikes_stn'], state['gpe']['V']
    )
    syn_gpe_stn, I_syn_stn_gpe = synapse_step(
        state['synapses']['gpe_to_stn'], config['synapses']['gpe_to_stn'],
        state['spikes_gpe'], state['stn']['V']
    )
    syn_stn_gpi, I_syn_gpi_stn = synapse_step(
        state['synapses']['stn_to_gpi'], config['synapses']['stn_to_gpi'],
        state['spikes_stn'], state['gpi']['V']
    )
    syn_gpe_gpi, I_syn_gpi_gpe = synapse_step(
        state['synapses']['gpe_to_gpi'], config['synapses']['gpe_to_gpi'],
        state['spikes_gpe'], state['gpi']['V']
    )
    
    # Neurons
    new_stn, (V_stn, s_stn) = config['neuron_step_fns']['stn'](
        state['stn'], config['neuron_params']['stn'], dt, I_n_stn, I_syn_stn_gpe, t_ms
    )
    new_gpe, (V_gpe, s_gpe) = config['neuron_step_fns']['gpe'](
        state['gpe'], config['neuron_params']['gpe'], dt, I_n_gpe, I_syn_gpe_stn, t_ms
    )
    new_gpi, (V_gpi, s_gpi) = config['neuron_step_fns']['gpi'](
        state['gpi'], config['neuron_params']['gpi'], dt, I_n_gpi, I_syn_gpi_stn + I_syn_gpi_gpe, t_ms
    )
    
    new_state = {
        'stn': new_stn, 'gpe': new_gpe, 'gpi': new_gpi,
        'spikes_stn': s_stn, 'spikes_gpe': s_gpe, 'spikes_gpi': s_gpi,
        'synapses': {
            'stn_to_gpe': syn_stn_gpe, 'gpe_to_stn': syn_gpe_stn,
            'stn_to_gpi': syn_stn_gpi, 'gpe_to_gpi': syn_gpe_gpi
        },
        'noise': {'stn': ns_stn, 'gpe': ns_gpe, 'gpi': ns_gpi}
    }
    
    observables = {
        'V_stn': V_stn, 'V_gpe': V_gpe, 'V_gpi': V_gpi,
        'spikes_stn': s_stn, 'spikes_gpe': s_gpe, 'spikes_gpi': s_gpi
    }
    
    return new_state, observables
