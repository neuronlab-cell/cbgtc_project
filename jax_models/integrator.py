"""
Network integrator - steps all populations and synapses.
Supports both AdEx and Rubin-Terman HH neurons.
"""

import jax.numpy as jnp
from typing import Dict, Tuple


def network_step(state: Dict, config: Dict, t_ms: float) -> Tuple[Dict, Dict]:
    """
    Single network timestep.
    
    Args:
        state: Current state of all populations and synapses
        config: Network configuration
        t_ms: Current time in ms
        
    Returns:
        new_state: Updated state
        observables: Voltages and spikes for recording
    """
    dt = config['dt_ms']
    
    # Get step functions
    stn_step = config['neuron_step_fns']['stn']
    gpe_step = config['neuron_step_fns']['gpe']
    gpi_step = config['neuron_step_fns']['gpi']
    
    # Get parameters
    stn_params = config['neuron_params']['stn']
    gpe_params = config['neuron_params']['gpe']
    gpi_params = config['neuron_params']['gpi']
    
    # =========================================================================
    # COMPUTE SYNAPTIC CURRENTS
    # =========================================================================
    
    from .synapses_jax import compute_synaptic_current, update_synapse_state
    
    # STN receives from GPe (inhibitory)
    I_syn_stn, syn_state_gpe_stn = compute_synaptic_current(
        state['synapses']['gpe_to_stn'],
        config['synapses']['gpe_to_stn'],
        state['spikes_gpe'],
        state['stn']['V']
    )
    
    # GPe receives from STN (excitatory)
    I_syn_gpe, syn_state_stn_gpe = compute_synaptic_current(
        state['synapses']['stn_to_gpe'],
        config['synapses']['stn_to_gpe'],
        state['spikes_stn'],
        state['gpe']['V']
    )
    
    # GPi receives from STN (excitatory) and GPe (inhibitory)
    I_syn_gpi_from_stn, syn_state_stn_gpi = compute_synaptic_current(
        state['synapses']['stn_to_gpi'],
        config['synapses']['stn_to_gpi'],
        state['spikes_stn'],
        state['gpi']['V']
    )
    
    I_syn_gpi_from_gpe, syn_state_gpe_gpi = compute_synaptic_current(
        state['synapses']['gpe_to_gpi'],
        config['synapses']['gpe_to_gpi'],
        state['spikes_gpe'],
        state['gpi']['V']
    )
    
    I_syn_gpi = I_syn_gpi_from_stn + I_syn_gpi_from_gpe
    
    # =========================================================================
    # COMPUTE NOISE CURRENTS
    # =========================================================================
    
    from .noise_jax import step_ou_noise
    
    noise_state_stn, I_noise_stn = step_ou_noise(
        state['noise']['stn'], config['noise']['stn'], dt
    )
    noise_state_gpe, I_noise_gpe = step_ou_noise(
        state['noise']['gpe'], config['noise']['gpe'], dt
    )
    noise_state_gpi, I_noise_gpi = step_ou_noise(
        state['noise']['gpi'], config['noise']['gpi'], dt
    )
    
    # =========================================================================
    # STEP NEURONS
    # =========================================================================
    
    # STN (Hodgkin-Huxley based)
    V_stn, n_stn, h_stn, Ca_stn, ref_stn, spikes_stn = stn_step(
        state['stn']['V'],
        state['stn']['n'],
        state['stn']['h'],
        state['stn']['Ca'],
        state['stn']['refractory'],
        I_syn_stn,
        I_noise_stn,
        dt,
        stn_params
    )
    
    # GPe - Check if using HH or AdEx
    if 'Ca' in state['gpe']:
        # Rubin-Terman HH model
        V_gpe, h_gpe, n_gpe, r_gpe, Ca_gpe, spikes_gpe = gpe_step(
            state['gpe']['V'],
            state['gpe']['h'],
            state['gpe']['n'],
            state['gpe']['r'],
            state['gpe']['Ca'],
            I_syn_gpe,
            I_noise_gpe,
            dt,
            gpe_params
        )
        new_gpe_state = {
            'V': V_gpe, 'h': h_gpe, 'n': n_gpe, 'r': r_gpe, 'Ca': Ca_gpe,
            'refractory': state['gpe']['refractory']
        }
    else:
        # AdEx model
        V_gpe, w_gpe, ref_gpe, spikes_gpe = gpe_step(
            state['gpe']['V'],
            state['gpe']['w'],
            state['gpe']['refractory'],
            I_syn_gpe,
            I_noise_gpe,
            dt,
            gpe_params
        )
        new_gpe_state = {'V': V_gpe, 'w': w_gpe, 'refractory': ref_gpe}
    
    # GPi - Check if using HH or AdEx
    if 'Ca' in state['gpi']:
        # Rubin-Terman HH model
        V_gpi, h_gpi, n_gpi, r_gpi, Ca_gpi, spikes_gpi = gpi_step(
            state['gpi']['V'],
            state['gpi']['h'],
            state['gpi']['n'],
            state['gpi']['r'],
            state['gpi']['Ca'],
            I_syn_gpi,
            I_noise_gpi,
            dt,
            gpi_params
        )
        new_gpi_state = {
            'V': V_gpi, 'h': h_gpi, 'n': n_gpi, 'r': r_gpi, 'Ca': Ca_gpi,
            'refractory': state['gpi']['refractory']
        }
    else:
        # AdEx model
        V_gpi, w_gpi, ref_gpi, spikes_gpi = gpi_step(
            state['gpi']['V'],
            state['gpi']['w'],
            state['gpi']['refractory'],
            I_syn_gpi,
            I_noise_gpi,
            dt,
            gpi_params
        )
        new_gpi_state = {'V': V_gpi, 'w': w_gpi, 'refractory': ref_gpi}
    
    # =========================================================================
    # UPDATE SYNAPSE STATES
    # =========================================================================
    
    syn_state_gpe_stn = update_synapse_state(
        syn_state_gpe_stn, config['synapses']['gpe_to_stn'], dt
    )
    syn_state_stn_gpe = update_synapse_state(
        syn_state_stn_gpe, config['synapses']['stn_to_gpe'], dt
    )
    syn_state_stn_gpi = update_synapse_state(
        syn_state_stn_gpi, config['synapses']['stn_to_gpi'], dt
    )
    syn_state_gpe_gpi = update_synapse_state(
        syn_state_gpe_gpi, config['synapses']['gpe_to_gpi'], dt
    )
    
    # =========================================================================
    # ASSEMBLE NEW STATE
    # =========================================================================
    
    new_state = {
        'stn': {
            'V': V_stn, 'n': n_stn, 'h': h_stn, 'Ca': Ca_stn, 'refractory': ref_stn
        },
        'gpe': new_gpe_state,
        'gpi': new_gpi_state,
        'spikes_stn': spikes_stn,
        'spikes_gpe': spikes_gpe,
        'spikes_gpi': spikes_gpi,
        'synapses': {
            'gpe_to_stn': syn_state_gpe_stn,
            'stn_to_gpe': syn_state_stn_gpe,
            'stn_to_gpi': syn_state_stn_gpi,
            'gpe_to_gpi': syn_state_gpe_gpi,
        },
        'noise': {
            'stn': noise_state_stn,
            'gpe': noise_state_gpe,
            'gpi': noise_state_gpi,
        }
    }
    
    observables = {
        'V_stn': V_stn,
        'V_gpe': V_gpe,
        'V_gpi': V_gpi,
        'spikes_stn': spikes_stn,
        'spikes_gpe': spikes_gpe,
        'spikes_gpi': spikes_gpi,
    }
    
    return new_state, observables
