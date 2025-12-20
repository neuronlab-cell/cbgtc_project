# observables.py

import jax.numpy as jnp

def compute_firing_rates(spikes_dict, dt_ms):
    """
    Compute mean firing rates for each population.
    
    Args:
        spikes_dict: {'stn': (n_steps, n_neurons), 'gpe': ..., 'gpi': ...}
        dt_ms: Timestep
        
    Returns:
        rates: {'stn': Hz, 'gpe': Hz, 'gpi': Hz}
    """
    rates = {}
    for pop_name, spike_array in spikes_dict.items():
        n_steps, n_neurons = spike_array.shape
        total_time_sec = (n_steps * dt_ms) / 1000.0
        total_spikes = jnp.sum(spike_array)
        rates[pop_name] = (total_spikes / n_neurons) / total_time_sec
    return rates


def compute_beta_power(V_trace, dt_ms, freq_range=(13, 30)):
    """
    Compute power in beta band using FFT.
    
    Args:
        V_trace: (n_steps, n_neurons) voltage traces
        dt_ms: Timestep
        freq_range: (low, high) Hz
        
    Returns:
        beta_power: Scalar power in beta band
    """
    # LFP proxy: mean across neurons
    lfp = jnp.mean(V_trace, axis=1)
    
    # FFT
    fft_vals = jnp.fft.rfft(lfp)
    freqs = jnp.fft.rfftfreq(len(lfp), d=dt_ms/1000.0)
    psd = jnp.abs(fft_vals)**2
    
    # Extract beta band
    idx = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    beta_power = jnp.sum(psd[idx])
    
    return beta_power


def compute_mean_voltage(V_trace):
    """Mean voltage across time and neurons."""
    return jnp.mean(V_trace)


def compute_all_metrics(observables_dict, dt_ms):
    """
    Convenience function to compute all metrics at once.
    
    Args:
        observables_dict: Output from simulation with keys like:
            'V_stn', 'V_gpe', 'V_gpi', 'spikes_stn', 'spikes_gpe', 'spikes_gpi'
        dt_ms: Timestep
        
    Returns:
        metrics: Dict with firing_rates, beta_powers, mean_voltages
    """
    # Firing rates
    spikes = {
        'stn': observables_dict['spikes_stn'],
        'gpe': observables_dict['spikes_gpe'],
        'gpi': observables_dict['spikes_gpi']
    }
    firing_rates = compute_firing_rates(spikes, dt_ms)
    
    # Beta power (per population)
    beta_stn = compute_beta_power(observables_dict['V_stn'], dt_ms)
    beta_gpe = compute_beta_power(observables_dict['V_gpe'], dt_ms)
    beta_gpi = compute_beta_power(observables_dict['V_gpi'], dt_ms)
    
    # Mean voltages
    mean_V = {
        'stn': compute_mean_voltage(observables_dict['V_stn']),
        'gpe': compute_mean_voltage(observables_dict['V_gpe']),
        'gpi': compute_mean_voltage(observables_dict['V_gpi'])
    }
    
    return {
        'firing_rates': firing_rates,
        'beta_power': {'stn': beta_stn, 'gpe': beta_gpe, 'gpi': beta_gpi},
        'mean_V': mean_V
    }
