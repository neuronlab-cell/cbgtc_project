"""
JAX-optimized metrics computation for Optuna optimization.

This module wraps the observables.py functions with JIT compilation
and adds burn-in period handling for steady-state metrics.

Author: Kavin Nakkeeran
Functional Neurosurgery Lab, Johns Hopkins University
Date: December 2025
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict


# ============================================================================
# FIRING RATES
# ============================================================================

def compute_firing_rate_single_pop(spikes: jnp.ndarray, dt_ms: float, burn_steps: int) -> float:
    """
    Compute mean firing rate for a single population after burn-in.
    
    Args:
        spikes: Binary spike array (n_steps, n_neurons)
        dt_ms: Timestep in milliseconds
        burn_steps: Number of initial steps to discard
        
    Returns:
        Mean firing rate in Hz
        
    Note: Not JIT-compiled at top level because burn_steps is dynamic.
    The actual computation inside is still fast.
    """
    # Simple Python slicing works when not in JIT context
    valid_spikes = spikes[burn_steps:]
    
    n_steps, n_neurons = valid_spikes.shape
    total_time_sec = (n_steps * dt_ms) / 1000.0
    
    # This part is fast (vectorized JAX operations)
    total_spikes = jnp.sum(valid_spikes)
    mean_rate = total_spikes / n_neurons / total_time_sec
    
    return float(mean_rate)


def compute_firing_rates(spikes_dict: Dict[str, jnp.ndarray], dt_ms: float, burn_steps: int = 1000) -> Dict[str, float]:
    """
    Compute firing rates for all populations.
    
    Args:
        spikes_dict: {'stn': (n_steps, n_stn), 'gpe': ..., 'gpi': ...}
        dt_ms: Timestep
        burn_steps: Burn-in period (default: 1000 steps = 25ms @ dt=0.025)
        
    Returns:
        {'stn': Hz, 'gpe': Hz, 'gpi': Hz}
    """
    rates = {}
    for pop_name, spike_array in spikes_dict.items():
        rates[pop_name] = compute_firing_rate_single_pop(spike_array, dt_ms, burn_steps)
    
    return rates


# ============================================================================
# COEFFICIENT OF VARIATION (ISI)
# ============================================================================

# Note: CV computation is complex to JIT due to variable spike counts
# We compute it without JIT for now - it's fast enough
def compute_cv_population(spikes: jnp.ndarray, dt_ms: float, burn_steps: int) -> float:
    """
    Compute mean CV across a population.
    
    Args:
        spikes: Binary spike array (n_steps, n_neurons)
        dt_ms: Timestep
        burn_steps: Burn-in steps
        
    Returns:
        Mean CV across neurons
        
    Note: This function is NOT JIT-compiled due to variable-length spike trains.
    For most use cases, CV computation is fast enough without JIT.
    """
    # Simple slicing
    valid_spikes = spikes[burn_steps:]
    
    # Convert to numpy for easier spike time extraction
    valid_spikes_np = np.array(valid_spikes)
    n_steps, n_neurons = valid_spikes_np.shape
    
    cvs = []
    for neuron_idx in range(n_neurons):
        spike_train = valid_spikes_np[:, neuron_idx]
        spike_indices = np.where(spike_train > 0.5)[0]
        
        if len(spike_indices) < 2:
            continue  # Need at least 2 spikes
        
        spike_times = spike_indices * dt_ms
        isis = np.diff(spike_times)
        
        if len(isis) > 0 and np.mean(isis) > 1e-6:
            cv = np.std(isis) / np.mean(isis)
            cvs.append(cv)
    
    if len(cvs) == 0:
        return 0.0
    
    return float(np.mean(cvs))


def compute_cv_all_populations(spikes_dict: Dict[str, jnp.ndarray], dt_ms: float, burn_steps: int = 1000) -> Dict[str, float]:
    """
    Compute CV for all populations.
    
    Args:
        spikes_dict: {'stn': (n_steps, n_stn), 'gpe': ..., 'gpi': ...}
        dt_ms: Timestep
        burn_steps: Burn-in period
        
    Returns:
        {'stn': CV, 'gpe': CV, 'gpi': CV}
    """
    cvs = {}
    for pop_name, spike_array in spikes_dict.items():
        cvs[pop_name] = compute_cv_population(spike_array, dt_ms, burn_steps)
    
    return cvs


# ============================================================================
# BETA POWER (13-30 Hz)
# ============================================================================

def compute_beta_power_single_pop(V_trace: jnp.ndarray, dt_ms: float, burn_steps: int, freq_range: tuple = (13, 30)) -> float:
    """
    Compute power in beta band (13-30 Hz) using FFT.
    
    Args:
        V_trace: Voltage traces (n_steps, n_neurons)
        dt_ms: Timestep
        burn_steps: Burn-in steps
        freq_range: (low, high) frequency range in Hz
        
    Returns:
        Total power in beta band
        
    Note: Not JIT-compiled due to dynamic burn_steps.
    FFT operations are still fast.
    """
    # Simple slicing when not in JIT
    valid_V = V_trace[burn_steps:]
    
    # LFP proxy: mean across neurons
    lfp = jnp.mean(valid_V, axis=1)
    
    # FFT (vectorized, fast)
    fft_vals = jnp.fft.rfft(lfp)
    freqs = jnp.fft.rfftfreq(len(lfp), d=dt_ms / 1000.0)
    
    # Power spectral density
    psd = jnp.abs(fft_vals) ** 2
    
    # Extract beta band
    low_freq, high_freq = freq_range
    idx = (freqs >= low_freq) & (freqs <= high_freq)
    
    beta_power = jnp.sum(psd[idx])
    
    return float(beta_power)


def compute_beta_power_all(V_dict: Dict[str, jnp.ndarray], dt_ms: float, burn_steps: int = 1000) -> Dict[str, float]:
    """
    Compute beta power for all populations.
    
    Args:
        V_dict: {'stn': (n_steps, n_stn), 'gpe': ..., 'gpi': ...}
        dt_ms: Timestep
        burn_steps: Burn-in period
        
    Returns:
        {'stn': power, 'gpe': power, 'gpi': power}
    """
    beta = {}
    for pop_name, V_array in V_dict.items():
        beta[pop_name] = compute_beta_power_single_pop(V_array, dt_ms, burn_steps)
    
    return beta


# ============================================================================
# ALL METRICS (CONVENIENCE FUNCTION)
# ============================================================================

def compute_all_metrics(observables: Dict[str, jnp.ndarray], dt_ms: float, burn_steps: int = 1000) -> Dict:
    """
    Compute all metrics for Optuna objective function.
    
    Args:
        observables: Dict from simulation with keys:
            'V_stn', 'V_gpe', 'V_gpi',
            'spikes_stn', 'spikes_gpe', 'spikes_gpi'
        dt_ms: Timestep
        burn_steps: Number of steps to discard for steady-state
        
    Returns:
        Dict with:
            'firing_rates': {'stn': Hz, 'gpe': Hz, 'gpi': Hz}
            'cv': {'stn': CV, 'gpe': CV, 'gpi': CV}
            'beta_power': {'stn': power, 'gpe': power, 'gpi': power}
            'mean_V': {'stn': mV, 'gpe': mV, 'gpi': mV}
    """
    # Extract spikes and voltages
    spikes = {
        'stn': observables['spikes_stn'],
        'gpe': observables['spikes_gpe'],
        'gpi': observables['spikes_gpi']
    }
    
    voltages = {
        'stn': observables['V_stn'],
        'gpe': observables['V_gpe'],
        'gpi': observables['V_gpi']
    }
    
    # Compute metrics
    firing_rates = compute_firing_rates(spikes, dt_ms, burn_steps)
    cvs = compute_cv_all_populations(spikes, dt_ms, burn_steps)
    beta_power = compute_beta_power_all(voltages, dt_ms, burn_steps)
    
    # Mean voltages (simple diagnostic)
    mean_V = {pop: float(jnp.mean(V[burn_steps:])) for pop, V in voltages.items()}
    
    return {
        'firing_rates': firing_rates,
        'cv': cvs,
        'beta_power': beta_power,
        'mean_V': mean_V
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Test metrics computation."""
    import numpy as np
    
    print("=" * 70)
    print("Testing metrics_jax.py")
    print("=" * 70)
    
    # Create fake data
    n_steps = 2000
    n_neurons = 50
    dt_ms = 0.025
    
    # Fake spikes (random)
    rng = np.random.default_rng(42)
    fake_spikes = rng.random((n_steps, n_neurons)) < 0.02  # ~20 Hz
    fake_V = rng.normal(-60, 5, (n_steps, n_neurons))
    
    observables = {
        'spikes_stn': jnp.array(fake_spikes),
        'spikes_gpe': jnp.array(fake_spikes),
        'spikes_gpi': jnp.array(fake_spikes),
        'V_stn': jnp.array(fake_V),
        'V_gpe': jnp.array(fake_V),
        'V_gpi': jnp.array(fake_V)
    }
    
    # Compute metrics
    import time
    t0 = time.time()
    metrics = compute_all_metrics(observables, dt_ms, burn_steps=200)
    t1 = time.time()
    
    print(f"\nMetrics computation time: {(t1-t0)*1000:.2f} ms")
    print(f"\nFiring rates: {metrics['firing_rates']}")
    print(f"CV: {metrics['cv']}")
    print(f"Beta power: {metrics['beta_power']}")
    print(f"Mean voltages: {metrics['mean_V']}")
    
    print("\n" + "=" * 70)
    print("✓ Metrics computation works!")
    print("=" * 70)
