"""
Diagnose whether the network can produce beta oscillations.

Test hypothesis: Weaker GPe→STN (reduced g_gpe_stn) should allow 
STN-GPe loop to oscillate in beta range.
"""

import sys, os
sys.path.insert(0, os.getcwd())
import copy
import numpy as np
import jax.numpy as jnp

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print("=" * 80)
print("DIAGNOSING BETA OSCILLATION CAPABILITY")
print("=" * 80)

# Build network
state, config = build_network_state(50, 100, 75, 0.025)

def compute_beta_fraction(V_trace, dt_ms, burn_steps):
    """Compute beta fraction with mean centering."""
    valid_V = V_trace[burn_steps:]
    lfp = jnp.mean(valid_V, axis=1)
    lfp = lfp - jnp.mean(lfp)  # Mean center
    
    fft_vals = jnp.fft.rfft(lfp)
    freqs = jnp.fft.rfftfreq(len(lfp), d=dt_ms / 1000.0)
    psd = jnp.abs(fft_vals) ** 2
    
    beta_idx = (freqs >= 13) & (freqs <= 30)
    broad_idx = (freqs >= 1) & (freqs <= 100)
    
    beta_power = float(jnp.sum(psd[beta_idx]))
    total_power = float(jnp.sum(psd[broad_idx]))
    
    if total_power > 1e-10:
        return beta_power / total_power, beta_power
    return 0.0, 0.0

# Test different GPe→STN strengths
# Hypothesis: WEAKER g_gpe_stn allows oscillations
print("\n--- Testing GPe→STN Strength Effect on Beta ---")
print(f"{'g_gpe_stn':>10} | {'STN Hz':>8} | {'GPe Hz':>8} | {'Beta %':>8} | {'Beta Raw':>12}")
print("-" * 60)

N_STEPS = 16000  # 400ms for better frequency resolution
BURN = 4000      # 100ms burn-in

for g_gpe_stn in [15.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.5]:
    
    trial_config = copy.deepcopy(config)
    trial_config['synapses']['stn_to_gpe'] = trial_config['synapses']['stn_to_gpe']._replace(g_max=2.5)
    trial_config['synapses']['gpe_to_stn'] = trial_config['synapses']['gpe_to_stn']._replace(g_max=g_gpe_stn)
    trial_config['synapses']['stn_to_gpi'] = trial_config['synapses']['stn_to_gpi']._replace(g_max=2.5)
    trial_config['synapses']['gpe_to_gpi'] = trial_config['synapses']['gpe_to_gpi']._replace(g_max=3.0)
    
    simulate = create_simulation_fn(trial_config, n_steps=N_STEPS)
    
    params = {
        'ISTN': 70.0,
        'I_gpe': 300.0,
        'I_gpi': 250.0,
        'noise_stn_sigma': 0.5,
        'noise_gpe_sigma': 40.0,
        'noise_gpi_sigma': 40.0
    }
    
    obs = simulate(params, state)
    metrics = compute_all_metrics(obs, 0.025, BURN)
    
    beta_frac, beta_raw = compute_beta_fraction(obs['V_stn'], 0.025, BURN)
    
    print(f"{g_gpe_stn:>10.1f} | {metrics['firing_rates']['stn']:>8.1f} | "
          f"{metrics['firing_rates']['gpe']:>8.1f} | {beta_frac*100:>7.1f}% | {beta_raw:>12.2e}")

print("\n" + "=" * 80)
print("Testing different I_gpe (dopamine depletion simulation)")
print("=" * 80)
print(f"{'I_gpe':>10} | {'STN Hz':>8} | {'GPe Hz':>8} | {'Beta %':>8} | {'CV STN':>8}")
print("-" * 60)

# Use moderate g_gpe_stn, vary I_gpe
trial_config = copy.deepcopy(config)
trial_config['synapses']['stn_to_gpe'] = trial_config['synapses']['stn_to_gpe']._replace(g_max=2.5)
trial_config['synapses']['gpe_to_stn'] = trial_config['synapses']['gpe_to_stn']._replace(g_max=5.0)  # Moderate
trial_config['synapses']['stn_to_gpi'] = trial_config['synapses']['stn_to_gpi']._replace(g_max=2.5)
trial_config['synapses']['gpe_to_gpi'] = trial_config['synapses']['gpe_to_gpi']._replace(g_max=3.0)

simulate = create_simulation_fn(trial_config, n_steps=N_STEPS)

for I_gpe in [400.0, 350.0, 300.0, 250.0, 200.0, 150.0, 100.0]:
    params = {
        'ISTN': 70.0,
        'I_gpe': I_gpe,
        'I_gpi': 250.0,
        'noise_stn_sigma': 0.5,
        'noise_gpe_sigma': 40.0,
        'noise_gpi_sigma': 40.0
    }
    
    obs = simulate(params, state)
    metrics = compute_all_metrics(obs, 0.025, BURN)
    
    beta_frac, _ = compute_beta_fraction(obs['V_stn'], 0.025, BURN)
    
    print(f"{I_gpe:>10.1f} | {metrics['firing_rates']['stn']:>8.1f} | "
          f"{metrics['firing_rates']['gpe']:>8.1f} | {beta_frac*100:>7.1f}% | "
          f"{metrics['cv']['stn']:>8.3f}")

print("\n" + "=" * 80)
print("Testing with very low noise (noise can mask oscillations)")
print("=" * 80)

for noise in [50.0, 20.0, 10.0, 5.0, 1.0]:
    params = {
        'ISTN': 70.0,
        'I_gpe': 200.0,  # Reduced (PD-like)
        'I_gpi': 250.0,
        'noise_stn_sigma': noise * 0.01,
        'noise_gpe_sigma': noise,
        'noise_gpi_sigma': noise
    }
    
    obs = simulate(params, state)
    metrics = compute_all_metrics(obs, 0.025, BURN)
    
    beta_frac, beta_raw = compute_beta_fraction(obs['V_stn'], 0.025, BURN)
    
    print(f"noise={noise:>5.1f} | STN={metrics['firing_rates']['stn']:>6.1f}Hz | "
          f"Beta={beta_frac*100:>5.1f}% | CV={metrics['cv']['stn']:.3f}")

print("\n" + "=" * 80)
