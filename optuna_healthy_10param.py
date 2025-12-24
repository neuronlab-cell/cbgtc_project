"""
Healthy State Optimization - 10 Parameters

Parameters:
- 3 intrinsic currents (ISTN, I_gpe, I_gpi)
- 4 synaptic weight multipliers (g_stn_gpe, g_gpe_stn, g_stn_gpi, g_gpe_gpi)
- 3 noise sigmas

Targets:
- STN: 20 Hz, CV=0.4
- GPe: 70 Hz, CV=0.35
- GPi: 80 Hz, CV=0.2
- Beta < 8%
"""

import sys, os
sys.path.insert(0, os.getcwd())

import time
import copy
import optuna
from optuna.samplers import CmaEsSampler
import pickle
import jax.numpy as jnp
import numpy as np

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

# =============================================================================
# BUILD BASE NETWORK
# =============================================================================

print("=" * 60)
print("HEALTHY STATE - 10 PARAMETER OPTIMIZATION")
print("=" * 60)

print("\nBuilding base network...")
state, base_config = build_network_state(400, 800, 600, 0.025)

# Store reference g_max values (already scaled for network size)
REF_WEIGHTS = {
    'g_stn_gpe': base_config['synapses']['stn_to_gpe'].g_max,
    'g_gpe_stn': base_config['synapses']['gpe_to_stn'].g_max,
    'g_stn_gpi': base_config['synapses']['stn_to_gpi'].g_max,
    'g_gpe_gpi': base_config['synapses']['gpe_to_gpi'].g_max,
}

print(f"Reference weights (scaled for 1800 neurons):")
for k, v in REF_WEIGHTS.items():
    print(f"  {k}: {v:.4f}")

# =============================================================================
# SIMULATOR CACHE
# =============================================================================

# Cache simulators to avoid recompilation
simulator_cache = {}

def get_simulator(weight_multipliers):
    """Get or create simulator for given weight multipliers."""
    # Round to 1 decimal to increase cache hits
    cache_key = tuple(round(v, 1) for v in weight_multipliers.values())
    
    if cache_key not in simulator_cache:
        # Create new config with modified weights
        config = copy.deepcopy(base_config)
        
        config['synapses']['stn_to_gpe'] = config['synapses']['stn_to_gpe']._replace(
            g_max=REF_WEIGHTS['g_stn_gpe'] * weight_multipliers['g_stn_gpe'])
        config['synapses']['gpe_to_stn'] = config['synapses']['gpe_to_stn']._replace(
            g_max=REF_WEIGHTS['g_gpe_stn'] * weight_multipliers['g_gpe_stn'])
        config['synapses']['stn_to_gpi'] = config['synapses']['stn_to_gpi']._replace(
            g_max=REF_WEIGHTS['g_stn_gpi'] * weight_multipliers['g_stn_gpi'])
        config['synapses']['gpe_to_gpi'] = config['synapses']['gpe_to_gpi']._replace(
            g_max=REF_WEIGHTS['g_gpe_gpi'] * weight_multipliers['g_gpe_gpi'])
        
        # Compile
        simulator = create_simulation_fn(config, n_steps=16000)
        
        # Warm up
        dummy = {'ISTN': 80.0, 'I_gpe': 400.0, 'I_gpi': 500.0,
                 'noise_stn_sigma': 0.5, 'noise_gpe_sigma': 40.0, 'noise_gpi_sigma': 40.0}
        obs = simulator(dummy, state)
        obs['V_stn'].block_until_ready()
        
        simulator_cache[cache_key] = simulator
        print(f"  [Compiled new simulator, cache size: {len(simulator_cache)}]")
    
    return simulator_cache[cache_key]

# =============================================================================
# TARGETS & HELPERS
# =============================================================================

TARGETS = {
    'rate_stn': 20.0, 'rate_gpe': 70.0, 'rate_gpi': 80.0,
    'cv_stn': 0.4, 'cv_gpe': 0.35, 'cv_gpi': 0.2,
}

def compute_beta_fraction(V_trace, burn_steps=4000):
    valid_V = V_trace[burn_steps:]
    lfp = jnp.mean(valid_V, axis=1)
    lfp = lfp - jnp.mean(lfp)
    fft_vals = jnp.fft.rfft(lfp)
    freqs = jnp.fft.rfftfreq(len(lfp), d=0.025 / 1000.0)
    psd = jnp.abs(fft_vals) ** 2
    beta_idx = (freqs >= 13) & (freqs <= 30)
    broad_idx = (freqs >= 1) & (freqs <= 100)
    beta = float(jnp.sum(psd[beta_idx]))
    total = float(jnp.sum(psd[broad_idx]))
    return beta / total if total > 1e-10 else 0.0

def save_study(study, filename='results/healthy_10param_study.pkl'):
    os.makedirs('results', exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(study, f)

# =============================================================================
# OBJECTIVE
# =============================================================================

def objective(trial):
    # Intrinsic currents (3)
    intrinsic_params = {
        'ISTN': trial.suggest_float('ISTN', 30.0, 150.0),
        'I_gpe': trial.suggest_float('I_gpe', 250.0, 600.0),
        'I_gpi': trial.suggest_float('I_gpi', 200.0, 500.0),
    }
    
    # Synaptic weight multipliers (4)
    weight_multipliers = {
        'g_stn_gpe': trial.suggest_float('g_stn_gpe_mult', 0.3, 3.0),
        'g_gpe_stn': trial.suggest_float('g_gpe_stn_mult', 0.3, 3.0),
        'g_stn_gpi': trial.suggest_float('g_stn_gpi_mult', 0.3, 3.0),
        'g_gpe_gpi': trial.suggest_float('g_gpe_gpi_mult', 0.3, 3.0),
    }
    
    # Noise (3)
    noise_params = {
        'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.1, 3.0),
        'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 10.0, 150.0),
        'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 10.0, 150.0),
    }
    
    # Get simulator for these weights
    simulator = get_simulator(weight_multipliers)
    
    # Combine params for simulation
    sim_params = {**intrinsic_params, **noise_params}
    
    # Run simulation
    obs = simulator(sim_params, state)
    metrics = compute_all_metrics(obs, 0.025, burn_steps=4000)
    
    # Extract metrics
    r_stn = metrics['firing_rates']['stn']
    r_gpe = metrics['firing_rates']['gpe']
    r_gpi = metrics['firing_rates']['gpi']
    cv_stn = metrics['cv']['stn']
    cv_gpe = metrics['cv']['gpe']
    cv_gpi = metrics['cv']['gpi']
    
    # Check validity
    if r_stn < 1.0 or r_gpe < 1.0 or r_gpi < 1.0:
        return float('inf')
    if any(np.isnan(x) for x in [r_stn, r_gpe, r_gpi, cv_stn, cv_gpe, cv_gpi]):
        return float('inf')
    
    beta = compute_beta_fraction(obs['V_stn'])
    
    # Score
    rate_error = (
        (r_stn - TARGETS['rate_stn'])**2 +
        (r_gpe - TARGETS['rate_gpe'])**2 +
        (r_gpi - TARGETS['rate_gpi'])**2
    )
    
    cv_error = (
        (cv_stn - TARGETS['cv_stn'])**2 +
        (cv_gpe - TARGETS['cv_gpe'])**2 +
        (cv_gpi - TARGETS['cv_gpi'])**2
    )
    
    beta_penalty = 100.0 * max(0, beta - 0.08)
    order_penalty = 0.0 if (r_gpi > r_gpe > r_stn) else 50.0
    
    score = rate_error + 50.0 * cv_error + beta_penalty + order_penalty
    
    # Log
    trial.set_user_attr('r_stn', float(r_stn))
    trial.set_user_attr('r_gpe', float(r_gpe))
    trial.set_user_attr('r_gpi', float(r_gpi))
    trial.set_user_attr('cv_stn', float(cv_stn))
    trial.set_user_attr('cv_gpe', float(cv_gpe))
    trial.set_user_attr('cv_gpi', float(cv_gpi))
    trial.set_user_attr('beta', float(beta))
    
    return score

# =============================================================================
# CALLBACK
# =============================================================================

def callback(study, trial):
    if trial.value is not None and trial.value != float('inf'):
        print(f"Trial {trial.number}: score={trial.value:.1f}, "
              f"STN={trial.user_attrs.get('r_stn',0):.1f}Hz/{trial.user_attrs.get('cv_stn',0):.2f}, "
              f"GPe={trial.user_attrs.get('r_gpe',0):.1f}Hz, "
              f"GPi={trial.user_attrs.get('r_gpi',0):.1f}Hz")
    
    if (trial.number + 1) % 50 == 0:
        save_study(study)
        print(f"  [Auto-saved at trial {trial.number + 1}]")

# =============================================================================
# RUN
# =============================================================================

print("\n" + "=" * 60)
print("STARTING 10-PARAMETER CMA-ES (1000 trials)")
print("=" * 60 + "\n")

study = optuna.create_study(
    direction='minimize',
    sampler=CmaEsSampler(seed=42)
)

t_start = time.time()

try:
    study.optimize(objective, n_trials=1000, callbacks=[callback])
except KeyboardInterrupt:
    print("\n\n⚠️ Interrupted! Saving...")

save_study(study)
total_time = time.time() - t_start

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nCompleted {len(study.trials)} trials in {total_time/60:.1f} minutes")
print(f"Simulator cache size: {len(simulator_cache)} compiled variants")

if study.best_trial:
    best = study.best_trial
    
    print("\n--- BEST PARAMETERS ---")
    print("\nIntrinsic currents:")
    print(f"  ISTN: {study.best_params['ISTN']:.1f} pA")
    print(f"  I_gpe: {study.best_params['I_gpe']:.1f} pA")
    print(f"  I_gpi: {study.best_params['I_gpi']:.1f} pA")
    
    print("\nSynaptic weight multipliers:")
    print(f"  g_stn_gpe: {study.best_params['g_stn_gpe_mult']:.2f}x")
    print(f"  g_gpe_stn: {study.best_params['g_gpe_stn_mult']:.2f}x")
    print(f"  g_stn_gpi: {study.best_params['g_stn_gpi_mult']:.2f}x")
    print(f"  g_gpe_gpi: {study.best_params['g_gpe_gpi_mult']:.2f}x")
    
    print("\nNoise:")
    print(f"  noise_stn_sigma: {study.best_params['noise_stn_sigma']:.2f}")
    print(f"  noise_gpe_sigma: {study.best_params['noise_gpe_sigma']:.1f}")
    print(f"  noise_gpi_sigma: {study.best_params['noise_gpi_sigma']:.1f}")

    print("\n--- METRICS ---")
    print(f"  STN: {best.user_attrs['r_stn']:.1f} Hz (target: 20), CV={best.user_attrs['cv_stn']:.3f} (target: 0.40)")
    print(f"  GPe: {best.user_attrs['r_gpe']:.1f} Hz (target: 70), CV={best.user_attrs['cv_gpe']:.3f} (target: 0.35)")
    print(f"  GPi: {best.user_attrs['r_gpi']:.1f} Hz (target: 80), CV={best.user_attrs['cv_gpi']:.3f} (target: 0.20)")
    print(f"  Beta: {best.user_attrs['beta']*100:.1f}%")
    print(f"\n  Score: {study.best_value:.2f}")

print("\n✓ Saved to results/healthy_10param_study.pkl")
print("=" * 60)
