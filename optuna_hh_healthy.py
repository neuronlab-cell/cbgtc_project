"""
Optuna optimization for HH GPe/GPi network - Healthy state.
Target: STN ~20 Hz, GPe ~70 Hz, GPi ~80 Hz
"""

import sys
sys.path.insert(0, '.')

import optuna
from optuna.samplers import CmaEsSampler
import jax
import jax.numpy as jnp
import time
import pickle

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# NETWORK SETUP (450 neurons)
# =============================================================================

N_STN, N_GPE, N_GPI = 100, 200, 150
DT_MS = 0.025
N_STEPS = 16000  # 400ms
BURN_STEPS = 4000  # 100ms burn-in

print(f"\nBuilding {N_STN + N_GPE + N_GPI}-neuron HH network...")
state, config = build_network_state(N_STN, N_GPE, N_GPI, DT_MS, use_hh=True)
simulator = create_simulation_fn(config, n_steps=N_STEPS)

# Warm-up JIT
print("Warming up JIT...")
dummy_params = {
    'ISTN': 100.0, 'I_gpe': 3.0, 'I_gpi': 3.0,
    'noise_stn_sigma': 1.0, 'noise_gpe_sigma': 30.0, 'noise_gpi_sigma': 30.0,
}
obs = simulator(dummy_params, state)
obs['V_stn'].block_until_ready()
print("JIT ready!\n")

# =============================================================================
# TARGETS (Healthy state)
# =============================================================================

TARGETS = {
    'rate_stn': 20.0,
    'rate_gpe': 70.0,
    'rate_gpi': 80.0,
    'cv_stn': 0.40,
    'cv_gpe': 0.35,
    'cv_gpi': 0.20,
}

# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def objective(trial):
    # Sample parameters
    params = {
        'ISTN': trial.suggest_float('ISTN', 80.0, 200.0),
        'I_gpe': trial.suggest_float('I_gpe', 1.0, 8.0),
        'I_gpi': trial.suggest_float('I_gpi', 1.0, 8.0),
        'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.5, 5.0),
        'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 10.0, 100.0),
        'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 10.0, 100.0),
    }
    
    # Run simulation
    try:
        obs = simulator(params, state)
        obs['V_stn'].block_until_ready()
        
        # Check for NaN
        if jnp.any(jnp.isnan(obs['V_stn'])) or jnp.any(jnp.isnan(obs['V_gpe'])):
            return 1e6
        
        # Compute metrics
        metrics = compute_all_metrics(obs, DT_MS, burn_steps=BURN_STEPS)
        
        # Extract values
        r_stn = metrics['firing_rates']['stn']
        r_gpe = metrics['firing_rates']['gpe']
        r_gpi = metrics['firing_rates']['gpi']
        cv_stn = metrics['cv']['stn']
        cv_gpe = metrics['cv']['gpe']
        cv_gpi = metrics['cv']['gpi']
        
        # Compute loss (weighted squared errors)
        loss = 0.0
        
        # Firing rate errors (weight = 1)
        loss += ((r_stn - TARGETS['rate_stn']) / TARGETS['rate_stn']) ** 2
        loss += ((r_gpe - TARGETS['rate_gpe']) / TARGETS['rate_gpe']) ** 2
        loss += ((r_gpi - TARGETS['rate_gpi']) / TARGETS['rate_gpi']) ** 2
        
        # CV errors (weight = 0.5)
        loss += 0.5 * ((cv_stn - TARGETS['cv_stn']) / TARGETS['cv_stn']) ** 2
        loss += 0.5 * ((cv_gpe - TARGETS['cv_gpe']) / TARGETS['cv_gpe']) ** 2
        loss += 0.5 * ((cv_gpi - TARGETS['cv_gpi']) / TARGETS['cv_gpi']) ** 2
        
        # Penalty for silent populations
        if r_stn < 1.0:
            loss += 100.0
        if r_gpe < 10.0:
            loss += 100.0
        if r_gpi < 10.0:
            loss += 100.0
            
        # Report intermediate values
        trial.set_user_attr('r_stn', float(r_stn))
        trial.set_user_attr('r_gpe', float(r_gpe))
        trial.set_user_attr('r_gpi', float(r_gpi))
        trial.set_user_attr('cv_stn', float(cv_stn))
        trial.set_user_attr('cv_gpe', float(cv_gpe))
        trial.set_user_attr('cv_gpi', float(cv_gpi))
        
        return loss
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 1e6

# =============================================================================
# RUN OPTIMIZATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HH Network Optimization - Healthy State")
    print("=" * 60)
    print(f"Network: {N_STN}/{N_GPE}/{N_GPI} neurons")
    print(f"Targets: STN={TARGETS['rate_stn']}Hz, GPe={TARGETS['rate_gpe']}Hz, GPi={TARGETS['rate_gpi']}Hz")
    print("=" * 60)
    
    # Create study with CMA-ES
    study = optuna.create_study(
        direction='minimize',
        sampler=CmaEsSampler(seed=42),
        study_name='hh_healthy_450'
    )
    
    # Run optimization
    t0 = time.time()
    study.optimize(objective, n_trials=500, show_progress_bar=True)
    elapsed = time.time() - t0
    
    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed/500*1000:.0f}ms/trial)")
    print(f"Best score: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.3f}")
    
    print(f"\nBest metrics:")
    best = study.best_trial
    print(f"  STN: {best.user_attrs['r_stn']:.1f} Hz (target: {TARGETS['rate_stn']})")
    print(f"  GPe: {best.user_attrs['r_gpe']:.1f} Hz (target: {TARGETS['rate_gpe']})")
    print(f"  GPi: {best.user_attrs['r_gpi']:.1f} Hz (target: {TARGETS['rate_gpi']})")
    print(f"  STN CV: {best.user_attrs['cv_stn']:.2f} (target: {TARGETS['cv_stn']})")
    print(f"  GPe CV: {best.user_attrs['cv_gpe']:.2f} (target: {TARGETS['cv_gpe']})")
    print(f"  GPi CV: {best.user_attrs['cv_gpi']:.2f} (target: {TARGETS['cv_gpi']})")
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_metrics': best.user_attrs,
        'targets': TARGETS,
        'network_size': (N_STN, N_GPE, N_GPI),
        'n_trials': len(study.trials),
        'elapsed_seconds': elapsed,
    }
    
    with open('results/hh_healthy_study.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to results/hh_healthy_study.pkl")
