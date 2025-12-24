import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import time

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print("=" * 70)
print("Optuna with EXPANDED Search Ranges")
print("=" * 70)

# Setup
state, config = build_network_state(50, 100, 75, 0.025)
simulate = create_simulation_fn(config, n_steps=4000)

# Warm up
dummy = {'ISTN': 42.0, 'I_gpe': 1200.0, 'I_gpi': 600.0, 
         'noise_stn_sigma': 0.15, 'noise_gpe_sigma': 30.0, 'noise_gpi_sigma': 30.0}
obs = simulate(dummy, state)
obs['V_stn'].block_until_ready()

TARGET_RATES = {'stn': 20.0, 'gpe': 60.0, 'gpi': 70.0}

def objective(trial):
    params = {
        'ISTN': trial.suggest_float('ISTN', 25.0, 60.0),           # Slightly expanded
        'I_gpe': trial.suggest_float('I_gpe', 800.0, 2000.0),      # MUCH wider (was 400-700)
        'I_gpi': trial.suggest_float('I_gpi', 400.0, 1200.0),      # MUCH wider (was 200-350)
        'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.05, 0.3),
        'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 10.0, 80.0),  # Wider
        'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 10.0, 80.0)   # Wider
    }
    
    obs = simulate(params, state)
    metrics = compute_all_metrics(obs, dt_ms=0.025, burn_steps=1000)
    
    # Scoring
    rate_error = sum([
        (metrics['firing_rates'][pop] - TARGET_RATES[pop]) ** 2 
        for pop in ['stn', 'gpe', 'gpi']
    ])
    
    beta_penalty = sum(metrics['beta_power'].values()) / 1e6
    
    cv_error = sum([
        (metrics['cv'][pop] - 0.4) ** 2 
        for pop in ['stn', 'gpe', 'gpi']
    ])
    
    score = 1.0 * rate_error + 0.01 * beta_penalty + 0.1 * cv_error
    
    # Log metrics
    trial.set_user_attr('rate_stn', float(metrics['firing_rates']['stn']))
    trial.set_user_attr('rate_gpe', float(metrics['firing_rates']['gpe']))
    trial.set_user_attr('rate_gpi', float(metrics['firing_rates']['gpi']))
    
    return score

print("\nStarting 20-trial optimization with expanded ranges...")
print("GPe range: 800-2000 (was 400-700)")
print("GPi range: 400-1200 (was 200-350)")
print()

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("\n--- Best Parameters ---")
for k, v in study.best_params.items():
    print(f"  {k:20s}: {v:.3f}")

print(f"\n--- Best Metrics ---")
print(f"  Score: {study.best_value:.3f}")
print(f"  STN rate: {study.best_trial.user_attrs['rate_stn']:.1f} Hz (target: 20 Hz)")
print(f"  GPe rate: {study.best_trial.user_attrs['rate_gpe']:.1f} Hz (target: 60 Hz)")
print(f"  GPi rate: {study.best_trial.user_attrs['rate_gpi']:.1f} Hz (target: 70 Hz)")

print("\n" + "=" * 70)
