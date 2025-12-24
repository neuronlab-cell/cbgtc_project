"""
Healthy State Optimization - 1,800 Neurons

Targets (from literature):
- STN: 20 Hz, CV=0.4
- GPe: 70 Hz, CV=0.35  
- GPi: 80 Hz, CV=0.2
- Beta fraction < 5-10%
"""

import sys, os
sys.path.insert(0, os.getcwd())

import time
import copy
import optuna
from optuna.samplers import TPESampler
import pickle
import jax.numpy as jnp

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

# =============================================================================
# BUILD NETWORK ONCE
# =============================================================================

print("=" * 60)
print("HEALTHY STATE OPTIMIZATION - 1,800 NEURONS")
print("=" * 60)

print("\nBuilding network...")
t0 = time.time()
state, config = build_network_state(400, 800, 600, 0.025)
print(f"✓ Built in {time.time()-t0:.1f}s")

print("Compiling simulator...")
t0 = time.time()
simulator = create_simulation_fn(config, n_steps=16000)

# Warm up
dummy = {'ISTN': 80.0, 'I_gpe': 300.0, 'I_gpi': 300.0,
         'noise_stn_sigma': 0.5, 'noise_gpe_sigma': 40.0, 'noise_gpi_sigma': 40.0}
obs = simulator(dummy, state)
obs['V_stn'].block_until_ready()
print(f"✓ Compiled in {time.time()-t0:.1f}s")

# =============================================================================
# TARGETS
# =============================================================================

TARGETS = {
    'rate_stn': 20.0,
    'rate_gpe': 70.0,
    'rate_gpi': 80.0,
    'cv_stn': 0.4,
    'cv_gpe': 0.35,
    'cv_gpi': 0.2,
}

BETA_MAX = 0.08  # < 8% beta fraction

print("\nTargets:")
print(f"  STN: {TARGETS['rate_stn']} Hz, CV={TARGETS['cv_stn']}")
print(f"  GPe: {TARGETS['rate_gpe']} Hz, CV={TARGETS['cv_gpe']}")
print(f"  GPi: {TARGETS['rate_gpi']} Hz, CV={TARGETS['cv_gpi']}")
print(f"  Beta: < {BETA_MAX*100:.0f}%")

# =============================================================================
# BETA COMPUTATION
# =============================================================================

def compute_beta_fraction(V_trace, dt_ms=0.025, burn_steps=4000):
    valid_V = V_trace[burn_steps:]
    lfp = jnp.mean(valid_V, axis=1)
    lfp = lfp - jnp.mean(lfp)
    
    fft_vals = jnp.fft.rfft(lfp)
    freqs = jnp.fft.rfftfreq(len(lfp), d=dt_ms / 1000.0)
    psd = jnp.abs(fft_vals) ** 2
    
    beta_idx = (freqs >= 13) & (freqs <= 30)
    broad_idx = (freqs >= 1) & (freqs <= 100)
    
    beta_power = float(jnp.sum(psd[beta_idx]))
    total_power = float(jnp.sum(psd[broad_idx]))
    
    if total_power > 1e-10:
        return beta_power / total_power
    return 0.0

# =============================================================================
# OBJECTIVE
# =============================================================================

def objective(trial):
    # Sample parameters
    params = {
        'ISTN': trial.suggest_float('ISTN', 30.0, 120.0),
        'I_gpe': trial.suggest_float('I_gpe', 200.0, 500.0),
        'I_gpi': trial.suggest_float('I_gpi', 200.0, 500.0),
        'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.1, 2.0),
        'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 10.0, 100.0),
        'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 10.0, 100.0),
    }
    
    # Simulate
    obs = simulator(params, state)
    metrics = compute_all_metrics(obs, 0.025, burn_steps=4000)
    
    # Get values
    r_stn = metrics['firing_rates']['stn']
    r_gpe = metrics['firing_rates']['gpe']
    r_gpi = metrics['firing_rates']['gpi']
    cv_stn = metrics['cv']['stn']
    cv_gpe = metrics['cv']['gpe']
    cv_gpi = metrics['cv']['gpi']
    
    # Beta
    beta_stn = compute_beta_fraction(obs['V_stn'])
    
    # Score: squared errors
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
    
    # Beta penalty (only if too high)
    beta_penalty = 0.0
    if beta_stn > BETA_MAX:
        beta_penalty = 100.0 * (beta_stn - BETA_MAX)
    
    # Ordering penalty
    order_penalty = 0.0
    if not (r_gpi > r_gpe > r_stn):
        order_penalty = 50.0
    
    score = rate_error + 50.0 * cv_error + beta_penalty + order_penalty
    
    # Log
    trial.set_user_attr('r_stn', float(r_stn))
    trial.set_user_attr('r_gpe', float(r_gpe))
    trial.set_user_attr('r_gpi', float(r_gpi))
    trial.set_user_attr('cv_stn', float(cv_stn))
    trial.set_user_attr('cv_gpe', float(cv_gpe))
    trial.set_user_attr('cv_gpi', float(cv_gpi))
    trial.set_user_attr('beta', float(beta_stn))
    
    return score

# =============================================================================
# RUN
# =============================================================================

print("\n" + "=" * 60)
print("STARTING OPTIMIZATION (1000 trials)")
print("=" * 60)

study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42, n_startup_trials=50, multivariate=True)
)

t_start = time.time()

study.optimize(
    objective,
    n_trials=1000,
    show_progress_bar=True,
    callbacks=[
        lambda study, trial: print(
            f"Trial {trial.number}: score={trial.value:.1f}, "
            f"STN={trial.user_attrs.get('r_stn',0):.1f}Hz, "
            f"GPe={trial.user_attrs.get('r_gpe',0):.1f}Hz, "
            f"GPi={trial.user_attrs.get('r_gpi',0):.1f}Hz, "
            f"beta={trial.user_attrs.get('beta',0)*100:.1f}%"
        ) if trial.value is not None else None
    ]
)

total_time = time.time() - t_start

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nCompleted in {total_time/60:.1f} minutes")

best = study.best_trial
print("\n--- BEST PARAMETERS ---")
for p, v in study.best_params.items():
    print(f"  {p}: {v:.3f}")

print("\n--- METRICS ---")
print(f"  STN: {best.user_attrs['r_stn']:.1f} Hz (target: 20), CV={best.user_attrs['cv_stn']:.3f} (target: 0.40)")
print(f"  GPe: {best.user_attrs['r_gpe']:.1f} Hz (target: 70), CV={best.user_attrs['cv_gpe']:.3f} (target: 0.35)")
print(f"  GPi: {best.user_attrs['r_gpi']:.1f} Hz (target: 80), CV={best.user_attrs['cv_gpi']:.3f} (target: 0.20)")
print(f"  Beta: {best.user_attrs['beta']*100:.1f}% (target: <8%)")
print(f"\n  Score: {study.best_value:.2f}")

# Check ordering
r_ok = best.user_attrs['r_gpi'] > best.user_attrs['r_gpe'] > best.user_attrs['r_stn']
print(f"\n  Rate ordering (GPi>GPe>STN): {'✓' if r_ok else '✗'}")

# Save
with open('results/healthy_1800_study.pkl', 'wb') as f:
    pickle.dump(study, f)

print("\n✓ Saved to results/healthy_1800_study.pkl")
print("=" * 60)
