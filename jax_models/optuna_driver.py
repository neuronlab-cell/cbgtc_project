"""
Optuna driver for JAX-accelerated parameter optimization.

This script optimizes network parameters to match target firing rates
and minimize pathological beta oscillations.

Author: Kavin Nakkeeran
Functional Neurosurgery Lab, Johns Hopkins University
Date: December 2025
"""

import optuna
import time
from typing import Dict
import jax.numpy as jnp

from network_builder import build_network_state
from sim_jax import create_simulation_fn
from metrics_jax import compute_all_metrics


# ============================================================================
# GLOBAL SETUP (Build network and JIT-compile once)
# ============================================================================

print("=" * 70)
print("JAX-Optuna Parameter Optimization Pipeline")
print("=" * 70)

# Network configuration
N_STN = 50
N_GPE = 100
N_GPI = 75
DT_MS = 0.025
N_STEPS = 4000  # 100ms simulation
BURN_STEPS = 1000  # Discard first 25ms

print(f"\nNetwork size: {N_STN} STN, {N_GPE} GPe, {N_GPI} GPi")
print(f"Simulation: {N_STEPS} steps ({N_STEPS * DT_MS:.1f} ms)")
print(f"Burn-in: {BURN_STEPS} steps ({BURN_STEPS * DT_MS:.1f} ms)")

# Build network once (reused for all trials)
print("\nBuilding network...")
t0 = time.time()
state, config = build_network_state(
    n_stn=N_STN, 
    n_gpe=N_GPE, 
    n_gpi=N_GPI, 
    dt_ms=DT_MS
)
t1 = time.time()
print(f"✓ Network built in {(t1-t0)*1000:.1f} ms")

# Create JIT-compiled simulator (compile once, reuse many times)
print("\nCreating JIT-compiled simulator...")
t0 = time.time()
simulate = create_simulation_fn(config, n_steps=N_STEPS)

# Warm up JIT with dummy run
dummy_params = {
    'ISTN': 42.0, 'I_gpe': 580.0, 'I_gpi': 240.0,
    'noise_stn_sigma': 0.15, 'noise_gpe_sigma': 30.0, 'noise_gpi_sigma': 30.0
}
obs = simulate(dummy_params, state)
obs['V_stn'].block_until_ready()  # Wait for GPU
t1 = time.time()
print(f"✓ Simulator compiled in {(t1-t0)*1000:.1f} ms")

# Test simulation speed
print("\nTesting simulation speed...")
t0 = time.time()
obs = simulate(dummy_params, state)
obs['V_stn'].block_until_ready()
t1 = time.time()
sim_time_ms = (t1-t0) * 1000
print(f"✓ Cached simulation: {sim_time_ms:.2f} ms per trial")
print(f"✓ Expected throughput: ~{int(1000/sim_time_ms)} trials/second")

print("\n" + "=" * 70)


# ============================================================================
# TARGET VALUES (Literature-based)
# ============================================================================

# Firing rate targets (Hz)
TARGET_RATES = {
    'stn': 20.0,   # Levy et al. 2001: 15-25 Hz
    'gpe': 60.0,   # DeLong 1971: 60-80 Hz
    'gpi': 70.0    # DeLong 1971: 60-80 Hz
}

# CV targets (optional - for now just minimize rate error)
TARGET_CVS = {
    'stn': 0.4,    # Wichmann & Soares 2006: 0.31 ± 0.14
    'gpe': 0.45,   # Benazzouz et al. 2002: 0.3-0.6
    'gpi': 0.2     # Miller & DeLong 1987: 0.1-0.3
}


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Score (lower is better)
    """
    # Sample parameters
    params = {
        'ISTN': trial.suggest_float('ISTN', 25.0, 45.0),
        'I_gpe': trial.suggest_float('I_gpe', 400.0, 700.0),
        'I_gpi': trial.suggest_float('I_gpi', 200.0, 350.0),
        'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.05, 0.3),
        'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 10.0, 50.0),
        'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 10.0, 50.0)
    }
    
    # Run simulation (FAST!)
    obs = simulate(params, state)
    
    # Compute metrics
    metrics = compute_all_metrics(obs, dt_ms=DT_MS, burn_steps=BURN_STEPS)
    
    # --- Scoring function ---
    
    # 1. Firing rate error (primary objective)
    rate_error = 0.0
    for pop in ['stn', 'gpe', 'gpi']:
        actual_rate = metrics['firing_rates'][pop]
        target_rate = TARGET_RATES[pop]
        # Squared error (penalizes large deviations more)
        rate_error += (actual_rate - target_rate) ** 2
    
    # 2. Beta power penalty (minimize pathological oscillations)
    # Total beta power across all populations
    beta_penalty = sum(metrics['beta_power'].values())
    
    # Normalize beta penalty (optional - depends on scale)
    beta_penalty = beta_penalty / 1e6  # Scale down
    
    # 3. CV penalty (optional - encourage realistic irregularity)
    cv_error = 0.0
    for pop in ['stn', 'gpe', 'gpi']:
        actual_cv = metrics['cv'][pop]
        target_cv = TARGET_CVS[pop]
        cv_error += (actual_cv - target_cv) ** 2
    
    # Combined score (weighted sum)
    score = (
        1.0 * rate_error +      # Weight: 1.0 (most important)
        0.01 * beta_penalty +   # Weight: 0.01 (secondary)
        0.1 * cv_error          # Weight: 0.1 (tertiary)
    )
    
    # Log metrics for this trial
    trial.set_user_attr('rate_stn', float(metrics['firing_rates']['stn']))
    trial.set_user_attr('rate_gpe', float(metrics['firing_rates']['gpe']))
    trial.set_user_attr('rate_gpi', float(metrics['firing_rates']['gpi']))
    trial.set_user_attr('beta_stn', float(metrics['beta_power']['stn']))
    trial.set_user_attr('cv_stn', float(metrics['cv']['stn']))
    
    return score


# ============================================================================
# OPTIMIZATION
# ============================================================================

def run_optimization(n_trials: int = 100, study_name: str = "jax_optuna_test"):
    """
    Run Optuna optimization.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name for this study
    """
    print("\n" + "=" * 70)
    print(f"Starting Optimization: {n_trials} trials")
    print("=" * 70)
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42)  # Reproducible
    )
    
    # Run optimization with timing
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    t1 = time.time()
    
    total_time_sec = t1 - t0
    time_per_trial_ms = (total_time_sec / n_trials) * 1000
    
    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    print(f"\nTotal time: {total_time_sec:.1f} seconds")
    print(f"Time per trial: {time_per_trial_ms:.2f} ms")
    print(f"Throughput: {n_trials / total_time_sec:.1f} trials/second")
    
    print(f"\n--- Best Parameters ---")
    for param, value in study.best_params.items():
        print(f"  {param:20s}: {value:.3f}")
    
    print(f"\n--- Best Metrics ---")
    print(f"  Score: {study.best_value:.3f}")
    print(f"  STN rate: {study.best_trial.user_attrs['rate_stn']:.1f} Hz (target: {TARGET_RATES['stn']:.1f} Hz)")
    print(f"  GPe rate: {study.best_trial.user_attrs['rate_gpe']:.1f} Hz (target: {TARGET_RATES['gpe']:.1f} Hz)")
    print(f"  GPi rate: {study.best_trial.user_attrs['rate_gpi']:.1f} Hz (target: {TARGET_RATES['gpi']:.1f} Hz)")
    print(f"  STN beta: {study.best_trial.user_attrs['beta_stn']:.2e}")
    print(f"  STN CV: {study.best_trial.user_attrs['cv_stn']:.3f} (target: {TARGET_CVS['stn']:.3f})")
    
    print("\n" + "=" * 70)
    
    return study


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Quick test with 10 trials
    print("\n*** QUICK TEST: 10 trials ***")
    study_test = run_optimization(n_trials=10, study_name="quick_test")
    
    # Uncomment for full optimization
    # print("\n\n*** FULL OPTIMIZATION: 1000 trials ***")
    # study_full = run_optimization(n_trials=1000, study_name="full_optimization")
    
    # Save results (optional)
    # import pickle
    # with open('optuna_results.pkl', 'wb') as f:
    #     pickle.dump(study_full, f)
    
    print("\n✓ Optimization pipeline ready!")
    print("  - Increase n_trials for better results")
    print("  - Adjust scoring weights in objective() as needed")
    print("  - Use Optuna dashboard for visualization:")
    print("    optuna-dashboard sqlite:///optuna_study.db")
