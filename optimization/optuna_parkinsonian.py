"""
Parkinsonian state optimization using SAME parameters as healthy state.

Based on literature-accepted PD firing statistics:
- STN: 20-30 Hz, CV 0.6-1.0 (bursty, beta-locked)
- GPe: 30-60 Hz, CV 0.2-0.5 (rhythmic pauses)
- GPi: 60-90 Hz, CV 0.2-0.4 (high-rate, modulated)

We optimize the SAME 10 biological parameters to discover what changes
produce Parkinsonian dynamics.

Reference: Rubin & Terman, Nevado-Holgado, Bergman, Wichmann, Brown
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.samplers import TPESampler
import time
import pickle
from pathlib import Path

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print("=" * 80)
print("PARKINSONIAN STATE OPTIMIZATION")
print("Discovering parameter regime for PD dynamics")
print("=" * 80)

# =============================================================================
# NETWORK SETUP (Same as healthy)
# =============================================================================

print("\nBuilding network...")
t0 = time.time()
state, config = build_network_state(n_stn=50, n_gpe=100, n_gpi=75, dt_ms=0.025)
print(f"✓ Network built in {(time.time()-t0)*1000:.1f} ms")

print("\nCreating JIT-compiled simulator...")
t0 = time.time()
simulate_base = create_simulation_fn(config, n_steps=4000)
print(f"✓ Simulator compiled in {(time.time()-t0)*1000:.1f} ms")

# Warm up
dummy_params = {
    'ISTN': 60.0, 'I_gpe': 200.0, 'I_gpi': 250.0,
    'noise_stn_sigma': 0.8, 'noise_gpe_sigma': 40.0, 'noise_gpi_sigma': 40.0
}
obs = simulate_base(dummy_params, state)
obs['V_stn'].block_until_ready()

# =============================================================================
# PARKINSONIAN TARGETS (Literature-validated)
# =============================================================================

TARGET_RATES = {
    'stn': 25.0,   # Hz - middle of 20-30 range
    'gpe': 45.0,   # Hz - middle of 30-60 range (reduced from healthy)
    'gpi': 75.0    # Hz - middle of 60-90 range
}

TARGET_CV = {
    'stn': 0.8,    # High irregularity (bursty, beta-locked)
    'gpe': 0.35,   # Moderate irregularity (rhythmic pauses)
    'gpi': 0.3     # Relatively preserved regularity
}

print("\n" + "=" * 80)
print("PARKINSONIAN TARGETS (from literature)")
print("=" * 80)
print(f"\nFiring Rates:")
print(f"  STN: {TARGET_RATES['stn']:.1f} Hz (range: 20-30 Hz)")
print(f"  GPe: {TARGET_RATES['gpe']:.1f} Hz (range: 30-60 Hz) ← REDUCED")
print(f"  GPi: {TARGET_RATES['gpi']:.1f} Hz (range: 60-90 Hz)")

print(f"\nCoefficient of Variation:")
print(f"  STN: {TARGET_CV['stn']:.2f} (range: 0.6-1.0) ← HIGHLY IRREGULAR")
print(f"  GPe: {TARGET_CV['gpe']:.2f} (range: 0.2-0.5)")
print(f"  GPi: {TARGET_CV['gpi']:.2f} (range: 0.2-0.4)")

print("\nCritical ordering constraints:")
print("  CV:   STN > GPe > GPi  (irregularity hierarchy)")
print("  Rate: GPi > GPe > STN  (firing rate hierarchy)")

# =============================================================================
# OBJECTIVE FUNCTION (Same parameters, different targets)
# =============================================================================

def objective(trial):
    """
    Optimize for Parkinsonian dynamics using SAME 10 parameters as healthy.
    
    Key hypothesis: Reduced GPe drive (dopamine depletion) should emerge
    as the primary difference from healthy state.
    """
    
    # -------------------------------------------------------------------------
    # SAME PARAMETER RANGES AS HEALTHY
    # -------------------------------------------------------------------------
    
    params = {
        'ISTN': trial.suggest_float('ISTN', 40.0, 100.0),
        'I_gpe': trial.suggest_float('I_gpe', 250.0, 500.0),  # Same range
        'I_gpi': trial.suggest_float('I_gpi', 150.0, 350.0),
    }
    
    synaptic_weights = {
        'g_stn_gpe': trial.suggest_float('g_stn_gpe', 1.0, 4.0),
        'g_gpe_stn': trial.suggest_float('g_gpe_stn', 3.0, 15.0),
        'g_stn_gpi': trial.suggest_float('g_stn_gpi', 1.0, 4.0),
        'g_gpe_gpi': trial.suggest_float('g_gpe_gpi', 1.5, 5.0),
    }
    
    params['noise_stn_sigma'] = trial.suggest_float('noise_stn_sigma', 0.2, 1.0)
    params['noise_gpe_sigma'] = trial.suggest_float('noise_gpe_sigma', 20.0, 80.0)
    params['noise_gpi_sigma'] = trial.suggest_float('noise_gpi_sigma', 20.0, 80.0)
    
    # -------------------------------------------------------------------------
    # BUILD NETWORK WITH TRIAL SYNAPTIC WEIGHTS
    # -------------------------------------------------------------------------
    
    import copy
    trial_config = copy.deepcopy(config)
    
    trial_config['synapses']['stn_to_gpe'] = trial_config['synapses']['stn_to_gpe']._replace(
        g_max=synaptic_weights['g_stn_gpe']
    )
    trial_config['synapses']['gpe_to_stn'] = trial_config['synapses']['gpe_to_stn']._replace(
        g_max=synaptic_weights['g_gpe_stn']
    )
    trial_config['synapses']['stn_to_gpi'] = trial_config['synapses']['stn_to_gpi']._replace(
        g_max=synaptic_weights['g_stn_gpi']
    )
    trial_config['synapses']['gpe_to_gpi'] = trial_config['synapses']['gpe_to_gpi']._replace(
        g_max=synaptic_weights['g_gpe_gpi']
    )
    
    simulate = create_simulation_fn(trial_config, n_steps=4000)
    
    # -------------------------------------------------------------------------
    # RUN SIMULATION
    # -------------------------------------------------------------------------
    
    try:
        obs = simulate(params, state)
        metrics = compute_all_metrics(obs, dt_ms=0.025, burn_steps=1000)
        
        # Check for NaN
        if any(v != v for v in metrics['firing_rates'].values()):
            return float('inf')
        
        # -----------------------------------------------------------------
        # PARKINSONIAN SCORING
        # -----------------------------------------------------------------
        
        # Firing rate error
        rate_error = sum([
            (metrics['firing_rates'][pop] - TARGET_RATES[pop]) ** 2
            for pop in ['stn', 'gpe', 'gpi']
        ])
        
        # CV error (CRITICAL for PD - must be irregular!)
        cv_error = sum([
            (metrics['cv'][pop] - TARGET_CV[pop]) ** 2
            for pop in ['stn', 'gpe', 'gpi']
        ])
        
        # Beta power bonus (WANT high beta in PD!)
        # Negative penalty = reward for beta
        beta_power_total = sum(metrics['beta_power'].values())
        beta_bonus = -0.001 * beta_power_total  # Encourage beta
        
        # Ordering constraints (enforce physiological hierarchy)
        cv_ordering_penalty = 0.0
        if not (metrics['cv']['stn'] > metrics['cv']['gpe'] > metrics['cv']['gpi']):
            cv_ordering_penalty = 100.0  # Strong penalty for violating CV hierarchy
        
        rate_ordering_penalty = 0.0
        if not (metrics['firing_rates']['gpi'] > metrics['firing_rates']['gpe'] > 
                metrics['firing_rates']['stn']):
            rate_ordering_penalty = 100.0  # Strong penalty for violating rate hierarchy
        
        # Weighted score
        score = (
            1.0 * rate_error +           # Match PD firing rates
            2.0 * cv_error +              # CRITICAL: match PD irregularity (higher weight!)
            beta_bonus +                  # Encourage beta oscillations
            cv_ordering_penalty +         # Enforce CV: STN > GPe > GPi
            rate_ordering_penalty         # Enforce Rate: GPi > GPe > STN
        )
        
        # -----------------------------------------------------------------
        # LOG METRICS
        # -----------------------------------------------------------------
        
        trial.set_user_attr('rate_stn', float(metrics['firing_rates']['stn']))
        trial.set_user_attr('rate_gpe', float(metrics['firing_rates']['gpe']))
        trial.set_user_attr('rate_gpi', float(metrics['firing_rates']['gpi']))
        trial.set_user_attr('cv_stn', float(metrics['cv']['stn']))
        trial.set_user_attr('cv_gpe', float(metrics['cv']['gpe']))
        trial.set_user_attr('cv_gpi', float(metrics['cv']['gpi']))
        trial.set_user_attr('beta_stn', float(metrics['beta_power']['stn']))
        trial.set_user_attr('beta_gpe', float(metrics['beta_power']['gpe']))
        trial.set_user_attr('beta_gpi', float(metrics['beta_power']['gpi']))
        
        return float(score)
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')


# =============================================================================
# RUN OPTIMIZATION
# =============================================================================

def run_parkinsonian_optimization(n_trials=1000, study_name="parkinsonian_full"):
    """
    Discover Parkinsonian parameter regime.
    """
    
    print("\n" + "=" * 80)
    print(f"STARTING PARKINSONIAN OPTIMIZATION: {n_trials} trials")
    print("=" * 80)
    print(f"\nEstimated time: ~{n_trials * 0.3 / 60:.1f} minutes")
    print(f"\nGoal: Discover what parameter changes produce PD dynamics")
    print(f"Hypothesis: Reduced I_gpe will emerge as key difference\n")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(
            seed=43,  # Different seed than healthy
            n_startup_trials=50,
            multivariate=True
        ),
        study_name=study_name
    )
    
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: print(
                    f"Trial {trial.number}: score={trial.value:.2f}, "
                    f"STN={trial.user_attrs.get('rate_stn', 0):.1f} Hz, "
                    f"CV_STN={trial.user_attrs.get('cv_stn', 0):.3f}"
                )
            ]
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Optimization interrupted by user")
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f"{study_name}_study.pkl", 'wb') as f:
        pickle.dump(study, f)
    
    with open(results_dir / f"{study_name}_best_params.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PARKINSONIAN STATE - BEST PARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        for param, value in study.best_params.items():
            f.write(f"{param:20s}: {value:.4f}\n")
        f.write(f"\nBest score: {study.best_value:.4f}\n")
    
    # ==========================================================================
    # DISPLAY RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("PARKINSONIAN OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Completed: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
    
    if len(study.trials) > 0:
        print("\n--- PARKINSONIAN PARAMETERS ---")
        for param, value in study.best_params.items():
            print(f"  {param:20s}: {value:.3f}")
        
        print("\n--- PARKINSONIAN METRICS ---")
        print(f"  Score: {study.best_value:.3f}")
        print(f"\n  Firing Rates:")
        print(f"    STN: {study.best_trial.user_attrs['rate_stn']:.1f} Hz (target: 25 Hz)")
        print(f"    GPe: {study.best_trial.user_attrs['rate_gpe']:.1f} Hz (target: 45 Hz)")
        print(f"    GPi: {study.best_trial.user_attrs['rate_gpi']:.1f} Hz (target: 75 Hz)")
        
        print(f"\n  Coefficient of Variation:")
        print(f"    STN: {study.best_trial.user_attrs['cv_stn']:.3f} (target: 0.80)")
        print(f"    GPe: {study.best_trial.user_attrs['cv_gpe']:.3f} (target: 0.35)")
        print(f"    GPi: {study.best_trial.user_attrs['cv_gpi']:.3f} (target: 0.30)")
        
        print(f"\n  Beta Power:")
        print(f"    STN: {study.best_trial.user_attrs['beta_stn']:.2e}")
        print(f"    GPe: {study.best_trial.user_attrs['beta_gpe']:.2e}")
        print(f"    GPi: {study.best_trial.user_attrs['beta_gpi']:.2e}")
        
        # Check ordering constraints
        cv_ok = (study.best_trial.user_attrs['cv_stn'] > 
                 study.best_trial.user_attrs['cv_gpe'] > 
                 study.best_trial.user_attrs['cv_gpi'])
        rate_ok = (study.best_trial.user_attrs['rate_gpi'] > 
                   study.best_trial.user_attrs['rate_gpe'] > 
                   study.best_trial.user_attrs['rate_stn'])
        
        print(f"\n  Ordering Constraints:")
        print(f"    CV hierarchy (STN>GPe>GPi): {'✓' if cv_ok else '✗'}")
        print(f"    Rate hierarchy (GPi>GPe>STN): {'✓' if rate_ok else '✗'}")
    
    print(f"\n✓ Results saved to: {results_dir}/")
    print("=" * 80)
    
    return study


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run 1000 trials to discover Parkinsonian regime
    study = run_parkinsonian_optimization(n_trials=1000, study_name="parkinsonian_full")
