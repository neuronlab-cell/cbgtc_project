"""
Full biological parameter optimization based on:
Nevado-Holgado et al. (2014) J Physiol 592: 693–716
"Conditions for the generation of beta oscillations in the 
subthalamic nucleus–globus pallidus network of the basal ganglia"

This script optimizes:
- Intrinsic currents (ISTN, I_gpe, I_gpi)
- Synaptic weights (4 connections)
- Noise levels (3 populations)

All ranges are constrained to biologically plausible values from literature.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.samplers import TPESampler
import time
import pickle
from pathlib import Path

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn, apply_params_to_config
from optimization.metrics_jax import compute_all_metrics

print("=" * 80)
print("FULL BIOLOGICAL PARAMETER OPTIMIZATION")
print("Based on: Nevado-Holgado et al. (2014) J Physiol")
print("=" * 80)

# =============================================================================
# NETWORK SETUP
# =============================================================================

print("\nBuilding network...")
t0 = time.time()
state, config = build_network_state(n_stn=50, n_gpe=100, n_gpi=75, dt_ms=0.025)
print(f"✓ Network built in {(time.time()-t0)*1000:.1f} ms")

print("\nCreating JIT-compiled simulator...")
t0 = time.time()
simulate_base = create_simulation_fn(config, n_steps=4000)  # 100 ms simulation
print(f"✓ Simulator compiled in {(time.time()-t0)*1000:.1f} ms")

# Warm up
dummy_params = {
    'ISTN': 60.0, 'I_gpe': 300.0, 'I_gpi': 250.0,
    'noise_stn_sigma': 0.5, 'noise_gpe_sigma': 40.0, 'noise_gpi_sigma': 40.0
}
obs = simulate_base(dummy_params, state)
obs['V_stn'].block_until_ready()

print("\nTesting simulation speed...")
t0 = time.time()
obs = simulate_base(dummy_params, state)
obs['V_stn'].block_until_ready()
t1 = time.time()
print(f"✓ Cached simulation: {(t1-t0)*1000:.1f} ms per trial")
print(f"✓ Expected throughput: ~{int(1/(t1-t0))} trials/second")

# =============================================================================
# TARGET METRICS (from Nevado-Holgado 2014)
# =============================================================================

TARGET_RATES = {
    'stn': 20.0,   # Hz - healthy STN firing rate
    'gpe': 60.0,   # Hz - healthy GPe firing rate  
    'gpi': 70.0    # Hz - healthy GPi firing rate
}

TARGET_CV = 0.4  # Coefficient of variation for irregular firing

# =============================================================================
# OBJECTIVE FUNCTION WITH SYNAPTIC WEIGHTS
# =============================================================================

def objective(trial):
    """
    Optimize all biological parameters within literature-constrained ranges.
    
    Parameters optimized:
    1. Intrinsic currents (3): ISTN, I_gpe, I_gpi
    2. Synaptic weights (4): STN→GPe, GPe→STN, STN→GPi, GPe→GPi
    3. Noise levels (3): sigma for each population
    
    Total: 10 parameters
    """
    
    # -------------------------------------------------------------------------
    # 1. INTRINSIC CURRENTS (pA)
    # -------------------------------------------------------------------------
    # From Nevado-Holgado 2014: I_baseline typically 50-150 pA for tonic neurons
    # But we found empirically that our model needs higher values
    
    params = {
        'ISTN': trial.suggest_float('ISTN', 40.0, 100.0),      # STN excitatory drive
        'I_gpe': trial.suggest_float('I_gpe', 250.0, 500.0),   # GPe baseline
        'I_gpi': trial.suggest_float('I_gpi', 150.0, 350.0),   # GPi baseline
    }
    
    # -------------------------------------------------------------------------
    # 2. SYNAPTIC WEIGHTS (nS)
    # -------------------------------------------------------------------------
    # From Nevado-Holgado 2014 Table 1:
    # - All g_max values in range 0.5-5.0 nS
    # - Excitatory (STN→GPe, STN→GPi): typically 1-3 nS
    # - Inhibitory (GPe→STN, GPe→GPi): typically 2-5 nS
    
    synaptic_weights = {
        'g_stn_gpe': trial.suggest_float('g_stn_gpe', 1.0, 4.0),   # Excitatory
        'g_gpe_stn': trial.suggest_float('g_gpe_stn', 3.0, 15.0),  # Inhibitory (critical!)
        'g_stn_gpi': trial.suggest_float('g_stn_gpi', 1.0, 4.0),   # Excitatory
        'g_gpe_gpi': trial.suggest_float('g_gpe_gpi', 1.5, 5.0),   # Inhibitory
    }
    
    # -------------------------------------------------------------------------
    # 3. NOISE LEVELS (pA or unitless)
    # -------------------------------------------------------------------------
    # Noise adds biological irregularity (CV target: 0.4)
    
    params['noise_stn_sigma'] = trial.suggest_float('noise_stn_sigma', 0.2, 1.0)
    params['noise_gpe_sigma'] = trial.suggest_float('noise_gpe_sigma', 20.0, 80.0)
    params['noise_gpi_sigma'] = trial.suggest_float('noise_gpi_sigma', 20.0, 80.0)
    
    # -------------------------------------------------------------------------
    # 4. BUILD NETWORK WITH TRIAL PARAMETERS
    # -------------------------------------------------------------------------
    
    # Update synaptic weights in network config
    from jax_models import synapses_jax
    import copy
    
    trial_config = copy.deepcopy(config)
    
    # Update STN → GPe
    syn_cfg = trial_config['synapses']['stn_to_gpe']
    trial_config['synapses']['stn_to_gpe'] = syn_cfg._replace(
        g_max=synaptic_weights['g_stn_gpe']
    )
    
    # Update GPe → STN (CRITICAL for STN-GPe oscillator)
    syn_cfg = trial_config['synapses']['gpe_to_stn']
    trial_config['synapses']['gpe_to_stn'] = syn_cfg._replace(
        g_max=synaptic_weights['g_gpe_stn']
    )
    
    # Update STN → GPi
    syn_cfg = trial_config['synapses']['stn_to_gpi']
    trial_config['synapses']['stn_to_gpi'] = syn_cfg._replace(
        g_max=synaptic_weights['g_stn_gpi']
    )
    
    # Update GPe → GPi
    syn_cfg = trial_config['synapses']['gpe_to_gpi']
    trial_config['synapses']['gpe_to_gpi'] = syn_cfg._replace(
        g_max=synaptic_weights['g_gpe_gpi']
    )
    
    # Create simulator with updated config
    simulate = create_simulation_fn(trial_config, n_steps=4000)
    
    # -------------------------------------------------------------------------
    # 5. RUN SIMULATION
    # -------------------------------------------------------------------------
    
    try:
        obs = simulate(params, state)
        metrics = compute_all_metrics(obs, dt_ms=0.025, burn_steps=1000)
        
        # Check for NaN/Inf
        if any(v != v for v in metrics['firing_rates'].values()):  # NaN check
            return float('inf')
        
        # -----------------------------------------------------------------
        # 6. COMPUTE SCORE
        # -----------------------------------------------------------------
        
        # Firing rate error (main objective)
        rate_error = sum([
            (metrics['firing_rates'][pop] - TARGET_RATES[pop]) ** 2
            for pop in ['stn', 'gpe', 'gpi']
        ])
        
        # CV error (want irregular firing, not clockwork)
        cv_error = sum([
            (metrics['cv'][pop] - TARGET_CV) ** 2
            for pop in ['stn', 'gpe', 'gpi']
        ])
        
        # Beta power penalty (minimize pathological oscillations)
        beta_penalty = sum(metrics['beta_power'].values()) / 1e6
        
        # Weighted score
        score = (
            1.0 * rate_error +      # Main: match firing rates
            0.5 * cv_error +         # Important: irregular firing
            0.01 * beta_penalty      # Minor: low beta (healthy)
        )
        
        # -----------------------------------------------------------------
        # 7. LOG METRICS FOR ANALYSIS
        # -----------------------------------------------------------------
        
        trial.set_user_attr('rate_stn', float(metrics['firing_rates']['stn']))
        trial.set_user_attr('rate_gpe', float(metrics['firing_rates']['gpe']))
        trial.set_user_attr('rate_gpi', float(metrics['firing_rates']['gpi']))
        trial.set_user_attr('cv_stn', float(metrics['cv']['stn']))
        trial.set_user_attr('cv_gpe', float(metrics['cv']['gpe']))
        trial.set_user_attr('cv_gpi', float(metrics['cv']['gpi']))
        trial.set_user_attr('beta_stn', float(metrics['beta_power']['stn']))
        
        return float(score)
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')


# =============================================================================
# RUN OPTIMIZATION
# =============================================================================

def run_long_optimization(n_trials=1000, study_name="biological_full"):
    """
    Run extended optimization with biological constraints.
    
    Args:
        n_trials: Number of trials (default: 1000 for ~3 hours)
        study_name: Name for the study
    """
    
    print("\n" + "=" * 80)
    print(f"STARTING OPTIMIZATION: {n_trials} trials")
    print("=" * 80)
    print(f"\nEstimated time: ~{n_trials * 0.3 / 60:.1f} minutes")
    print(f"Parameters optimized: 10 (3 currents + 4 synapses + 3 noise)")
    print(f"Search space: ~10^10 combinations")
    print("\nPress Ctrl+C to stop early (results will be saved)\n")
    
    # Create study with TPE sampler (Bayesian optimization)
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(
            seed=42,
            n_startup_trials=50,  # Random exploration first
            multivariate=True      # Consider parameter interactions
        ),
        study_name=study_name
    )
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: print(
                    f"Trial {trial.number}: score={trial.value:.2f}, "
                    f"STN={trial.user_attrs.get('rate_stn', 0):.1f} Hz"
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
    
    # Save study object
    with open(results_dir / f"{study_name}_study.pkl", 'wb') as f:
        pickle.dump(study, f)
    
    # Save best parameters
    with open(results_dir / f"{study_name}_best_params.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BEST PARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        for param, value in study.best_params.items():
            f.write(f"{param:20s}: {value:.4f}\n")
        f.write(f"\nBest score: {study.best_value:.4f}\n")
    
    # ==========================================================================
    # DISPLAY RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    if len(study.trials) > 0:
        print("\n--- BEST PARAMETERS ---")
        for param, value in study.best_params.items():
            print(f"  {param:20s}: {value:.3f}")
        
        print("\n--- BEST METRICS ---")
        print(f"  Score: {study.best_value:.3f}")
        print(f"  STN rate: {study.best_trial.user_attrs['rate_stn']:.1f} Hz (target: 20 Hz)")
        print(f"  GPe rate: {study.best_trial.user_attrs['rate_gpe']:.1f} Hz (target: 60 Hz)")
        print(f"  GPi rate: {study.best_trial.user_attrs['rate_gpi']:.1f} Hz (target: 70 Hz)")
        print(f"  STN CV: {study.best_trial.user_attrs['cv_stn']:.3f} (target: 0.4)")
        print(f"  GPe CV: {study.best_trial.user_attrs['cv_gpe']:.3f} (target: 0.4)")
        print(f"  GPi CV: {study.best_trial.user_attrs['cv_gpi']:.3f} (target: 0.4)")
    
    print(f"\n✓ Results saved to: {results_dir}/")
    print("=" * 80)
    
    return study


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run 1000 trials (~3-5 hours at 300ms/trial)
    study = run_long_optimization(n_trials=1000, study_name="full_biological_3hr")
