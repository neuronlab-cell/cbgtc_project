"""
Parkinsonian state optimization v3

Key insight from diagnosis:
- Beta emerges when I_gpe < 150 (GPe nearly silent)
- But we need GPe to still fire (30-60 Hz)
- Solution: Lower I_gpe + stronger STN→GPe drive

This version:
1. Expands I_gpe range DOWN (100-350)
2. Expands g_stn_gpe UP (to drive GPe from STN)
3. Uses beta_fraction as constraint
4. Longer simulation (400ms)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.samplers import TPESampler
import time
import pickle
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import copy

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print("=" * 80)
print("PARKINSONIAN STATE OPTIMIZATION v3")
print("Key: Expanded I_gpe range to include beta-generating regime")
print("=" * 80)

# =============================================================================
# BETA FRACTION COMPUTATION
# =============================================================================

def compute_beta_fraction(V_trace, dt_ms, burn_steps):
    """Compute beta as fraction of broadband power."""
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
        return min(beta_power / total_power, 1.0)
    return 0.0


# =============================================================================
# NETWORK SETUP
# =============================================================================

print("\nBuilding network...")
state, config = build_network_state(n_stn=50, n_gpe=100, n_gpi=75, dt_ms=0.025)

# Longer simulation for better beta detection
N_STEPS = 16000   # 400ms total
BURN_STEPS = 4000  # 100ms burn-in, 300ms analysis

print(f"Simulation: {N_STEPS * 0.025:.0f}ms total, {(N_STEPS-BURN_STEPS)*0.025:.0f}ms analysis window")

simulate_base = create_simulation_fn(config, n_steps=N_STEPS)

# Warm up
dummy = {'ISTN': 70.0, 'I_gpe': 200.0, 'I_gpi': 250.0,
         'noise_stn_sigma': 0.5, 'noise_gpe_sigma': 40.0, 'noise_gpi_sigma': 40.0}
obs = simulate_base(dummy, state)
obs['V_stn'].block_until_ready()

# =============================================================================
# PARKINSONIAN TARGETS
# =============================================================================

TARGET_RATES = {'stn': 25.0, 'gpe': 45.0, 'gpi': 75.0}
TARGET_CV = {'stn': 0.8, 'gpe': 0.35, 'gpi': 0.30}
BETA_THRESHOLD = 0.10  # 10% beta fraction minimum

print(f"\nTargets: STN={TARGET_RATES['stn']}Hz, GPe={TARGET_RATES['gpe']}Hz, GPi={TARGET_RATES['gpi']}Hz")
print(f"CV: STN={TARGET_CV['stn']}, GPe={TARGET_CV['gpe']}, GPi={TARGET_CV['gpi']}")
print(f"Beta threshold: {BETA_THRESHOLD*100:.0f}%")

# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def objective(trial):
    """
    v3: Expanded search space to find beta-generating PD regime.
    """
    
    # EXPANDED RANGES based on diagnosis
    params = {
        'ISTN': trial.suggest_float('ISTN', 50.0, 120.0),      # Higher max for STN drive
        'I_gpe': trial.suggest_float('I_gpe', 100.0, 350.0),   # LOWER min (beta regime!)
        'I_gpi': trial.suggest_float('I_gpi', 150.0, 350.0),
    }
    
    synaptic_weights = {
        'g_stn_gpe': trial.suggest_float('g_stn_gpe', 2.0, 8.0),   # HIGHER (drive GPe from STN)
        'g_gpe_stn': trial.suggest_float('g_gpe_stn', 1.0, 10.0),  # LOWER min (allow oscillation)
        'g_stn_gpi': trial.suggest_float('g_stn_gpi', 1.0, 5.0),
        'g_gpe_gpi': trial.suggest_float('g_gpe_gpi', 1.0, 5.0),
    }
    
    params['noise_stn_sigma'] = trial.suggest_float('noise_stn_sigma', 0.2, 1.0)
    params['noise_gpe_sigma'] = trial.suggest_float('noise_gpe_sigma', 10.0, 60.0)
    params['noise_gpi_sigma'] = trial.suggest_float('noise_gpi_sigma', 10.0, 60.0)
    
    # Build config with trial synapses
    trial_config = copy.deepcopy(config)
    trial_config['synapses']['stn_to_gpe'] = trial_config['synapses']['stn_to_gpe']._replace(
        g_max=synaptic_weights['g_stn_gpe'])
    trial_config['synapses']['gpe_to_stn'] = trial_config['synapses']['gpe_to_stn']._replace(
        g_max=synaptic_weights['g_gpe_stn'])
    trial_config['synapses']['stn_to_gpi'] = trial_config['synapses']['stn_to_gpi']._replace(
        g_max=synaptic_weights['g_stn_gpi'])
    trial_config['synapses']['gpe_to_gpi'] = trial_config['synapses']['gpe_to_gpi']._replace(
        g_max=synaptic_weights['g_gpe_gpi'])
    
    simulate = create_simulation_fn(trial_config, n_steps=N_STEPS)
    
    try:
        obs = simulate(params, state)
        metrics = compute_all_metrics(obs, dt_ms=0.025, burn_steps=BURN_STEPS)
        
        if any(np.isnan(v) for v in metrics['firing_rates'].values()):
            return float('inf')
        
        # Beta fractions
        beta_stn = compute_beta_fraction(obs['V_stn'], 0.025, BURN_STEPS)
        beta_gpi = compute_beta_fraction(obs['V_gpi'], 0.025, BURN_STEPS)
        
        # Get metrics
        r_stn = metrics['firing_rates']['stn']
        r_gpe = metrics['firing_rates']['gpe']
        r_gpi = metrics['firing_rates']['gpi']
        cv_stn = metrics['cv']['stn']
        cv_gpe = metrics['cv']['gpe']
        cv_gpi = metrics['cv']['gpi']
        
        # -----------------------------------------------------------------
        # SCORING
        # -----------------------------------------------------------------
        
        # Rate error
        rate_error = ((r_stn - TARGET_RATES['stn'])**2 + 
                      (r_gpe - TARGET_RATES['gpe'])**2 + 
                      (r_gpi - TARGET_RATES['gpi'])**2)
        
        # CV error (weighted higher for PD)
        cv_error = ((cv_stn - TARGET_CV['stn'])**2 + 
                    (cv_gpe - TARGET_CV['gpe'])**2 + 
                    (cv_gpi - TARGET_CV['gpi'])**2)
        
        # Ordering penalties
        cv_order_penalty = 0.0 if (cv_stn > cv_gpe > cv_gpi) else 200.0
        rate_order_penalty = 0.0 if (r_gpi > r_gpe > r_stn) else 200.0
        
        # GPe minimum rate (must not go silent!)
        gpe_silent_penalty = 0.0
        if r_gpe < 20.0:  # GPe should be at least 20 Hz
            gpe_silent_penalty = 100.0 * (20.0 - r_gpe)
        
        # Beta constraint (penalize if too low)
        beta_penalty = 0.0
        if beta_stn < BETA_THRESHOLD:
            beta_penalty = 100.0 * (BETA_THRESHOLD - beta_stn)
        
        # Small beta bonus (bounded)
        beta_bonus = -2.0 * min(beta_stn, 0.5)  # Max -1.0 contribution
        
        # Total score
        score = (1.0 * rate_error + 
                 2.0 * cv_error + 
                 cv_order_penalty + 
                 rate_order_penalty +
                 gpe_silent_penalty +
                 beta_penalty +
                 beta_bonus)
        
        # Log
        trial.set_user_attr('rate_stn', float(r_stn))
        trial.set_user_attr('rate_gpe', float(r_gpe))
        trial.set_user_attr('rate_gpi', float(r_gpi))
        trial.set_user_attr('cv_stn', float(cv_stn))
        trial.set_user_attr('cv_gpe', float(cv_gpe))
        trial.set_user_attr('cv_gpi', float(cv_gpi))
        trial.set_user_attr('beta_stn', float(beta_stn))
        trial.set_user_attr('beta_gpi', float(beta_gpi))
        
        return float(score)
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')


# =============================================================================
# RUN
# =============================================================================

def run_optimization(n_trials=1000, study_name="parkinsonian_v3"):
    print("\n" + "=" * 80)
    print(f"STARTING OPTIMIZATION: {n_trials} trials")
    print("=" * 80)
    print("\nSearch space changes from v2:")
    print("  I_gpe: 100-350 (was 250-500) - includes beta regime!")
    print("  g_stn_gpe: 2-8 (was 1-4) - stronger STN→GPe drive")
    print("  g_gpe_stn: 1-10 (was 3-15) - allows weaker feedback")
    print("  + GPe silent penalty (must stay > 20 Hz)")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=45, n_startup_trials=50, multivariate=True),
        study_name=study_name
    )
    
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: print(
                    f"Trial {trial.number}: score={trial.value if trial.value else float('inf'):.1f}, "
                    f"STN={trial.user_attrs.get('rate_stn',0):.1f}Hz, "
                    f"GPe={trial.user_attrs.get('rate_gpe',0):.1f}Hz, "
                    f"beta={trial.user_attrs.get('beta_stn',0)*100:.1f}%"
                )
            ]
        )
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    
    # Save
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f"{study_name}_study.pkl", 'wb') as f:
        pickle.dump(study, f)
    
    # Display
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if study.best_trial:
        best = study.best_trial
        
        print("\n--- PARAMETERS ---")
        for p, v in study.best_params.items():
            print(f"  {p:20s}: {v:.3f}")
        
        print("\n--- METRICS ---")
        print(f"  STN: {best.user_attrs['rate_stn']:.1f} Hz, CV={best.user_attrs['cv_stn']:.3f}")
        print(f"  GPe: {best.user_attrs['rate_gpe']:.1f} Hz, CV={best.user_attrs['cv_gpe']:.3f}")
        print(f"  GPi: {best.user_attrs['rate_gpi']:.1f} Hz, CV={best.user_attrs['cv_gpi']:.3f}")
        print(f"\n  Beta STN: {best.user_attrs['beta_stn']*100:.1f}%")
        print(f"  Beta GPi: {best.user_attrs['beta_gpi']*100:.1f}%")
        
        cv_ok = best.user_attrs['cv_stn'] > best.user_attrs['cv_gpe'] > best.user_attrs['cv_gpi']
        rate_ok = best.user_attrs['rate_gpi'] > best.user_attrs['rate_gpe'] > best.user_attrs['rate_stn']
        
        print(f"\n  CV order (STN>GPe>GPi): {'✓' if cv_ok else '✗'}")
        print(f"  Rate order (GPi>GPe>STN): {'✓' if rate_ok else '✗'}")
        print(f"\n  Score: {study.best_value:.2f}")
    
    return study


if __name__ == "__main__":
    study = run_optimization(n_trials=1000, study_name="parkinsonian_v3")
