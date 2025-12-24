"""
Parkinsonian state optimization v2 - FIXED

Key fixes:
1. Beta is a CONSTRAINT, not an unbounded reward
2. Beta computed as FRACTION (0-1), not raw power
3. Longer simulation (8000 steps = 200ms after burn-in)
4. Beta measured at STN (canonical biomarker) + GPi (output)
5. Rate/CV matching is the PRIMARY objective

Reference: Rubin & Terman, Nevado-Holgado, Bergman, Wichmann, Brown
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

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print("=" * 80)
print("PARKINSONIAN STATE OPTIMIZATION v2")
print("Fixed: Beta as constraint, normalized metric, longer simulation")
print("=" * 80)

# =============================================================================
# IMPROVED BETA COMPUTATION
# =============================================================================

def compute_beta_fraction(V_trace: jnp.ndarray, dt_ms: float, burn_steps: int) -> dict:
    """
    Compute beta power as a FRACTION of total power (bounded 0-1).
    
    This prevents beta from dominating the objective function.
    
    Args:
        V_trace: Voltage traces (n_steps, n_neurons)
        dt_ms: Timestep in ms
        burn_steps: Steps to discard
        
    Returns:
        dict with 'beta_power', 'total_power', 'beta_fraction'
    """
    valid_V = V_trace[burn_steps:]
    
    # LFP proxy: mean across neurons
    lfp = jnp.mean(valid_V, axis=1)
    
    # CRITICAL: Mean-center to remove DC offset
    lfp = lfp - jnp.mean(lfp)
    
    # FFT
    fft_vals = jnp.fft.rfft(lfp)
    freqs = jnp.fft.rfftfreq(len(lfp), d=dt_ms / 1000.0)
    psd = jnp.abs(fft_vals) ** 2
    
    # Beta band (13-30 Hz)
    beta_idx = (freqs >= 13) & (freqs <= 30)
    beta_power = float(jnp.sum(psd[beta_idx]))
    
    # Broadband (1-100 Hz) for normalization
    broad_idx = (freqs >= 1) & (freqs <= 100)
    total_power = float(jnp.sum(psd[broad_idx]))
    
    # Beta fraction (bounded 0-1)
    if total_power > 1e-10:
        beta_fraction = beta_power / total_power
    else:
        beta_fraction = 0.0
    
    return {
        'beta_power': beta_power,
        'total_power': total_power,
        'beta_fraction': min(beta_fraction, 1.0)  # Clamp to 0-1
    }


def compute_all_beta_fractions(V_dict: dict, dt_ms: float, burn_steps: int) -> dict:
    """Compute beta fractions for all populations."""
    results = {}
    for pop, V in V_dict.items():
        results[pop] = compute_beta_fraction(V, dt_ms, burn_steps)
    return results


# =============================================================================
# NETWORK SETUP
# =============================================================================

print("\nBuilding network...")
t0 = time.time()
state, config = build_network_state(n_stn=50, n_gpe=100, n_gpi=75, dt_ms=0.025)
print(f"✓ Network built in {(time.time()-t0)*1000:.1f} ms")

# LONGER SIMULATION: 8000 steps = 200ms total, 175ms after burn-in
# This gives ~5-13 beta cycles (much better for FFT)
N_STEPS = 8000
BURN_STEPS = 1000

print(f"\nSimulation: {N_STEPS} steps ({N_STEPS * 0.025:.1f} ms)")
print(f"Burn-in: {BURN_STEPS} steps ({BURN_STEPS * 0.025:.1f} ms)")
print(f"Analysis window: {(N_STEPS - BURN_STEPS) * 0.025:.1f} ms (~5-13 beta cycles)")

print("\nCreating JIT-compiled simulator...")
t0 = time.time()
simulate_base = create_simulation_fn(config, n_steps=N_STEPS)
print(f"✓ Simulator compiled in {(time.time()-t0)*1000:.1f} ms")

# Warm up
dummy_params = {
    'ISTN': 60.0, 'I_gpe': 200.0, 'I_gpi': 250.0,
    'noise_stn_sigma': 0.8, 'noise_gpe_sigma': 40.0, 'noise_gpi_sigma': 40.0
}
obs = simulate_base(dummy_params, state)
obs['V_stn'].block_until_ready()

# =============================================================================
# PARKINSONIAN TARGETS
# =============================================================================

TARGET_RATES = {
    'stn': 25.0,   # Hz (20-30 range)
    'gpe': 45.0,   # Hz (30-60 range, reduced from healthy)
    'gpi': 75.0    # Hz (60-90 range)
}

TARGET_CV = {
    'stn': 0.8,    # High irregularity (0.6-1.0)
    'gpe': 0.35,   # Moderate (0.2-0.5)
    'gpi': 0.3     # Relatively regular (0.2-0.4)
}

# Beta threshold for PD (beta_fraction should exceed this)
BETA_THRESHOLD_STN = 0.15   # 15% of power in beta band
BETA_THRESHOLD_GPi = 0.10   # 10% of power in beta band

print("\n" + "=" * 80)
print("PARKINSONIAN TARGETS")
print("=" * 80)
print(f"\nFiring Rates: STN={TARGET_RATES['stn']}, GPe={TARGET_RATES['gpe']}, GPi={TARGET_RATES['gpi']} Hz")
print(f"CV Targets: STN={TARGET_CV['stn']}, GPe={TARGET_CV['gpe']}, GPi={TARGET_CV['gpi']}")
print(f"Beta thresholds: STN>{BETA_THRESHOLD_STN*100:.0f}%, GPi>{BETA_THRESHOLD_GPi*100:.0f}%")

# =============================================================================
# OBJECTIVE FUNCTION v2
# =============================================================================

def objective(trial):
    """
    Parkinsonian objective with beta as CONSTRAINT.
    
    Primary goal: Match rates + CV + ordering
    Secondary: Beta must exceed threshold (constraint, not reward)
    """
    
    # Sample parameters (SAME as healthy)
    params = {
        'ISTN': trial.suggest_float('ISTN', 40.0, 100.0),
        'I_gpe': trial.suggest_float('I_gpe', 250.0, 500.0),
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
    
    # Build trial config
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
    
    simulate = create_simulation_fn(trial_config, n_steps=N_STEPS)
    
    try:
        obs = simulate(params, state)
        metrics = compute_all_metrics(obs, dt_ms=0.025, burn_steps=BURN_STEPS)
        
        # Check for NaN in firing rates
        if any(np.isnan(v) for v in metrics['firing_rates'].values()):
            return float('inf')
        
        # -----------------------------------------------------------------
        # COMPUTE BETA FRACTIONS (bounded 0-1)
        # -----------------------------------------------------------------
        
        V_dict = {
            'stn': obs['V_stn'],
            'gpe': obs['V_gpe'],
            'gpi': obs['V_gpi']
        }
        beta_results = compute_all_beta_fractions(V_dict, dt_ms=0.025, burn_steps=BURN_STEPS)
        
        beta_frac_stn = beta_results['stn']['beta_fraction']
        beta_frac_gpe = beta_results['gpe']['beta_fraction']
        beta_frac_gpi = beta_results['gpi']['beta_fraction']
        
        # -----------------------------------------------------------------
        # PRIMARY OBJECTIVE: Rate + CV matching
        # -----------------------------------------------------------------
        
        rate_error = sum([
            (metrics['firing_rates'][pop] - TARGET_RATES[pop]) ** 2
            for pop in ['stn', 'gpe', 'gpi']
        ])
        
        cv_error = sum([
            (metrics['cv'][pop] - TARGET_CV[pop]) ** 2
            for pop in ['stn', 'gpe', 'gpi']
        ])
        
        # -----------------------------------------------------------------
        # ORDERING CONSTRAINTS (hard penalties)
        # -----------------------------------------------------------------
        
        cv_ordering_penalty = 0.0
        cv_stn = metrics['cv']['stn']
        cv_gpe = metrics['cv']['gpe']
        cv_gpi = metrics['cv']['gpi']
        
        if not (cv_stn > cv_gpe > cv_gpi):
            cv_ordering_penalty = 200.0
        
        rate_ordering_penalty = 0.0
        r_stn = metrics['firing_rates']['stn']
        r_gpe = metrics['firing_rates']['gpe']
        r_gpi = metrics['firing_rates']['gpi']
        
        if not (r_gpi > r_gpe > r_stn):
            rate_ordering_penalty = 200.0
        
        # -----------------------------------------------------------------
        # BETA CONSTRAINT (not reward!)
        # -----------------------------------------------------------------
        # Penalize if beta is TOO LOW (we want elevated beta in PD)
        # But don't reward infinitely high beta
        
        beta_penalty = 0.0
        
        # STN beta should exceed threshold
        if beta_frac_stn < BETA_THRESHOLD_STN:
            beta_penalty += 50.0 * (BETA_THRESHOLD_STN - beta_frac_stn)
        
        # GPi beta should exceed threshold  
        if beta_frac_gpi < BETA_THRESHOLD_GPi:
            beta_penalty += 50.0 * (BETA_THRESHOLD_GPi - beta_frac_gpi)
        
        # Small tiebreaker: prefer higher beta (but bounded!)
        # This only matters when constraints are satisfied
        beta_tiebreaker = -0.5 * (beta_frac_stn + beta_frac_gpi)  # Max contribution: -1.0
        
        # -----------------------------------------------------------------
        # FINAL SCORE
        # -----------------------------------------------------------------
        
        score = (
            1.0 * rate_error +          # Match PD firing rates
            2.0 * cv_error +             # Match PD irregularity (critical!)
            cv_ordering_penalty +        # Enforce CV: STN > GPe > GPi
            rate_ordering_penalty +      # Enforce Rate: GPi > GPe > STN
            beta_penalty +               # Penalize if beta too low
            beta_tiebreaker              # Small preference for higher beta
        )
        
        # -----------------------------------------------------------------
        # LOG METRICS
        # -----------------------------------------------------------------
        
        trial.set_user_attr('rate_stn', float(r_stn))
        trial.set_user_attr('rate_gpe', float(r_gpe))
        trial.set_user_attr('rate_gpi', float(r_gpi))
        trial.set_user_attr('cv_stn', float(cv_stn))
        trial.set_user_attr('cv_gpe', float(cv_gpe))
        trial.set_user_attr('cv_gpi', float(cv_gpi))
        trial.set_user_attr('beta_frac_stn', float(beta_frac_stn))
        trial.set_user_attr('beta_frac_gpe', float(beta_frac_gpe))
        trial.set_user_attr('beta_frac_gpi', float(beta_frac_gpi))
        
        return float(score)
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')


# =============================================================================
# RUN OPTIMIZATION
# =============================================================================

def run_parkinsonian_optimization_v2(n_trials=1000, study_name="parkinsonian_v2"):
    """Run fixed Parkinsonian optimization."""
    
    print("\n" + "=" * 80)
    print(f"STARTING PARKINSONIAN OPTIMIZATION v2: {n_trials} trials")
    print("=" * 80)
    print(f"\nKey fixes:")
    print(f"  1. Beta as CONSTRAINT (penalty if too low)")
    print(f"  2. Beta_fraction (0-1) instead of raw power")
    print(f"  3. Longer simulation ({N_STEPS} steps)")
    print(f"  4. Small tiebreaker for beta (max -1.0, can't dominate)")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=44, n_startup_trials=50, multivariate=True),
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
                    f"STN={trial.user_attrs.get('rate_stn', 0):.1f}Hz, "
                    f"CV_STN={trial.user_attrs.get('cv_stn', 0):.3f}, "
                    f"beta_STN={trial.user_attrs.get('beta_frac_stn', 0)*100:.1f}%"
                )
            ]
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Optimization interrupted")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f"{study_name}_study.pkl", 'wb') as f:
        pickle.dump(study, f)
    
    # Display results
    print("\n" + "=" * 80)
    print("PARKINSONIAN OPTIMIZATION v2 COMPLETE")
    print("=" * 80)
    
    if study.best_trial:
        best = study.best_trial
        
        print("\n--- BEST PARAMETERS ---")
        for param, value in study.best_params.items():
            print(f"  {param:20s}: {value:.3f}")
        
        print("\n--- FIRING RATES ---")
        print(f"  STN: {best.user_attrs['rate_stn']:.1f} Hz (target: 25)")
        print(f"  GPe: {best.user_attrs['rate_gpe']:.1f} Hz (target: 45)")
        print(f"  GPi: {best.user_attrs['rate_gpi']:.1f} Hz (target: 75)")
        
        print("\n--- COEFFICIENT OF VARIATION ---")
        print(f"  STN: {best.user_attrs['cv_stn']:.3f} (target: 0.80)")
        print(f"  GPe: {best.user_attrs['cv_gpe']:.3f} (target: 0.35)")
        print(f"  GPi: {best.user_attrs['cv_gpi']:.3f} (target: 0.30)")
        
        print("\n--- BETA FRACTION (% of power) ---")
        print(f"  STN: {best.user_attrs['beta_frac_stn']*100:.1f}% (threshold: {BETA_THRESHOLD_STN*100:.0f}%)")
        print(f"  GPe: {best.user_attrs['beta_frac_gpe']*100:.1f}%")
        print(f"  GPi: {best.user_attrs['beta_frac_gpi']*100:.1f}% (threshold: {BETA_THRESHOLD_GPi*100:.0f}%)")
        
        # Check orderings
        cv_ok = (best.user_attrs['cv_stn'] > best.user_attrs['cv_gpe'] > best.user_attrs['cv_gpi'])
        rate_ok = (best.user_attrs['rate_gpi'] > best.user_attrs['rate_gpe'] > best.user_attrs['rate_stn'])
        
        print("\n--- ORDERING CONSTRAINTS ---")
        print(f"  CV hierarchy (STN>GPe>GPi): {'✓' if cv_ok else '✗'}")
        print(f"  Rate hierarchy (GPi>GPe>STN): {'✓' if rate_ok else '✗'}")
        
        print(f"\n  Final Score: {study.best_value:.3f}")
    
    print(f"\n✓ Results saved to: {results_dir}/")
    print("=" * 80)
    
    return study


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    study = run_parkinsonian_optimization_v2(n_trials=1000, study_name="parkinsonian_v2")
