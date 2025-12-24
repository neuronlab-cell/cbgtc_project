"""
Parkinsonian State Optimization - 10 Parameters

Targets:
- STN: 25-30 Hz, CV=0.7-0.9, Beta=0.30-0.45
- GPe: 35-50 Hz, CV=0.3-0.5, Beta=0.15-0.30
- GPi: 75-90 Hz, CV=0.2-0.35, Beta=0.25-0.40

Parameters:
- 3 intrinsic currents
- 4 synaptic weight multipliers
- 3 noise sigmas
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
print("PARKINSONIAN STATE - 10 PARAMETER OPTIMIZATION")
print("=" * 60)

print("\nBuilding base network...")
state, base_config = build_network_state(400, 800, 600, 0.025)

# Store reference g_max values
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

simulator_cache = {}

def get_simulator(weight_multipliers):
    """Get or create simulator for given weight multipliers."""
    cache_key = tuple(round(v, 1) for v in weight_multipliers.values())
    
    if cache_key not in simulator_cache:
        config = copy.deepcopy(base_config)
        
        config['synapses']['stn_to_gpe'] = config['synapses']['stn_to_gpe']._replace(
            g_max=REF_WEIGHTS['g_stn_gpe'] * weight_multipliers['g_stn_gpe'])
        config['synapses']['gpe_to_stn'] = config['synapses']['gpe_to_stn']._replace(
            g_max=REF_WEIGHTS['g_gpe_stn'] * weight_multipliers['g_gpe_stn'])
        config['synapses']['stn_to_gpi'] = config['synapses']['stn_to_gpi']._replace(
            g_max=REF_WEIGHTS['g_stn_gpi'] * weight_multipliers['g_stn_gpi'])
        config['synapses']['gpe_to_gpi'] = config['synapses']['gpe_to_gpi']._replace(
            g_max=REF_WEIGHTS['g_gpe_gpi'] * weight_multipliers['g_gpe_gpi'])
        
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
# TARGETS (Parkinsonian)
# =============================================================================

RATE_TARGETS = {'stn': 27.5, 'gpe': 42.5, 'gpi': 82.5}
CV_TARGETS = {'stn': 0.8, 'gpe': 0.4, 'gpi': 0.275}
BETA_MIN = {'stn': 0.30, 'gpe': 0.15, 'gpi': 0.25}
BETA_MID = {'stn': 0.375, 'gpe': 0.225, 'gpi': 0.325}
BETA_MAX = {'stn': 0.45, 'gpe': 0.30, 'gpi': 0.40}

print("\nParkinsonian Targets:")
print(f"  STN: {RATE_TARGETS['stn']} Hz, CV={CV_TARGETS['stn']}, Beta={BETA_MIN['stn']}-{BETA_MAX['stn']}")
print(f"  GPe: {RATE_TARGETS['gpe']} Hz, CV={CV_TARGETS['gpe']}, Beta={BETA_MIN['gpe']}-{BETA_MAX['gpe']}")
print(f"  GPi: {RATE_TARGETS['gpi']} Hz, CV={CV_TARGETS['gpi']}, Beta={BETA_MIN['gpi']}-{BETA_MAX['gpi']}")

# =============================================================================
# HELPERS
# =============================================================================

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

def save_study(study, filename='results/parkinsonian_10param_study.pkl'):
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
        'I_gpe': trial.suggest_float('I_gpe', 150.0, 500.0),  # Lower range for PD
        'I_gpi': trial.suggest_float('I_gpi', 200.0, 600.0),
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
    
    # Get simulator
    simulator = get_simulator(weight_multipliers)
    
    # Run simulation
    sim_params = {**intrinsic_params, **noise_params}
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
    
    # Compute beta fractions
    beta_stn = compute_beta_fraction(obs['V_stn'])
    beta_gpe = compute_beta_fraction(obs['V_gpe'])
    beta_gpi = compute_beta_fraction(obs['V_gpi'])
    
    # =========================================================================
    # SCORING
    # =========================================================================
    
    # 1. Rate error (squared, midpoints)
    rate_error = (
        (r_stn - RATE_TARGETS['stn'])**2 +
        (r_gpe - RATE_TARGETS['gpe'])**2 +
        (r_gpi - RATE_TARGETS['gpi'])**2
    )
    
    # 2. CV error (squared, midpoints)
    cv_error = (
        (cv_stn - CV_TARGETS['stn'])**2 +
        (cv_gpe - CV_TARGETS['gpe'])**2 +
        (cv_gpi - CV_TARGETS['gpi'])**2
    )
    
    # 3. Beta: CONSTRAINT (penalize if too LOW)
    beta_penalty = 0.0
    if beta_stn < BETA_MIN['stn']:
        beta_penalty += 50.0 * (BETA_MIN['stn'] - beta_stn)
    if beta_gpe < BETA_MIN['gpe']:
        beta_penalty += 50.0 * (BETA_MIN['gpe'] - beta_gpe)
    if beta_gpi < BETA_MIN['gpi']:
        beta_penalty += 50.0 * (BETA_MIN['gpi'] - beta_gpi)
    
    # 4. Beta: Small bounded bonus for being in range
    beta_bonus = 0.0
    if beta_stn >= BETA_MIN['stn']:
        beta_bonus -= 5.0 * min(beta_stn, BETA_MAX['stn']) / BETA_MAX['stn']
    if beta_gpe >= BETA_MIN['gpe']:
        beta_bonus -= 3.0 * min(beta_gpe, BETA_MAX['gpe']) / BETA_MAX['gpe']
    if beta_gpi >= BETA_MIN['gpi']:
        beta_bonus -= 4.0 * min(beta_gpi, BETA_MAX['gpi']) / BETA_MAX['gpi']
    
    # 5. Ordering constraints
    rate_order_penalty = 0.0 if (r_gpi > r_gpe > r_stn) else 50.0
    cv_order_penalty = 0.0 if (cv_stn > cv_gpe > cv_gpi) else 50.0
    
    # 6. Final score
    score = (
        1.0 * rate_error +
        50.0 * cv_error +
        beta_penalty +
        beta_bonus +
        rate_order_penalty +
        cv_order_penalty
    )
    
    # Log
    trial.set_user_attr('r_stn', float(r_stn))
    trial.set_user_attr('r_gpe', float(r_gpe))
    trial.set_user_attr('r_gpi', float(r_gpi))
    trial.set_user_attr('cv_stn', float(cv_stn))
    trial.set_user_attr('cv_gpe', float(cv_gpe))
    trial.set_user_attr('cv_gpi', float(cv_gpi))
    trial.set_user_attr('beta_stn', float(beta_stn))
    trial.set_user_attr('beta_gpe', float(beta_gpe))
    trial.set_user_attr('beta_gpi', float(beta_gpi))
    
    return score

# =============================================================================
# CALLBACK
# =============================================================================

def callback(study, trial):
    if trial.value is not None and trial.value != float('inf'):
        print(f"Trial {trial.number}: score={trial.value:.1f}, "
              f"STN={trial.user_attrs.get('r_stn',0):.1f}Hz/CV{trial.user_attrs.get('cv_stn',0):.2f}/β{trial.user_attrs.get('beta_stn',0)*100:.0f}%, "
              f"GPe={trial.user_attrs.get('r_gpe',0):.1f}Hz, "
              f"GPi={trial.user_attrs.get('r_gpi',0):.1f}Hz")
    
    if (trial.number + 1) % 50 == 0:
        save_study(study)
        print(f"  [Auto-saved at trial {trial.number + 1}]")

# =============================================================================
# RUN
# =============================================================================

print("\n" + "=" * 60)
print("STARTING PARKINSONIAN 10-PARAM CMA-ES (1000 trials)")
print("=" * 60 + "\n")

study = optuna.create_study(
    direction='minimize',
    sampler=CmaEsSampler(seed=43)  # Different seed than healthy
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
print("PARKINSONIAN RESULTS")
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
    print(f"  STN: {best.user_attrs['r_stn']:.1f} Hz (target: 27.5), CV={best.user_attrs['cv_stn']:.3f} (target: 0.80), Beta={best.user_attrs['beta_stn']*100:.1f}% (target: 30-45%)")
    print(f"  GPe: {best.user_attrs['r_gpe']:.1f} Hz (target: 42.5), CV={best.user_attrs['cv_gpe']:.3f} (target: 0.40), Beta={best.user_attrs['beta_gpe']*100:.1f}% (target: 15-30%)")
    print(f"  GPi: {best.user_attrs['r_gpi']:.1f} Hz (target: 82.5), CV={best.user_attrs['cv_gpi']:.3f} (target: 0.28), Beta={best.user_attrs['beta_gpi']*100:.1f}% (target: 25-40%)")
    print(f"\n  Score: {study.best_value:.2f}")
    
    # Check orderings
    rate_ok = best.user_attrs['r_gpi'] > best.user_attrs['r_gpe'] > best.user_attrs['r_stn']
    cv_ok = best.user_attrs['cv_stn'] > best.user_attrs['cv_gpe'] > best.user_attrs['cv_gpi']
    print(f"\n  Rate ordering (GPi>GPe>STN): {'✓' if rate_ok else '✗'}")
    print(f"  CV ordering (STN>GPe>GPi): {'✓' if cv_ok else '✗'}")

print("\n✓ Saved to results/parkinsonian_10param_study.pkl")
print("=" * 60)
