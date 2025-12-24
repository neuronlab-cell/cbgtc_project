"""
Optuna optimization for HH GPe/GPi network - Parkinsonian state.
10 parameters: 6 intrinsic + 4 synaptic multipliers
Uses BETA FRACTION (not raw power)
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
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# NETWORK SETUP
# =============================================================================

N_STN, N_GPE, N_GPI = 100, 200, 150
DT_MS = 0.025
N_STEPS = 16000  # 400ms
BURN_STEPS = 4000

print(f"\nBuilding {N_STN + N_GPE + N_GPI}-neuron HH network...")
base_state, base_config = build_network_state(N_STN, N_GPE, N_GPI, DT_MS, use_hh=True)

# =============================================================================
# CUSTOM SIMULATOR WITH SYNAPTIC SCALING
# =============================================================================

def create_pd_simulator(base_config, n_steps):
    from optimization.sim_jax import apply_params_to_config
    from jax_models.integrator import network_step
    from jax import lax
    
    @jax.jit
    def simulate(trial_params, init_state):
        config = apply_params_to_config(trial_params, base_config)
        syn_configs = dict(config['synapses'])
        
        # Scale synaptic weights
        for syn_name, mult_name in [
            ('stn_to_gpe', 'g_stn_gpe_mult'),
            ('gpe_to_stn', 'g_gpe_stn_mult'),
            ('stn_to_gpi', 'g_stn_gpi_mult'),
            ('gpe_to_gpi', 'g_gpe_gpi_mult'),
        ]:
            old_cfg = syn_configs[syn_name]
            new_weights = old_cfg.weights * trial_params.get(mult_name, 1.0)
            syn_configs[syn_name] = old_cfg._replace(weights=new_weights)
        
        config['synapses'] = syn_configs
        
        def step_fn(carry_state, t_idx):
            t_ms = t_idx * config['dt_ms']
            new_state, obs = network_step(carry_state, config, t_ms)
            return new_state, obs
        
        final_state, obs_history = lax.scan(step_fn, init=init_state, xs=jnp.arange(n_steps))
        return obs_history
    
    return simulate

simulator = create_pd_simulator(base_config, N_STEPS)

# Warm-up
print("Warming up JIT...")
dummy_params = {
    'ISTN': 100.0, 'I_gpe': 3.0, 'I_gpi': 3.0,
    'noise_stn_sigma': 1.0, 'noise_gpe_sigma': 30.0, 'noise_gpi_sigma': 30.0,
    'g_stn_gpe_mult': 1.0, 'g_gpe_stn_mult': 1.0,
    'g_stn_gpi_mult': 1.0, 'g_gpe_gpi_mult': 1.0,
}
obs = simulator(dummy_params, base_state)
obs['V_stn'].block_until_ready()
print("JIT ready!\n")

# =============================================================================
# TARGETS - PD state with realistic beta
# =============================================================================

TARGETS = {
    'rate_stn': 27.5,
    'rate_gpe': 42.5,
    'rate_gpi': 82.5,
    'cv_stn': 0.60,
    'cv_gpe': 0.35,
    'cv_gpi': 0.25,
    'beta_gpe': 0.20,  # 20% beta fraction (realistic target)
}

WEIGHTS = {
    'rate': 1.0,
    'cv': 0.2,
    'beta': 15.0,  # VERY high weight on beta
}

# =============================================================================
# OBJECTIVE
# =============================================================================

def objective(trial):
    params = {
        'ISTN': trial.suggest_float('ISTN', 60.0, 150.0),
        'I_gpe': trial.suggest_float('I_gpe', 0.5, 4.0),
        'I_gpi': trial.suggest_float('I_gpi', 1.0, 5.0),
        'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 1.0, 8.0),
        'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 20.0, 150.0),
        'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 20.0, 150.0),
        # Synaptic multipliers - key for beta!
        'g_stn_gpe_mult': trial.suggest_float('g_stn_gpe_mult', 1.5, 5.0),  # Increased
        'g_gpe_stn_mult': trial.suggest_float('g_gpe_stn_mult', 0.1, 0.8),  # REDUCED - critical!
        'g_stn_gpi_mult': trial.suggest_float('g_stn_gpi_mult', 1.0, 4.0),
        'g_gpe_gpi_mult': trial.suggest_float('g_gpe_gpi_mult', 0.2, 1.2),
    }
    
    try:
        obs = simulator(params, base_state)
        obs['V_stn'].block_until_ready()
        
        if jnp.any(jnp.isnan(obs['V_stn'])) or jnp.any(jnp.isnan(obs['V_gpe'])):
            return 1e6
        
        # Compute metrics
        metrics = compute_all_metrics(obs, DT_MS, burn_steps=BURN_STEPS)
        beta_fractions = compute_beta_fraction_all(obs, DT_MS, burn_steps=BURN_STEPS)
        
        r_stn = metrics['firing_rates']['stn']
        r_gpe = metrics['firing_rates']['gpe']
        r_gpi = metrics['firing_rates']['gpi']
        cv_stn = metrics['cv']['stn']
        cv_gpe = metrics['cv']['gpe']
        cv_gpi = metrics['cv']['gpi']
        beta_gpe = beta_fractions['gpe']
        beta_stn = beta_fractions['stn']
        
        # Loss
        loss = 0.0
        
        # Rates
        loss += WEIGHTS['rate'] * ((r_stn - TARGETS['rate_stn']) / TARGETS['rate_stn']) ** 2
        loss += WEIGHTS['rate'] * ((r_gpe - TARGETS['rate_gpe']) / TARGETS['rate_gpe']) ** 2
        loss += WEIGHTS['rate'] * ((r_gpi - TARGETS['rate_gpi']) / TARGETS['rate_gpi']) ** 2
        
        # CV
        loss += WEIGHTS['cv'] * ((cv_stn - TARGETS['cv_stn']) / TARGETS['cv_stn']) ** 2
        loss += WEIGHTS['cv'] * ((cv_gpe - TARGETS['cv_gpe']) / TARGETS['cv_gpe']) ** 2
        loss += WEIGHTS['cv'] * ((cv_gpi - TARGETS['cv_gpi']) / TARGETS['cv_gpi']) ** 2
        
        # BETA - maximize!
        # Penalize if below target
        if beta_gpe < TARGETS['beta_gpe']:
            loss += WEIGHTS['beta'] * ((TARGETS['beta_gpe'] - beta_gpe) / TARGETS['beta_gpe']) ** 2
        
        # Bonus for high beta
        loss -= 5.0 * beta_gpe  # Direct reward
        
        # PD constraints
        if r_gpe > 55.0:
            loss += 5.0 * ((r_gpe - 55.0) / 20.0) ** 2
        if r_stn < 18.0:
            loss += 5.0 * ((18.0 - r_stn) / 18.0) ** 2
        
        # Silent penalty
        if r_stn < 3.0 or r_gpe < 5.0 or r_gpi < 10.0:
            loss += 100.0
        
        trial.set_user_attr('r_stn', float(r_stn))
        trial.set_user_attr('r_gpe', float(r_gpe))
        trial.set_user_attr('r_gpi', float(r_gpi))
        trial.set_user_attr('cv_stn', float(cv_stn))
        trial.set_user_attr('cv_gpe', float(cv_gpe))
        trial.set_user_attr('cv_gpi', float(cv_gpi))
        trial.set_user_attr('beta_gpe', float(beta_gpe))
        trial.set_user_attr('beta_stn', float(beta_stn))
        
        return loss
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 1e6

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HH Network - PARKINSONIAN (10 params, BETA FRACTION)")
    print("=" * 60)
    print(f"Targets: STN={TARGETS['rate_stn']}Hz, GPe={TARGETS['rate_gpe']}Hz")
    print(f"         Beta_GPe={TARGETS['beta_gpe']*100:.0f}% (weight={WEIGHTS['beta']}x)")
    print("=" * 60)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=CmaEsSampler(seed=42),
        study_name='hh_pd_beta_fraction'
    )
    
    t0 = time.time()
    study.optimize(objective, n_trials=1000, show_progress_bar=True)
    elapsed = time.time() - t0
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Best score: {study.best_value:.4f}")
    
    print(f"\nBest intrinsic parameters:")
    for k in ['ISTN', 'I_gpe', 'I_gpi', 'noise_stn_sigma', 'noise_gpe_sigma', 'noise_gpi_sigma']:
        print(f"  {k}: {study.best_params[k]:.3f}")
    
    print(f"\nBest synaptic multipliers:")
    for k in ['g_stn_gpe_mult', 'g_gpe_stn_mult', 'g_stn_gpi_mult', 'g_gpe_gpi_mult']:
        print(f"  {k}: {study.best_params[k]:.3f}")
    
    best = study.best_trial
    print(f"\nBest metrics:")
    print(f"  STN: {best.user_attrs['r_stn']:.1f} Hz (target: {TARGETS['rate_stn']})")
    print(f"  GPe: {best.user_attrs['r_gpe']:.1f} Hz (target: {TARGETS['rate_gpe']})")
    print(f"  GPi: {best.user_attrs['r_gpi']:.1f} Hz (target: {TARGETS['rate_gpi']})")
    print(f"\n  *** GPe Beta: {best.user_attrs['beta_gpe']*100:.1f}% (target: {TARGETS['beta_gpe']*100:.0f}%) ***")
    print(f"  *** STN Beta: {best.user_attrs['beta_stn']*100:.1f}% ***")
    
    # Save
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_metrics': best.user_attrs,
        'targets': TARGETS,
        'weights': WEIGHTS,
        'network_size': (N_STN, N_GPE, N_GPI),
        'n_trials': len(study.trials),
        'elapsed_seconds': elapsed,
    }
    
    with open('results/hh_parkinsonian_beta_study.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to results/hh_parkinsonian_beta_study.pkl")
