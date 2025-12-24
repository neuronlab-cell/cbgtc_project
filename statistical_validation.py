"""
Statistical Validation - Multiple Random Seeds
Author: Kavin Nakkeeran, Johns Hopkins University
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn, apply_params_to_config
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all
from jax_models.integrator import network_step
from jax import lax

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# PARAMETERS
# =============================================================================

N_SEEDS = 10
N_STEPS = 24000  # 600ms

healthy_params = {
    'ISTN': 140.0,
    'I_gpe': 3.379,
    'I_gpi': 2.188,
    'noise_stn_sigma': 0.996,
    'noise_gpe_sigma': 97.760,
    'noise_gpi_sigma': 69.678,
}

pd_params = {
    'ISTN': 80.0,
    'I_gpe': 0.672,
    'I_gpi': 2.430,
    'noise_stn_sigma': 4.333,
    'noise_gpe_sigma': 139.364,
    'noise_gpi_sigma': 109.012,
    'g_stn_gpe_mult': 1.592,
    'g_gpe_stn_mult': 0.182,
    'g_stn_gpi_mult': 1.975,
    'g_gpe_gpi_mult': 0.419,
}

# =============================================================================
# RUN MULTIPLE SEEDS
# =============================================================================

def create_pd_simulator(base_config, n_steps):
    @jax.jit
    def simulate(trial_params, init_state):
        config = apply_params_to_config(trial_params, base_config)
        syn_configs = dict(config['synapses'])
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
        _, obs_history = lax.scan(step_fn, init=init_state, xs=jnp.arange(n_steps))
        return obs_history
    return simulate

# Storage for results
results = {
    'healthy': {'stn_rate': [], 'gpe_rate': [], 'gpi_rate': [], 
                'stn_cv': [], 'gpe_cv': [], 'gpi_cv': [],
                'stn_beta': [], 'gpe_beta': [], 'gpi_beta': []},
    'pd': {'stn_rate': [], 'gpe_rate': [], 'gpi_rate': [],
           'stn_cv': [], 'gpe_cv': [], 'gpi_cv': [],
           'stn_beta': [], 'gpe_beta': [], 'gpi_beta': []}
}

print(f"\nRunning {N_SEEDS} seeds for statistical validation...")
print("="*60)

t_total = time.time()

for seed in range(N_SEEDS):
    print(f"\nSeed {seed+1}/{N_SEEDS}...")
    
    # Build network with different seed
    state, config = build_network_state(400, 800, 600, 0.025, use_hh=True, seed=seed)
    
    # Create simulators
    simulator = create_simulation_fn(config, n_steps=N_STEPS)
    pd_simulator = create_pd_simulator(config, N_STEPS)
    
    # Run healthy
    t0 = time.time()
    obs_h = simulator(healthy_params, state)
    obs_h['V_stn'].block_until_ready()
    
    metrics_h = compute_all_metrics(obs_h, 0.025, burn_steps=4000)
    beta_h = compute_beta_fraction_all(obs_h, 0.025, burn_steps=4000)
    
    results['healthy']['stn_rate'].append(metrics_h['firing_rates']['stn'])
    results['healthy']['gpe_rate'].append(metrics_h['firing_rates']['gpe'])
    results['healthy']['gpi_rate'].append(metrics_h['firing_rates']['gpi'])
    results['healthy']['stn_cv'].append(metrics_h['cv']['stn'])
    results['healthy']['gpe_cv'].append(metrics_h['cv']['gpe'])
    results['healthy']['gpi_cv'].append(metrics_h['cv']['gpi'])
    results['healthy']['stn_beta'].append(beta_h['stn']*100)
    results['healthy']['gpe_beta'].append(beta_h['gpe']*100)
    results['healthy']['gpi_beta'].append(beta_h['gpi']*100)
    
    # Run PD
    obs_pd = pd_simulator(pd_params, state)
    obs_pd['V_stn'].block_until_ready()
    
    metrics_pd = compute_all_metrics(obs_pd, 0.025, burn_steps=4000)
    beta_pd = compute_beta_fraction_all(obs_pd, 0.025, burn_steps=4000)
    
    results['pd']['stn_rate'].append(metrics_pd['firing_rates']['stn'])
    results['pd']['gpe_rate'].append(metrics_pd['firing_rates']['gpe'])
    results['pd']['gpi_rate'].append(metrics_pd['firing_rates']['gpi'])
    results['pd']['stn_cv'].append(metrics_pd['cv']['stn'])
    results['pd']['gpe_cv'].append(metrics_pd['cv']['gpe'])
    results['pd']['gpi_cv'].append(metrics_pd['cv']['gpi'])
    results['pd']['stn_beta'].append(beta_pd['stn']*100)
    results['pd']['gpe_beta'].append(beta_pd['gpe']*100)
    results['pd']['gpi_beta'].append(beta_pd['gpi']*100)
    
    print(f"  Healthy: STN={metrics_h['firing_rates']['stn']:.1f}Hz, GPe Beta={beta_h['gpe']*100:.1f}%")
    print(f"  PD:      STN={metrics_pd['firing_rates']['stn']:.1f}Hz, GPe Beta={beta_pd['gpe']*100:.1f}%")
    print(f"  Time: {time.time()-t0:.1f}s")

total_time = time.time() - t_total
print(f"\nTotal time: {total_time/60:.1f} minutes")

# =============================================================================
# COMPUTE STATISTICS
# =============================================================================

print("\n" + "="*70)
print("STATISTICAL RESULTS (n={})".format(N_SEEDS))
print("="*70)

def print_stats(name, healthy_vals, pd_vals):
    h_mean, h_std = np.mean(healthy_vals), np.std(healthy_vals)
    pd_mean, pd_std = np.mean(pd_vals), np.std(pd_vals)
    
    # t-test
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(healthy_vals, pd_vals)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    print(f"{name:<15} {h_mean:>8.2f} ± {h_std:<6.2f} {pd_mean:>8.2f} ± {pd_std:<6.2f}  p={p_val:.4f} {sig}")

print(f"{'Metric':<15} {'Healthy':>16} {'Parkinsonian':>16}  {'p-value':>12}")
print("-"*70)

print_stats("STN Rate (Hz)", results['healthy']['stn_rate'], results['pd']['stn_rate'])
print_stats("GPe Rate (Hz)", results['healthy']['gpe_rate'], results['pd']['gpe_rate'])
print_stats("GPi Rate (Hz)", results['healthy']['gpi_rate'], results['pd']['gpi_rate'])
print("-"*70)
print_stats("STN CV", results['healthy']['stn_cv'], results['pd']['stn_cv'])
print_stats("GPe CV", results['healthy']['gpe_cv'], results['pd']['gpe_cv'])
print_stats("GPi CV", results['healthy']['gpi_cv'], results['pd']['gpi_cv'])
print("-"*70)
print_stats("STN Beta (%)", results['healthy']['stn_beta'], results['pd']['stn_beta'])
print_stats("GPe Beta (%)", results['healthy']['gpe_beta'], results['pd']['gpe_beta'])
print_stats("GPi Beta (%)", results['healthy']['gpi_beta'], results['pd']['gpi_beta'])

print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

# =============================================================================
# FIGURE: Bar Charts with Error Bars
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f'Statistical Validation (n={N_SEEDS} seeds)', fontsize=14, fontweight='bold')

x = np.arange(3)
width = 0.35
populations = ['STN', 'GPe', 'GPi']

# Panel A: Firing Rates
h_rates = [np.mean(results['healthy']['stn_rate']), 
           np.mean(results['healthy']['gpe_rate']),
           np.mean(results['healthy']['gpi_rate'])]
h_rates_err = [np.std(results['healthy']['stn_rate']),
               np.std(results['healthy']['gpe_rate']),
               np.std(results['healthy']['gpi_rate'])]

pd_rates = [np.mean(results['pd']['stn_rate']),
            np.mean(results['pd']['gpe_rate']),
            np.mean(results['pd']['gpi_rate'])]
pd_rates_err = [np.std(results['pd']['stn_rate']),
                np.std(results['pd']['gpe_rate']),
                np.std(results['pd']['gpi_rate'])]

axes[0].bar(x - width/2, h_rates, width, yerr=h_rates_err, label='Healthy', 
            color='steelblue', edgecolor='black', capsize=5)
axes[0].bar(x + width/2, pd_rates, width, yerr=pd_rates_err, label='Parkinsonian',
            color='firebrick', edgecolor='black', capsize=5)
axes[0].set_ylabel('Firing Rate (Hz)', fontsize=11)
axes[0].set_xticks(x)
axes[0].set_xticklabels(populations)
axes[0].legend()
axes[0].set_title('A. Firing Rates', fontweight='bold')

# Panel B: CV
h_cv = [np.mean(results['healthy']['stn_cv']),
        np.mean(results['healthy']['gpe_cv']),
        np.mean(results['healthy']['gpi_cv'])]
h_cv_err = [np.std(results['healthy']['stn_cv']),
            np.std(results['healthy']['gpe_cv']),
            np.std(results['healthy']['gpi_cv'])]

pd_cv = [np.mean(results['pd']['stn_cv']),
         np.mean(results['pd']['gpe_cv']),
         np.mean(results['pd']['gpi_cv'])]
pd_cv_err = [np.std(results['pd']['stn_cv']),
             np.std(results['pd']['gpe_cv']),
             np.std(results['pd']['gpi_cv'])]

axes[1].bar(x - width/2, h_cv, width, yerr=h_cv_err, label='Healthy',
            color='steelblue', edgecolor='black', capsize=5)
axes[1].bar(x + width/2, pd_cv, width, yerr=pd_cv_err, label='Parkinsonian',
            color='firebrick', edgecolor='black', capsize=5)
axes[1].set_ylabel('CV (Coefficient of Variation)', fontsize=11)
axes[1].set_xticks(x)
axes[1].set_xticklabels(populations)
axes[1].legend()
axes[1].set_title('B. Firing Irregularity (CV)', fontweight='bold')

# Panel C: Beta Power
h_beta = [np.mean(results['healthy']['stn_beta']),
          np.mean(results['healthy']['gpe_beta']),
          np.mean(results['healthy']['gpi_beta'])]
h_beta_err = [np.std(results['healthy']['stn_beta']),
              np.std(results['healthy']['gpe_beta']),
              np.std(results['healthy']['gpi_beta'])]

pd_beta = [np.mean(results['pd']['stn_beta']),
           np.mean(results['pd']['gpe_beta']),
           np.mean(results['pd']['gpi_beta'])]
pd_beta_err = [np.std(results['pd']['stn_beta']),
               np.std(results['pd']['gpe_beta']),
               np.std(results['pd']['gpi_beta'])]

axes[2].bar(x - width/2, h_beta, width, yerr=h_beta_err, label='Healthy',
            color='steelblue', edgecolor='black', capsize=5)
axes[2].bar(x + width/2, pd_beta, width, yerr=pd_beta_err, label='Parkinsonian',
            color='firebrick', edgecolor='black', capsize=5)
axes[2].set_ylabel('Beta Power (%)', fontsize=11)
axes[2].set_xticks(x)
axes[2].set_xticklabels(populations)
axes[2].legend()
axes[2].set_title('C. Beta Band Power (13-30 Hz)', fontweight='bold')

plt.tight_layout()
plt.savefig('results/fig6_statistical_validation.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig6_statistical_validation.pdf', bbox_inches='tight')
print("\nSaved: results/fig6_statistical_validation.png/pdf")

# =============================================================================
# SAVE RESULTS
# =============================================================================

import pickle
with open('results/statistical_validation.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Saved: results/statistical_validation.pkl")

print("\n✓ Statistical validation complete!")
