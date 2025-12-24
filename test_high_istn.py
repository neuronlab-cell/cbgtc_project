import sys, os
sys.path.insert(0, os.getcwd())

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print("=" * 70)
print("Testing Higher ISTN and Noise")
print("=" * 70)

state, config = build_network_state(50, 100, 75, 0.025)
sim_fn = create_simulation_fn(config, n_steps=4000)

test_params = [
    {'ISTN': 60.0, 'noise_stn_sigma': 0.5},
    {'ISTN': 70.0, 'noise_stn_sigma': 0.5},
    {'ISTN': 80.0, 'noise_stn_sigma': 0.5},
]

for p in test_params:
    params = {
        'ISTN': p['ISTN'],
        'I_gpe': 300.0,
        'I_gpi': 250.0,
        'noise_stn_sigma': p['noise_stn_sigma'],
        'noise_gpe_sigma': 50.0,
        'noise_gpi_sigma': 50.0
    }
    
    obs = sim_fn(params, state)
    metrics = compute_all_metrics(obs, 0.025, 1000)
    
    print(f"\nISTN = {params['ISTN']:.0f}, noise = {p['noise_stn_sigma']:.1f}:")
    print(f"  STN: {metrics['firing_rates']['stn']:.1f} Hz")
    print(f"  GPe: {metrics['firing_rates']['gpe']:.1f} Hz")
    print(f"  GPi: {metrics['firing_rates']['gpi']:.1f} Hz")
    print(f"  CV: {metrics['cv']['stn']:.3f}")

print("\n" + "=" * 70)
