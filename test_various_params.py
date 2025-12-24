import sys, os
sys.path.insert(0, os.getcwd())

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics
import numpy as np

print("=" * 70)
print("Testing Parameter Range")
print("=" * 70)

state, config = build_network_state(50, 100, 75, 0.025)
sim_fn = create_simulation_fn(config, n_steps=4000)

# Test a range of ISTN values
test_params = [
    {'ISTN': 30.0, 'I_gpe': 300.0, 'I_gpi': 250.0},
    {'ISTN': 35.0, 'I_gpe': 300.0, 'I_gpi': 250.0},
    {'ISTN': 40.0, 'I_gpe': 300.0, 'I_gpi': 250.0},
    {'ISTN': 45.0, 'I_gpe': 300.0, 'I_gpi': 250.0},
]

for params_base in test_params:
    params = {
        **params_base,
        'noise_stn_sigma': 0.15,
        'noise_gpe_sigma': 30.0,
        'noise_gpi_sigma': 30.0
    }
    
    try:
        obs = sim_fn(params, state)
        metrics = compute_all_metrics(obs, 0.025, 1000)
        
        print(f"\nISTN = {params['ISTN']:.0f} pA:")
        print(f"  STN: {metrics['firing_rates']['stn']:.1f} Hz")
        print(f"  GPe: {metrics['firing_rates']['gpe']:.1f} Hz")
        print(f"  GPi: {metrics['firing_rates']['gpi']:.1f} Hz")
        print(f"  Beta: {metrics['beta_power']['stn']:.2e}")
        print(f"  CV: {metrics['cv']['stn']:.3f}")
        
        # Check for NaN
        if np.isnan(metrics['beta_power']['stn']):
            print("  ⚠ Beta returned NaN")
            
    except Exception as e:
        print(f"\nISTN = {params['ISTN']:.0f} pA: ERROR - {e}")

print("\n" + "=" * 70)
