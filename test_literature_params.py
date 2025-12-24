import sys, os
sys.path.insert(0, os.getcwd())

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print("=" * 70)
print("Testing Literature-Corrected AdEx Parameters")
print("=" * 70)

state, config = build_network_state(50, 100, 75, 0.025)
sim_fn = create_simulation_fn(config, n_steps=4000)

# Test with default parameters (should work now!)
params = {
    'ISTN': 35.0,
    'I_gpe': 300.0,   # New default
    'I_gpi': 250.0,   # New default
    'noise_stn_sigma': 0.15,
    'noise_gpe_sigma': 30.0,
    'noise_gpi_sigma': 30.0
}

print("\nRunning simulation with literature defaults...")
print(f"  ISTN: {params['ISTN']} pA")
print(f"  I_gpe: {params['I_gpe']} pA")
print(f"  I_gpi: {params['I_gpi']} pA")

obs = sim_fn(params, state)
metrics = compute_all_metrics(obs, 0.025, 1000)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"STN: {metrics['firing_rates']['stn']:.1f} Hz (target: 20 Hz)")
print(f"GPe: {metrics['firing_rates']['gpe']:.1f} Hz (target: 60 Hz)")
print(f"GPi: {metrics['firing_rates']['gpi']:.1f} Hz (target: 70 Hz)")
print(f"\nBeta (STN): {metrics['beta_power']['stn']:.2e}")
print(f"CV (STN): {metrics['cv']['stn']:.3f} (target: 0.4)")

if metrics['firing_rates']['gpi'] > 50:
    print("\n✓✓✓ SUCCESS! GPi is firing! ✓✓✓")
else:
    print("\n⚠ GPi still not firing properly")

print("=" * 70)
