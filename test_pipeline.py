import sys, os
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("JAX Pipeline Test")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from jax_models.network_builder import build_network_state
    from optimization.sim_jax import create_simulation_fn
    from optimization.metrics_jax import compute_all_metrics
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Build network
print("\n2. Building network...")
import time
state, config = build_network_state(n_stn=50, n_gpe=100, n_gpi=75, dt_ms=0.025)
print("   ✓ Network built (50-100-75 neurons)")

# Create simulator
print("\n3. Compiling JIT simulator...")
sim_fn = create_simulation_fn(config, n_steps=4000)
params = {
    'ISTN': 42.0, 'I_gpe': 580.0, 'I_gpi': 240.0,
    'noise_stn_sigma': 0.15, 'noise_gpe_sigma': 30.0, 'noise_gpi_sigma': 30.0
}

# First run (compile)
t0 = time.time()
obs = sim_fn(params, state)
obs['V_stn'].block_until_ready()
t1 = time.time()
compile_time = (t1-t0)*1000
print(f"   ✓ Compiled in {compile_time:.0f} ms")

# Second run (cached)
t0 = time.time()
obs = sim_fn(params, state)
obs['V_stn'].block_until_ready()
t1 = time.time()
cached_time = (t1-t0)*1000
print(f"   ✓ Cached run: {cached_time:.1f} ms")
print(f"   ✓ Speedup: {compile_time/cached_time:.0f}x")

# Compute metrics
print("\n4. Computing metrics...")
t0 = time.time()
metrics = compute_all_metrics(obs, dt_ms=0.025, burn_steps=1000)
t1 = time.time()
print(f"   ✓ Metrics computed in {(t1-t0)*1000:.0f} ms")

# Show results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Firing rates:")
print(f"  STN: {metrics['firing_rates']['stn']:.1f} Hz (target: 20 Hz)")
print(f"  GPe: {metrics['firing_rates']['gpe']:.1f} Hz (target: 60 Hz)")
print(f"  GPi: {metrics['firing_rates']['gpi']:.1f} Hz (target: 70 Hz)")
print(f"\nBeta power (STN): {metrics['beta_power']['stn']:.2e}")
print(f"CV (STN): {metrics['cv']['stn']:.3f} (target: 0.4)")

print("\n" + "=" * 60)
print("✓✓✓ PIPELINE WORKING! ✓✓✓")
print("=" * 60)
print(f"\nPerformance: {cached_time:.1f} ms per trial")
print(f"Estimated: {int(3600000/cached_time)} trials/hour")
print("=" * 60)
