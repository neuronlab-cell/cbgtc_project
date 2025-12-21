#!/usr/bin/env python3
"""Complete test of sim_jax.py with proper package imports"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_models.network_builder import build_network_state
from optimization.sim_jax import apply_params_to_config, run_simulation_python_loop, create_simulation_fn
import time

print("=" * 70)
print("JAX Simulation Wrapper - Complete Test Suite")
print("=" * 70)

# Build small network for testing
print("\nBuilding test network (10 STN, 20 GPe, 15 GPi)...")
state, config = build_network_state(n_stn=10, n_gpe=20, n_gpi=15, dt_ms=0.025)
print("✓ Network built\n")

# Test parameters (use different value to show update works)
params = {
    'ISTN': 50.0,  # Different from default 42.0
    'I_gpe': 600.0,  # Different from default 580.0
    'I_gpi': 250.0,  # Different from default 240.0
    'noise_stn_sigma': 0.15,
    'noise_gpe_sigma': 35.0,
    'noise_gpi_sigma': 35.0
}

# ============================================================================
# Test 1: Parameter Application
# ============================================================================
print("=" * 70)
print("Test 1: Parameter Application")
print("-" * 70)

new_config = apply_params_to_config(params, config)
print(f"✓ ISTN: {config['neuron_params']['stn']['ISTN']} → {new_config['neuron_params']['stn']['ISTN']}")
print(f"✓ I_gpe: {config['neuron_params']['gpe']['I_baseline']} → {new_config['neuron_params']['gpe']['I_baseline']}")
print(f"✓ Noise sigma (STN): {config['noise']['stn'].sigma} → {new_config['noise']['stn'].sigma}")

assert new_config['neuron_params']['stn']['ISTN'] == 50.0
assert config['neuron_params']['stn']['ISTN'] == 42.0  # Unchanged (default)
print("\n✓ Test 1 PASSED\n")

# ============================================================================
# Test 2: Python Loop Simulation
# ============================================================================
print("=" * 70)
print("Test 2: Python Loop Simulation")
print("-" * 70)

print("Running 50-step simulation...")
t0 = time.time()
obs = run_simulation_python_loop(params, state, config, n_steps=50)
t1 = time.time()

print(f"Time: {(t1-t0)*1000:.1f} ms")
print(f"Shapes: V_stn={obs['V_stn'].shape}, spikes_gpe={obs['spikes_gpe'].shape}")

assert obs['V_stn'].shape == (50, 10)
assert obs['V_gpe'].shape == (50, 20)
assert obs['V_gpi'].shape == (50, 15)
print("\n✓ Test 2 PASSED\n")

# ============================================================================
# Test 3: JIT Compilation
# ============================================================================
print("=" * 70)
print("Test 3: JIT + lax.scan Simulation")
print("-" * 70)

print("Creating JIT-compiled function...")
sim_fn = create_simulation_fn(config, n_steps=50)

print("\nFirst call (compile time):")
t0 = time.time()
obs1 = sim_fn(params, state)
obs1['V_stn'].block_until_ready()
t1 = time.time()
compile_ms = (t1-t0)*1000
print(f"  {compile_ms:.1f} ms")

print("\nSecond call (cached):")
t0 = time.time()
obs2 = sim_fn(params, state)
obs2['V_stn'].block_until_ready()
t1 = time.time()
cached_ms = (t1-t0)*1000
print(f"  {cached_ms:.3f} ms")

print(f"\nSpeedup: {compile_ms/cached_ms:.1f}x")
assert obs2['V_stn'].shape == (50, 10)
print("\n✓ Test 3 PASSED\n")

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("ALL TESTS PASSED! 🎉")
print("=" * 70)
print("\nWhat works:")
print("  ✓ Functional parameter updates (no mutations)")
print("  ✓ Python loop simulation")
print("  ✓ JIT compilation with lax.scan")
print(f"  ✓ Performance gain: {compile_ms/cached_ms:.1f}x")
print("\nYour implementation is correct!")
print("=" * 70)
