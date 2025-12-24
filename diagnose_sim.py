"""Diagnose simulation bottleneck."""

import sys, os
sys.path.insert(0, os.getcwd())

import time
import jax
import jax.numpy as jnp

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn

print("=" * 60)
print("SIMULATION BOTTLENECK DIAGNOSIS")
print("=" * 60)

# Build small network first
print("\n1. Testing with SMALL network (1,800 neurons)...")
state, config = build_network_state(400, 800, 600, 0.025)

simulator = create_simulation_fn(config, n_steps=16000)

params = {'ISTN': 80.0, 'I_gpe': 300.0, 'I_gpi': 300.0,
          'noise_stn_sigma': 0.5, 'noise_gpe_sigma': 40.0, 'noise_gpi_sigma': 40.0}

# Warm up
obs = simulator(params, state)
obs['V_stn'].block_until_ready()

# Time it
t0 = time.time()
obs = simulator(params, state)
obs['V_stn'].block_until_ready()
small_time = (time.time() - t0) * 1000
print(f"   Small network: {small_time:.0f}ms")

# Check if simulation function is JIT compiled
print(f"\n2. Checking if simulator is JIT compiled...")
print(f"   Type: {type(simulator)}")

# Test with explicit JIT
print("\n3. Testing raw JAX operations on large arrays...")

# Simulate what happens in the simulation
n_neurons = 30000
n_steps = 16000

V = jnp.zeros((n_steps, n_neurons))
spikes = jnp.zeros((n_steps, n_neurons), dtype=jnp.bool_)

# Time array operations
t0 = time.time()
V_sum = jnp.sum(V)
V_sum.block_until_ready()
print(f"   Sum 30K×16K array: {(time.time()-t0)*1000:.1f}ms")

t0 = time.time()
V_mean = jnp.mean(V, axis=1)
V_mean.block_until_ready()
print(f"   Mean across neurons: {(time.time()-t0)*1000:.1f}ms")

print("\n4. The issue is likely in the simulation loop structure.")
print("   Let's check sim_jax.py...")

print("=" * 60)
