"""Diagnose why simulation is slow."""

import sys, os
sys.path.insert(0, os.getcwd())

import jax
import time

print("=" * 60)
print("SPEED DIAGNOSIS")
print("=" * 60)

# Check JAX backend
print(f"\nJAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Check if GPU is being used
if jax.default_backend() != 'gpu':
    print("\n⚠️  NOT USING GPU!")
else:
    print("\n✓ GPU detected")

# Simple GPU test
import jax.numpy as jnp

print("\nTesting GPU speed with matrix multiply...")
x = jnp.ones((5000, 5000))

# Warm up
y = jnp.dot(x, x)
y.block_until_ready()

# Timed
t0 = time.time()
for _ in range(10):
    y = jnp.dot(x, x)
    y.block_until_ready()
gpu_time = (time.time() - t0) / 10 * 1000

print(f"5000x5000 matmul: {gpu_time:.1f}ms")

if gpu_time > 100:
    print("⚠️  GPU seems slow - may be falling back to CPU")
else:
    print("✓ GPU working properly")

print("=" * 60)
