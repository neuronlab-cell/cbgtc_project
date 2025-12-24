"""Test build speed improvement."""
import time
import sys
sys.path.insert(0, '.')

from jax_models.network_builder import build_network_state

print("=" * 60)
print("BUILD SPEED TEST (Optimized)")
print("=" * 60)

sizes = [(50, 100, 75), (100, 200, 150), (200, 400, 300), (400, 800, 600), (800, 1600, 1200)]

for n_stn, n_gpe, n_gpi in sizes:
    total = n_stn + n_gpe + n_gpi
    
    t0 = time.time()
    state, config = build_network_state(n_stn, n_gpe, n_gpi, 0.025)
    build_time = time.time() - t0
    
    print(f"{total:>6} neurons: {build_time:>6.2f}s")
    
    del state, config

print("=" * 60)
