"""Test massive network builds with optimized code."""
import time
import sys
sys.path.insert(0, '.')

from jax_models.network_builder import build_network_state

print("=" * 70)
print("MASSIVE NETWORK BUILD TEST")
print("=" * 70)

# Push to 100k+ neurons!
sizes = [
    (400, 800, 600),       # 1,800
    (800, 1600, 1200),     # 3,600
    (1600, 3200, 2400),    # 7,200
    (3200, 6400, 4800),    # 14,400
    (6400, 12800, 9600),   # 28,800
    (12800, 25600, 19200), # 57,600
    (25600, 51200, 38400), # 115,200
]

print(f"{'Neurons':>10} | {'Build (s)':>10} | {'Status'}")
print("-" * 40)

for n_stn, n_gpe, n_gpi in sizes:
    total = n_stn + n_gpe + n_gpi
    
    try:
        t0 = time.time()
        state, config = build_network_state(n_stn, n_gpe, n_gpi, 0.025)
        build_time = time.time() - t0
        
        n_conn = sum(cfg.connections.shape[0] for cfg in config['synapses'].values())
        
        print(f"{total:>10,} | {build_time:>10.2f} | ✓ ({n_conn:,} connections)")
        
        del state, config
        
    except Exception as e:
        print(f"{total:>10,} | {'FAILED':>10} | {str(e)[:30]}")
        break

print("=" * 70)
