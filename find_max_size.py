"""Find maximum network size that fits in GPU memory."""

import sys, os
sys.path.insert(0, os.getcwd())

import time
import copy
import jax
jax.clear_caches()

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn

def test_size(n_stn, n_gpe, n_gpi):
    """Test if network size fits in memory."""
    total = n_stn + n_gpe + n_gpi
    
    try:
        # Build
        state, config = build_network_state(n_stn, n_gpe, n_gpi, 0.025)
        n_conn = sum(c.connections.shape[0] for c in config['synapses'].values())
        
        # Compile & run
        simulator = create_simulation_fn(config, n_steps=16000)
        params = {'ISTN': 80.0, 'I_gpe': 300.0, 'I_gpi': 300.0,
                  'noise_stn_sigma': 0.5, 'noise_gpe_sigma': 40.0, 'noise_gpi_sigma': 40.0}
        
        obs = simulator(params, state)
        obs['V_stn'].block_until_ready()
        
        # Timed run
        t0 = time.time()
        obs = simulator(params, state)
        obs['V_stn'].block_until_ready()
        sim_time = (time.time() - t0) * 1000
        
        print(f"✓ {total:>6,} neurons | {n_conn:>12,} conn | {sim_time:>6.0f}ms")
        
        del state, config, simulator, obs
        jax.clear_caches()
        
        return True
        
    except Exception as e:
        print(f"✗ {total:>6,} neurons | FAILED: {str(e)[:50]}")
        jax.clear_caches()
        return False

print("=" * 60)
print("FINDING MAXIMUM NETWORK SIZE")
print("=" * 60)

# Test increasing sizes
sizes = [
    (2000, 4000, 3000),     # 9,000
    (4000, 8000, 6000),     # 18,000
    (5000, 10000, 7500),    # 22,500
    (6000, 12000, 9000),    # 27,000
    (7000, 14000, 10500),   # 31,500
    (8000, 16000, 12000),   # 36,000
    (9000, 18000, 13500),   # 40,500
    (10000, 20000, 15000),  # 45,000
]

for n_stn, n_gpe, n_gpi in sizes:
    success = test_size(n_stn, n_gpe, n_gpi)
    if not success:
        break

print("=" * 60)
