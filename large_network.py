"""
30,000 Neuron Basal Ganglia Network

Fits in 24GB GPU VRAM.
"""

import sys, os
sys.path.insert(0, os.getcwd())

import time
import copy
import jax.numpy as jnp

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics


class LargeNetwork:
    """30K neuron network - fits in L4 GPU."""
    
    def __init__(self):
        # ~30K neurons (STN:GPe:GPi = 2:4:3 ratio)
        self.n_stn = 6667
        self.n_gpe = 13333
        self.n_gpi = 10000
        self.n_total = self.n_stn + self.n_gpe + self.n_gpi  # 30,000
        
        self.dt_ms = 0.025
        self.state = None
        self.config = None
        self.simulator = None
        
        # Optimized synaptic weights
        self.synaptic_weights = {
            'g_stn_gpe': 3.421,
            'g_gpe_stn': 8.829,
            'g_stn_gpi': 3.724,
            'g_gpe_gpi': 4.082
        }
    
    def build(self, n_steps=16000):
        """Build network and compile simulator."""
        
        print("=" * 60)
        print(f"BUILDING {self.n_total:,} NEURON NETWORK")
        print("=" * 60)
        
        # Build
        t0 = time.time()
        self.state, self.config = build_network_state(
            self.n_stn, self.n_gpe, self.n_gpi, self.dt_ms
        )
        build_time = time.time() - t0
        
        n_conn = sum(c.connections.shape[0] for c in self.config['synapses'].values())
        print(f"✓ Built in {build_time:.1f}s ({n_conn:,} connections)")
        
        # Apply synaptic weights
        self.config = copy.deepcopy(self.config)
        for name, key in [('stn_to_gpe', 'g_stn_gpe'), ('gpe_to_stn', 'g_gpe_stn'),
                          ('stn_to_gpi', 'g_stn_gpi'), ('gpe_to_gpi', 'g_gpe_gpi')]:
            self.config['synapses'][name] = self.config['synapses'][name]._replace(
                g_max=self.synaptic_weights[key]
            )
        
        # Compile
        print("Compiling (this may take a minute)...")
        t0 = time.time()
        self.simulator = create_simulation_fn(self.config, n_steps=n_steps)
        
        # Warm-up
        dummy = {'ISTN': 80.0, 'I_gpe': 300.0, 'I_gpi': 300.0,
                 'noise_stn_sigma': 0.5, 'noise_gpe_sigma': 40.0, 'noise_gpi_sigma': 40.0}
        obs = self.simulator(dummy, self.state)
        obs['V_stn'].block_until_ready()
        compile_time = time.time() - t0
        print(f"✓ Compiled in {compile_time:.1f}s")
        
        # Speed test
        t0 = time.time()
        obs = self.simulator(dummy, self.state)
        obs['V_stn'].block_until_ready()
        sim_time = (time.time() - t0) * 1000
        print(f"✓ Simulation: {sim_time:.0f}ms per trial")
        
        print("=" * 60)
        print("READY")
        print("=" * 60)
        
        return self
    
    def simulate(self, params):
        """Run simulation, return metrics."""
        obs = self.simulator(params, self.state)
        metrics = compute_all_metrics(obs, self.dt_ms, burn_steps=4000)
        return metrics, obs


if __name__ == "__main__":
    net = LargeNetwork()
    net.build()
    
    # Test
    print("\nTest simulation...")
    params = {
        'ISTN': 80.0,
        'I_gpe': 300.0,
        'I_gpi': 300.0,
        'noise_stn_sigma': 0.5,
        'noise_gpe_sigma': 40.0,
        'noise_gpi_sigma': 40.0
    }
    
    metrics, _ = net.simulate(params)
    print(f"STN: {metrics['firing_rates']['stn']:.1f} Hz")
    print(f"GPe: {metrics['firing_rates']['gpe']:.1f} Hz")
    print(f"GPi: {metrics['firing_rates']['gpi']:.1f} Hz")
