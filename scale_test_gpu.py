"""
GPU-enabled scaling test for JAX basal ganglia network.

NVIDIA L4: 24GB VRAM, Ada Lovelace architecture
"""

import sys, os
sys.path.insert(0, os.getcwd())

import time
import copy
import jax
import jax.numpy as jnp
import numpy as np

print("=" * 80)
print("JAX GPU SCALING TEST")
print("=" * 80)

# GPU Info
print(f"\nJAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

if jax.default_backend() != 'gpu':
    print("\n⚠️  WARNING: GPU not detected! Install jax[cuda12]")
    print("Run: pip3 install --upgrade 'jax[cuda12]' --break-system-packages")

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

# Best PD parameters from v3
best_params = {
    'ISTN': 116.519,
    'I_gpe': 274.789,
    'I_gpi': 345.139,
    'noise_stn_sigma': 0.676,
    'noise_gpe_sigma': 49.955,
    'noise_gpi_sigma': 27.587
}

best_synapses = {
    'g_stn_gpe': 3.421,
    'g_gpe_stn': 8.829,
    'g_stn_gpi': 3.724,
    'g_gpe_gpi': 4.082
}

def compute_beta_fraction(V_trace, dt_ms, burn_steps):
    valid_V = V_trace[burn_steps:]
    lfp = jnp.mean(valid_V, axis=1)
    lfp = lfp - jnp.mean(lfp)
    fft_vals = jnp.fft.rfft(lfp)
    freqs = jnp.fft.rfftfreq(len(lfp), d=dt_ms / 1000.0)
    psd = jnp.abs(fft_vals) ** 2
    beta_idx = (freqs >= 13) & (freqs <= 30)
    broad_idx = (freqs >= 1) & (freqs <= 100)
    beta_power = float(jnp.sum(psd[beta_idx]))
    total_power = float(jnp.sum(psd[broad_idx]))
    if total_power > 1e-10:
        return beta_power / total_power
    return 0.0

# GPU-optimized scale factors (much more aggressive!)
scale_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

print("\n" + "-" * 90)
print(f"{'Scale':>6} | {'Neurons':>10} | {'Build':>8} | {'Compile':>10} | {'Run (ms)':>10} | {'Beta GPi':>10} | {'Status'}")
print("-" * 90)

N_STEPS = 16000  # 400ms simulation
BURN = 4000

results = []

for scale in scale_factors:
    n_stn = 50 * scale
    n_gpe = 100 * scale
    n_gpi = 75 * scale
    total = n_stn + n_gpe + n_gpi
    
    print(f"{scale:>6}x | {total:>10,} | ", end="", flush=True)
    
    try:
        # Build network
        t0 = time.time()
        state, config = build_network_state(n_stn, n_gpe, n_gpi, 0.025)
        build_time = time.time() - t0
        print(f"{build_time:>7.1f}s | ", end="", flush=True)
        
        # Update synapses
        config = copy.deepcopy(config)
        config['synapses']['stn_to_gpe'] = config['synapses']['stn_to_gpe']._replace(g_max=best_synapses['g_stn_gpe'])
        config['synapses']['gpe_to_stn'] = config['synapses']['gpe_to_stn']._replace(g_max=best_synapses['g_gpe_stn'])
        config['synapses']['stn_to_gpi'] = config['synapses']['stn_to_gpi']._replace(g_max=best_synapses['g_stn_gpi'])
        config['synapses']['gpe_to_gpi'] = config['synapses']['gpe_to_gpi']._replace(g_max=best_synapses['g_gpe_gpi'])
        
        # Compile
        t0 = time.time()
        simulate = create_simulation_fn(config, n_steps=N_STEPS)
        obs = simulate(best_params, state)
        obs['V_stn'].block_until_ready()
        compile_time = time.time() - t0
        print(f"{compile_time:>9.1f}s | ", end="", flush=True)
        
        # Timed run (average of 3)
        times = []
        for _ in range(3):
            t0 = time.time()
            obs = simulate(best_params, state)
            obs['V_stn'].block_until_ready()
            times.append((time.time() - t0) * 1000)
        run_time = np.mean(times)
        print(f"{run_time:>10.1f} | ", end="", flush=True)
        
        # Beta
        beta_gpi = compute_beta_fraction(obs['V_gpi'], 0.025, BURN)
        print(f"{beta_gpi*100:>9.1f}% | ", end="", flush=True)
        
        # Metrics
        metrics = compute_all_metrics(obs, 0.025, BURN)
        
        results.append({
            'scale': scale,
            'neurons': total,
            'build_time': build_time,
            'compile_time': compile_time,
            'run_time': run_time,
            'beta_gpi': beta_gpi,
            'rate_stn': metrics['firing_rates']['stn'],
            'rate_gpe': metrics['firing_rates']['gpe'],
            'rate_gpi': metrics['firing_rates']['gpi'],
            'cv_stn': metrics['cv']['stn']
        })
        
        # Real-time ratio
        realtime_ratio = 400 / run_time
        print(f"{realtime_ratio:.1f}x RT ✓")
        
        # Clear memory
        del state, config, simulate, obs
        jax.clear_caches()
        
    except Exception as e:
        print(f"FAILED: {str(e)[:40]}")
        break

print("-" * 90)

# Summary
print("\n" + "=" * 80)
print("🏆 SCALING RESULTS SUMMARY")
print("=" * 80)

if results:
    max_result = results[-1]
    base_result = results[0]
    
    print(f"\n📊 Maximum Network Size: {max_result['neurons']:,} neurons")
    print(f"⚡ Simulation Time: {max_result['run_time']:.1f} ms for 400ms neural activity")
    print(f"🚀 Speed: {400 / max_result['run_time']:.1f}x faster than real-time!")
    print(f"📈 Speedup vs base: {base_result['run_time'] / max_result['run_time'] * max_result['neurons'] / base_result['neurons']:.0f}x efficiency gain")
    
    print("\n--- Neural Dynamics at Maximum Scale ---")
    print(f"  STN: {max_result['rate_stn']:.1f} Hz (CV={max_result['cv_stn']:.3f})")
    print(f"  GPe: {max_result['rate_gpe']:.1f} Hz")
    print(f"  GPi: {max_result['rate_gpi']:.1f} Hz")
    print(f"  Beta GPi: {max_result['beta_gpi']*100:.1f}%")
    
    if max_result['beta_gpi'] > 0.1:
        print("\n  ✅ BETA OSCILLATIONS PERSIST AT SCALE!")
    
    print("\n--- Scaling Table ---")
    for r in results:
        rt_ratio = 400 / r['run_time']
        print(f"  {r['neurons']:>8,} neurons | {r['run_time']:>8.1f} ms | {rt_ratio:>6.1f}x RT | Beta: {r['beta_gpi']*100:>5.1f}%")

print("\n" + "=" * 80)
