import sys, os
sys.path.insert(0, os.getcwd())
import copy

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print("=" * 80)
print("Testing Beta Oscillation Emergence")
print("=" * 80)

# Best healthy parameters
healthy = {
    'ISTN': 81.233,
    'I_gpe': 349.465,
    'I_gpi': 349.481,
    'noise_stn_sigma': 0.585,
    'noise_gpe_sigma': 43.928,
    'noise_gpi_sigma': 36.680
}

# Build network with optimized synapses
state, config = build_network_state(50, 100, 75, 0.025)
config = copy.deepcopy(config)
config['synapses']['stn_to_gpe'] = config['synapses']['stn_to_gpe']._replace(g_max=2.622)
config['synapses']['gpe_to_stn'] = config['synapses']['gpe_to_stn']._replace(g_max=9.991)
config['synapses']['stn_to_gpi'] = config['synapses']['stn_to_gpi']._replace(g_max=2.449)
config['synapses']['gpe_to_gpi'] = config['synapses']['gpe_to_gpi']._replace(g_max=2.899)

simulate = create_simulation_fn(config, n_steps=40000)  # 1 second for better FFT

# Test progression from healthy to Parkinsonian
gpe_currents = [349.5, 300.0, 250.0, 200.0, 150.0]  # Reduce GPe drive

print("\n" + "-" * 80)
print("I_gpe   | STN rate | GPe rate | GPi rate | Beta Power | State")
print("-" * 80)

for I_gpe in gpe_currents:
    params = healthy.copy()
    params['I_gpe'] = I_gpe
    
    obs = simulate(params, state)
    metrics = compute_all_metrics(obs, 0.025, burn_steps=10000)
    
    state_label = "Healthy" if I_gpe > 300 else "Transitional" if I_gpe > 200 else "Parkinsonian"
    
    print(f"{I_gpe:6.1f}  | {metrics['firing_rates']['stn']:8.1f} | "
          f"{metrics['firing_rates']['gpe']:8.1f} | "
          f"{metrics['firing_rates']['gpi']:8.1f} | "
          f"{metrics['beta_power']['stn']:10.2e} | {state_label}")

print("-" * 80)
print("\n✓ If beta increases as I_gpe decreases, your model captures PD pathophysiology!")
print("=" * 80)
