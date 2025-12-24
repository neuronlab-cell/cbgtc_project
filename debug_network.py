import sys, os
sys.path.insert(0, os.getcwd())

from jax_models.network_builder import build_network_state

# Build network
state, config = build_network_state(n_stn=10, n_gpe=20, n_gpi=15, dt_ms=0.025)

print("=" * 60)
print("Network Connectivity Debug")
print("=" * 60)

# Check synaptic configs
print("\n--- STN → GPe Synapse ---")
syn = config['synapses']['stn_to_gpe']
print(f"Connections: {syn.connections.shape[0]} (out of {syn.n_pre * syn.n_post} possible)")
print(f"Connection prob: {syn.connection_prob}")
print(f"Max weight: {syn.g_max}")
print(f"E_syn: {syn.E_syn}")

print("\n--- STN → GPi Synapse ---")
syn = config['synapses']['stn_to_gpi']
print(f"Connections: {syn.connections.shape[0]} (out of {syn.n_pre * syn.n_post} possible)")
print(f"Max weight: {syn.g_max}")

print("\n--- Baseline Currents ---")
print(f"ISTN: {config['neuron_params']['stn']['ISTN']}")
print(f"I_gpe: {config['neuron_params']['gpe']['I_baseline']}")
print(f"I_gpi: {config['neuron_params']['gpi']['I_baseline']}")

print("\n" + "=" * 60)
