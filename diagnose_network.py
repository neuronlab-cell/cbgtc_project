import sys, os
sys.path.insert(0, os.getcwd())

from jax_models.network_builder import build_network_state

print("=" * 70)
print("Network Connectivity Diagnosis")
print("=" * 70)

state, config = build_network_state(50, 100, 75, 0.025)

print("\n--- SYNAPTIC CONNECTIONS ---")

print("\nSTN → GPe:")
syn = config['synapses']['stn_to_gpe']
print(f"  Connections: {syn.connections.shape[0]} / {syn.n_pre * syn.n_post}")
print(f"  Probability: {syn.connection_prob}")
print(f"  Max weight (g_max): {syn.g_max} nS")
print(f"  E_syn: {syn.E_syn} mV (excitatory)")

print("\nGPe → STN:")
syn = config['synapses']['gpe_to_stn']
print(f"  Connections: {syn.connections.shape[0]} / {syn.n_pre * syn.n_post}")
print(f"  Probability: {syn.connection_prob}")
print(f"  Max weight (g_max): {syn.g_max} nS")
print(f"  E_syn: {syn.E_syn} mV (inhibitory)")

print("\nSTN → GPi:")
syn = config['synapses']['stn_to_gpi']
print(f"  Connections: {syn.connections.shape[0]} / {syn.n_pre * syn.n_post}")
print(f"  Probability: {syn.connection_prob}")
print(f"  Max weight (g_max): {syn.g_max} nS")
print(f"  E_syn: {syn.E_syn} mV (excitatory)")

print("\nGPe → GPi:")
syn = config['synapses']['gpe_to_gpi']
print(f"  Connections: {syn.connections.shape[0]} / {syn.n_pre * syn.n_post}")
print(f"  Probability: {syn.connection_prob}")
print(f"  Max weight (g_max): {syn.g_max} nS")
print(f"  E_syn: {syn.E_syn} mV (inhibitory)")

print("\n--- NEURON BASELINE CURRENTS ---")
print(f"\nSTN I_baseline: {config['neuron_params']['stn']['ISTN']} pA")
print(f"GPe I_baseline: {config['neuron_params']['gpe']['I_baseline']} pA")
print(f"GPi I_baseline: {config['neuron_params']['gpi']['I_baseline']} pA")

print("\n--- ANALYSIS ---")

# Check if STN can drive GPe
stn_to_gpe_weight = config['synapses']['stn_to_gpe'].g_max
print(f"\n✓ STN → GPe excitation: {stn_to_gpe_weight} nS")
if stn_to_gpe_weight < 10:
    print(f"  ⚠ Too weak! Literature: 15-25 nS")

# Check if GPe inhibits itself
gpe_to_stn_weight = config['synapses']['gpe_to_stn'].g_max
print(f"\n✓ GPe → STN inhibition: {gpe_to_stn_weight} nS")
if gpe_to_stn_weight < 0.05:
    print(f"  ⚠ Too weak!")

# Check if GPe can inhibit GPi
gpe_to_gpi_weight = config['synapses']['gpe_to_gpi'].g_max
print(f"\n✓ GPe → GPi inhibition: {gpe_to_gpi_weight} nS")

print("\n" + "=" * 70)
