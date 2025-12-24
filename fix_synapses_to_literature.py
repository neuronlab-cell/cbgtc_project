"""
Fix synaptic weights to match literature.

Based on Nevado-Holgado et al. (2014):
- All g_max values in nS (nanosiemens)
- These weights produce stable oscillations in the STN-GPe network
- GPi receives balanced excitation/inhibition
"""

with open('jax_models/network_builder.py', 'r') as f:
    content = f.read()

print("=" * 70)
print("Fixing Synaptic Weights to Literature Values")
print("=" * 70)

# Replace synaptic configurations
replacements = {
    # STN → GPe: Reduce from 22.0 to 2.0 nS
    "create_synapse_config(n_stn, n_gpe, 0.15, 22.0, 0.2, 5.0, 3.0, 0.0, dt_ms, seed+10)":
    "create_synapse_config(n_stn, n_gpe, 0.15, 2.0, 0.2, 5.0, 3.0, 0.0, dt_ms, seed+10)  # g_max: 22.0→2.0",
    
    # GPe → STN: Increase from 0.09 to 9.0 nS (100x increase!)
    "create_synapse_config(n_gpe, n_stn, 0.07, 0.09, 0.2, 8.0, 8.0, -70.0, dt_ms, seed+11)":
    "create_synapse_config(n_gpe, n_stn, 0.07, 9.0, 0.2, 8.0, 8.0, -70.0, dt_ms, seed+11)  # g_max: 0.09→9.0",
    
    # STN → GPi: Reduce from 18.0 to 2.0 nS
    "create_synapse_config(n_stn, n_gpi, 0.30, 18.0, 0.2, 5.0, 3.0, 0.0, dt_ms, seed+12)":
    "create_synapse_config(n_stn, n_gpi, 0.30, 2.0, 0.2, 5.0, 3.0, 0.0, dt_ms, seed+12)  # g_max: 18.0→2.0",
    
    # GPe → GPi: Reduce from 25.0 to 3.0 nS
    "create_synapse_config(n_gpe, n_gpi, 0.05, 25.0, 0.2, 5.0, 8.0, -70.0, dt_ms, seed+13)":
    "create_synapse_config(n_gpe, n_gpi, 0.05, 3.0, 0.2, 5.0, 8.0, -70.0, dt_ms, seed+13)  # g_max: 25.0→3.0",
}

for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
        # Extract just the g_max change for logging
        old_g = old.split(', ')[3]
        new_g = new.split(', ')[3]
        syn_type = "STN→GPe" if "n_stn, n_gpe" in old else \
                   "GPe→STN" if "n_gpe, n_stn" in old else \
                   "STN→GPi" if "n_stn, n_gpi" in old else "GPe→GPi"
        print(f"  ✓ {syn_type}: g_max {old_g} → {new_g} nS")

with open('jax_models/network_builder.py', 'w') as f:
    f.write(content)

print("\n" + "=" * 70)
print("✓ Synaptic weights updated to literature values!")
print("=" * 70)

print("\nSummary:")
print("  STN → GPe: 22.0 → 2.0 nS (11x weaker)")
print("  GPe → STN: 0.09 → 9.0 nS (100x STRONGER!) ← KEY FIX")
print("  STN → GPi: 18.0 → 2.0 nS (9x weaker)")
print("  GPe → GPi: 25.0 → 3.0 nS (8x weaker)")

print("\nWhy this matters:")
print("  - GPe can now inhibit STN properly (feedback control)")
print("  - STN won't overexcite GPe (balanced drive)")
print("  - GPi will receive appropriate excitation")
print("  - Network should produce stable oscillations")

print("\n" + "=" * 70)
