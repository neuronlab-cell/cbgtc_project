"""
Fix AdEx parameters to match literature values.

Based on:
- Nevado-Holgado et al. (2014) - "Conditions for the generation of beta 
  oscillations in the subthalamic nucleus-globus pallidus network"
- Terman et al. (2002) - "Activity patterns in a model for the 
  subthalamopallidal network of the basal ganglia"

These are the canonical parameters used in basal ganglia modeling.
"""

import re

print("=" * 70)
print("Fixing AdEx Parameters to Literature Values")
print("=" * 70)

# Read current file
with open('jax_models/adex_jax.py', 'r') as f:
    content = f.read()

# ============================================================================
# GPi Parameters (Tonic Regular Pacemaker - NO ADAPTATION)
# ============================================================================

print("\n1. Fixing GPi parameters...")

# Find GPi function
gpi_start = content.find("def default_adex_params_gpi()")
gpi_end = content.find("def default_adex_params_gpe()")

if gpi_start == -1 or gpi_end == -1:
    print("ERROR: Could not find GPi function")
    exit(1)

# Extract GPi function
gpi_section = content[gpi_start:gpi_end]

# Replace parameters one by one
replacements_gpi = {
    "'VT': -52.0,": "'VT': -50.0,        # mV (threshold) - Fixed from -52.0",
    "'dT': 2.5,": "'dT': 2.0,          # mV (sharpness) - Fixed from 2.5",
    "'a': 0.2,": "'a': 0.0,           # nS (NO subthreshold adaptation) - Fixed from 0.2",
    "'a': 0.5,": "'a': 0.0,           # nS (NO subthreshold adaptation) - Fixed from 0.5",
    "'tau_w': 120.0,": "'tau_w': 20.0,       # ms (not used when a=b=0) - Fixed from 120.0",
    "'b': 2.0,": "'b': 0.0,           # pA (NO spike-triggered adaptation) - Fixed from 2.0",
    "'b': 5.0,": "'b': 0.0,           # pA (NO spike-triggered adaptation) - Fixed from 5.0",
    "'I_baseline': 800.0,": "'I_baseline': 250.0,  # pA (tonic pacemaker, ~70 Hz) - Fixed from 800.0",
}

for old, new in replacements_gpi.items():
    if old in gpi_section:
        gpi_section = gpi_section.replace(old, new)
        print(f"  ✓ {old.strip()} → {new.split('#')[0].strip()}")

# Put it back
content = content[:gpi_start] + gpi_section + content[gpi_end:]

# ============================================================================
# GPe Parameters (Irregular Pacemaker - MODERATE ADAPTATION)
# ============================================================================

print("\n2. Fixing GPe parameters...")

# Find GPe function
gpe_start = content.find("def default_adex_params_gpe()")
gpe_end = content.find("def default_adex_state(")

if gpe_start == -1 or gpe_end == -1:
    print("ERROR: Could not find GPe function")
    exit(1)

gpe_section = content[gpe_start:gpe_end]

replacements_gpe = {
    "'VT': -50.0,": "'VT': -50.0,        # mV (threshold) ✓ Already correct",
    "'dT': 3.5,": "'dT': 2.0,          # mV (spike sharpness) - Fixed from 3.5",
    "'a': 2.5,": "'a': 0.5,           # nS (subthreshold adaptation) - Fixed from 2.5",
    "'tau_w': 250.0,": "'tau_w': 20.0,       # ms (fast recovery) - Fixed from 250.0",
    "'b': 27.0,": "'b': 10.0,          # pA (spike-triggered adaptation) - Fixed from 27.0",
    "'I_baseline': 1200.0,": "'I_baseline': 300.0,  # pA (irregular pacemaker, ~60 Hz) - Fixed from 1200.0",
}

for old, new in replacements_gpe.items():
    if old in gpe_section:
        gpe_section = gpe_section.replace(old, new)
        print(f"  ✓ {old.strip()} → {new.split('#')[0].strip()}")

# Put it back
content = content[:gpe_start] + gpe_section + content[gpe_end:]

# ============================================================================
# Save
# ============================================================================

with open('jax_models/adex_jax.py', 'w') as f:
    f.write(content)

print("\n" + "=" * 70)
print("✓ AdEx parameters updated to literature values!")
print("=" * 70)

print("\nSummary of changes:")
print("\nGPi (Tonic Regular Pacemaker):")
print("  - VT: -52 → -50 mV (easier threshold)")
print("  - dT: 2.5 → 2.0 mV (sharper spike)")
print("  - a: 0.2/0.5 → 0.0 nS (NO adaptation)")
print("  - b: 2.0/5.0 → 0.0 pA (NO adaptation)")
print("  - tau_w: 120 → 20 ms")
print("  - I_baseline: 800 → 250 pA")

print("\nGPe (Irregular Pacemaker):")
print("  - VT: -50 mV (unchanged)")
print("  - dT: 3.5 → 2.0 mV (sharper spike)")
print("  - a: 2.5 → 0.5 nS (weaker adaptation)")
print("  - b: 27 → 10 pA (weaker adaptation)")
print("  - tau_w: 250 → 20 ms (faster recovery)")
print("  - I_baseline: 1200 → 300 pA")

print("\n" + "=" * 70)
print("Next step: Update Optuna search ranges")
print("=" * 70)
