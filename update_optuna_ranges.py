"""Update Optuna search ranges to match new literature-based parameters."""

with open('optimization/optuna_driver.py', 'r') as f:
    content = f.read()

# Update search ranges
replacements = {
    "'I_gpe': trial.suggest_float('I_gpe', 1000.0, 1500.0)": 
        "'I_gpe': trial.suggest_float('I_gpe', 200.0, 500.0)",
    
    "'I_gpi': trial.suggest_float('I_gpi', 600.0, 1000.0)": 
        "'I_gpi': trial.suggest_float('I_gpi', 150.0, 400.0)",
    
    "'ISTN': trial.suggest_float('ISTN', 25.0, 45.0)": 
        "'ISTN': trial.suggest_float('ISTN', 25.0, 50.0)",
}

for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
        print(f"✓ Updated: {old.split('(')[1].split(',')[0]} range")

with open('optimization/optuna_driver.py', 'w') as f:
    f.write(content)

print("\n✓ Optuna ranges updated!")
print("\nNew ranges:")
print("  I_gpe: 200-500 pA (was 1000-1500)")
print("  I_gpi: 150-400 pA (was 600-1000)")
print("  ISTN: 25-50 pA (slightly expanded)")
