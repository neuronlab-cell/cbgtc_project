"""Adjust search ranges for STN firing problem."""

with open('optimization/optuna_driver.py', 'r') as f:
    content = f.read()

# Increase ISTN range (STN needs more drive against GPe inhibition)
content = content.replace(
    "'ISTN': trial.suggest_float('ISTN', 25.0, 50.0)",
    "'ISTN': trial.suggest_float('ISTN', 50.0, 100.0)"
)

# Increase noise ranges (for CV)
content = content.replace(
    "'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.05, 0.3)",
    "'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.3, 1.0)"
)

content = content.replace(
    "'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 10.0, 50.0)",
    "'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 30.0, 80.0)"
)

content = content.replace(
    "'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 10.0, 50.0)",
    "'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 30.0, 80.0)"
)

with open('optimization/optuna_driver.py', 'w') as f:
    f.write(content)

print("✓ Updated search ranges:")
print("  ISTN: 50-100 pA (was 25-50)")
print("  noise_stn_sigma: 0.3-1.0 (was 0.05-0.3)")
print("  noise_gpe_sigma: 30-80 pA (was 10-50)")
print("  noise_gpi_sigma: 30-80 pA (was 10-50)")
