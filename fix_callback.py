with open('optimization/optuna_parkinsonian_v3.py', 'r') as f:
    content = f.read()

old_callback = '''lambda study, trial: print(
                    f"Trial {trial.number}: score={trial.value:.1f}, "
                    f"STN={trial.user_attrs.get('rate_stn',0):.1f}Hz, "
                    f"GPe={trial.user_attrs.get('rate_gpe',0):.1f}Hz, "
                    f"beta={trial.user_attrs.get('beta_stn',0)*100:.1f}%"
                )'''

new_callback = '''lambda study, trial: print(
                    f"Trial {trial.number}: score={trial.value if trial.value is not None else 'FAIL'}, "
                    f"STN={trial.user_attrs.get('rate_stn',0):.1f}Hz, "
                    f"GPe={trial.user_attrs.get('rate_gpe',0):.1f}Hz, "
                    f"beta={trial.user_attrs.get('beta_stn',0)*100:.1f}%"
                ) if trial.value is not None else print(f"Trial {trial.number}: FAILED")'''

content = content.replace(old_callback, new_callback)

with open('optimization/optuna_parkinsonian_v3.py', 'w') as f:
    f.write(content)

print("✓ Fixed callback to handle NaN trials")
