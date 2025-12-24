import pickle
import numpy as np
import pandas as pd

# Load the study
with open('results/full_biological_3hr_study.pkl', 'rb') as f:
    study = pickle.load(f)

print("=" * 80)
print("OPTUNA STUDY ANALYSIS")
print("=" * 80)

# Get all completed trials
completed = [t for t in study.trials if t.state.name == 'COMPLETE']
print(f"\nTotal trials: {len(study.trials)}")
print(f"Completed: {len(completed)}")

# Extract metrics from best trial
best = study.best_trial
print("\n" + "=" * 80)
print("BEST TRIAL METRICS")
print("=" * 80)
print(f"\nTrial number: {best.number}")
print(f"Score: {best.value:.3f}")

print("\n--- Parameters ---")
for param, value in best.params.items():
    print(f"  {param:20s}: {value:.3f}")

print("\n--- Firing Rates ---")
print(f"  STN: {best.user_attrs['rate_stn']:.1f} Hz")
print(f"  GPe: {best.user_attrs['rate_gpe']:.1f} Hz")
print(f"  GPi: {best.user_attrs['rate_gpi']:.1f} Hz")

print("\n--- Coefficient of Variation ---")
print(f"  STN: {best.user_attrs['cv_stn']:.3f}")
print(f"  GPe: {best.user_attrs['cv_gpe']:.3f}")
print(f"  GPi: {best.user_attrs['cv_gpi']:.3f}")

print("\n--- Beta Power ---")
print(f"  STN: {best.user_attrs['beta_stn']:.2e}")

# Analyze beta power across all trials
beta_values = [t.user_attrs.get('beta_stn', 0) for t in completed if 'beta_stn' in t.user_attrs]
beta_nonzero = [b for b in beta_values if b > 0]

print("\n" + "=" * 80)
print("BETA POWER STATISTICS (across all trials)")
print("=" * 80)
print(f"\nTrials with beta > 0: {len(beta_nonzero)} / {len(completed)}")
if len(beta_nonzero) > 0:
    print(f"Mean beta (non-zero): {np.mean(beta_nonzero):.2e}")
    print(f"Median beta (non-zero): {np.median(beta_nonzero):.2e}")
    print(f"Min beta (non-zero): {np.min(beta_nonzero):.2e}")
    print(f"Max beta (non-zero): {np.max(beta_nonzero):.2e}")
else:
    print("All trials had beta = 0 (NaN was converted to 0)")

# Find trials with highest beta
if len(beta_nonzero) > 0:
    print("\n--- Top 5 Trials by Beta Power ---")
    trials_with_beta = [(t.number, t.user_attrs['beta_stn'], t.user_attrs['rate_stn']) 
                        for t in completed if t.user_attrs.get('beta_stn', 0) > 0]
    trials_with_beta.sort(key=lambda x: x[1], reverse=True)
    
    for trial_num, beta, stn_rate in trials_with_beta[:5]:
        print(f"  Trial {trial_num}: beta={beta:.2e}, STN rate={stn_rate:.1f} Hz")

# Score components analysis
print("\n" + "=" * 80)
print("SCORE BREAKDOWN (Best Trial)")
print("=" * 80)

rate_error = ((best.user_attrs['rate_stn'] - 20)**2 + 
              (best.user_attrs['rate_gpe'] - 60)**2 + 
              (best.user_attrs['rate_gpi'] - 70)**2)

cv_error = ((best.user_attrs['cv_stn'] - 0.4)**2 + 
            (best.user_attrs['cv_gpe'] - 0.4)**2 + 
            (best.user_attrs['cv_gpi'] - 0.4)**2)

beta_penalty = best.user_attrs['beta_stn'] / 1e6

print(f"\nRate error (weight=1.0):  {rate_error:.3f} × 1.0 = {1.0 * rate_error:.3f}")
print(f"CV error (weight=0.5):    {cv_error:.3f} × 0.5 = {0.5 * cv_error:.3f}")
print(f"Beta penalty (weight=0.01): {beta_penalty:.3f} × 0.01 = {0.01 * beta_penalty:.3f}")
print(f"\nTotal score: {best.value:.3f}")
print(f"Expected: {1.0*rate_error + 0.5*cv_error + 0.01*beta_penalty:.3f}")

print("\n" + "=" * 80)
