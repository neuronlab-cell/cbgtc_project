# optuna_driver.py
# Main orchestration script for δ-paired parameter optimization using Optuna.
#
# Responsibilities:
#   • Define parameter search space (θ)
#   • Run two simulations per trial: δ=0 (normal), δ=1 (PD)
#   • Score both with objectives.py
#   • Return combined score to Optuna
#   • Save best parameters + study metadata
#
# How to run (on AWS EC2 with venv active):
#   cd ~/cbgtc_project
#   source .venv/bin/activate
#   python -m stn_gp.optuna.optuna_driver

from __future__ import annotations
import json
from pathlib import Path
import optuna
import numpy as np

from stn_gp.optuna.sim_api import run_simulation
from stn_gp.optuna.objectives import (
    PairedObjectiveConfig,
    compute_normal_score,
    compute_pathological_score,
    compute_combined_score,
)


# ============================================================
# PARAMETER SEARCH SPACE (θ) — BASE PARAMETERS
# ============================================================

def sample_theta(trial: optuna.Trial) -> dict:
    """
    Sample base parameters θ from biologically plausible ranges.
    These are *normal-state baseline* values BEFORE δ-modulation.
    """
    theta = {}

    # -------------------------
    # Synaptic weights
    # -------------------------
    theta["stn_to_gpe_mean"] = trial.suggest_float(
        "stn_to_gpe_mean",
        10.0, 80.0,   # pA range for AMPA → tune later if needed
    )

    theta["gpe_to_stn_mean"] = trial.suggest_float(
        "gpe_to_stn_mean",
        0.03, 0.30,   # μA/cm² range (inhibitory conductance)
    )

    # -------------------------
    # Intrinsic drives
    # -------------------------
    theta["stn_ISTN"] = trial.suggest_float(
        "stn_ISTN",
        10.0, 35.0,    # µA/cm² — baseline tonic drive
    )

    theta["gpe_I_baseline"] = trial.suggest_float(
        "gpe_I_baseline",
        80.0, 250.0,   # pA baseline current for AdEx GPe
    )

    # -------------------------
    # Delays (ms)
    # -------------------------
    theta["delay_stn_to_gpe_ms"] = trial.suggest_float(
        "delay_stn_to_gpe_ms",
        2.0, 8.0,
    )
    theta["delay_gpe_to_stn_ms"] = trial.suggest_float(
        "delay_gpe_to_stn_ms",
        5.0, 12.0,
    )

    # -------------------------
    # Noise
    # -------------------------
    theta["noise_sigma_stn"] = trial.suggest_float(
        "noise_sigma_stn",
        0.5, 3.0,
    )
    theta["noise_sigma_gpe"] = trial.suggest_float(
        "noise_sigma_gpe",
        0.5, 3.0,
    )

    # -------------------------
    # Population sizes (optional but allowed)
    # -------------------------
    theta["n_stn"] = trial.suggest_int("n_stn", 40, 80)
    theta["n_gpe"] = trial.suggest_int("n_gpe", 80, 160)

    return theta


# ============================================================
# OBJECTIVE FUNCTION FOR OPTUNA
# ============================================================

def objective(trial: optuna.Trial) -> float:
    """
    Full paired δ objective.
    """
    # -------------------------
    # 1. Sample θ
    # -------------------------
    theta = sample_theta(trial)

    # -------------------------
    # 2. Load PD target PSD if available
    # -------------------------
    # By default we leave these None; you can drop in real data later.
    cfg = PairedObjectiveConfig()

    # -------------------------
    # 3. Simulate δ = 0 (normal)
    # -------------------------
    sim_norm = run_simulation(
        theta,
        delta=0.0,
        t_total_s=4.0,     # keep short for early optimization
        burn_in_s=1.0,
        dt_ms=0.025,
    )
    J_normal = compute_normal_score(sim_norm, cfg)

    # -------------------------
    # 4. Simulate δ = 1 (PD)
    # -------------------------
    sim_pd = run_simulation(
        theta,
        delta=1.0,
        t_total_s=4.0,
        burn_in_s=1.0,
        dt_ms=0.025,
    )
    J_path = compute_pathological_score(sim_pd, cfg)

    # -------------------------
    # 5. Combine
    # -------------------------
    J = compute_combined_score(J_normal, J_path, gamma_normal=1.0, gamma_path=1.0)

    # -------------------------
    # 6. Log trial metrics
    # -------------------------
    trial.set_user_attr("J_normal", float(J_normal))
    trial.set_user_attr("J_path", float(J_path))

    return J


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    print("🔧 Starting Optuna paired δ-study for STN–GPe network...")

    # Path to store the study DB
    db_path = Path("stn_gp_optuna.db").absolute()
    storage = f"sqlite:///{db_path}"

    # Create study
    study = optuna.create_study(
        study_name="stn_gpe_paired_delta",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    print(f"📁 Study DB = {db_path}")

    # Run optimization
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Print results
    print("\n=== BEST TRIAL ===")
    best = study.best_trial
    print(f"Value: {best.value}")
    print("Params:", best.params)
    print("Attributes:", best.user_attrs)

    # Save best parameters to JSON
    best_params_path = Path("best_theta.json")
    with open(best_params_path, "w") as f:
        json.dump(best.params, f, indent=2)

    print(f"💾 Best parameters saved to {best_params_path}")


if __name__ == "__main__":
    main()
