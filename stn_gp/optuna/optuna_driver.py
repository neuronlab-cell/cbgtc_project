# optuna_driver.py
# Main orchestration script for paired δ optimization using Optuna.
#
# Responsibilities:
#   • Define parameter search space (θ)
#   • Run two simulations per trial: δ=0 (normal), δ=1 (PD-like)
#   • Score both using objectives.py (normal + PD + δ-contrast)
#   • Save best parameters + store study in a SQLite DB
#
# Usage (from repo root, with venv active):
#   cd ~/cbgtc_project
#   source .venv/bin/activate
#   python -m stn_gp.optuna.optuna_driver
#
# To do a quick smoke test, edit n_trials in main() to something small (e.g., 5).

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import optuna

from stn_gp.optuna.sim_api import run_simulation
from stn_gp.optuna.objectives import (
    PairedObjectiveConfig,
    compute_combined_score,
    _analyze_condition,  # internal but useful to get metrics
)


# ============================================================
# PARAMETER SEARCH SPACE (θ)
# ============================================================

def sample_theta(trial: optuna.Trial) -> Dict:
    """
    Sample base parameters θ from biologically plausible ranges.
    These are baseline (δ = 0) values before dopamine modulation.
    """

    theta: Dict = {}

    # -------------------------
    # Synaptic weights
    # -------------------------
    theta["stn_to_gpe_mean"] = trial.suggest_float(
        "stn_to_gpe_mean",
        20.0, 70.0,  # pA; moderate–strong AMPA
    )

    theta["gpe_to_stn_mean"] = trial.suggest_float(
        "gpe_to_stn_mean",
        0.05, 0.25,  # μA/cm²; realistic inhibitory range
    )

    # -------------------------
    # Intrinsic drives
    # -------------------------
    theta["stn_ISTN"] = trial.suggest_float(
        "stn_ISTN",
        15.0, 30.0,  # µA/cm²; modest–high tonic drive
    )

    theta["gpe_I_baseline"] = trial.suggest_float(
        "gpe_I_baseline",
        120.0, 230.0,  # pA baseline AdEx current
    )

    # -------------------------
    # Delays (ms)
    # -------------------------
    theta["delay_stn_to_gpe_ms"] = trial.suggest_float(
        "delay_stn_to_gpe_ms",
        2.0, 5.0,  # literature ~2–4 ms
    )
    theta["delay_gpe_to_stn_ms"] = trial.suggest_float(
        "delay_gpe_to_stn_ms",
        5.0, 10.0,  # literature ~5–8 ms
    )

    # -------------------------
    # Noise (OU sigma, arbitrary units)
    # -------------------------
    theta["noise_sigma_stn"] = trial.suggest_float(
        "noise_sigma_stn",
        0.5, 2.0,
    )
    theta["noise_sigma_gpe"] = trial.suggest_float(
        "noise_sigma_gpe",
        0.3, 1.5,
    )

    # -------------------------
    # Population sizes
    # -------------------------
    theta["n_stn"] = trial.suggest_int("n_stn", 50, 80)
    theta["n_gpe"] = trial.suggest_int("n_gpe", 100, 180)

    return theta


# ============================================================
# OBJECTIVE FUNCTION FOR OPTUNA
# ============================================================

def objective(trial: optuna.Trial) -> float:
    """
    Full paired-δ objective:
      • Simulate δ = 0 (normal)
      • Simulate δ = 1 (PD-like)
      • Score both (normal / PD) with objectives.py
      • Add δ-contrast term (beta & sharpness differences)
      • Return combined score for maximization
    """

    # 1) Sample base parameters
    theta = sample_theta(trial)

    # 2) Objective configuration (can later be made trial-dependent)
    cfg = PairedObjectiveConfig()

    # 3) Run simulations for δ = 0 (normal) and δ = 1 (PD)
    sim_norm = run_simulation(
        theta,
        delta=0.0,
        t_total_s=4.0,
        burn_in_s=1.0,
        dt_ms=0.025,
    )
    sim_pd = run_simulation(
        theta,
        delta=1.0,
        t_total_s=4.0,
        burn_in_s=1.0,
        dt_ms=0.025,
    )

    # 4) Analyze each condition to get scores + metrics
    normal_score, normal_metrics = _analyze_condition(sim_norm, cfg, mode="normal")
    path_score, path_metrics = _analyze_condition(sim_pd, cfg, mode="pd")

    # 5) Combine with explicit δ-contrast (in compute_combined_score)
    J = compute_combined_score(
        normal_score,
        path_score,
        gamma_normal=1.0,
        gamma_path=1.0,
        cfg=cfg,
        normal_metrics=normal_metrics,
        path_metrics=path_metrics,
    )

    # 6) Log useful attributes for later inspection
    trial.set_user_attr("normal_score", float(normal_score))
    trial.set_user_attr("path_score", float(path_score))

    # store a few key metrics too
    trial.set_user_attr("normal_stn_rate", normal_metrics["stn_rate"])
    trial.set_user_attr("normal_gpe_rate", normal_metrics["gpe_rate"])
    trial.set_user_attr("normal_beta_ratio", normal_metrics["beta_ratio"])
    trial.set_user_attr("normal_beta_sharp", normal_metrics["beta_sharp"])

    trial.set_user_attr("pd_stn_rate", path_metrics["stn_rate"])
    trial.set_user_attr("pd_gpe_rate", path_metrics["gpe_rate"])
    trial.set_user_attr("pd_beta_ratio", path_metrics["beta_ratio"])
    trial.set_user_attr("pd_beta_sharp", path_metrics["beta_sharp"])

    return float(J)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    print("🔧 Starting Optuna paired-δ study for STN–GPe network...")

    # SQLite DB for persistent study storage
    db_path = Path("stn_gp_optuna.db").absolute()
    storage = f"sqlite:///{db_path}"

    # Create or load study
    study = optuna.create_study(
        study_name="stn_gpe_paired_delta",
        direction="maximize",  # higher score is better
        storage=storage,
        load_if_exists=True,
    )

    print(f"📁 Study DB = {db_path}")

    # Number of trials
    # For quick validation: use n_trials=5
    # For more serious search: bump to 50–200 depending on budget
    n_trials = 50

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print best trial summary
    print("\n=== BEST TRIAL ===")
    best = study.best_trial
    print(f"Value: {best.value}")
    print("Params:", best.params)
    print("Attributes:", best.user_attrs)

    # Save best parameters to JSON for later reuse
    best_params_path = Path("best_theta.json")
    with open(best_params_path, "w") as f:
        json.dump(best.params, f, indent=2)

    print(f"💾 Best parameters saved to {best_params_path}")


if __name__ == "__main__":
    main()
