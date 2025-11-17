# optuna_driver.py
# Driver for optimization studies involving the STN–GPe model.
#
# Responsibilities:
#   • Define optuna-compatible objective wrapper
#   • Sample parameter sets from Optuna trial
#   • Run a single simulation via sim_api.run_simulation()
#   • Feed results into objectives.stn_gpe_beta_objective
#   • Manage study, storage, and logging
#
# Non-responsibilities:
#   • Detailed simulation logic (delegated to sim_api)
#   • PSD, firing rate, or scoring math (delegated to objectives)


from __future__ import annotations

import optuna
from typing import Dict, Any, Optional

from stn_gp.optuna.sim_api import run_simulation   # YOU will define this wrapper inside sim_api.py
from stn_gp.optuna.objectives import (
    STNGPObjectiveConfig,
    stn_gpe_beta_objective,
)


# ---------------------------------------------------------------------
# Trial → Parameter dictionary
# ---------------------------------------------------------------------

def sample_stn_gpe_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample all parameters you want Optuna to tune.

    Only defines the *free parameters*. sim_api.run_simulation()
    decides how to insert these into the actual model/configs.

    YOU can change these as needed when the scientific design evolves.
    """
    return {
        # Example parameters to tune — change freely
        "ISTN": trial.suggest_float("ISTN", 0.0, 60.0),
        "gT":   trial.suggest_float("gT", 0.1, 2.0),
        "gAHP": trial.suggest_float("gAHP", 2.0, 20.0),

        # GPe adaptation strength
        "a_gpe": trial.suggest_float("a_gpe", 0.1, 6.0),
        "tauw_gpe": trial.suggest_float("tauw_gpe", 80.0, 400.0),

        # Synaptic weights — example ranges
        "w_stn_to_gpe": trial.suggest_float("w_stn_to_gpe", 0.1, 4.0),
        "w_gpe_to_stn": trial.suggest_float("w_gpe_to_stn", 0.1, 4.0),
    }


# ---------------------------------------------------------------------
# Optuna objective wrapper
# ---------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    obj_cfg: STNGPObjectiveConfig,
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Optuna objective function.  Converts trial → params,
    runs one simulation, evaluates scalar loss.

    Parameters
    ----------
    trial : optuna.Trial
        Provided by optuna.
    obj_cfg : STNGPObjectiveConfig
        Fixed objective configuration (targets, weights, etc.)
    runtime_cfg : dict or None
        Non-optimized config like sim duration, population sizes, seed, etc.

    Returns
    -------
    loss : float
        Scalar to minimize.
    """
    if runtime_cfg is None:
        runtime_cfg = {}

    # ---- 1) Sample free parameters from trial ----
    trial_params = sample_stn_gpe_params(trial)

    # ---- 2) Run one STN–GPe simulation ----
    # run_simulation() MUST return a dict containing:
    #   {
    #       "stn_pop_spikes": 1D array (T,),
    #       "gpe_pop_spikes": 1D array (T,),
    #       "stn_lfp":        1D array (T,),
    #       "dt_ms":          float,
    #       "n_stn":          int,
    #       "n_gpe":          int,
    #   }
    sim_out = run_simulation(
        trial_params=trial_params,
        runtime_cfg=runtime_cfg,
    )

    # ---- 3) Compute scalar loss using objectives.py ----
    loss = stn_gpe_beta_objective(
        stn_pop_spikes=sim_out["stn_pop_spikes"],
        gpe_pop_spikes=sim_out["gpe_pop_spikes"],
        stn_lfp=sim_out["stn_lfp"],
        dt_ms=sim_out["dt_ms"],
        cfg=obj_cfg,
        n_stn=sim_out["n_stn"],
        n_gpe=sim_out["n_gpe"],
    )

    # Optional: record metrics in trial.user_attrs
    trial.set_user_attr("loss", loss)

    return float(loss)


# ---------------------------------------------------------------------
# Study creation + running
# ---------------------------------------------------------------------

def run_optimization(
    study_name: str,
    storage: Optional[str],
    obj_cfg: STNGPObjectiveConfig,
    runtime_cfg: Optional[Dict[str, Any]] = None,
    n_trials: int = 50,
    direction: str = "minimize",
) -> optuna.Study:
    """
    Create or load an Optuna study and run optimization.

    Parameters
    ----------
    study_name : str
        Name of the study.
    storage : str or None
        Optuna storage URI, e.g. "sqlite:///optuna_stn_gpe.db"
        or None for in-memory study (not saved).
    obj_cfg : STNGPObjectiveConfig
        Objective configuration.
    runtime_cfg : dict or None
        Runtime config for sim_api (duration, dt, seed, etc.)
    n_trials : int
        Number of Optuna trials to run.
    direction : {'minimize','maximize'}
        Typically "minimize" for a loss.

    Returns
    -------
    study : optuna.Study
    """
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: optuna_objective(trial, obj_cfg, runtime_cfg),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study
