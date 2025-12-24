# CBGTC Project: Hodgkin-Huxley Network Optimization Report

**Author:** Kavin Nakkeeran  
**Affiliation:** Functional Neurosurgery Lab, Johns Hopkins University  
**Date:** December 2025

## Executive Summary

This report documents the implementation and optimization of a JAX-accelerated basal ganglia network model using Hodgkin-Huxley (HH) neurons. Using Optuna's CMA-ES optimizer, we successfully reproduced healthy and Parkinsonian firing patterns, with the key finding that **reduced GPe→STN inhibition (82% reduction) enables pathological beta oscillations, increasing GPe beta power from 0% (healthy) to 30% (Parkinsonian)**.

## Key Results

### Healthy vs Parkinsonian Comparison

| Metric | Healthy | Parkinsonian | 
|--------|---------|--------------|
| STN Rate | 19.7 Hz | 27.6 Hz |
| GPe Rate | 66.3 Hz | 36.8 Hz |
| GPi Rate | 78.0 Hz | 84.3 Hz |
| STN CV | 0.22 | 0.28 |
| GPe CV | 0.27 | 0.40 |
| **GPe Beta** | **0.0%** | **30.0%** |
| **STN Beta** | **7.7%** | **18.0%** |

### Synaptic Changes in Parkinsonism

| Pathway | Healthy | PD | Change |
|---------|---------|-----|--------|
| STN→GPe | 1.0x | 1.59x | ↑ 59% |
| **GPe→STN** | **1.0x** | **0.18x** | **↓ 82%** ⭐ |
| STN→GPi | 1.0x | 1.98x | ↑ 98% |
| GPe→GPi | 1.0x | 0.42x | ↓ 58% |

**Critical Finding:** Reduced GPe→STN inhibition (0.18x) is the primary mechanism enabling beta oscillations.

## Optimized Parameters

### Healthy State (6 parameters)
```python
healthy_params = {
    'ISTN': 117.025,
    'I_gpe': 3.379,
    'I_gpi': 2.188,
    'noise_stn_sigma': 0.996,
    'noise_gpe_sigma': 97.760,
    'noise_gpi_sigma': 69.678,
}
```

### Parkinsonian State (10 parameters)
```python
pd_params = {
    # Intrinsic
    'ISTN': 66.470,
    'I_gpe': 0.672,
    'I_gpi': 2.430,
    'noise_stn_sigma': 4.333,
    'noise_gpe_sigma': 139.364,
    'noise_gpi_sigma': 109.012,
    # Synaptic multipliers
    'g_stn_gpe_mult': 1.592,
    'g_gpe_stn_mult': 0.182,   # CRITICAL
    'g_stn_gpi_mult': 1.975,
    'g_gpe_gpi_mult': 0.419,
}
```

## Usage

### Running a Simulation
```python
from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all

# Build 450-neuron HH network
state, config = build_network_state(100, 200, 150, 0.025, use_hh=True)
simulator = create_simulation_fn(config, n_steps=16000)

# Run with healthy params
obs = simulator(healthy_params, state)
metrics = compute_all_metrics(obs, 0.025, burn_steps=4000)
beta = compute_beta_fraction_all(obs, 0.025, burn_steps=4000)
```

### Running Optimization
```bash
# Healthy state (6 params, ~10 min)
python3 optuna_hh_healthy.py

# Parkinsonian state (10 params, ~20 min)
python3 optuna_hh_parkinsonian.py
```

## Model Details

- **STN:** Modified Hodgkin-Huxley with T-type calcium
- **GPe/GPi:** Rubin-Terman (2004) with rebound bursting
- **Network:** 100 STN, 200 GPe, 150 GPi neurons
- **Timestep:** 0.025 ms
- **Simulation:** 400 ms (100 ms burn-in)

## References

- Rubin JE, Terman D (2004). J Comput Neurosci 16:211-235.
- Bergman H et al. (1994). Science 265:1346-1348.
- Brown P (2003). Mov Disord 18:357-363.
