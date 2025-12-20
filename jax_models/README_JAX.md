# JAX Models for CBGTC Simulation

This directory contains JAX implementations of neuron models for high-performance simulation of the cortico-basal ganglia-thalamo-cortical (CBGTC) loop.

## Overview

JAX implementations provide:
- **10-1000x speedup** over pure Python through JIT compilation
- **Efficient vectorization** across neuron populations using `jax.vmap`
- **GPU/TPU compatibility** for scaling to 100K+ neurons
- **Functional programming** paradigm for reliability and composability

## Current Models

### STN (Subthalamic Nucleus) - `stn_jax.py`

Modified Hodgkin-Huxley model with 7 ionic currents:
- Fast Na+ (I_Na): Action potential upstroke
- Delayed rectifier K+ (I_K): Repolarization
- Leak (I_L): Resting potential
- T-type Ca2+ (I_T): Burst firing
- High-voltage Ca2+ (I_CaH): Calcium influx
- AHP (I_AHP): Afterhyperpolarization
- H-current (I_H): Pacemaking

**Performance:**
- Single neuron: ~0.001 ms/step (JIT compiled)
- 10,000 neurons: ~3 ms/step on CPU
- **3,757x faster** than Python loop implementation

**Status:** ✅ Validated against original Python implementation

## Installation

```bash
# CPU-only (works everywhere)
pip install jax jaxlib

# GPU (CUDA 12)
pip install jax[cuda12]
```

## Quick Start

### Single Neuron

```python
from stn_jax import stn_step, default_stn_params, default_stn_state

# Initialize
state = default_stn_state()
params = default_stn_params()

# Run one timestep
new_state, (V, spiked) = stn_step(
    state, params, 
    dt_ms=0.025, 
    I_ext=0.0, 
    I_syn=0.0, 
    t_ms=0.0
)

print(f"Voltage: {V:.2f} mV, Spiked: {spiked}")
```

### Population (Vectorized)

```python
from stn_jax import create_vectorized_stn, create_population_state
import jax.numpy as jnp

# Create 100 neurons
n_neurons = 100
states = create_population_state(n_neurons, heterogeneity=0.05)
stn_pop = create_vectorized_stn(compile=True)

# Inputs (one per neuron)
I_ext = jnp.zeros(n_neurons)
I_syn = jnp.zeros(n_neurons)

# Step all neurons simultaneously
new_states, (V_array, spike_array) = stn_pop(
    states, params, 0.025, I_ext, I_syn, 0.0
)

print(f"Spikes: {jnp.sum(spike_array)} / {n_neurons}")
```

## Validation

All JAX models have been validated against original Python implementations:
- Voltage traces match within 0.05 mV over 100ms
- Spike times match exactly
- Gate variables match to 1e-7 precision

See `validate_jax_vs_python.py` for validation code.

## File Structure

```
jax_models/
├── README.md              # This file
├── stn_jax.py            # STN neuron model
├── adex_jax.py           # AdEx model (GPe/GPi) [TODO]
├── synapses_jax.py       # Synaptic connectivity [TODO]
└── tests/
    ├── test_stn.py       # Unit tests for STN
    └── test_vectorization.py
```

## Performance Benchmarks

### STN Neuron (CPU - Intel Xeon)

| Neurons | Time/step | Speedup vs Python |
|---------|-----------|-------------------|
| 1       | 0.001 ms  | ~10x              |
| 100     | 0.23 ms   | ~42x              |
| 1,000   | 1.1 ms    | ~300x             |
| 10,000  | 3.3 ms    | ~3,700x           |

### Scaling Efficiency

Time per neuron decreases with population size (better cache usage):
- 10 neurons: 87 µs/neuron
- 100 neurons: 2.3 µs/neuron (38x improvement!)
- 10,000 neurons: 0.33 µs/neuron (262x improvement!)

## Design Principles

### 1. Pure Functions
All neuron step functions are pure (no side effects):
```python
new_state, outputs = neuron_step(state, params, inputs)
```

### 2. Immutable State
State is never modified in-place:
```python
# ✗ BAD (mutation)
state['V'] += dt * dV_dt

# ✓ GOOD (functional)
V_new = state['V'] + dt * dV_dt
new_state = {**state, 'V': V_new}
```

### 3. Vectorization
Use `jax.vmap` to map single-neuron functions over populations:
```python
neuron_pop = jax.vmap(neuron_step, in_axes=(0, None, None, 0, 0, None))
```

### 4. JIT Compilation
Compile performance-critical functions:
```python
neuron_fast = jax.jit(neuron_step)
```

## Roadmap

- [x] STN neuron (single compartment)
- [ ] STN neuron (multi-compartment with DBS)
- [ ] AdEx neuron (GPe/GPi)
- [ ] LIF neuron (cortex, thalamus)
- [ ] Sparse synaptic connectivity
- [ ] Delay buffers (optimized)
- [ ] Full network integration
- [ ] GPU benchmarks

## Citation

If you use this code, please cite:

```
Kavin Nakkeeran
Functional Neurosurgery Lab
Johns Hopkins University
December 2025
```

## License

[Your license here]

## Contact

[Your contact info]
