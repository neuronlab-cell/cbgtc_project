# JAX Models for CBGTC Simulation

High-performance JAX implementation of the STN-GPe-GPi circuit for studying phase-amplitude coupling (PAC) in Parkinson's disease and deep brain stimulation (DBS) effects.

**Author:** Kavin Nakkeeran  
**Lab:** Functional Neurosurgery Lab, Johns Hopkins University  
**Date:** December 2025

---

## Overview

This JAX implementation provides **10-1000x speedup** over pure Python, enabling:
- Large-scale network simulations (100K+ neurons)
- Rapid parameter optimization (Bayesian/evolutionary methods)
- GPU/TPU acceleration for cloud deployment
- Reproducible functional programming paradigm

**Key Innovation:** Sparse synaptic connectivity using connection lists instead of dense matrices, reducing memory from ~80 GB to ~50 MB for 100K neuron networks.

---

## Project Structure

```
jax_models/
├── stn_jax.py           # STN neurons (Hodgkin-Huxley)
├── adex_jax.py          # GPe/GPi neurons (Adaptive Exponential)
├── synapses_jax.py      # Sparse synaptic connectivity
├── noise_jax.py         # Ornstein-Uhlenbeck background noise
├── network_builder.py   # Assembles STN-GPe-GPi circuit
├── integrator.py        # Simulation loop
├── observables.py       # Metric computation (beta power, firing rates)
└── README.md            # This file
```

---

## Quick Start

### Installation

```bash
# CPU-only
pip install jax jaxlib

# GPU (CUDA 12)
pip install jax[cuda12]
```

### Run a Simulation

```python
from network_builder import build_network_state
from integrator import network_step
from observables import compute_all_metrics
import jax.numpy as jnp

# 1. Build network (100 STN, 200 GPe, 150 GPi neurons)
state, config = build_network_state(
    n_stn=100, n_gpe=200, n_gpi=150,
    dt_ms=0.025, seed=42
)

# 2. Run simulation (4000 steps = 100ms)
V_history = {'stn': [], 'gpe': [], 'gpi': []}
spikes_history = {'stn': [], 'gpe': [], 'gpi': []}

for i in range(4000):
    state, obs = network_step(state, config, i * 0.025)
    V_history['stn'].append(obs['V_stn'])
    spikes_history['stn'].append(obs['spikes_stn'])
    # ... (collect gpe, gpi similarly)

# 3. Compute metrics
observables = {
    'V_stn': jnp.stack(V_history['stn']),
    'spikes_stn': jnp.stack(spikes_history['stn']),
    # ... (add gpe, gpi)
}

metrics = compute_all_metrics(observables, dt_ms=0.025)
print(f"Beta power (STN): {metrics['beta_power']['stn']:.2e}")
print(f"Firing rate (STN): {metrics['firing_rates']['stn']:.1f} Hz")
```

---

## Module Details

### 1. Neuron Models

#### **STN Neurons** (`stn_jax.py`)
- **Model:** Modified Hodgkin-Huxley with 7 ionic currents
- **Currents:** I_Na, I_K, I_L, I_T (T-type Ca²⁺), I_CaH (high-voltage Ca²⁺), I_AHP, I_H (pacemaker)
- **Validation:** Matches Python version within 0.05 mV
- **Performance:** 3,757x faster than Python loop

```python
from stn_jax import create_vectorized_stn, default_stn_params, create_population_state

# Create 1000 STN neurons
stn_step = create_vectorized_stn(compile=True)
params = default_stn_params()
states = create_population_state(1000, heterogeneity=0.05)

# Step forward
I_ext = jnp.zeros(1000)
I_syn = jnp.zeros(1000)
new_states, (V, spikes) = stn_step(states, params, 0.025, I_ext, I_syn, 0.0)
```

#### **GPe/GPi Neurons** (`adex_jax.py`)
- **Model:** Adaptive Exponential Integrate-and-Fire
- **Variants:** 
  - GPe: Irregular pacemaker (40-55 Hz, CV > 0.3)
  - GPi: Regular pacemaker (60-70 Hz, CV < 0.2)
- **Features:** Spike-frequency adaptation, absolute refractory period

```python
from adex_jax import create_vectorized_adex, default_adex_params_gpe

gpe_step = create_vectorized_adex(compile=True)
params = default_adex_params_gpe()
states = create_population_state(2000, cell_type='gpe')

new_states, (V, spikes) = gpe_step(states, params, 0.025, I_ext, I_syn, 0.0)
```

---

### 2. Synaptic Connectivity (`synapses_jax.py`)

**Sparse Connection List Representation:**

Instead of dense matrices (n_post × n_pre), stores only actual synapses:
- `pre_ids`: Presynaptic neuron indices
- `post_ids`: Postsynaptic neuron indices  
- `weights`: Synaptic strengths
- `delays_steps`: Axonal delays (in timesteps)

**Memory Comparison (100K → 200K neurons @ 15% connectivity):**
- Dense: 80,000 MB (80 GB) ❌
- Sparse: 48 MB ✅ **(1,666x reduction!)**

```python
from synapses_jax import create_synapse_config, init_synapse_state, synapse_step

# Create STN → GPe connection
config = create_synapse_config(
    n_pre=10000, n_post=20000, p_connect=0.15,
    weight_mean=22.0, weight_cv=0.2, delay_ms=5.0,
    tau_decay_ms=3.0, E_rev_mV=0.0, dt_ms=0.025
)
state = init_synapse_state(config)

# Step synapses
spikes_stn = jnp.zeros(10000)  # Binary spike array
V_gpe = jnp.ones(20000) * -60.0  # Postsynaptic voltages

new_state, I_syn = synapse_step(state, config, spikes_stn, V_gpe)
# I_syn has shape (20000,) - current to each GPe neuron
```

---

### 3. Background Noise (`noise_jax.py`)

**Ornstein-Uhlenbeck Process:** Temporally correlated Gaussian noise

Parameters:
- `tau_ms`: Correlation time (5-10 ms typical)
- `mu`: Mean current (µA/cm² for HH, pA for AdEx)
- `sigma`: Standard deviation

```python
from noise_jax import create_ou_for_population, ou_step

# STN background drive
config, state = create_ou_for_population(
    n_neurons=10000, dt_ms=0.025,
    tau_ms=8.0, mu=1.8, sigma=0.15
)

# Step noise
new_state, I_noise = ou_step(state, config)
# I_noise has shape (10000,) - current to each neuron
```

---

### 4. Network Assembly (`network_builder.py`)

Creates the full STN-GPe-GPi circuit with:
- 4 synaptic projections (STN→GPe, GPe→STN, STN→GPi, GPe→GPi)
- Background noise for each population
- Neuron parameters (healthy state defaults)

**Default Connectivity:**
| Projection | p_connect | Delay (ms) | Type |
|------------|-----------|------------|------|
| STN → GPe  | 0.15      | 5.0        | Excitatory (glutamate) |
| GPe → STN  | 0.07      | 8.0        | Inhibitory (GABA) |
| STN → GPi  | 0.30      | 5.0        | Excitatory |
| GPe → GPi  | 0.05      | 5.0        | Inhibitory |

```python
from network_builder import build_network_state

state, config = build_network_state(
    n_stn=1000, n_gpe=2000, n_gpi=1500,
    dt_ms=0.025, seed=42
)

# state: Network state (all dynamic variables)
# config: Network configuration (fixed parameters)
```

---

### 5. Integration (`integrator.py`)

Steps the entire network forward by one timestep.

**Order of operations:**
1. Update noise → get I_noise
2. Update synapses (with previous spikes) → get I_syn
3. Step neurons (with I_noise + I_syn) → get V, spikes
4. Return new state + observables

```python
from integrator import network_step

state, config = build_network_state(...)

for t in range(n_steps):
    state, obs = network_step(state, config, t * dt)
    # obs contains: V_stn, V_gpe, V_gpi, spikes_stn, spikes_gpe, spikes_gpi
```

**Note:** First call is slow (JIT compilation), subsequent calls are fast.

---

### 6. Observables (`observables.py`)

Computes metrics from simulation data:

**Firing Rates:**
```python
from observables import compute_firing_rates

spikes = {
    'stn': spike_array,  # shape (n_steps, n_neurons)
    'gpe': ...,
    'gpi': ...
}
rates = compute_firing_rates(spikes, dt_ms=0.025)
# Returns: {'stn': X Hz, 'gpe': Y Hz, 'gpi': Z Hz}
```

**Beta Power (13-30 Hz):**
```python
from observables import compute_beta_power

beta = compute_beta_power(V_trace, dt_ms=0.025, freq_range=(13, 30))
# V_trace shape: (n_steps, n_neurons)
# Returns: Scalar power in beta band
```

**All Metrics:**
```python
from observables import compute_all_metrics

observables = {
    'V_stn': V_stn_history,  # (n_steps, n_neurons)
    'spikes_stn': spikes_stn_history,
    # ... (gpe, gpi)
}

metrics = compute_all_metrics(observables, dt_ms=0.025)
# Returns: {
#   'firing_rates': {'stn': ..., 'gpe': ..., 'gpi': ...},
#   'beta_power': {'stn': ..., 'gpe': ..., 'gpi': ...},
#   'mean_V': {'stn': ..., 'gpe': ..., 'gpi': ...}
# }
```

---

## Performance Benchmarks

### STN Neurons (CPU - Intel Xeon)

| Neurons | Time/step | Speedup vs Python |
|---------|-----------|-------------------|
| 10      | 0.09 ms   | 10x               |
| 100     | 0.23 ms   | 42x               |
| 1,000   | 1.1 ms    | 300x              |
| 10,000  | 3.3 ms    | **3,757x**        |

### Network Simulation

| Configuration | Synapses | Step Time (CPU) | Notes |
|---------------|----------|-----------------|-------|
| 10-20-15      | 104      | 0.3 ms          | After JIT warmup |
| 100-200-150   | 10,400   | 5 ms            | First call (compile) |
| 1K-2K-1.5K    | ~1M      | ~50 ms          | Estimated |

**GPU Performance:** Expected **10-100x faster** than CPU (not yet benchmarked).

---

## Design Principles

### 1. Functional Programming
All functions are pure (no side effects):
```python
# ❌ Bad (mutation):
state.V += dt * dV

# ✅ Good (functional):
new_state = {**state, 'V': state['V'] + dt * dV}
```

### 2. State as PyTrees
State is nested dictionaries of JAX arrays:
```python
state = {
    'stn': {'V': ..., 'n': ..., 'h': ...},
    'gpe': {'V': ..., 'w': ...},
    'synapses': {...},
    'noise': {...}
}
```

### 3. Explicit State Threading
State flows through the program explicitly:
```python
new_state, outputs = function(old_state, config, inputs)
```

### 4. JIT Compilation
Performance-critical functions are JIT-compiled:
```python
step_fn = jax.jit(network_step)
# First call: slow (compiles)
# Subsequent calls: fast (cached)
```

---

## Validation

All models validated against original Python implementations:

| Model | Voltage Match | Spike Timing | Status |
|-------|---------------|--------------|--------|
| STN   | < 0.05 mV     | Exact        | ✅     |
| AdEx  | < 0.003 mV    | Exact        | ✅     |
| Network | N/A         | Functional   | ✅     |

---

## Next Steps

### Immediate (Ready Now)
1. **Parameter Optimization:** Use Optuna to find parameters that produce:
   - Healthy state: Low beta, weak PAC
   - PD state: High beta (13-30 Hz), strong PAC
2. **Longer Simulations:** Run 1-10 seconds to observe oscillation emergence
3. **GPU Deployment:** Move to cloud GPU for 10-100x speedup

### Near-Term (1-2 weeks)
1. **LIF Neurons:** Add cortex, thalamus, striatum (simple integrate-and-fire)
2. **Full CBGTC Loop:** Assemble complete circuit
3. **PAC Computation:** Add phase-amplitude coupling metric

### Long-Term (2-4 weeks)
1. **Multicompartment STN:** Extend to 20-100 compartments per neuron
2. **DBS Electrode:** 3D electric field model
3. **Spatial Arrangement:** 3D neuron positioning for realistic DBS

---

## Common Issues

**Q: JIT compilation takes forever on first call**  
A: This is normal. First call compiles the function (~5-30 sec depending on network size). Subsequent calls are fast. Compile once, reuse many times.

**Q: Out of memory on large networks**  
A: Use sparse synapses (`synapses_jax.py`), not dense matrices. Check you're not accidentally storing all history in memory.

**Q: Simulations are slow**  
A: Make sure you're using JIT-compiled functions. Call `create_vectorized_stn(compile=True)`. First step is slow, rest should be fast.

**Q: Random numbers aren't reproducible**  
A: JAX uses functional RNG. Always split keys: `key, subkey = random.split(key)`. Never reuse keys.

---

## File Sizes & Memory

**Small Network (100-200-150 neurons):**
- Code: ~50 KB total
- Network state: ~500 KB
- 100ms simulation data: ~5 MB

**Large Network (10K-20K-15K neurons):**
- Network state: ~50 MB (sparse synapses)
- 1 second simulation data: ~2 GB (if storing all timesteps)

**Tip:** Don't store every timestep. Downsample or compute metrics online.

---

## Citation

```
Kavin Nakkeeran
Functional Neurosurgery Lab
Johns Hopkins University
December 2025
```

---

## License

[Your license here]

---

## Contact

[Your contact here]

---

## Acknowledgments

Built with JAX for high-performance numerical computing on CPU/GPU/TPU.
