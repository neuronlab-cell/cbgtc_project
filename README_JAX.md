# CBGTC Project: JAX-Accelerated Basal Ganglia Network Model

**Author:** Kavin Nakkeeran  
**Affiliation:** Functional Neurosurgery Lab, Johns Hopkins University  
**Date:** December 2025

---

## 🧠 Overview

A high-performance computational model of the cortico-basal ganglia-thalamo-cortical (CBGTC) circuit for studying Parkinson's disease and deep brain stimulation. The model uses **Hodgkin-Huxley neurons** and achieves **1000x speedup** over traditional Python implementations through JAX GPU acceleration.

### Key Features

- **Biologically realistic**: Rubin-Terman HH neurons with T-type calcium currents for rebound bursting
- **GPU-accelerated**: JAX JIT compilation enables ~2s simulations for 1800 neurons
- **Automated optimization**: Optuna + CMA-ES for parameter fitting
- **Publication-ready**: Reproduces healthy and Parkinsonian firing patterns with beta oscillations

### Main Finding

**Reduced GPe→STN inhibition (82% reduction) is the primary mechanism enabling pathological beta oscillations in Parkinson's disease.**

---

## 🚀 Quick Start (No GPU Required!)

You don't need a local GPU - just a web browser! Use Google Cloud Platform's free tier or paid GPU instances.

### Option 1: Google Colab (Free, Limited)

1. Go to [Google Colab](https://colab.research.google.com)
2. Create new notebook
3. Runtime → Change runtime type → GPU
4. Run:

```python
!git clone https://github.com/neuronlab-cell/cbgtc_project.git
%cd cbgtc_project
!pip install jax[cuda12_pip] optuna scipy matplotlib -q
!python generate_figures.py
```

### Option 2: Google Cloud Platform (Recommended)

See [Cloud Setup Guide](#cloud-setup-guide) below.

---

## 📋 Requirements

### Python Version
- Python 3.9 - 3.11 (tested on 3.10)

### Core Dependencies

```
jax>=0.4.20
jaxlib>=0.4.20
optuna>=3.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

### For GPU Acceleration (Recommended)

```
# CUDA 12.x
jax[cuda12_pip]>=0.4.20

# Or CUDA 11.x
jax[cuda11_pip]>=0.4.20
```

### Installation

```bash
# Clone repository
git clone https://github.com/neuronlab-cell/cbgtc_project.git
cd cbgtc_project

# Install dependencies (CPU only)
pip install -r requirements.txt

# Or with GPU support (CUDA 12)
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install optuna scipy matplotlib
```

### requirements.txt

```
jax>=0.4.20
jaxlib>=0.4.20
optuna>=3.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

---

## 🖥️ Cloud Setup Guide

### Google Cloud Platform (GCP) Setup

#### Step 1: Create GCP Account

1. Go to [cloud.google.com](https://cloud.google.com)
2. Sign up (free $300 credit for new users)
3. Enable billing

#### Step 2: Create GPU VM

**Via Console:**

1. Go to Compute Engine → VM Instances
2. Click "Create Instance"
3. Configure:
   - **Name:** `cbgtc-gpu`
   - **Region:** `us-central1-a` (good GPU availability)
   - **Machine type:** `g2-standard-8` (8 vCPU, 32GB RAM)
   - **GPU:** Add GPU → NVIDIA L4 (1 GPU)
   - **Boot disk:** 
     - Click "Change"
     - Select "Deep Learning on Linux"
     - Choose "Deep Learning VM with CUDA 12.1"
     - Size: 100 GB
   - **Firewall:** Allow HTTP/HTTPS

4. Click "Create"

**Via gcloud CLI:**

```bash
gcloud compute instances create cbgtc-gpu \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-cuda121-debian-11 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE
```

#### Step 3: Connect to VM

```bash
# SSH via gcloud
gcloud compute ssh cbgtc-gpu --zone=us-central1-a

# Or use the SSH button in GCP Console
```

#### Step 4: Setup Environment

```bash
# Verify GPU
nvidia-smi

# Clone repository
git clone https://github.com/neuronlab-cell/cbgtc_project.git
cd cbgtc_project

# Install JAX with CUDA support
pip install "jax[cuda12_pip]>=0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --break-system-packages

# Install other dependencies
pip install optuna scipy matplotlib --break-system-packages

# Verify JAX sees GPU
python3 -c "import jax; print(jax.devices())"
# Should output: [cuda(id=0)]
```

#### Step 5: Run Simulations

```bash
# Generate all figures
python3 generate_figures.py

# Run optimization
python3 optuna_hh_healthy.py

# Run statistical validation
python3 statistical_validation.py
```

#### Cost Estimate

| Resource | Cost/Hour | Notes |
|----------|-----------|-------|
| g2-standard-8 + L4 GPU | ~$0.70 | US regions |
| Storage (100GB) | ~$0.04/day | Standard SSD |

**Typical session:** 2-3 hours = ~$2-3

#### Stop VM When Done!

```bash
# From local terminal
gcloud compute instances stop cbgtc-gpu --zone=us-central1-a

# Or click "Stop" in GCP Console
```

---

## 📁 Project Structure

```
cbgtc_project/
├── jax_models/                    # Core neural models
│   ├── __init__.py
│   ├── stn_jax.py                 # STN Hodgkin-Huxley model
│   ├── gpe_gpi_hh.py              # Rubin-Terman GPe/GPi model
│   ├── adex_jax.py                # AdEx model (alternative)
│   ├── integrator.py              # Network integration
│   ├── network_builder.py         # Build network with connectivity
│   ├── synapses_jax.py            # Synaptic dynamics
│   └── noise_jax.py               # Ornstein-Uhlenbeck noise
│
├── optimization/                   # Simulation & metrics
│   ├── sim_jax.py                 # JIT-compiled simulator
│   └── metrics_jax.py             # Firing rates, CV, beta power
│
├── results/                        # Output files
│   ├── fig1_raster_plots.png/pdf
│   ├── fig2_power_spectra.png/pdf
│   ├── fig3_firing_rates_beta.png/pdf
│   ├── fig4_network_schematic.png/pdf
│   ├── fig5_lfp_traces.png/pdf
│   ├── fig6_statistical_validation.png/pdf
│   ├── fig7_dbs_effect.png/pdf
│   ├── hh_healthy_study.pkl
│   ├── hh_parkinsonian_beta_study.pkl
│   └── statistical_validation.pkl
│
├── docs/                           # Documentation
│   └── CBGTC_HH_OPTIMIZATION.md
│
├── optuna_hh_healthy.py           # 6-param healthy optimization
├── optuna_hh_parkinsonian.py      # 10-param PD optimization
├── generate_figures.py            # Publication figures
├── statistical_validation.py      # Multi-seed validation
├── dbs_simulation.py              # DBS proof-of-concept
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔬 Usage Guide

### Basic Simulation

```python
import sys
sys.path.insert(0, '.')

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all

# Build network (400 STN, 800 GPe, 600 GPi neurons)
state, config = build_network_state(
    n_stn=400, 
    n_gpe=800, 
    n_gpi=600, 
    dt_ms=0.025,      # 0.025ms timestep
    use_hh=True       # Use Hodgkin-Huxley (vs AdEx)
)

# Create JIT-compiled simulator
simulator = create_simulation_fn(config, n_steps=16000)  # 400ms

# Define parameters
params = {
    'ISTN': 140.0,           # STN drive current
    'I_gpe': 3.379,          # GPe applied current
    'I_gpi': 2.188,          # GPi applied current
    'noise_stn_sigma': 0.996,
    'noise_gpe_sigma': 97.760,
    'noise_gpi_sigma': 69.678,
}

# Run simulation
obs = simulator(params, state)
obs['V_stn'].block_until_ready()  # Wait for GPU

# Compute metrics
metrics = compute_all_metrics(obs, dt_ms=0.025, burn_steps=4000)
beta = compute_beta_fraction_all(obs, dt_ms=0.025, burn_steps=4000)

print(f"STN: {metrics['firing_rates']['stn']:.1f} Hz")
print(f"GPe: {metrics['firing_rates']['gpe']:.1f} Hz")
print(f"GPe Beta: {beta['gpe']*100:.1f}%")
```

### Running Optimization

```bash
# Healthy state (6 parameters, ~10 min)
python3 optuna_hh_healthy.py

# Parkinsonian state (10 parameters, ~20 min)
python3 optuna_hh_parkinsonian.py
```

### Custom Optuna Study

```python
import optuna
from optuna.samplers import CmaEsSampler

def objective(trial):
    params = {
        'ISTN': trial.suggest_float('ISTN', 80.0, 200.0),
        'I_gpe': trial.suggest_float('I_gpe', 1.0, 8.0),
        'I_gpi': trial.suggest_float('I_gpi', 1.0, 8.0),
        # ... more parameters
    }
    
    obs = simulator(params, state)
    metrics = compute_all_metrics(obs, 0.025, burn_steps=4000)
    
    # Define loss
    loss = (metrics['firing_rates']['stn'] - 20.0)**2
    return loss

study = optuna.create_study(
    direction='minimize',
    sampler=CmaEsSampler(seed=42)
)
study.optimize(objective, n_trials=500)
print(study.best_params)
```

### Generate Figures

```bash
python3 generate_figures.py
# Outputs: results/fig1-5.png/pdf

python3 statistical_validation.py
# Outputs: results/fig6_statistical_validation.png/pdf

python3 dbs_simulation.py
# Outputs: results/fig7_dbs_effect.png/pdf
```

---

## 📊 Key Results

### Healthy vs Parkinsonian (n=10 seeds)

| Metric | Healthy | Parkinsonian | p-value |
|--------|---------|--------------|---------|
| STN Rate | 22.6 ± 0.8 Hz | 37.6 ± 0.03 Hz | < 0.0001 |
| GPe Rate | 66.0 ± 0.1 Hz | 37.5 ± 0.1 Hz | < 0.0001 |
| GPi Rate | 78.1 ± 0.1 Hz | 90.4 ± 0.1 Hz | < 0.0001 |
| GPe CV | 0.27 ± 0.00 | 0.40 ± 0.00 | < 0.0001 |
| **GPe Beta** | **4.6 ± 1.7%** | **10.1 ± 2.7%** | **0.0001** |

### Synaptic Changes in Parkinsonism

| Pathway | Healthy | PD | Change |
|---------|---------|-----|--------|
| STN→GPe | 1.0x | 1.59x | +59% |
| **GPe→STN** | **1.0x** | **0.18x** | **-82%** |
| STN→GPi | 1.0x | 1.98x | +98% |
| GPe→GPi | 1.0x | 0.42x | -58% |

### DBS Effect

| Metric | PD (OFF) | PD + DBS | Change |
|--------|----------|----------|--------|
| GPe Beta | 11.6% | 7.4% | -36% |

---

## 🧪 Optimized Parameters

### Healthy State

```python
healthy_params = {
    'ISTN': 140.0,      # For 1800 neurons (scale-adjusted)
    'I_gpe': 3.379,
    'I_gpi': 2.188,
    'noise_stn_sigma': 0.996,
    'noise_gpe_sigma': 97.760,
    'noise_gpi_sigma': 69.678,
}
```

### Parkinsonian State

```python
pd_params = {
    # Intrinsic parameters
    'ISTN': 80.0,
    'I_gpe': 0.672,
    'I_gpi': 2.430,
    'noise_stn_sigma': 4.333,
    'noise_gpe_sigma': 139.364,
    'noise_gpi_sigma': 109.012,
    # Synaptic multipliers
    'g_stn_gpe_mult': 1.592,
    'g_gpe_stn_mult': 0.182,   # KEY: 82% reduction
    'g_stn_gpi_mult': 1.975,
    'g_gpe_gpi_mult': 0.419,
}
```

---

## ⚡ Performance Benchmarks

| Network Size | Neurons | Build Time | Sim Time (JIT) | Sim Time (cached) |
|--------------|---------|------------|----------------|-------------------|
| Small | 225 | 0.5s | 2.0s | 0.5s |
| Medium | 450 | 1.2s | 3.0s | 0.6s |
| Large | 1800 | 4.8s | 5.5s | 2.2s |

**Hardware:** NVIDIA L4 GPU, Google Cloud Platform

---

## 📚 References

### Model References

- Rubin JE, Terman D (2004). High frequency stimulation of the subthalamic nucleus eliminates pathological thalamic rhythmicity in a computational model. *J Comput Neurosci* 16:211-235.
- Gillies A, Willshaw D (2006). Membrane channel interactions underlying rat subthalamic projection neuron rhythmic and bursting activity. *J Neurophysiol* 95:2352-2365.

### Experimental Data

- Bergman H, Wichmann T, DeLong MR (1994). Reversal of experimental parkinsonism by lesions of the subthalamic nucleus. *Science* 265:1346-1348.
- Filion M, Tremblay L (1991). Abnormal spontaneous activity of globus pallidus neurons in monkeys with MPTP-induced parkinsonism. *Brain Res* 547:142-151.
- Brown P (2003). Oscillatory nature of human basal ganglia activity. *Mov Disord* 18:357-363.

### Software

- Bradbury J, et al. (2018). JAX: composable transformations of Python+NumPy programs.
- Akiba T, et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD 2019*.

---

## 🐛 Troubleshooting

### JAX doesn't see GPU

```bash
# Check CUDA
nvidia-smi

# Reinstall JAX with CUDA
pip uninstall jax jaxlib -y
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify
python3 -c "import jax; print(jax.devices())"
```

### NumPy/Matplotlib conflict

```bash
pip install "numpy<2" --break-system-packages
# Or
pip install --upgrade matplotlib --break-system-packages
```

### Out of GPU memory

Reduce network size:

```python
# Instead of 1800 neurons
state, config = build_network_state(100, 200, 150, 0.025, use_hh=True)
```

### Simulation produces NaN

This is usually due to numerical instability. The current code includes voltage clamping to prevent this. If you modify the model, ensure:

```python
V_new = jnp.clip(V_new, -100.0, 60.0)
```

---

## 📄 License

MIT License - See LICENSE file

---

## 🤝 Contact

- **Author:** Kavin Nakkeeran
- **Lab:** Functional Neurosurgery Lab, Johns Hopkins University
- **GitHub:** [neuronlab-cell/cbgtc_project](https://github.com/neuronlab-cell/cbgtc_project)

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{nakkeeran2025cbgtc,
  author = {Nakkeeran, Kavin},
  title = {CBGTC: JAX-Accelerated Basal Ganglia Network Model},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/neuronlab-cell/cbgtc_project}
}
```
