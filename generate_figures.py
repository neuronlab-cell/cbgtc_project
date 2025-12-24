"""
Publication Figure Generation for CBGTC HH Network
Author: Kavin Nakkeeran, Johns Hopkins University
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn, apply_params_to_config
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all
from jax_models.integrator import network_step
from jax import lax

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# SETUP
# =============================================================================

print("Building 1800-neuron network...")
state, config = build_network_state(400, 800, 600, 0.025, use_hh=True)

simulator = create_simulation_fn(config, n_steps=24000)

def create_pd_simulator(base_config, n_steps):
    @jax.jit
    def simulate(trial_params, init_state):
        config = apply_params_to_config(trial_params, base_config)
        syn_configs = dict(config['synapses'])
        for syn_name, mult_name in [
            ('stn_to_gpe', 'g_stn_gpe_mult'),
            ('gpe_to_stn', 'g_gpe_stn_mult'),
            ('stn_to_gpi', 'g_stn_gpi_mult'),
            ('gpe_to_gpi', 'g_gpe_gpi_mult'),
        ]:
            old_cfg = syn_configs[syn_name]
            new_weights = old_cfg.weights * trial_params.get(mult_name, 1.0)
            syn_configs[syn_name] = old_cfg._replace(weights=new_weights)
        config['synapses'] = syn_configs
        def step_fn(carry_state, t_idx):
            t_ms = t_idx * config['dt_ms']
            new_state, obs = network_step(carry_state, config, t_ms)
            return new_state, obs
        _, obs_history = lax.scan(step_fn, init=init_state, xs=jnp.arange(n_steps))
        return obs_history
    return simulate

pd_simulator = create_pd_simulator(config, 24000)

# Parameters (scaled for 1800 neurons)
healthy_params = {
    'ISTN': 140.0,
    'I_gpe': 3.379,
    'I_gpi': 2.188,
    'noise_stn_sigma': 0.996,
    'noise_gpe_sigma': 97.760,
    'noise_gpi_sigma': 69.678,
}

pd_params = {
    'ISTN': 80.0,
    'I_gpe': 0.672,
    'I_gpi': 2.430,
    'noise_stn_sigma': 4.333,
    'noise_gpe_sigma': 139.364,
    'noise_gpi_sigma': 109.012,
    'g_stn_gpe_mult': 1.592,
    'g_gpe_stn_mult': 0.182,
    'g_stn_gpi_mult': 1.975,
    'g_gpe_gpi_mult': 0.419,
}

# Run simulations
print("Running healthy simulation...")
obs_h = simulator(healthy_params, state)
obs_h['V_stn'].block_until_ready()

print("Running PD simulation...")
obs_pd = pd_simulator(pd_params, state)
obs_pd['V_stn'].block_until_ready()

# Compute metrics
metrics_h = compute_all_metrics(obs_h, 0.025, burn_steps=4000)
metrics_pd = compute_all_metrics(obs_pd, 0.025, burn_steps=4000)
beta_h = compute_beta_fraction_all(obs_h, 0.025, burn_steps=4000)
beta_pd = compute_beta_fraction_all(obs_pd, 0.025, burn_steps=4000)

print("Generating figures...")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_spike_times(spikes, dt_ms, burn_steps=4000):
    """Extract spike times from binary spike array."""
    spikes_valid = np.array(spikes[burn_steps:])
    times, neurons = [], []
    for t_idx in range(spikes_valid.shape[0]):
        spike_neurons = np.where(spikes_valid[t_idx])[0]
        for n in spike_neurons:
            times.append(t_idx * dt_ms)
            neurons.append(n)
    return np.array(times), np.array(neurons)

def compute_psd(V_trace, dt_ms, burn_steps=4000):
    """Compute power spectral density of LFP proxy."""
    valid_V = np.array(V_trace[burn_steps:])
    lfp = np.mean(valid_V, axis=1) - np.mean(np.mean(valid_V, axis=1))
    fft_vals = np.fft.rfft(lfp)
    freqs = np.fft.rfftfreq(len(lfp), d=dt_ms/1000)
    psd = np.abs(fft_vals)**2
    psd = psd / np.max(psd)
    return freqs, psd

# =============================================================================
# FIGURE 1: Raster Plots
# =============================================================================

fig1, axes = plt.subplots(2, 3, figsize=(14, 8))
fig1.suptitle('Figure 1: Raster Plots - Healthy vs Parkinsonian', fontsize=14, fontweight='bold')

populations = ['stn', 'gpe', 'gpi']
pop_labels = ['STN', 'GPe', 'GPi']
pop_colors = ['#E74C3C', '#3498DB', '#2ECC71']
n_neurons = [400, 800, 600]
t_start, t_end = 0, 200

for col, (pop, label, color, n_n) in enumerate(zip(populations, pop_labels, pop_colors, n_neurons)):
    times_h, neurons_h = get_spike_times(obs_h[f'spikes_{pop}'], 0.025)
    mask_h = (times_h >= t_start) & (times_h <= t_end)
    neuron_subsample = np.linspace(0, n_n-1, min(100, n_n), dtype=int)
    mask_neurons_h = np.isin(neurons_h, neuron_subsample)
    
    axes[0, col].scatter(times_h[mask_h & mask_neurons_h], neurons_h[mask_h & mask_neurons_h],
                         s=0.5, c=color, alpha=0.7)
    axes[0, col].set_xlim(t_start, t_end)
    axes[0, col].set_ylim(0, n_n)
    axes[0, col].set_title(f'{label} - Healthy\n({metrics_h["firing_rates"][pop]:.1f} Hz)', fontsize=11)
    if col == 0:
        axes[0, col].set_ylabel('Neuron #', fontsize=10)
    
    times_pd, neurons_pd = get_spike_times(obs_pd[f'spikes_{pop}'], 0.025)
    mask_pd = (times_pd >= t_start) & (times_pd <= t_end)
    mask_neurons_pd = np.isin(neurons_pd, neuron_subsample)
    
    axes[1, col].scatter(times_pd[mask_pd & mask_neurons_pd], neurons_pd[mask_pd & mask_neurons_pd],
                         s=0.5, c=color, alpha=0.7)
    axes[1, col].set_xlim(t_start, t_end)
    axes[1, col].set_ylim(0, n_n)
    axes[1, col].set_title(f'{label} - Parkinsonian\n({metrics_pd["firing_rates"][pop]:.1f} Hz)', fontsize=11)
    axes[1, col].set_xlabel('Time (ms)', fontsize=10)
    if col == 0:
        axes[1, col].set_ylabel('Neuron #', fontsize=10)

plt.tight_layout()
plt.savefig('results/fig1_raster_plots.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig1_raster_plots.pdf', bbox_inches='tight')
print("  Saved: fig1_raster_plots.png/pdf")

# =============================================================================
# FIGURE 2: Power Spectra
# =============================================================================

fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle('Figure 2: Power Spectra - Beta Band Emergence in Parkinsonism', fontsize=14, fontweight='bold')

for col, (pop, label, color) in enumerate(zip(populations, pop_labels, pop_colors)):
    freqs_h, psd_h = compute_psd(obs_h[f'V_{pop}'], 0.025)
    freqs_pd, psd_pd = compute_psd(obs_pd[f'V_{pop}'], 0.025)
    mask = freqs_h <= 50
    
    axes[col].semilogy(freqs_h[mask], psd_h[mask], 'b-', label='Healthy', linewidth=1.5, alpha=0.8)
    axes[col].semilogy(freqs_pd[mask], psd_pd[mask], 'r-', label='Parkinsonian', linewidth=1.5, alpha=0.8)
    axes[col].axvspan(13, 30, alpha=0.2, color='orange', label='Beta (13-30 Hz)')
    axes[col].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[col].set_ylabel('Normalized Power', fontsize=10)
    axes[col].set_title(f'{label}\nBeta: {beta_h[pop]*100:.1f}% → {beta_pd[pop]*100:.1f}%', fontsize=11)
    axes[col].legend(fontsize=8)
    axes[col].set_xlim(0, 50)
    axes[col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/fig2_power_spectra.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig2_power_spectra.pdf', bbox_inches='tight')
print("  Saved: fig2_power_spectra.png/pdf")

# =============================================================================
# FIGURE 3: Firing Rate Comparison
# =============================================================================

fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Figure 3: Firing Rates and Beta Power - Healthy vs Parkinsonian', fontsize=14, fontweight='bold')

x = np.arange(3)
width = 0.35

rates_h = [metrics_h['firing_rates']['stn'], metrics_h['firing_rates']['gpe'], metrics_h['firing_rates']['gpi']]
rates_pd = [metrics_pd['firing_rates']['stn'], metrics_pd['firing_rates']['gpe'], metrics_pd['firing_rates']['gpi']]

bars1 = axes[0].bar(x - width/2, rates_h, width, label='Healthy', color='steelblue', edgecolor='black')
bars2 = axes[0].bar(x + width/2, rates_pd, width, label='Parkinsonian', color='firebrick', edgecolor='black')
axes[0].set_ylabel('Firing Rate (Hz)', fontsize=11)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['STN', 'GPe', 'GPi'], fontsize=11)
axes[0].legend(fontsize=10)
axes[0].set_title('A. Firing Rates', fontsize=12, fontweight='bold')

for bar, val in zip(bars1, rates_h):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, rates_pd):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

beta_vals_h = [beta_h['stn']*100, beta_h['gpe']*100, beta_h['gpi']*100]
beta_vals_pd = [beta_pd['stn']*100, beta_pd['gpe']*100, beta_pd['gpi']*100]

bars3 = axes[1].bar(x - width/2, beta_vals_h, width, label='Healthy', color='steelblue', edgecolor='black')
bars4 = axes[1].bar(x + width/2, beta_vals_pd, width, label='Parkinsonian', color='firebrick', edgecolor='black')
axes[1].set_ylabel('Beta Power (%)', fontsize=11)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['STN', 'GPe', 'GPi'], fontsize=11)
axes[1].legend(fontsize=10)
axes[1].set_title('B. Beta Band Power (13-30 Hz)', fontsize=12, fontweight='bold')

for bar, val in zip(bars3, beta_vals_h):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars4, beta_vals_pd):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/fig3_firing_rates_beta.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig3_firing_rates_beta.pdf', bbox_inches='tight')
print("  Saved: fig3_firing_rates_beta.png/pdf")

# =============================================================================
# FIGURE 4: Network Schematic
# =============================================================================

fig4, axes = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle('Figure 4: Basal Ganglia Network - Synaptic Changes in Parkinsonism', fontsize=14, fontweight='bold')

def draw_network(ax, title, synaptic_weights):
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    positions = {'STN': (0, 1), 'GPe': (-1, 0), 'GPi': (1, 0)}
    colors = {'STN': '#E74C3C', 'GPe': '#3498DB', 'GPi': '#2ECC71'}
    
    for name, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.4, color=colors[name], ec='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold', color='white', zorder=11)
    
    connections = [
        ('STN', 'GPe', 'excitatory', synaptic_weights['stn_gpe']),
        ('GPe', 'STN', 'inhibitory', synaptic_weights['gpe_stn']),
        ('STN', 'GPi', 'excitatory', synaptic_weights['stn_gpi']),
        ('GPe', 'GPi', 'inhibitory', synaptic_weights['gpe_gpi']),
    ]
    
    for src, tgt, conn_type, weight in connections:
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/dist, dy/dist
        
        x1_arr, y1_arr = x1 + dx * 0.45, y1 + dy * 0.45
        x2_arr, y2_arr = x2 - dx * 0.45, y2 - dy * 0.45
        
        color = 'green' if conn_type == 'excitatory' else 'red'
        linewidth = 2 + weight * 2
        
        ax.annotate('', xy=(x2_arr, y2_arr), xytext=(x1_arr, y1_arr),
                   arrowprops=dict(arrowstyle='->', color=color, lw=linewidth))
        
        mid_x, mid_y = (x1_arr + x2_arr)/2, (y1_arr + y2_arr)/2
        offset_x = -0.2 if src == 'GPe' and tgt == 'STN' else 0.15
        offset_y = 0.15 if src == 'STN' else -0.15
        ax.text(mid_x + offset_x, mid_y + offset_y, f'{weight:.2f}x', fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.plot([], [], 'g-', linewidth=3, label='Excitatory')
    ax.plot([], [], 'r-', linewidth=3, label='Inhibitory')
    ax.legend(loc='lower center', fontsize=9)

healthy_weights = {'stn_gpe': 1.0, 'gpe_stn': 1.0, 'stn_gpi': 1.0, 'gpe_gpi': 1.0}
pd_weights = {'stn_gpe': 1.59, 'gpe_stn': 0.18, 'stn_gpi': 1.98, 'gpe_gpi': 0.42}

draw_network(axes[0], 'A. Healthy State', healthy_weights)
draw_network(axes[1], 'B. Parkinsonian State', pd_weights)

axes[1].annotate('82% reduction\n(KEY for beta)', xy=(-0.5, 0.6), fontsize=10,
                 ha='center', color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('results/fig4_network_schematic.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig4_network_schematic.pdf', bbox_inches='tight')
print("  Saved: fig4_network_schematic.png/pdf")

# =============================================================================
# FIGURE 5: LFP Traces
# =============================================================================

fig5, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig5.suptitle('Figure 5: GPe Local Field Potential - Beta Oscillations', fontsize=14, fontweight='bold')

lfp_h = np.mean(np.array(obs_h['V_gpe'][4000:]), axis=1)
lfp_pd = np.mean(np.array(obs_pd['V_gpe'][4000:]), axis=1)
t = np.arange(len(lfp_h)) * 0.025
t_start, t_end = 0, 300
mask = (t >= t_start) & (t <= t_end)

axes[0].plot(t[mask], lfp_h[mask], 'b-', linewidth=0.8)
axes[0].set_ylabel('LFP (mV)', fontsize=10)
axes[0].set_title(f'Healthy - Beta: {beta_h["gpe"]*100:.1f}%', fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t[mask], lfp_pd[mask], 'r-', linewidth=0.8)
axes[1].set_xlabel('Time (ms)', fontsize=10)
axes[1].set_ylabel('LFP (mV)', fontsize=10)
axes[1].set_title(f'Parkinsonian - Beta: {beta_pd["gpe"]*100:.1f}%', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/fig5_lfp_traces.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig5_lfp_traces.pdf', bbox_inches='tight')
print("  Saved: fig5_lfp_traces.png/pdf")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("FIGURE GENERATION COMPLETE")
print("="*60)
print("Generated figures in results/ folder:")
print("  1. fig1_raster_plots.png/pdf")
print("  2. fig2_power_spectra.png/pdf")
print("  3. fig3_firing_rates_beta.png/pdf")
print("  4. fig4_network_schematic.png/pdf")
print("  5. fig5_lfp_traces.png/pdf")
