"""
Minimal PING (Pyramidal-Interneuron Network Gamma) Model in JAX
Author: Kavin Nakkeeran, Johns Hopkins University

Simple E-I network generating gamma oscillations (~40 Hz)
- E cells: Regular spiking (AdEx)
- I cells: Fast spiking (AdEx with different params)
"""

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import numpy as np
import time

# =============================================================================
# AdEx NEURON PARAMETERS
# =============================================================================

# Regular Spiking (E cells - Pyramidal)
E_PARAMS = {
    'C': 200.0,        # pF - membrane capacitance
    'gL': 10.0,        # nS - leak conductance
    'EL': -70.0,       # mV - leak reversal
    'VT': -50.0,       # mV - threshold
    'DeltaT': 2.0,     # mV - slope factor
    'Vr': -60.0,       # mV - reset voltage
    'Vcut': 20.0,      # mV - spike cutoff
    'a': 2.0,          # nS - subthreshold adaptation
    'b': 60.0,         # pA - spike-triggered adaptation
    'tau_w': 100.0,    # ms - adaptation time constant
}

# Fast Spiking (I cells - PV+ Interneurons)
I_PARAMS = {
    'C': 100.0,        # pF - smaller, faster
    'gL': 10.0,        # nS
    'EL': -70.0,       # mV
    'VT': -50.0,       # mV
    'DeltaT': 0.5,     # mV - sharper threshold (more FS-like)
    'Vr': -60.0,       # mV
    'Vcut': 20.0,      # mV
    'a': 0.0,          # nS - no subthreshold adaptation
    'b': 0.0,          # pA - no spike adaptation (FS)
    'tau_w': 50.0,     # ms
}

# Synaptic parameters
SYN_PARAMS = {
    'E_exc': 0.0,      # mV - AMPA reversal
    'E_inh': -75.0,    # mV - GABA_A reversal
    'tau_ampa': 2.0,   # ms - AMPA decay
    'tau_gaba': 8.0,   # ms - GABA_A decay (sets gamma frequency!)
}


def create_ping_config(n_e=200, n_i=50, dt=0.1, seed=42):
    """Create PING network configuration."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    
    p_ei, p_ie, p_ee, p_ii = 0.5, 0.5, 0.1, 0.5
    
    W_ei = (jax.random.uniform(keys[0], (n_i, n_e)) < p_ei).astype(jnp.float32)
    W_ie = (jax.random.uniform(keys[1], (n_e, n_i)) < p_ie).astype(jnp.float32)
    W_ee = (jax.random.uniform(keys[2], (n_e, n_e)) < p_ee).astype(jnp.float32)
    W_ii = (jax.random.uniform(keys[3], (n_i, n_i)) < p_ii).astype(jnp.float32)
    
    W_ee = W_ee.at[jnp.diag_indices(n_e)].set(0.0)
    W_ii = W_ii.at[jnp.diag_indices(n_i)].set(0.0)
    
    g_ei = 0.5 / jnp.sqrt(n_e)
    g_ie = 2.0 / jnp.sqrt(n_i)
    g_ee = 0.05 / jnp.sqrt(n_e)
    g_ii = 0.5 / jnp.sqrt(n_i)
    
    return {
        'n_e': n_e, 'n_i': n_i, 'dt': dt,
        'W_ei': W_ei * g_ei, 'W_ie': W_ie * g_ie,
        'W_ee': W_ee * g_ee, 'W_ii': W_ii * g_ii,
        'E_params': E_PARAMS, 'I_params': I_PARAMS, 'syn_params': SYN_PARAMS,
    }


def create_ping_state(config, seed=0):
    """Initialize network state."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2)
    n_e, n_i = config['n_e'], config['n_i']
    
    return {
        'V_e': config['E_params']['EL'] + jax.random.uniform(keys[0], (n_e,)) * 10,
        'V_i': config['I_params']['EL'] + jax.random.uniform(keys[1], (n_i,)) * 10,
        'w_e': jnp.zeros(n_e), 'w_i': jnp.zeros(n_i),
        's_e': jnp.zeros(n_e), 's_i': jnp.zeros(n_i),
    }


def ping_step(state, config, I_ext_e, I_ext_i, noise_key):
    """Single timestep of PING network."""
    dt = config['dt']
    E_p, I_p, S_p = config['E_params'], config['I_params'], config['syn_params']
    V_e, V_i = state['V_e'], state['V_i']
    w_e, w_i = state['w_e'], state['w_i']
    s_e, s_i = state['s_e'], state['s_i']
    
    k1, k2 = jax.random.split(noise_key)
    noise_e = jax.random.normal(k1, V_e.shape) * 50.0
    noise_i = jax.random.normal(k2, V_i.shape) * 20.0
    
    # Synaptic currents
    I_syn_e = -(config['W_ee'] @ s_e * (V_e - S_p['E_exc'])) - (config['W_ie'] @ s_i * (V_e - S_p['E_inh']))
    I_syn_i = -(config['W_ei'] @ s_e * (V_i - S_p['E_exc'])) - (config['W_ii'] @ s_i * (V_i - S_p['E_inh']))
    
    # AdEx dynamics
    exp_e = E_p['DeltaT'] * jnp.exp((V_e - E_p['VT']) / E_p['DeltaT'])
    exp_i = I_p['DeltaT'] * jnp.exp((V_i - I_p['VT']) / I_p['DeltaT'])
    
    dV_e = (E_p['gL'] * (E_p['EL'] - V_e + exp_e) - w_e + I_ext_e + noise_e + I_syn_e) / E_p['C']
    dV_i = (I_p['gL'] * (I_p['EL'] - V_i + exp_i) - w_i + I_ext_i + noise_i + I_syn_i) / I_p['C']
    dw_e = (E_p['a'] * (V_e - E_p['EL']) - w_e) / E_p['tau_w']
    dw_i = (I_p['a'] * (V_i - I_p['EL']) - w_i) / I_p['tau_w']
    
    V_e_new = V_e + dt * dV_e
    V_i_new = V_i + dt * dV_i
    w_e_new = w_e + dt * dw_e
    w_i_new = w_i + dt * dw_i
    
    s_e_new = s_e + dt * (-s_e / S_p['tau_ampa'])
    s_i_new = s_i + dt * (-s_i / S_p['tau_gaba'])
    
    # Spikes
    spike_e = V_e_new >= E_p['Vcut']
    spike_i = V_i_new >= I_p['Vcut']
    
    V_e_new = jnp.where(spike_e, E_p['Vr'], V_e_new)
    V_i_new = jnp.where(spike_i, I_p['Vr'], V_i_new)
    w_e_new = jnp.where(spike_e, w_e_new + E_p['b'], w_e_new)
    w_i_new = jnp.where(spike_i, w_i_new + I_p['b'], w_i_new)
    s_e_new = jnp.where(spike_e, s_e_new + 1.0, s_e_new)
    s_i_new = jnp.where(spike_i, s_i_new + 1.0, s_i_new)
    
    V_e_new = jnp.clip(V_e_new, -100.0, E_p['Vcut'])
    V_i_new = jnp.clip(V_i_new, -100.0, I_p['Vcut'])
    
    new_state = {'V_e': V_e_new, 'V_i': V_i_new, 'w_e': w_e_new, 'w_i': w_i_new, 's_e': s_e_new, 's_i': s_i_new}
    obs = {'V_e': V_e_new, 'V_i': V_i_new, 'spike_e': spike_e.astype(jnp.float32), 'spike_i': spike_i.astype(jnp.float32)}
    return new_state, obs


def run_ping_simulation(config, state, n_steps, I_ext_e, I_ext_i, seed=42):
    """Run PING simulation."""
    @jax.jit
    def simulate(init_state, keys):
        def step_fn(carry, key):
            new_state, obs = ping_step(carry, config, I_ext_e, I_ext_i, key)
            return new_state, obs
        return lax.scan(step_fn, init_state, keys)
    
    keys = jax.random.split(jax.random.PRNGKey(seed), n_steps)
    _, obs = simulate(state, keys)
    return obs


def compute_lfp(V, burn_steps=1000):
    return jnp.mean(V[burn_steps:], axis=1)


def compute_psd(signal, dt_ms, max_freq=100):
    signal = np.array(signal) - np.mean(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=dt_ms/1000)
    psd = np.abs(fft_vals)**2
    mask = freqs <= max_freq
    return freqs[mask], psd[mask]


def compute_firing_rate(spikes, dt_ms, burn_steps=1000):
    spikes = np.array(spikes[burn_steps:])
    return np.sum(spikes) / (spikes.shape[1] * spikes.shape[0] * dt_ms / 1000)


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    print("\n" + "="*60)
    print("PING MODEL - Gamma Oscillation Test")
    print("="*60)
    
    n_e, n_i, dt = 200, 50, 0.1
    duration_ms, burn_steps = 500, 1000
    n_steps = int(duration_ms / dt)
    
    print(f"\nNetwork: {n_e} E cells, {n_i} I cells, dt={dt}ms")
    
    config = create_ping_config(n_e=n_e, n_i=n_i, dt=dt)
    state = create_ping_state(config)
    
    I_ext_e, I_ext_i = 250.0, 50.0
    print(f"Drive: E={I_ext_e}pA, I={I_ext_i}pA")
    
    print("\nRunning...")
    t0 = time.time()
    obs = run_ping_simulation(config, state, n_steps, I_ext_e, I_ext_i)
    obs['V_e'].block_until_ready()
    print(f"Time: {time.time()-t0:.2f}s")
    
    rate_e = compute_firing_rate(obs['spike_e'], dt, burn_steps)
    rate_i = compute_firing_rate(obs['spike_i'], dt, burn_steps)
    print(f"\nRates: E={rate_e:.1f}Hz, I={rate_i:.1f}Hz")
    
    lfp_e = compute_lfp(obs['V_e'], burn_steps)
    freqs, psd = compute_psd(lfp_e, dt)
    gamma_mask = (freqs >= 30) & (freqs <= 80)
    peak_freq = freqs[gamma_mask][np.argmax(psd[gamma_mask])] if np.any(gamma_mask) else 0
    print(f"Gamma peak: {peak_freq:.1f}Hz")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('PING Model - Gamma Oscillations', fontweight='bold')
    time_ms = np.arange(n_steps) * dt
    
    # Raster
    ax = axes[0, 0]
    for i in range(min(50, n_e)):
        st = time_ms[np.array(obs['spike_e'][:, i]) > 0]
        ax.scatter(st, np.ones_like(st)*i, c='red', s=1)
    for i in range(n_i):
        st = time_ms[np.array(obs['spike_i'][:, i]) > 0]
        ax.scatter(st, np.ones_like(st)*(i+50), c='blue', s=1)
    ax.set_xlim(200, 400)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Raster (E:red, I:blue)')
    
    # LFP
    ax = axes[0, 1]
    ps, pe = int(200/dt), int(400/dt)
    ax.plot(time_ms[ps:pe], np.array(obs['V_e'][ps:pe]).mean(axis=1), 'r-', label='E')
    ax.plot(time_ms[ps:pe], np.array(obs['V_i'][ps:pe]).mean(axis=1), 'b-', label='I')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mean V (mV)')
    ax.legend()
    ax.set_title('LFP')
    
    # PSD
    ax = axes[1, 0]
    ax.semilogy(freqs, psd/psd.max(), 'r-', lw=2)
    ax.axvspan(30, 80, alpha=0.2, color='green')
    ax.axvline(peak_freq, color='k', ls='--', label=f'{peak_freq:.0f}Hz')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.legend()
    ax.set_title('Power Spectrum')
    
    # Phase
    ax = axes[1, 1]
    zs, ze = int(250/dt), int(300/dt)
    ve = np.array(obs['V_e'][zs:ze]).mean(axis=1)
    vi = np.array(obs['V_i'][zs:ze]).mean(axis=1)
    ax.plot(time_ms[zs:ze], (ve-ve.min())/(ve.max()-ve.min()), 'r-', label='E')
    ax.plot(time_ms[zs:ze], (vi-vi.min())/(vi.max()-vi.min()), 'b-', label='I')
    ax.set_xlabel('Time (ms)')
    ax.legend()
    ax.set_title('E-I Phase (PING)')
    
    plt.tight_layout()
    plt.savefig('results/ping_gamma_test.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/ping_gamma_test.png")
    print("\n✓ PING test complete!")
