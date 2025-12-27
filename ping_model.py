"""
PING Model v2 - Fixed E-I coupling for proper gamma
"""
import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import numpy as np
import time

E_PARAMS = {'C': 200.0, 'gL': 10.0, 'EL': -70.0, 'VT': -50.0, 'DeltaT': 2.0, 'Vr': -60.0, 'Vcut': 20.0, 'a': 2.0, 'b': 60.0, 'tau_w': 100.0}
I_PARAMS = {'C': 100.0, 'gL': 10.0, 'EL': -65.0, 'VT': -50.0, 'DeltaT': 0.5, 'Vr': -60.0, 'Vcut': 20.0, 'a': 0.0, 'b': 0.0, 'tau_w': 50.0}
SYN_PARAMS = {'E_exc': 0.0, 'E_inh': -75.0, 'tau_ampa': 2.0, 'tau_gaba': 8.0}

def create_ping_config(n_e=200, n_i=50, dt=0.1, seed=42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    
    # Higher connectivity and MUCH stronger weights
    W_ei = (jax.random.uniform(keys[0], (n_i, n_e)) < 0.5).astype(jnp.float32)
    W_ie = (jax.random.uniform(keys[1], (n_e, n_i)) < 0.5).astype(jnp.float32)
    W_ee = (jax.random.uniform(keys[2], (n_e, n_e)) < 0.1).astype(jnp.float32)
    W_ii = (jax.random.uniform(keys[3], (n_i, n_i)) < 0.5).astype(jnp.float32)
    W_ee = W_ee.at[jnp.diag_indices(n_e)].set(0.0)
    W_ii = W_ii.at[jnp.diag_indices(n_i)].set(0.0)
    
    # KEY FIX: Much stronger synaptic weights
    g_ei = 5.0 / jnp.sqrt(n_e)    # E→I strong (drives I cells)
    g_ie = 10.0 / jnp.sqrt(n_i)   # I→E very strong (feedback inhibition)
    g_ee = 0.1 / jnp.sqrt(n_e)    # E→E weak
    g_ii = 2.0 / jnp.sqrt(n_i)    # I→I moderate
    
    return {'n_e': n_e, 'n_i': n_i, 'dt': dt,
            'W_ei': W_ei * g_ei, 'W_ie': W_ie * g_ie,
            'W_ee': W_ee * g_ee, 'W_ii': W_ii * g_ii,
            'E_params': E_PARAMS, 'I_params': I_PARAMS, 'syn_params': SYN_PARAMS}

def create_ping_state(config, seed=0):
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
    dt = config['dt']
    E_p, I_p, S_p = config['E_params'], config['I_params'], config['syn_params']
    V_e, V_i, w_e, w_i, s_e, s_i = state['V_e'], state['V_i'], state['w_e'], state['w_i'], state['s_e'], state['s_i']
    
    k1, k2 = jax.random.split(noise_key)
    noise_e = jax.random.normal(k1, V_e.shape) * 30.0
    noise_i = jax.random.normal(k2, V_i.shape) * 10.0
    
    # Synaptic currents - now using nS * mV = pA
    I_ampa_ee = config['W_ee'] @ s_e * (S_p['E_exc'] - V_e)  # Note: (E_rev - V) for correct sign
    I_ampa_ei = config['W_ei'] @ s_e * (S_p['E_exc'] - V_i)
    I_gaba_ie = config['W_ie'] @ s_i * (S_p['E_inh'] - V_e)
    I_gaba_ii = config['W_ii'] @ s_i * (S_p['E_inh'] - V_i)
    
    I_syn_e = I_ampa_ee + I_gaba_ie
    I_syn_i = I_ampa_ei + I_gaba_ii
    
    # AdEx dynamics
    exp_e = E_p['DeltaT'] * jnp.exp(jnp.clip((V_e - E_p['VT']) / E_p['DeltaT'], -20, 20))
    exp_i = I_p['DeltaT'] * jnp.exp(jnp.clip((V_i - I_p['VT']) / I_p['DeltaT'], -20, 20))
    
    dV_e = (E_p['gL'] * (E_p['EL'] - V_e + exp_e) - w_e + I_ext_e + noise_e + I_syn_e) / E_p['C']
    dV_i = (I_p['gL'] * (I_p['EL'] - V_i + exp_i) - w_i + I_ext_i + noise_i + I_syn_i) / I_p['C']
    dw_e = (E_p['a'] * (V_e - E_p['EL']) - w_e) / E_p['tau_w']
    dw_i = (I_p['a'] * (V_i - I_p['EL']) - w_i) / I_p['tau_w']
    
    V_e_new = V_e + dt * dV_e
    V_i_new = V_i + dt * dV_i
    w_e_new = w_e + dt * dw_e
    w_i_new = w_i + dt * dw_i
    s_e_new = s_e * jnp.exp(-dt / S_p['tau_ampa'])
    s_i_new = s_i * jnp.exp(-dt / S_p['tau_gaba'])
    
    spike_e = V_e_new >= E_p['Vcut']
    spike_i = V_i_new >= I_p['Vcut']
    
    V_e_new = jnp.where(spike_e, E_p['Vr'], jnp.clip(V_e_new, -100.0, E_p['Vcut']))
    V_i_new = jnp.where(spike_i, I_p['Vr'], jnp.clip(V_i_new, -100.0, I_p['Vcut']))
    w_e_new = jnp.where(spike_e, w_e_new + E_p['b'], w_e_new)
    s_e_new = jnp.where(spike_e, s_e_new + 1.0, s_e_new)
    s_i_new = jnp.where(spike_i, s_i_new + 1.0, s_i_new)
    
    return {'V_e': V_e_new, 'V_i': V_i_new, 'w_e': w_e_new, 'w_i': w_i_new, 's_e': s_e_new, 's_i': s_i_new}, \
           {'V_e': V_e_new, 'V_i': V_i_new, 'spike_e': spike_e.astype(jnp.float32), 'spike_i': spike_i.astype(jnp.float32)}

def run_ping(config, state, n_steps, I_e, I_i, seed=42):
    @jax.jit
    def sim(s, keys):
        def step(c, k):
            ns, obs = ping_step(c, config, I_e, I_i, k)
            return ns, obs
        return lax.scan(step, s, keys)
    _, obs = sim(state, jax.random.split(jax.random.PRNGKey(seed), n_steps))
    return obs

if __name__ == "__main__":
    print(f"JAX: {jax.devices()}")
    
    n_e, n_i, dt = 200, 50, 0.1
    n_steps = 5000  # 500ms
    burn = 1000
    
    config = create_ping_config(n_e, n_i, dt)
    state = create_ping_state(config)
    
    # Test different drive levels
    print("\nSweeping E drive to find PING regime...")
    for I_e in [200, 250, 300, 350, 400]:
        I_i = 100  # Give I cells some baseline drive too
        obs = run_ping(config, state, n_steps, float(I_e), float(I_i))
        obs['V_e'].block_until_ready()
        
        sp_e = np.array(obs['spike_e'][burn:])
        sp_i = np.array(obs['spike_i'][burn:])
        rate_e = sp_e.sum() / (n_e * (n_steps-burn) * dt / 1000)
        rate_i = sp_i.sum() / (n_i * (n_steps-burn) * dt / 1000)
        
        lfp = np.array(obs['V_e'][burn:]).mean(axis=1)
        lfp = lfp - lfp.mean()
        fft = np.abs(np.fft.rfft(lfp))**2
        freqs = np.fft.rfftfreq(len(lfp), dt/1000)
        gm = (freqs >= 30) & (freqs <= 80)
        peak = freqs[gm][np.argmax(fft[gm])] if gm.any() and fft[gm].max() > 0 else 0
        
        print(f"I_e={I_e}: E={rate_e:.1f}Hz, I={rate_i:.1f}Hz, gamma={peak:.1f}Hz")
    
    # Run best params and plot
    I_e, I_i = 350.0, 100.0
    print(f"\nRunning with I_e={I_e}, I_i={I_i}...")
    
    t0 = time.time()
    obs = run_ping(config, state, n_steps, I_e, I_i)
    obs['V_e'].block_until_ready()
    print(f"Time: {time.time()-t0:.2f}s")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    time_ms = np.arange(n_steps) * dt
    
    # Raster
    ax = axes[0,0]
    for i in range(min(50, n_e)):
        st = time_ms[np.array(obs['spike_e'][:,i]) > 0]
        ax.scatter(st, [i]*len(st), c='red', s=1)
    for i in range(n_i):
        st = time_ms[np.array(obs['spike_i'][:,i]) > 0]
        ax.scatter(st, [i+55]*len(st), c='blue', s=1)
    ax.axhline(52, c='k', ls='--', alpha=0.3)
    ax.set_xlim(200, 400)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron')
    ax.set_title('Raster (E:red, I:blue)')
    
    # LFP
    ax = axes[0,1]
    ps, pe = int(200/dt), int(400/dt)
    ax.plot(time_ms[ps:pe], np.array(obs['V_e'][ps:pe]).mean(1), 'r-', label='E')
    ax.plot(time_ms[ps:pe], np.array(obs['V_i'][ps:pe]).mean(1), 'b-', label='I')
    ax.legend()
    ax.set_xlabel('Time (ms)')
    ax.set_title('LFP')
    
    # PSD
    ax = axes[1,0]
    lfp = np.array(obs['V_e'][burn:]).mean(1)
    lfp = lfp - lfp.mean()
    fft = np.abs(np.fft.rfft(lfp))**2
    freqs = np.fft.rfftfreq(len(lfp), dt/1000)
    ax.semilogy(freqs[freqs<100], fft[freqs<100]/fft.max(), 'r-')
    ax.axvspan(30, 80, alpha=0.2, color='green')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Power Spectrum')
    
    # Zoom phase
    ax = axes[1,1]
    zs, ze = int(250/dt), int(300/dt)
    ve = np.array(obs['V_e'][zs:ze]).mean(1)
    vi = np.array(obs['V_i'][zs:ze]).mean(1)
    ve = (ve-ve.min())/(ve.max()-ve.min()+1e-6)
    vi = (vi-vi.min())/(vi.max()-vi.min()+1e-6)
    ax.plot(time_ms[zs:ze], ve, 'r-', lw=2, label='E')
    ax.plot(time_ms[zs:ze], vi, 'b-', lw=2, label='I')
    ax.legend()
    ax.set_xlabel('Time (ms)')
    ax.set_title('E-I Phase')
    
    plt.tight_layout()
    plt.savefig('results/ping_v2.png', dpi=150)
    print("Saved: results/ping_v2.png")
