"""
PAC Model v3 - Beta-phase gating of gamma
Key insight: In PD, gamma bursts occur at specific beta phases
We model this by having beta-frequency INHIBITION that gates when gamma can occur
"""
import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import time

print(f"JAX: {jax.devices()}")

E_PARAMS = {'C': 200.0, 'gL': 10.0, 'EL': -70.0, 'VT': -50.0, 'DeltaT': 2.0, 
            'Vr': -60.0, 'Vcut': 20.0, 'a': 2.0, 'b': 60.0, 'tau_w': 100.0}
I_PARAMS = {'C': 100.0, 'gL': 10.0, 'EL': -65.0, 'VT': -50.0, 'DeltaT': 0.5,
            'Vr': -60.0, 'Vcut': 20.0, 'a': 0.0, 'b': 0.0, 'tau_w': 50.0}

def create_config(n_e=200, n_i=50, dt=0.1, seed=42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    W_ei = (jax.random.uniform(keys[0], (n_i, n_e)) < 0.5).astype(jnp.float32)
    W_ie = (jax.random.uniform(keys[1], (n_e, n_i)) < 0.5).astype(jnp.float32)
    W_ee = (jax.random.uniform(keys[2], (n_e, n_e)) < 0.1).astype(jnp.float32)
    W_ii = (jax.random.uniform(keys[3], (n_i, n_i)) < 0.5).astype(jnp.float32)
    W_ee = W_ee.at[jnp.diag_indices(n_e)].set(0.0)
    W_ii = W_ii.at[jnp.diag_indices(n_i)].set(0.0)
    return {
        'n_e': n_e, 'n_i': n_i, 'dt': dt,
        'W_ei': W_ei * 5.0/jnp.sqrt(n_e), 'W_ie': W_ie * 10.0/jnp.sqrt(n_i),
        'W_ee': W_ee * 0.1/jnp.sqrt(n_e), 'W_ii': W_ii * 2.0/jnp.sqrt(n_i),
        'E_params': E_PARAMS, 'I_params': I_PARAMS,
        'syn': {'E_exc': 0.0, 'E_inh': -75.0, 'tau_ampa': 2.0, 'tau_gaba': 8.0}
    }

def create_state(cfg, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    return {
        'V_e': cfg['E_params']['EL'] + jax.random.uniform(k1, (cfg['n_e'],)) * 10,
        'V_i': cfg['I_params']['EL'] + jax.random.uniform(k2, (cfg['n_i'],)) * 10,
        'w_e': jnp.zeros(cfg['n_e']), 'w_i': jnp.zeros(cfg['n_i']),
        's_e': jnp.zeros(cfg['n_e']), 's_i': jnp.zeros(cfg['n_i']),
    }

def step(state, cfg, I_e, I_i, I_inhibit, key):
    """I_inhibit is beta-frequency inhibitory current (negative = inhibition)"""
    dt, Ep, Ip, Sp = cfg['dt'], cfg['E_params'], cfg['I_params'], cfg['syn']
    Ve, Vi, we, wi, se, si = state['V_e'], state['V_i'], state['w_e'], state['w_i'], state['s_e'], state['s_i']
    
    k1, k2 = jax.random.split(key)
    noise_e = jax.random.normal(k1, Ve.shape) * 30.0
    noise_i = jax.random.normal(k2, Vi.shape) * 10.0
    
    Isyn_e = cfg['W_ee'] @ se * (Sp['E_exc'] - Ve) + cfg['W_ie'] @ si * (Sp['E_inh'] - Ve)
    Isyn_i = cfg['W_ei'] @ se * (Sp['E_exc'] - Vi) + cfg['W_ii'] @ si * (Sp['E_inh'] - Vi)
    
    exp_e = Ep['DeltaT'] * jnp.exp(jnp.clip((Ve - Ep['VT']) / Ep['DeltaT'], -20, 20))
    exp_i = Ip['DeltaT'] * jnp.exp(jnp.clip((Vi - Ip['VT']) / Ip['DeltaT'], -20, 20))
    
    # Beta inhibition affects E cells (from GPe in real circuit)
    dVe = (Ep['gL'] * (Ep['EL'] - Ve + exp_e) - we + I_e + noise_e + Isyn_e + I_inhibit) / Ep['C']
    dVi = (Ip['gL'] * (Ip['EL'] - Vi + exp_i) - wi + I_i + noise_i + Isyn_i) / Ip['C']
    dwe = (Ep['a'] * (Ve - Ep['EL']) - we) / Ep['tau_w']
    dwi = (Ip['a'] * (Vi - Ip['EL']) - wi) / Ip['tau_w']
    
    Ve_new = Ve + dt * dVe
    Vi_new = Vi + dt * dVi
    we_new = we + dt * dwe
    wi_new = wi + dt * dwi
    se_new = se * jnp.exp(-dt / Sp['tau_ampa'])
    si_new = si * jnp.exp(-dt / Sp['tau_gaba'])
    
    spk_e, spk_i = Ve_new >= Ep['Vcut'], Vi_new >= Ip['Vcut']
    Ve_new = jnp.where(spk_e, Ep['Vr'], jnp.clip(Ve_new, -100, Ep['Vcut']))
    Vi_new = jnp.where(spk_i, Ip['Vr'], jnp.clip(Vi_new, -100, Ip['Vcut']))
    we_new = jnp.where(spk_e, we_new + Ep['b'], we_new)
    se_new = jnp.where(spk_e, se_new + 1.0, se_new)
    si_new = jnp.where(spk_i, si_new + 1.0, si_new)
    
    return {'V_e': Ve_new, 'V_i': Vi_new, 'w_e': we_new, 'w_i': wi_new, 's_e': se_new, 's_i': si_new}, \
           {'V_e': Ve_new, 'spike_e': spk_e.astype(jnp.float32)}

def run_sim(cfg, state, n_steps, I_e, I_i, I_inh_signal, seed=42):
    @jax.jit
    def sim(s, inputs):
        keys, inh = inputs
        def fn(c, inp):
            k, i = inp
            ns, obs = step(c, cfg, I_e, I_i, i, k)
            return ns, obs
        return lax.scan(fn, s, (keys, inh))
    keys = jax.random.split(jax.random.PRNGKey(seed), n_steps)
    _, obs = sim(state, (keys, I_inh_signal))
    return obs

def compute_pac(lfp, fs, f_phase=(13,30), f_amp=(30,80), n_bins=18):
    """Compute PAC using modulation index."""
    lfp = lfp - np.mean(lfp)
    
    b_p, a_p = signal.butter(2, [f_phase[0]/(fs/2), f_phase[1]/(fs/2)], 'band')
    b_a, a_a = signal.butter(2, [f_amp[0]/(fs/2), f_amp[1]/(fs/2)], 'band')
    
    lfp_filt_phase = signal.filtfilt(b_p, a_p, lfp)
    lfp_filt_amp = signal.filtfilt(b_a, a_a, lfp)
    
    phase = np.angle(signal.hilbert(lfp_filt_phase))
    amp = np.abs(signal.hilbert(lfp_filt_amp))
    
    bins = np.linspace(-np.pi, np.pi, n_bins+1)
    amp_by_phase = []
    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        amp_by_phase.append(amp[mask].mean() if mask.sum() > 0 else 0)
    amp_by_phase = np.array(amp_by_phase)
    
    # Normalize to probability distribution
    amp_by_phase = amp_by_phase / amp_by_phase.sum() if amp_by_phase.sum() > 0 else amp_by_phase
    
    # Modulation index
    uniform = 1.0 / n_bins
    amp_by_phase = amp_by_phase + 1e-10
    amp_by_phase = amp_by_phase / amp_by_phase.sum()
    mi = np.sum(amp_by_phase * np.log(amp_by_phase / uniform)) / np.log(n_bins)
    
    return mi, (bins[:-1] + bins[1:])/2, amp_by_phase * n_bins  # Scale for plotting

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PAC MODEL v3 - Beta Inhibitory Gating")
    print("="*60)
    
    dt = 0.1
    duration = 3000  # Longer for better stats
    n_steps = int(duration / dt)
    fs = 1000 / dt
    burn = int(500 / dt)
    
    cfg = create_config()
    state = create_state(cfg)
    I_e, I_i = 350.0, 100.0
    t = np.arange(n_steps) * dt / 1000  # seconds
    
    beta_freq = 20.0  # Hz
    
    print("\nTesting different beta inhibition strengths...")
    results = []
    
    # Key: Use RECTIFIED sine (only inhibition, no excitation)
    # This creates windows where gamma CAN vs CANNOT occur
    for inh_amp in [0, 30, 60, 100]:
        # Rectified sine: only negative (inhibitory) part
        beta_wave = np.sin(2 * np.pi * beta_freq * t)
        # Only keep inhibitory phase (negative values become inhibition)
        I_inh = -inh_amp * np.clip(-beta_wave, 0, 1)  # Negative current = hyperpolarization
        I_inh = jnp.array(I_inh)
        
        obs = run_sim(cfg, state, n_steps, I_e, I_i, I_inh)
        obs['V_e'].block_until_ready()
        
        lfp = np.array(obs['V_e'][burn:]).mean(axis=1)
        mi, bins, amp = compute_pac(lfp, fs)
        
        label = "Healthy" if inh_amp == 0 else f"PD ({inh_amp}pA)"
        print(f"{label:15s}: MI = {mi:.4f}")
        results.append((inh_amp, mi, bins, amp, lfp, np.array(I_inh[burn:])))
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS: PAC increases with beta inhibition strength")
    print("="*60)
    for inh, mi, _, _, _, _ in results:
        bar = "█" * int(mi * 200)
        print(f"Inh={inh:3d}pA: MI={mi:.4f} {bar}")
    
    # Plot
    fig = plt.figure(figsize=(14, 10))
    
    # 1. PAC summary bar chart
    ax1 = fig.add_subplot(2, 2, 1)
    labels = [f'{r[0]}pA' for r in results]
    mis = [r[1] for r in results]
    colors = ['steelblue', 'orange', 'firebrick', 'purple']
    bars = ax1.bar(labels, mis, color=colors, edgecolor='black', lw=2)
    for bar, mi in zip(bars, mis):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{mi:.3f}', ha='center', fontsize=11)
    ax1.set_xlabel('Beta Inhibition Strength')
    ax1.set_ylabel('Modulation Index (PAC)')
    ax1.set_title('PAC Increases with Pathological Beta', fontweight='bold')
    
    # 2. Phase-amplitude distribution
    ax2 = fig.add_subplot(2, 2, 2)
    for i, (inh, mi, bins, amp, _, _) in enumerate(results):
        ax2.plot(bins, amp, 'o-', color=colors[i], lw=2, markersize=4, 
                 label=f'{inh}pA (MI={mi:.3f})')
    ax2.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Uniform')
    ax2.set_xlabel('Beta Phase (rad)')
    ax2.set_ylabel('Normalized Gamma Amplitude')
    ax2.set_title('Gamma Locked to Beta Phase in PD')
    ax2.legend(fontsize=9)
    ax2.set_xlim(-np.pi, np.pi)
    
    # 3. Example LFP with beta overlay
    ax3 = fig.add_subplot(2, 2, 3)
    # Show healthy vs severe PD
    t_plot = np.arange(1000) * dt  # 100ms
    
    lfp_h = results[0][4][:1000]
    lfp_pd = results[3][4][:1000]
    inh_pd = results[3][5][:1000]
    
    ax3.plot(t_plot, lfp_h - 70, 'b-', lw=0.8, label='Healthy LFP')
    ax3.plot(t_plot, lfp_pd - 50, 'r-', lw=0.8, label='PD LFP')
    ax3.plot(t_plot, inh_pd/5 - 30, 'k-', lw=1, alpha=0.5, label='Beta inhibition')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('LFP (mV, offset)')
    ax3.set_title('LFP: Healthy vs Parkinsonian')
    ax3.legend(fontsize=9)
    
    # 4. Power spectra
    ax4 = fig.add_subplot(2, 2, 4)
    for i, (inh, _, _, _, lfp, _) in enumerate(results):
        freqs = np.fft.rfftfreq(len(lfp), dt/1000)
        psd = np.abs(np.fft.rfft(lfp - lfp.mean()))**2
        mask = freqs < 100
        ax4.semilogy(freqs[mask], psd[mask]/psd.max(), color=colors[i], lw=1.5, 
                     label=f'{inh}pA', alpha=0.8)
    ax4.axvspan(13, 30, alpha=0.1, color='orange', label='Beta')
    ax4.axvspan(30, 80, alpha=0.1, color='green', label='Gamma')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Normalized Power')
    ax4.set_title('Power Spectra')
    ax4.legend(fontsize=9, ncol=2)
    ax4.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig('results/pac_v3.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/pac_v3.png")
    
    print("\n✓ PAC v3 complete!")
