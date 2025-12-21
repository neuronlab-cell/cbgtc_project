# sim_jax.py
import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict
from integrator import network_step

def apply_params_to_config(trial_params: Dict[str, float], base_config: Dict) -> Dict:
    """
    Apply Optuna trial parameters to network config (functional update).
    
    Args:
        trial_params: {
            'ISTN': 42.0,
            'I_gpe': 580.0,
            'I_gpi': 240.0,
            'noise_stn_sigma': 0.15,
            'noise_gpe_sigma': 30.0,
            'noise_gpi_sigma': 30.0
        }
        base_config: Config from build_network_state()
        
    Returns:
        Modified config (base_config unchanged)
    """
    # 1. Start with shallow copy
    config = dict(base_config)
    
    # 2. Update neuron parameters
    stn_p = dict(config['neuron_params']['stn'])
    stn_p['ISTN'] = trial_params['ISTN']  # No float() needed in JIT
    
    gpe_p = dict(config['neuron_params']['gpe'])
    gpe_p['I_baseline'] = trial_params['I_gpe']
    
    gpi_p = dict(config['neuron_params']['gpi'])
    gpi_p['I_baseline'] = trial_params['I_gpi']
    
    config['neuron_params'] = {
        'stn': stn_p,
        'gpe': gpe_p,
        'gpi': gpi_p
    }
    
    # 3. Update noise configs (NamedTuples!)
    config['noise'] = {
        'stn': config['noise']['stn']._replace(
            sigma=trial_params.get('noise_stn_sigma', config['noise']['stn'].sigma)
        ),
        'gpe': config['noise']['gpe']._replace(
            sigma=trial_params.get('noise_gpe_sigma', config['noise']['gpe'].sigma)
        ),
        'gpi': config['noise']['gpi']._replace(
            sigma=trial_params.get('noise_gpi_sigma', config['noise']['gpi'].sigma)
        )
    }
    
    return config


def run_simulation_python_loop(
    trial_params: Dict[str, float],
    base_state: Dict,
    base_config: Dict,
    n_steps: int = 4000
) -> Dict:
    """
    Run simulation with Python loop (baseline, not optimized).
    
    Returns:
        observables: Dict of arrays with shape (n_steps, n_neurons)
    """
    config = apply_params_to_config(trial_params, base_config)
    state = base_state  # No copy needed - functional updates
    
    # Collect observables
    history = {
        'V_stn': [], 'V_gpe': [], 'V_gpi': [],
        'spikes_stn': [], 'spikes_gpe': [], 'spikes_gpi': []
    }
    
    for i in range(n_steps):
        t_ms = i * config['dt_ms']
        state, obs = network_step(state, config, t_ms)
        
        for key in history.keys():
            history[key].append(obs[key])
    
    # Stack into (n_steps, n_neurons) arrays
    return {k: jnp.stack(v) for k, v in history.items()}


def create_simulation_fn(base_config: Dict, n_steps: int = 4000):
    """
    Factory function creating JIT-compiled simulator.
    
    The base_config and n_steps are captured in closure to avoid passing
    non-traceable objects (functions) or dynamic shapes to JIT.
    
    Usage:
        sim_fn = create_simulation_fn(config, n_steps=4000)
        obs = sim_fn(params, state)  # Fast after first call
    
    Args:
        base_config: Network configuration (contains neuron step functions)
        n_steps: Number of simulation steps (must be fixed at compile time)
    """
    
    @jax.jit
    def simulate(trial_params: Dict[str, float], init_state: Dict):
        # Apply params to config captured from outer scope
        config = apply_params_to_config(trial_params, base_config)
        
        def step_fn(carry_state, t_idx):
            """Body function for lax.scan loop."""
            t_ms = t_idx * config['dt_ms']
            new_state, obs = network_step(carry_state, config, t_ms)
            return new_state, obs
        
        # Compiled loop replacing Python for
        # n_steps is static (from closure)
        final_state, obs_history = lax.scan(
            step_fn,
            init=init_state,
            xs=jnp.arange(n_steps)
        )
        
        # obs_history is already stacked: (n_steps, n_neurons) per observable
        return obs_history
    
    return simulate
