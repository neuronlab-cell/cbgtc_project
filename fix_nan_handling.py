"""Add NaN handling to metrics and objective function."""

# Fix metrics_jax.py
with open('optimization/metrics_jax.py', 'r') as f:
    content = f.read()

# Add NaN check to beta power calculation
old_beta = '''    beta_power = jnp.sum(psd[idx])
    
    return float(beta_power)'''

new_beta = '''    beta_power = jnp.sum(psd[idx])
    
    # Handle NaN/Inf
    if jnp.isnan(beta_power) or jnp.isinf(beta_power):
        return 0.0
    
    return float(beta_power)'''

if old_beta in content:
    content = content.replace(old_beta, new_beta)
    print("✓ Added NaN handling to beta power calculation")

with open('optimization/metrics_jax.py', 'w') as f:
    f.write(content)

# Fix optuna_driver.py objective function
with open('optimization/optuna_driver.py', 'r') as f:
    content = f.read()

# Add safety check in objective
old_obj = '''    # Compute score
    score = 1.0 * rate_error + 0.01 * beta_penalty + 0.1 * cv_error'''

new_obj = '''    # Compute score (with safety checks)
    if jnp.isnan(rate_error) or jnp.isnan(beta_penalty) or jnp.isnan(cv_error):
        return float('inf')  # Invalid trial
    
    score = 1.0 * rate_error + 0.01 * beta_penalty + 0.1 * cv_error
    
    if jnp.isnan(score) or jnp.isinf(score):
        return float('inf')'''

if old_obj in content:
    content = content.replace(old_obj, new_obj)
    print("✓ Added NaN handling to objective function")

with open('optimization/optuna_driver.py', 'w') as f:
    f.write(content)

print("\n✓ NaN handling added to metrics and objective!")
