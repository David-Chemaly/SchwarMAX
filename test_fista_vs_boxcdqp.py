"""
Quick synthetic test: FISTA vs BoxCDQP on the same NNLS problem.
Verifies that solve_fista_nnls matches solve_qp_boxcdqp.
"""
import jax
import jax.numpy as jnp
import time

# Ensure CPU for local testing
jax.config.update("jax_platform_name", "cpu")

from model import solve_fista_nnls, solve_qp_boxcdqp

# Reproducible random data
key = jax.random.PRNGKey(0)
n_obs_rz, n_obs_xy, n_obs_h, n_orb = 360, 2400, 2400, 500

keys = jax.random.split(key, 20)
A_Rzphi = jnp.abs(jax.random.normal(keys[0], (n_obs_rz, n_orb)))
A_xy    = jnp.abs(jax.random.normal(keys[1], (n_obs_xy, n_orb)))
A_h1    = jax.random.normal(keys[2], (n_obs_h, n_orb))
A_h2    = jax.random.normal(keys[3], (n_obs_h, n_orb))
A_h3    = jax.random.normal(keys[4], (n_obs_h, n_orb)) * 0.1
A_h4    = jax.random.normal(keys[5], (n_obs_h, n_orb)) * 0.1

# Create target data as weighted sum of columns (realistic scenario)
w_true = jnp.abs(jax.random.normal(keys[6], (n_orb,)))
y_Rzphi = A_Rzphi @ w_true + 0.01 * jnp.abs(jax.random.normal(keys[7], (n_obs_rz,)))
y_xy    = A_xy @ w_true + 0.01 * jnp.abs(jax.random.normal(keys[8], (n_obs_xy,)))
y_h1    = jax.random.normal(keys[9], (n_obs_h,)) * 0.1
y_h2    = jax.random.normal(keys[10], (n_obs_h,)) * 0.1
y_h3    = jax.random.normal(keys[11], (n_obs_h,)) * 0.05
y_h4    = jax.random.normal(keys[12], (n_obs_h,)) * 0.05

sig_Rzphi = jnp.ones(n_obs_rz) * 0.1
sig_xy    = jnp.ones(n_obs_xy) * 0.1
sig_A1    = jnp.ones(n_obs_h) * 0.1
sig_A2    = jnp.ones(n_obs_h) * 0.1
sig_A3    = jnp.ones(n_obs_h) * 0.1
sig_A4    = jnp.ones(n_obs_h) * 0.1

print(f"Problem size: {n_obs_rz+n_obs_xy+4*n_obs_h} obs x {n_orb} orbits")

# --- BoxCDQP (ground truth) ---
t0 = time.time()
w_qp = solve_qp_boxcdqp(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=100, tol=1e-1,
)
w_qp.block_until_ready()
t_qp = time.time() - t0

# --- FISTA ---
t0 = time.time()
w_fista = solve_fista_nnls(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=500,
)
w_fista.block_until_ready()
t_fista = time.time() - t0

# Compute objectives for both
eps = 1e-8
y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)
w_rzphi = jnp.sqrt(5.0 / A_Rzphi.shape[0])
w_xy    = jnp.sqrt(5.0 / A_xy.shape[0])
w_h     = jnp.sqrt(1.0 / A_h1.shape[0])

U_rz  = w_rzphi * (A_Rzphi / (sig_Rzphi[:, None] + eps))
b_rz  = w_rzphi * (y_Rzphi / (sig_Rzphi + eps))
U_xy_ = w_xy * (A_xy / (sig_xy[:, None] + eps))
b_xy  = w_xy * (y_xy / (sig_xy + eps))
U_h1_ = w_h * ((A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps))
U_h2_ = w_h * ((A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps))
U_h3_ = w_h * ((A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps))
U_h4_ = w_h * ((A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps))
b_h1  = w_h * (y_h1 / (sig_A1 + eps))
b_h2  = w_h * (y_h2 / (sig_A2 + eps))
b_h3  = w_h * (y_h3 / (sig_A3 + eps))
b_h4  = w_h * (y_h4 / (sig_A4 + eps))

U = jnp.vstack([U_rz, U_xy_, U_h1_, U_h2_, U_h3_, U_h4_])
b = jnp.concatenate([b_rz, b_xy, b_h1, b_h2, b_h3, b_h4])
reg = 1.0 / n_orb

def objective(w):
    r = U @ w - b
    return 0.5 * jnp.dot(r, r) + 0.5 * reg * jnp.dot(w, w)

obj_qp    = objective(w_qp)
obj_fista = objective(w_fista)

print(f"\n--- Results ---")
print(f"BoxCDQP  time: {t_qp:.3f}s,  objective: {obj_qp:.6f}")
print(f"FISTA    time: {t_fista:.3f}s,  objective: {obj_fista:.6f}")
print(f"Objective ratio (FISTA/QP): {obj_fista/obj_qp:.6f}")

# Weight comparison
diff = jnp.abs(w_fista - w_qp)
rel_diff = diff / (jnp.abs(w_qp) + 1e-10)
print(f"\nWeight differences:")
print(f"  Max abs diff:  {jnp.max(diff):.6e}")
print(f"  Mean abs diff: {jnp.mean(diff):.6e}")
print(f"  Max rel diff:  {jnp.max(rel_diff):.6e}")
print(f"  Mean rel diff: {jnp.mean(rel_diff):.6e}")
print(f"  Correlation:   {jnp.corrcoef(w_fista, w_qp)[0,1]:.6f}")

# Check non-negativity
print(f"\nNon-negativity:")
print(f"  FISTA min weight: {jnp.min(w_fista):.6e}")
print(f"  QP    min weight: {jnp.min(w_qp):.6e}")

# Downstream observable comparison (what actually matters)
density_fista = A_xy @ w_fista
density_qp    = A_xy @ w_qp
density_diff  = jnp.abs(density_fista - density_qp)
print(f"\nDensity observable:")
print(f"  Max abs diff:  {jnp.max(density_diff):.6e}")
print(f"  Mean rel diff: {jnp.mean(density_diff / (jnp.abs(density_qp) + 1e-10)):.6e}")
