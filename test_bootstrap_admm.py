"""
Benchmark: validate bootstrap ADMM solver + logL.
Each bootstrap logL is evaluated against its own bootstrapped data.
y_Rzphi is model-computed (not bootstrapped).
"""
import pickle
import time
import numpy as np

import jax
import jax.numpy as jnp

from model_bar import (
    solve_nnls_admm,
    solve_nnls_admm_bootstrap,
    compute_model_and_logl_bootstrap,
    _compute_logl_from_weights,
)

# ---- Load pre-computed orbital library ----
path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
lib_file = 'orbital_library_adaptive_Nmax1e4.pkl'

with open(path_data + lib_file, 'rb') as f:
    orb_lib = pickle.load(f)

(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4) = orb_lib[:18]

# v0, s for h_to_V_sigma
with open('/Users/hanyuan/Dropbox/python_script/SchwarMAX/mock_Nbody_bar_XY_withRot.pkl', 'rb') as f:
    bin_dict = pickle.load(f)
v0 = jnp.array(bin_dict['v0'])
s = jnp.array(bin_dict['s'])

N_BOOTSTRAP = 500

# =============================== 1. Single ADMM (reference) ===============================
print("=" * 80)
print("1. Single ADMM solve → reference logL")
print("=" * 80)

w_single = solve_nnls_admm(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=200,
)
jax.block_until_ready(w_single)

logl_ref = _compute_logl_from_weights(
    w_single, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)
jax.block_until_ready(logl_ref)
print(f"  logL (_compute_logl_from_weights): {float(logl_ref):.4f}")

# =============================== 2. Validate: unperturbed data through bootstrap pipeline ===============================
print(f"\n{'=' * 80}")
print("2. Validate: single weights + unperturbed data through vmapped logL")
print("=" * 80)

# Tile same weights and same (unperturbed) data
w_tiled = jnp.tile(w_single[None, :], (N_BOOTSTRAP, 1))
y_xy_tiled = jnp.tile(y_xy[None, :], (N_BOOTSTRAP, 1))
y_h1_tiled = jnp.tile(y_h1[None, :], (N_BOOTSTRAP, 1))
y_h2_tiled = jnp.tile(y_h2[None, :], (N_BOOTSTRAP, 1))
y_h3_tiled = jnp.tile(y_h3[None, :], (N_BOOTSTRAP, 1))
y_h4_tiled = jnp.tile(y_h4[None, :], (N_BOOTSTRAP, 1))

logl_marg_tiled, _, _, _, _, _, _, _, logl_all_tiled,_ = \
    compute_model_and_logl_bootstrap(
        w_tiled,
        A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi,
        y_xy_tiled, y_h1_tiled, y_h2_tiled, y_h3_tiled, y_h4_tiled,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
        v0, s,
    )
jax.block_until_ready(logl_marg_tiled)

print(f"  logl from vmapped func: {float(logl_all_tiled[0]):.4f}")
print(f"  logl from reference:    {float(logl_ref):.4f}")
print(f"  Diff: {float(logl_all_tiled[0]) - float(logl_ref):.6f}")

# =============================== 3. Varying noise scale ===============================
print(f"\n{'=' * 80}")
print(f"3. Bootstrap with varying noise scale (N_boot={N_BOOTSTRAP})")
print(f"   Each logL evaluated against its own bootstrapped data")
print("=" * 80)

y_xy_np = np.asarray(y_xy)
y_h1_np, y_h2_np = np.asarray(y_h1), np.asarray(y_h2)
y_h3_np, y_h4_np = np.asarray(y_h3), np.asarray(y_h4)
sig_xy_np = np.asarray(sig_xy)
sig_A1_np, sig_A2_np = np.asarray(sig_A1), np.asarray(sig_A2)
sig_A3_np, sig_A4_np = np.asarray(sig_A3), np.asarray(sig_A4)

print(f"  {'scale':>6} | {'std(logL)':>10} | {'range(logL)':>12} | {'penalty':>10} | {'logl_marg':>12} | {'m_eff':>10}")
print(f"  {'-'*70}")

for noise_scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    rng = np.random.default_rng(42)

    y_xy_boot = y_xy_np[None, :] + noise_scale * rng.normal(size=(N_BOOTSTRAP, len(y_xy_np))) * sig_xy_np[None, :]
    y_h1_boot = y_h1_np[None, :] + noise_scale * rng.normal(size=(N_BOOTSTRAP, len(y_h1_np))) * sig_A1_np[None, :]
    y_h2_boot = y_h2_np[None, :] + noise_scale * rng.normal(size=(N_BOOTSTRAP, len(y_h2_np))) * sig_A2_np[None, :]
    y_h3_boot = y_h3_np[None, :] + noise_scale * rng.normal(size=(N_BOOTSTRAP, len(y_h3_np))) * sig_A3_np[None, :]
    y_h4_boot = y_h4_np[None, :] + noise_scale * rng.normal(size=(N_BOOTSTRAP, len(y_h4_np))) * sig_A4_np[None, :]

    y_xy_boot_j = jnp.array(y_xy_boot, dtype=jnp.float32)
    y_h1_boot_j = jnp.array(y_h1_boot, dtype=jnp.float32)
    y_h2_boot_j = jnp.array(y_h2_boot, dtype=jnp.float32)
    y_h3_boot_j = jnp.array(y_h3_boot, dtype=jnp.float32)
    y_h4_boot_j = jnp.array(y_h4_boot, dtype=jnp.float32)

    w_all = solve_nnls_admm_bootstrap(
        A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy,
        y_xy_boot_j, y_h1_boot_j, y_h2_boot_j, y_h3_boot_j, y_h4_boot_j,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
        lambda_reg=1, maxiter=200,
    )
    jax.block_until_ready(w_all)

    logl_marg, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all, logl_all, m_eff = \
        compute_model_and_logl_bootstrap(
            w_all,
            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
            y_Rzphi,
            y_xy_boot_j, y_h1_boot_j, y_h2_boot_j, y_h3_boot_j, y_h4_boot_j,
            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
            v0, s,
        )
    jax.block_until_ready(logl_marg)

    std_val = float(jnp.std(logl_all))
    range_val = float(jnp.max(logl_all) - jnp.min(logl_all))
    penalty = float(logl_marg) - float(logl_ref)
    print(f"  {noise_scale:>6.1f} | {std_val:>10.2f} | {range_val:>12.2f} | {penalty:>10.2f} | {float(logl_marg):>12.2f} | {m_eff:>10.2f}")
