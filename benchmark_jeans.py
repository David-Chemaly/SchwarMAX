"""
Benchmark: optimized Jeans moments vs current implementation.

Current: 1000-pt trapezoid + numerical dPhi/dz (central diff, 2 pot calls)
Optimized: Gauss-Legendre quadrature + jax.grad for dPhi/dz and dPhi/dR

Usage:
    python benchmark_jeans.py
"""
import sys
import time
import numpy as np
from functools import partial

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
sys.path.append(path)

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from model_bar import (
    get_jeans_moments, density_func, potential_func, dPhi_dz, dPhi_dR,
)

# ============================================================
# Optimized version: Gauss-Legendre + jax.grad
# ============================================================

# Use jax.grad for exact derivatives (1 backward pass instead of 2 forward)
@jax.jit
def dPhi_dz_grad(x, y, z, params_baryon, params_halo):
    return jax.grad(potential_func, argnums=2)(x, y, z, params_baryon, params_halo)

@jax.jit
def dPhi_dR_grad(x, y, z, params_baryon, params_halo):
    """dPhi/dR using jax.grad through R = sqrt(x^2+y^2)."""
    R = jnp.sqrt(x**2 + y**2)
    return jax.grad(lambda r: potential_func(r, 0.0, z, params_baryon, params_halo))(R)


def make_get_jeans_moments_gl(n_quad):
    """Create optimized Jeans moments function with n_quad Gauss-Legendre points."""

    # Precompute GL nodes and weights on [0, 1]
    # We'll transform to [|z_star|, z_max] at runtime
    nodes_01, weights_01 = np.polynomial.legendre.leggauss(n_quad)
    # Transform from [-1, 1] to [0, 1]
    nodes_01 = jnp.array((nodes_01 + 1) / 2, dtype=jnp.float32)
    weights_01 = jnp.array(weights_01 / 2, dtype=jnp.float32)

    @jax.jit
    def get_jeans_moments_gl(x_star, y_star, z_star, params_baryon, params_disc, params_halo, anisotropy_b=1.0):
        R_star = jnp.sqrt(x_star**2 + y_star**2)
        z_lo = jnp.abs(z_star)
        z_hi = 10.0
        scale = z_hi - z_lo

        # GL points in [z_lo, z_hi]
        z_pts = z_lo + scale * nodes_01
        w_pts = weights_01 * scale

        # --- Step 1: sigma_z^2 = (1/nu) * integral(nu * dPhi/dz, z_star, inf) ---
        def integrand_z(z_prime):
            return density_func(x_star, y_star, z_prime, params_disc) * \
                   dPhi_dz_grad(x_star, y_star, z_prime, params_baryon, params_halo)

        int_vals = jax.vmap(integrand_z)(z_pts)
        integral_val = jnp.sum(w_pts * int_vals)

        nu_val = density_func(x_star, y_star, z_star, params_disc)
        sigma_z2 = jnp.maximum((1.0 / nu_val) * integral_val, 0.0)
        sigma_z = jnp.sqrt(sigma_z2)

        # --- Step 2: sigma_R^2 ---
        sigma_R2 = anisotropy_b * sigma_z2
        sigma_R = jnp.sqrt(sigma_R2)

        # --- Step 3: v_phi^2 via radial Jeans ---
        # d(nu*sigma_R^2)/dR via jax.grad of the vertical pressure integral
        def vertical_pressure(r_in):
            def integrand_r(z_prime):
                return density_func(r_in, 0.0, z_prime, params_disc) * \
                       dPhi_dz_grad(r_in, 0.0, z_prime, params_baryon, params_halo)
            int_vals = jax.vmap(integrand_r)(z_pts)
            return jnp.sum(w_pts * int_vals)

        dR = 5e-3
        Pzz_plus = vertical_pressure(R_star + dR)
        Pzz_minus = vertical_pressure(R_star - dR)
        d_nu_sigR2_dR = anisotropy_b * (Pzz_plus - Pzz_minus) / (2*dR)

        term1 = sigma_R2
        term2 = (R_star / nu_val) * d_nu_sigR2_dR
        term3 = R_star * dPhi_dR_grad(x_star, y_star, z_star, params_baryon, params_halo)

        v_phi_total_sq = term1 + term2 + term3

        # --- Step 4: Separate rotation vs dispersion ---
        sigma_phi = sigma_R
        v_streaming_sq = jnp.maximum(v_phi_total_sq - sigma_phi**2, 0.0)
        v_mean_phi = jnp.sqrt(v_streaming_sq)

        output = jax.lax.cond(
            nu_val <= 0,
            lambda: (0.0, 0.0, 0.0, 0.0),
            lambda: (v_mean_phi, sigma_R, sigma_z, sigma_phi),
        )
        return output

    return get_jeans_moments_gl


# ============================================================
# Setup params (same as model_bar.py)
# ============================================================
logM_disc, logM_bar = 10.71, 9.94
logRs_disk, logHs_disk, logL_bar = 0.62, 0.02, 0.56
L_bar = 10.0**logL_bar
Hs_disc = 10.0**logHs_disk

params_halo_pot = {
    'logM': 11.8, 'Rs': 10**1.24, 'a': 1.0, 'b': 1.0, 'c': 1.0,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
}

params_disc = {
    'logM_disc': logM_disc, 'Rs_disc': 10**logRs_disk, 'Hs_disc': Hs_disc,
    'logM_bar': logM_bar, 'L_bar': L_bar, 'a_bar': L_bar/5.0, 'b_bar': Hs_disc,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
}

params_baryon = params_disc.copy()

# ============================================================
# Test points
# ============================================================
rng = np.random.default_rng(42)
N_test_pts = 50
R_test = rng.uniform(0.5, 8.0, N_test_pts)
phi_test = rng.uniform(0, 2*np.pi, N_test_pts)
x_test = R_test * np.cos(phi_test)
y_test = R_test * np.sin(phi_test)
z_test = rng.uniform(0.1, 2.0, N_test_pts)

# ============================================================
# 1. Accuracy: compare dPhi_dz methods
# ============================================================
print("=" * 60)
print("Accuracy: dPhi/dz numerical vs jax.grad")
print("=" * 60)

diffs_dz = []
for i in range(min(10, N_test_pts)):
    v_num = float(dPhi_dz(x_test[i], y_test[i], z_test[i], params_baryon, params_halo_pot))
    v_grad = float(dPhi_dz_grad(x_test[i], y_test[i], z_test[i], params_baryon, params_halo_pot))
    rel = abs(v_num - v_grad) / (abs(v_num) + 1e-30)
    diffs_dz.append(rel)
    if i < 5:
        print(f"  pt {i}: numerical={v_num:.6f}, grad={v_grad:.6f}, rel_diff={rel:.2e}")
print(f"  Max rel diff: {max(diffs_dz):.2e}")

# ============================================================
# 2. Accuracy: compare dPhi_dR methods
# ============================================================
print()
print("=" * 60)
print("Accuracy: dPhi/dR numerical vs jax.grad")
print("=" * 60)

diffs_dR = []
for i in range(min(10, N_test_pts)):
    v_num = float(dPhi_dR(x_test[i], y_test[i], z_test[i], params_baryon, params_halo_pot))
    v_grad = float(dPhi_dR_grad(x_test[i], y_test[i], z_test[i], params_baryon, params_halo_pot))
    rel = abs(v_num - v_grad) / (abs(v_num) + 1e-30)
    diffs_dR.append(rel)
    if i < 5:
        print(f"  pt {i}: numerical={v_num:.6f}, grad={v_grad:.6f}, rel_diff={rel:.2e}")
print(f"  Max rel diff: {max(diffs_dR):.2e}")

# ============================================================
# 3. Accuracy: compare full Jeans moments (trap-1000 vs GL-N)
# ============================================================
print()
print("=" * 60)
print("Accuracy: full Jeans moments")
print("=" * 60)

# Reference: current implementation (1000-pt trapezoid + numerical derivs)
get_jeans_vmap_ref = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))

xs = jnp.array(x_test)
ys = jnp.array(y_test)
zs = jnp.array(z_test)

ref = get_jeans_vmap_ref(xs, ys, zs, params_baryon, params_disc, params_halo_pot, 1.0)
ref_vphi, ref_sigR, ref_sigz, ref_sigphi = [np.array(r) for r in ref]

for n_gl in [20, 30, 50, 100, 200]:
    get_jeans_gl = make_get_jeans_moments_gl(n_gl)
    get_jeans_gl_vmap = jax.vmap(get_jeans_gl, in_axes=(0,0,0,None,None,None,None))

    result = get_jeans_gl_vmap(xs, ys, zs, params_baryon, params_disc, params_halo_pot, 1.0)
    gl_vphi, gl_sigR, gl_sigz, gl_sigphi = [np.array(r) for r in result]

    # Relative errors (skip zero values)
    mask = ref_sigz > 1.0  # only compare meaningful values
    err_vphi = np.max(np.abs(gl_vphi[mask] - ref_vphi[mask]) / (np.abs(ref_vphi[mask]) + 1e-10))
    err_sigR = np.max(np.abs(gl_sigR[mask] - ref_sigR[mask]) / (np.abs(ref_sigR[mask]) + 1e-10))
    err_sigz = np.max(np.abs(gl_sigz[mask] - ref_sigz[mask]) / (np.abs(ref_sigz[mask]) + 1e-10))

    print(f"  GL-{n_gl:>3}: max rel err  v_phi={err_vphi:.4e}  sig_R={err_sigR:.4e}  sig_z={err_sigz:.4e}")

# ============================================================
# 4. Timing: current vs GL for different N_quad
# ============================================================
print()
print("=" * 60)
print("Timing comparison (vmapped over 50 stars)")
print("=" * 60)

# Warmup + time reference
_ = get_jeans_vmap_ref(xs, ys, zs, params_baryon, params_disc, params_halo_pot, 1.0)
jax.block_until_ready(_)

times_ref = []
for _ in range(10):
    t0 = time.perf_counter()
    r = get_jeans_vmap_ref(xs, ys, zs, params_baryon, params_disc, params_halo_pot, 1.0)
    jax.block_until_ready(r)
    times_ref.append(time.perf_counter() - t0)
t_ref = np.median(times_ref) * 1000
print(f"  Current (trap-1000 + numerical): {t_ref:.1f} ms")

for n_gl in [20, 30, 50, 100, 200]:
    get_jeans_gl = make_get_jeans_moments_gl(n_gl)
    get_jeans_gl_vmap = jax.vmap(get_jeans_gl, in_axes=(0,0,0,None,None,None,None))

    # Warmup
    _ = get_jeans_gl_vmap(xs, ys, zs, params_baryon, params_disc, params_halo_pot, 1.0)
    jax.block_until_ready(_)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        r = get_jeans_gl_vmap(xs, ys, zs, params_baryon, params_disc, params_halo_pot, 1.0)
        jax.block_until_ready(r)
        times.append(time.perf_counter() - t0)
    t_gl = np.median(times) * 1000
    print(f"  GL-{n_gl:>3} + jax.grad:             {t_gl:.1f} ms  ({t_ref/t_gl:.2f}x)")
