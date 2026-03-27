"""
Benchmark: early stopping in adaptive integrator.

Compares the current integrator (all N_max steps run acc_fn) vs
a modified version that skips acc_fn when t > T_total using jax.lax.cond.

Uses real orbital data from test_integration_convergence.py.

Usage:
    python benchmark_early_stop.py
"""
import sys
import time
import numpy as np
from functools import partial

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
sys.path.append(path)

import jax
import jax.numpy as jnp

from constants import EPSILON, KPCGYR_TO_KMS
from integrants_with_binning import (
    _deriv_barred, integrate_adaptive_barred,
    _integrate_adaptive_vmap,
    assign_regular_grid,
)

# ============================================================
# Load real setup
# ============================================================
from test_integration_convergence import (
    acc_fn, pot_fn, w0_new,
    T_orb, params_halo_pot, params_baryon,
    num_Vbin, bin_mapping, num_per_bin, v0, s, rotation_matrix,
    Omega_bar,
)

Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]])
Rzphi_n_grid = jnp.array([10,6,6])
xy_n_grid = jnp.array([60,40])
Rzphi_n_tot = 360

N_step_per_orb = 100
N_dynamical_time = 100
N_max = N_step_per_orb * N_dynamical_time  # 10000

T_total = T_orb * N_dynamical_time
dt_init = T_orb / N_step_per_orb
atol, rtol = 1e-7, 1e-4
dt_min, dt_max = 1e-5, 0.3

# ============================================================
# Modified integrator with early stop via jax.lax.cond
# ============================================================
def _compute_EJ(y, pot_fn, Omega):
    x, y_pos, z, vx, vy, vz = y[0], y[1], y[2], y[3], y[4], y[5]
    E_kin = 0.5 * (vx**2 + vy**2 + vz**2)
    E_pot = pot_fn(x, y_pos, z)
    Lz = x * vy - y_pos * vx
    return E_kin + E_pot - Omega * Lz


@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_barred_early_stop(
    w0, acc_fn, pot_fn, N_max, T_total,
    dt_init=0.010, Omega=0.0,
    atol=1e-8, rtol=1e-6,
    dt_min=1e-5, dt_max=0.1,
    num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
    num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
    Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
    XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
    nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
    num_segments_Rzphi=360,
    v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
    rotation_matrix=jnp.eye(3),
):
    """Same as integrate_adaptive_barred but skips acc_fn when t > T_total."""
    k1_init = _deriv_barred(w0, acc_fn, Omega)
    EJ_0 = _compute_EJ(w0, pot_fn, Omega)
    e1, e2, e3, e4 = -5.0/72.0, 1.0/12.0, 1.0/9.0, -1.0/8.0
    safety = 0.9

    def scan_body(carry, _):
        t, y, dt, k1 = carry

        # === Active step: run full RK integration ===
        def do_step(args):
            t_, y_, dt_, k1_ = args
            k2 = _deriv_barred(y_ + 0.5 * dt_ * k1_, acc_fn, Omega)
            k3 = _deriv_barred(y_ + 0.75 * dt_ * k2, acc_fn, Omega)
            y_new = y_ + dt_ * (2.0/9.0 * k1_ + 1.0/3.0 * k2 + 4.0/9.0 * k3)
            k4 = _deriv_barred(y_new, acc_fn, Omega)
            err_vec = dt_ * (e1 * k1_ + e2 * k2 + e3 * k3 + e4 * k4)
            scale = atol + rtol * jnp.maximum(jnp.fabs(y_), jnp.fabs(y_new))
            err_norm = jnp.sqrt(jnp.mean((err_vec / scale)**2))
            accept = err_norm <= 1.0
            y_out = jnp.where(accept, y_new, y_)
            t_new = jnp.where(accept, t_ + dt_, t_)
            k1_new = jnp.where(accept, k4, k1_)
            scale_factor = safety * jnp.power(jnp.maximum(err_norm, 1e-10), -1.0/3.0)
            scale_factor = jnp.clip(scale_factor, 0.2, 5.0)
            dt_new = jnp.clip(dt_ * scale_factor, dt_min, dt_max)
            dt_out = jnp.where(accept, dt_, 0.0)
            return t_new, y_out, dt_new, k1_new, dt_out

        # === Done step: no acc_fn, just carry forward ===
        def skip_step(args):
            t_, y_, dt_, k1_ = args
            return t_, y_, dt_, k1_, 0.0

        t_new, y_out, dt_new, k1_new, dt_out = jax.lax.cond(
            t >= T_total, skip_step, do_step, (t, y, dt, k1)
        )

        valid_flag = (dt_out > 0.0).astype(jnp.float32)
        return (t_new, y_out, dt_new, k1_new), (y_out, valid_flag, dt_out)

    init_carry = (0.0, w0, dt_init, k1_init)
    (t_final, y_final, dt_final, _), (y_traj, valid_mask, dt_traj) = jax.lax.scan(
        scan_body, init_carry, xs=None, length=N_max
    )

    n_accepted = jnp.sum(valid_mask)
    EJ_final = _compute_EJ(y_final, pot_fn, Omega)
    delta_EJ = jnp.fabs(EJ_final / EJ_0 - 1.0)
    valid = jnp.where((delta_EJ < 0.1) & (t_final > T_total / 10), 1.0, 0.0)

    # ── Apply 4-fold bar symmetry before binning ──
    sign_sym = jnp.array([
        [ 1,  1,  1,  1,  1,  1],
        [ 1,  1, -1,  1,  1, -1],
        [-1, -1,  1, -1, -1,  1],
        [-1, -1, -1, -1, -1, -1],
    ])
    y_traj = (y_traj[None, :, :] * sign_sym[:, None, :]).reshape(-1, 6)
    dt_traj = jnp.tile(dt_traj, 4)

    # ── Post-scan binning with dt-weighting ──
    R_vals = jnp.sqrt(y_traj[:, 0]**2 + y_traj[:, 1]**2)
    phi_vals = jnp.arctan2(y_traj[:, 1], y_traj[:, 0])
    Rzphi = jnp.stack([R_vals, y_traj[:, 2], phi_vals], axis=-1)

    x_pos = y_traj[:, :3]
    v_vel = y_traj[:, 3:]
    x_rot = (rotation_matrix @ x_pos.T).T
    v_rot = (rotation_matrix @ v_vel.T).T
    wN_rot = jnp.concatenate([x_rot, v_rot], axis=-1)
    XY = jnp.stack([wN_rot[:, 0], wN_rot[:, 2]], axis=-1)

    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_indices = assign_regular_grid(Rzphi, grid_min=Rzphi_minmax[:, 0],
                                         grid_max=Rzphi_minmax[:, 1],
                                         n_bins=nRzphi, strides=Rzphi_strides)
    XY_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    XY_indices = assign_regular_grid(XY, grid_min=XY_minmax[:, 0],
                                      grid_max=XY_minmax[:, 1],
                                      n_bins=nXY, strides=XY_strides)

    Vbin_indices = bin_mapping[XY_indices]
    T_integrated = jnp.sum(dt_traj) + EPSILON
    dt_norm = dt_traj / T_integrated

    Rzphi_bin_counts = jax.ops.segment_sum(dt_norm, Rzphi_indices,
                                            num_segments=num_segments_Rzphi)

    v0_cell = v0[Vbin_indices]
    s_cell = s[Vbin_indices]
    vy = wN_rot[:, 4] * KPCGYR_TO_KMS
    w = (vy - v0_cell) / s_cell
    eps = EPSILON

    counts = jax.ops.segment_sum(dt_norm, Vbin_indices, num_segments=num_Vbin)
    sum_w1 = jax.ops.segment_sum(dt_norm * w, Vbin_indices, num_segments=num_Vbin)
    sum_w2 = jax.ops.segment_sum(dt_norm * w**2, Vbin_indices, num_segments=num_Vbin)
    sum_w3 = jax.ops.segment_sum(dt_norm * w**3, Vbin_indices, num_segments=num_Vbin)
    sum_w4 = jax.ops.segment_sum(dt_norm * w**4, Vbin_indices, num_segments=num_Vbin)

    norm = counts + eps
    w1 = sum_w1 / norm
    w2 = sum_w2 / norm
    w3 = sum_w3 / norm
    w4 = sum_w4 / norm

    h1 = w1
    h2 = w2 - 1.0
    h3 = (w3 - 3 * w1) / jnp.sqrt(3.0)
    h4 = (w4 - 6 * w2 + 3) / (2 * jnp.sqrt(6.0))

    surface_density = counts

    return Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, n_accepted, t_final


# ============================================================
# Vmap both versions
# ============================================================
_integrate_early_stop_vmap = jax.vmap(
    integrate_adaptive_barred_early_stop,
    in_axes=(0, None, None, None, 0,
             0, None,
             None, None,
             None, None,
             None, None, None,
             None, None,
             None, None, None,
             None, None, None),
)

# ============================================================
# Test
# ============================================================
N_test = 200
w0_test = w0_new[:N_test]
T_total_test = T_total[:N_test]
dt_init_test = dt_init[:N_test]
T_orb_test = T_orb[:N_test]

print(f"N_test = {N_test}, N_max = {N_max}")
print(f"T_orb range: [{float(T_orb_test.min()):.4f}, {float(T_orb_test.max()):.4f}] Gyr")

# ── Current integrator ──
print("\nWarming up current integrator...")
t0 = time.time()
res_current = _integrate_adaptive_vmap(
    w0_test, acc_fn, pot_fn, N_max, T_total_test,
    dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim_grid, xy_lim_grid, Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
    v0, s, rotation_matrix,
)
jax.block_until_ready(res_current)
print(f"  JIT compile: {time.time()-t0:.1f}s")

# ── Early stop integrator ──
print("Warming up early-stop integrator...")
t0 = time.time()
res_early = _integrate_early_stop_vmap(
    w0_test, acc_fn, pot_fn, N_max, T_total_test,
    dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim_grid, xy_lim_grid, Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
    v0, s, rotation_matrix,
)
jax.block_until_ready(res_early)
print(f"  JIT compile: {time.time()-t0:.1f}s")

# ── Verify outputs match ──
t_current = np.array(res_current[-1])
t_early = np.array(res_early[-1])
# t_final should match up to T_total (early stop doesn't go past)
# Current goes past T_total, early stop caps at T_total
print(f"\nt_final (current):    mean={t_current.mean():.2f}, min={t_current.min():.2f}")
print(f"t_final (early stop): mean={t_early.mean():.2f}, min={t_early.min():.2f}")

N_dyn_current = t_current / np.array(T_orb_test)
N_dyn_early = t_early / np.array(T_orb_test)
print(f"Dyn times (current):    mean={N_dyn_current.mean():.0f}")
print(f"Dyn times (early stop): mean={N_dyn_early.mean():.0f}")

# Check kinematic outputs agree (surface density, h1-h4)
sd_curr = np.array(res_current[1])
sd_early = np.array(res_early[1])
h1_curr = np.array(res_current[2])
h1_early = np.array(res_early[2])
print(f"\nSurface density max rel diff: {np.max(np.abs(sd_curr - sd_early) / (np.abs(sd_curr) + 1e-30)):.4f}")
print(f"h1 max rel diff: {np.max(np.abs(h1_curr - h1_early) / (np.abs(h1_curr) + 1e-30)):.4f}")

# ── Timing ──
print("\n" + "=" * 60)
print("Timing comparison")
print("=" * 60)

n_iter = 5
times_curr = []
times_early = []

for _ in range(n_iter):
    t0 = time.time()
    r = _integrate_adaptive_vmap(
        w0_test, acc_fn, pot_fn, N_max, T_total_test,
        dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_lim_grid, xy_lim_grid, Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
        v0, s, rotation_matrix,
    )
    jax.block_until_ready(r)
    times_curr.append(time.time() - t0)

    t0 = time.time()
    r = _integrate_early_stop_vmap(
        w0_test, acc_fn, pot_fn, N_max, T_total_test,
        dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_lim_grid, xy_lim_grid, Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
        v0, s, rotation_matrix,
    )
    jax.block_until_ready(r)
    times_early.append(time.time() - t0)

tc = np.median(times_curr)
te = np.median(times_early)
print(f"Current:    {tc:.3f}s")
print(f"Early stop: {te:.3f}s")
print(f"Speedup: {tc/te:.2f}x")
