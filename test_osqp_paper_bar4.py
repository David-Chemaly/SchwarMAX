"""OSQP test for bar orbital library using paper-style quadratic setup.

This script solves a convex QP equivalent to the Sec. 2.7 formulation after
eliminating slack variables s:

  min_w  0.5 ||U w - y||^2 + 0.5 * lambda_reg * ||w||^2
  s.t.   w >= 0

where U,y are built from the Schwarzschild linear constraints.

Then it reports the same log_likelihood metric used in test_weights_optimisation.py.
"""

from __future__ import annotations

import os
import time
import pickle
import numpy as np

import jax
import jax.numpy as jnp
from jaxopt import OSQP

from constants import EPSILON


def build_weighted_blocks(
    A_Rzphi,
    A_xy,
    A_h1,
    A_h2,
    A_h3,
    A_h4,
    y_Rzphi,
    y_xy,
    y_h1,
    y_h2,
    y_h3,
    y_h4,
    sig_Rzphi,
    sig_xy,
    sig_A1,
    sig_A2,
    sig_A3,
    sig_A4,
):
    """Build weighted linear system blocks matching (unclipped, no-entropy) _nll_z terms."""
    eps = 1e-8

    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    # Same relative term weighting as model._nll_z
    w_rzphi = jnp.sqrt(5.0 / A_Rzphi.shape[0])
    w_xy = jnp.sqrt(5.0 / A_xy.shape[0])
    w_h = jnp.sqrt(1.0 / A_h1.shape[0])

    U_rz = w_rzphi * (A_Rzphi / (sig_Rzphi[:, None] + eps))
    y_rz = w_rzphi * (y_Rzphi / (sig_Rzphi + eps))

    U_xy = w_xy * (A_xy / (sig_xy[:, None] + eps))
    y_xy_obs = w_xy * (y_xy / (sig_xy + eps))

    # _nll_z uses (A_hi * A_xy) @ w / y_xy.
    U_h1 = w_h * ((A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps))
    U_h2 = w_h * ((A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps))
    U_h3 = w_h * ((A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps))
    U_h4 = w_h * ((A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps))

    y_h1_obs = w_h * (y_h1 / (sig_A1 + eps))
    y_h2_obs = w_h * (y_h2 / (sig_A2 + eps))
    y_h3_obs = w_h * (y_h3 / (sig_A3 + eps))
    y_h4_obs = w_h * (y_h4 / (sig_A4 + eps))

    U_blocks = (U_rz, U_xy, U_h1, U_h2, U_h3, U_h4)
    y_blocks = (y_rz, y_xy_obs, y_h1_obs, y_h2_obs, y_h3_obs, y_h4_obs)
    return U_blocks, y_blocks


def solve_osqp_nonnegative(U_blocks, y_blocks, lambda_reg=1e-2, maxiter=6000, tol=1e-4):
    """Solve nonnegative ridge least-squares with matrix-free OSQP."""
    n_orb = U_blocks[0].shape[1]

    def apply_U(w):
        return jnp.concatenate([U @ w for U in U_blocks], axis=0)

    # Static split indices for U^T v (must be Python ints for JAX-traced slicing).
    split_idx = tuple(np.cumsum([U.shape[0] for U in U_blocks])[:-1].tolist())

    def apply_UT(v):
        v0, v1, v2, v3, v4, v5 = jnp.split(v, split_idx, axis=0)
        return (
            U_blocks[0].T @ v0
            + U_blocks[1].T @ v1
            + U_blocks[2].T @ v2
            + U_blocks[3].T @ v3
            + U_blocks[4].T @ v4
            + U_blocks[5].T @ v5
        )

    y_stack = jnp.concatenate(y_blocks, axis=0)
    c = -apply_UT(y_stack)

    def matvec_Q(_, w):
        return apply_UT(apply_U(w)) + lambda_reg * w

    # w >= 0  <=>  -w <= 0
    def matvec_G(_, w):
        return -w

    solver = OSQP(
        matvec_Q=matvec_Q,
        matvec_G=matvec_G,
        maxiter=maxiter,
        tol=tol,
        # This check can return false positives in this jaxopt build.
        check_primal_dual_infeasability=False,
    )

    t0 = time.time()
    sol = solver.run(
        params_obj=(None, c),
        params_ineq=(None, jnp.zeros((n_orb,), dtype=c.dtype)),
    )
    w = sol.params.primal
    w.block_until_ready()
    t1 = time.time()

    return w, sol, t1 - t0


def compute_metric_logL(
    w,
    A_xy,
    A_h1,
    A_h2,
    A_h3,
    A_h4,
    y_xy,
    y_h1,
    y_h2,
    y_h3,
    y_h4,
    sig_xy,
    sig_A1,
    sig_A2,
    sig_A3,
    sig_A4,
):
    """Same metric as test_weights_optimisation.py."""
    A_h1w = A_h1 * A_xy
    A_h2w = A_h2 * A_xy
    A_h3w = A_h3 * A_xy
    A_h4w = A_h4 * A_xy

    density_2DXY = A_xy @ w
    h1_model = (A_h1w @ w) / y_xy
    h2_model = (A_h2w @ w) / y_xy
    h3_model = (A_h3w @ w) / y_xy
    h4_model = (A_h4w @ w) / y_xy

    h3_model = jnp.where(jnp.isnan(h3_model), 0.0, h3_model)
    h4_model = jnp.where(jnp.isnan(h4_model), 0.0, h4_model)

    res_density = ((density_2DXY - y_xy) / (sig_xy + EPSILON)) ** 2
    res_h1 = ((h1_model - y_h1) / (sig_A1 + 1e-3)) ** 2
    res_h2 = ((h2_model - y_h2) / (sig_A2 + 1e-3)) ** 2
    res_h3 = ((h3_model - y_h3) / (sig_A3 + 1e-3)) ** 2
    res_h4 = ((h4_model - y_h4) / (sig_A4 + 1e-3)) ** 2

    res_h1 = jnp.where((h1_model < 9.9), res_h1, 0)
    res_h2 = jnp.where((h2_model < 9.9), res_h2, 0)
    res_h3 = jnp.where((h3_model < 9.9), res_h3, 0)
    res_h4 = jnp.where((h4_model < 9.9), res_h4, 0)

    val1 = jnp.nansum(-0.5 * res_density) / len(density_2DXY)
    val4 = jnp.nansum(-0.5 * res_h1) / len(h1_model)
    val5 = jnp.nansum(-0.5 * res_h2) / len(h2_model)
    val6 = jnp.nansum(-0.5 * res_h3) / len(h3_model)
    val7 = jnp.nansum(-0.5 * res_h4) / len(h4_model)

    return val1 + val4 + val5 + val6 + val7


def main():
    path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
    pkl = os.path.join(path_data, 'orbital_library_bar_4.pkl')

    lambda_reg = float(os.environ.get('QP_LAMBDA', '1e-2'))
    maxiter = int(os.environ.get('QP_MAXITER', '6000'))
    tol = float(os.environ.get('QP_TOL', '1e-4'))

    with open(pkl, 'rb') as f:
        (
            A_Rzphi,
            A_xy,
            A_h1,
            A_h2,
            A_h3,
            A_h4,
            y_Rzphi,
            y_xy,
            y_h1,
            y_h2,
            y_h3,
            y_h4,
            sig_Rzphi,
            sig_xy,
            sig_A1,
            sig_A2,
            sig_A3,
            sig_A4,
        ) = pickle.load(f)

    # Use float32 to match rest of pipeline behavior.
    A_Rzphi = jnp.asarray(A_Rzphi, dtype=jnp.float32)
    A_xy = jnp.asarray(A_xy, dtype=jnp.float32)
    A_h1 = jnp.asarray(A_h1, dtype=jnp.float32)
    A_h2 = jnp.asarray(A_h2, dtype=jnp.float32)
    A_h3 = jnp.asarray(A_h3, dtype=jnp.float32)
    A_h4 = jnp.asarray(A_h4, dtype=jnp.float32)
    y_Rzphi = jnp.asarray(y_Rzphi, dtype=jnp.float32)
    y_xy = jnp.asarray(y_xy, dtype=jnp.float32)
    y_h1 = jnp.asarray(y_h1, dtype=jnp.float32)
    y_h2 = jnp.asarray(y_h2, dtype=jnp.float32)
    y_h3 = jnp.asarray(y_h3, dtype=jnp.float32)
    y_h4 = jnp.asarray(y_h4, dtype=jnp.float32)
    sig_Rzphi = jnp.asarray(sig_Rzphi, dtype=jnp.float32)
    sig_xy = jnp.asarray(sig_xy, dtype=jnp.float32)
    sig_A1 = jnp.asarray(sig_A1, dtype=jnp.float32)
    sig_A2 = jnp.asarray(sig_A2, dtype=jnp.float32)
    sig_A3 = jnp.asarray(sig_A3, dtype=jnp.float32)
    sig_A4 = jnp.asarray(sig_A4, dtype=jnp.float32)

    U_blocks, y_blocks = build_weighted_blocks(
        A_Rzphi,
        A_xy,
        A_h1,
        A_h2,
        A_h3,
        A_h4,
        y_Rzphi,
        y_xy,
        y_h1,
        y_h2,
        y_h3,
        y_h4,
        sig_Rzphi,
        sig_xy,
        sig_A1,
        sig_A2,
        sig_A3,
        sig_A4,
    )

    w_hat, sol, elapsed = solve_osqp_nonnegative(
        U_blocks,
        y_blocks,
        lambda_reg=lambda_reg,
        maxiter=maxiter,
        tol=tol,
    )

    logL = compute_metric_logL(
        w_hat,
        A_xy,
        A_h1,
        A_h2,
        A_h3,
        A_h4,
        y_xy,
        y_h1,
        y_h2,
        y_h3,
        y_h4,
        sig_xy,
        sig_A1,
        sig_A2,
        sig_A3,
        sig_A4,
    )

    print('QP lambda      :', lambda_reg)
    print('QP maxiter/tol :', maxiter, tol)
    print('OSQP status    :', int(sol.state.status))
    print('OSQP error     :', float(sol.state.error))
    print('Solve time [s] :', float(elapsed))
    print('weights sum/min/max:', float(jnp.sum(w_hat)), float(jnp.min(w_hat)), float(jnp.max(w_hat)))
    print('log_likelihood :', float(logL))


if __name__ == '__main__':
    main()
