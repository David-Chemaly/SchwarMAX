"""Paper-style Schwarzschild QP example using jaxopt.OSQP.

Implements Sec. 2.7 style formulation with variables x = [w, s]:

    min_{w,s}  sum_n (s_n / eps_n)^2 + lambda * sum_i w_i^2

subject to:
    U w + s = y_obs        (N_obs equations)
    C w = d                (optional N_cons equations)
    w >= 0

Converted to OSQP canonical form:
    min_x 0.5 x^T Q x + c^T x
    s.t.  A_eq x = b_eq,   G x <= h
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxopt import OSQP


def build_paper_qp(
    U: jnp.ndarray,
    y_obs: jnp.ndarray,
    eps: jnp.ndarray,
    lambda_reg: float,
    C: jnp.ndarray | None = None,
    d: jnp.ndarray | None = None,
):
    """Build (Q, c), (A_eq, b_eq), (G, h) for OSQP.

    Args:
        U: (N_obs, N_orb) orbit-response matrix, U[n, i] = u_{in}
        y_obs: (N_obs,) observed constraints U_n
        eps: (N_obs,) uncertainties epsilon_n
        lambda_reg: regularization strength for sum_i w_i^2
        C: optional (N_cons, N_orb) exact linear constraints
        d: optional (N_cons,) RHS for exact constraints
    """
    n_obs, n_orb = U.shape
    x_dim = n_orb + n_obs

    # Objective: 0.5 x^T Q x + c^T x, with x=[w;s].
    # To represent sum (s/eps)^2 + lambda sum w^2:
    #   Q_ww = 2*lambda*I, Q_ss = 2*diag(1/eps^2), c=0.
    Q = jnp.zeros((x_dim, x_dim), dtype=U.dtype)
    Q = Q.at[:n_orb, :n_orb].set(2.0 * lambda_reg * jnp.eye(n_orb, dtype=U.dtype))
    Q = Q.at[n_orb:, n_orb:].set(2.0 * jnp.diag(1.0 / (eps**2)))
    c = jnp.zeros((x_dim,), dtype=U.dtype)

    # Equality constraints:
    #   U w + s = y_obs
    # and optionally C w = d.
    A_obs = jnp.hstack([U, jnp.eye(n_obs, dtype=U.dtype)])
    b_obs = y_obs

    if C is None:
        A_eq = A_obs
        b_eq = b_obs
    else:
        if d is None:
            raise ValueError("d must be provided when C is provided.")
        n_cons = C.shape[0]
        A_cons = jnp.hstack([C, jnp.zeros((n_cons, n_obs), dtype=U.dtype)])
        A_eq = jnp.vstack([A_cons, A_obs])
        b_eq = jnp.concatenate([d, b_obs])

    # Inequality constraints for nonnegative orbit weights: w >= 0.
    # In Gx <= h form: -w <= 0.
    G = jnp.hstack([-jnp.eye(n_orb, dtype=U.dtype), jnp.zeros((n_orb, n_obs), dtype=U.dtype)])
    h = jnp.zeros((n_orb,), dtype=U.dtype)

    return (Q, c), (A_eq, b_eq), (G, h)


def solve_paper_qp_with_osqp(
    U: jnp.ndarray,
    y_obs: jnp.ndarray,
    eps: jnp.ndarray,
    lambda_reg: float,
    C: jnp.ndarray | None = None,
    d: jnp.ndarray | None = None,
    *,
    maxiter: int = 4000,
    tol: float = 1e-4,
):
    """Solve the paper-style QP with jaxopt.OSQP."""
    params_obj, params_eq, params_ineq = build_paper_qp(U, y_obs, eps, lambda_reg, C=C, d=d)

    solver = OSQP(
        maxiter=maxiter,
        tol=tol,
        # In this jaxopt build, enabling infeasibility checks may prematurely
        # report status=2 on otherwise feasible problems. Keep it off for a
        # straightforward numerical solve in this example.
        check_primal_dual_infeasability=False,
    )
    sol = solver.run(params_obj=params_obj, params_eq=params_eq, params_ineq=params_ineq)

    x = sol.params.primal
    n_obs, n_orb = U.shape
    w = x[:n_orb]
    s = x[n_orb:]

    return w, s, sol


def _paper_objective(w: jnp.ndarray, s: jnp.ndarray, eps: jnp.ndarray, lambda_reg: float):
    return jnp.sum((s / eps) ** 2) + lambda_reg * jnp.sum(w**2)


def demo():
    """Small synthetic demo showing how to call the solver."""
    key = jax.random.PRNGKey(0)
    n_orb = 8
    n_obs = 5

    k1, k2 = jax.random.split(key)

    # Synthetic orbit-response matrix and true nonnegative weights.
    U = jax.random.uniform(k1, (n_obs, n_orb), minval=0.1, maxval=1.0)
    w_true = jnp.array([0.8, 0.4, 0.0, 0.2, 0.0, 0.5, 0.1, 0.3], dtype=U.dtype)
    eps = 0.05 * jnp.ones((n_obs,), dtype=U.dtype)

    noise = 0.02 * jax.random.normal(k2, (n_obs,), dtype=U.dtype)
    y_obs = U @ w_true + noise

    # Optional exact linear constraint (like a normalization/self-consistency row):
    # sum_i w_i = known total mass.
    C = jnp.ones((1, n_orb), dtype=U.dtype)
    d = jnp.array([jnp.sum(w_true)], dtype=U.dtype)

    lambda_reg = 1e-2
    w_hat, s_hat, sol = solve_paper_qp_with_osqp(
        U,
        y_obs,
        eps,
        lambda_reg,
        C=C,
        d=d,
        maxiter=4000,
        tol=1e-4,
    )

    # Consistency check: s = y_obs - U w from equality constraints.
    slack_check = jnp.max(jnp.abs(s_hat - (y_obs - U @ w_hat)))

    print("OSQP status:", int(sol.state.status))
    print("OSQP error :", float(sol.state.error))
    print("w_hat      :", w_hat)
    print("s_hat      :", s_hat)
    print("max|s-(y-Uw)|:", float(slack_check))
    print("objective  :", float(_paper_objective(w_hat, s_hat, eps, lambda_reg)))


if __name__ == "__main__":
    demo()
