# ADMM NNLS Solver For Orbital Weights

This note summarizes the orbital-weight solver used in
`model_bar.py`, especially:

- `solve_nnls_admm()` at `model_bar.py:533`
- `solve_nnls_admm_bootstrap()` at `model_bar.py:656`
- the call sites in `model()` and `model_bootstrap()`

The goal is to connect the math to the exact Python implementation.

## 1. What problem is being solved?

After the orbit library has been integrated, the code has a set of linear
response matrices:

- `A_Rzphi`: intrinsic 3D density contribution of each orbit
- `A_xy`: projected surface-density contribution of each orbit
- `A_h1`, `A_h2`, `A_h3`, `A_h4`: Gauss-Hermite numerator-like orbit terms

If `w` is the vector of non-negative orbital weights, then the model is built
as a weighted sum of orbit contributions.

The solver is trying to find weights `w >= 0` such that the orbit library
matches the target observables:

- intrinsic density `y_Rzphi`
- projected density `y_xy`
- kinematic moments `y_h1` ... `y_h4`

after each block has been divided by its uncertainty.

In compact form, the solver minimizes

    min_{w >= 0}  1/2 ||U w - y||^2 + (reg/2) ||w||^2

This is a ridge-regularized non-negative least-squares problem.

Here:

    U = [ U_rz   ]
        [ U_xy   ]
        [ U_h1   ]
        [ U_h2   ]
        [ U_h3   ]
        [ U_h4   ]

    y = [ y_rz ]
        [ y_xy ]
        [ y_h1 ]
        [ y_h2 ]
        [ y_h3 ]
        [ y_h4 ]

The code constructs those blocks in `solve_nnls_admm()`:

    U_rz  = A_Rzphi / sig_Rzphi
    U_xy  = A_xy    / sig_xy
    U_hj  = (A_hj * A_xy) / y_xy_safe / sig_Aj

with matching normalized target vectors.

## 2. Why the GH terms still look linear

The surface density is linear in the weights:

    density_model = A_xy @ w

But the Gauss-Hermite moments are ratios, schematically:

    h_j,model = ((A_hj * A_xy) @ w) / density_model

That ratio is not linear in `w`.

To keep the weight solve as NNLS, the code freezes the denominator to the
observed projected density:

    y_xy_safe = where(abs(y_xy) > eps, y_xy, 1.0)

and uses

    h_j,linearized(w) = ((A_hj * A_xy) @ w) / y_xy_safe

inside the optimizer.

So the ADMM solve is a linearized subproblem for the weights. After solving,
the code reconstructs the final model maps explicitly from the weights.

## 3. Quadratic form used by ADMM

Define

    Q = U^T U + reg I
    c = - U^T y

Then the objective can be written as

    min_{w >= 0}  1/2 w^T Q w + c^T w + const

This is a convex quadratic program with only one constraint:

    w >= 0

That is why ADMM is a natural fit.

In the code:

- `n_orb = U.shape[1]`
- `reg = lambda_reg / n_orb`
- `Q = U.T @ U + reg * I`
- `c = -(U.T @ y)`

## 4. ADMM reformulation

ADMM introduces a copy variable `z`:

    min_{w,z}  1/2 w^T Q w + c^T w + I_+(z)
    subject to w = z

where `I_+(z)` is the indicator function of the non-negative orthant:

    I_+(z) = 0      if z >= 0 componentwise
           = +inf   otherwise

Using the scaled-dual form, the augmented Lagrangian is

    L_rho(w,z,u)
      = 1/2 w^T Q w + c^T w + I_+(z)
        + (rho/2) ||w - z + u||^2
        - (rho/2) ||u||^2

The solver then alternates between:

### 4.1 w-update

Solve the unconstrained quadratic subproblem

    w^{k+1} = argmin_w
        1/2 w^T Q w + c^T w + (rho/2) ||w - z^k + u^k||^2

Taking the gradient and setting it to zero gives

    (Q + rho I) w^{k+1} = rho (z^k - u^k) - c

This is the linear system solved every ADMM iteration.

### 4.2 z-update

Project the relaxed primal variable onto `w >= 0`:

    w_hat = alpha w^{k+1} + (1 - alpha) z^k
    z^{k+1} = Pi_+(w_hat + u^k)

where `Pi_+` is just componentwise clipping:

    Pi_+(x) = max(0, x)

### 4.3 Dual update

    u^{k+1} = u^k + w_hat - z^{k+1}

This tracks the disagreement between the unconstrained quadratic solution and
the constrained non-negative solution.

## 5. Mapping the math directly to `solve_nnls_admm()`

The core implementation is:

```python
Q = U.T @ U + reg * jnp.eye(n_orb, dtype=U.dtype)
c = -(U.T @ y)

rho = jnp.trace(Q) / n_orb
L_chol = jnp.linalg.cholesky(Q + rho * jnp.eye(n_orb, dtype=U.dtype))

w_init = jnp.ones(n_orb, dtype=U.dtype) * (jnp.sum(y_Rzphi) / n_orb)
z_init = w_init.copy()
u_init = jnp.zeros(n_orb, dtype=U.dtype)

alpha = 1.6

def admm_step(carry, _):
    w, z, u = carry
    rhs = rho * (z - u) - c
    w_new = jax.scipy.linalg.cho_solve((L_chol, True), rhs)
    w_hat = alpha * w_new + (1.0 - alpha) * z
    z_new = jnp.maximum(0.0, w_hat + u)
    u_new = u + w_hat - z_new
    return (w_new, z_new, u_new), None
```

Line-by-line interpretation:

- `Q` and `c` are the quadratic objective
- `rho` is the ADMM penalty parameter
- `L_chol` is a cached Cholesky factor of `Q + rho I`
- `rhs = rho * (z - u) - c` is the right-hand side of the `w`-update system
- `cho_solve()` performs the `w`-update
- `w_hat` is over-relaxation with parameter `alpha`
- `maximum(0.0, ...)` is the projection onto `w >= 0`
- `u_new = u + w_hat - z_new` is the scaled-dual update

The whole iteration is run inside `jax.lax.scan`, so the loop JIT-compiles well.

## 6. Why Cholesky is important

Every ADMM step must solve

    (Q + rho I) w = rhs

The matrix is the same at every iteration. Only `rhs` changes.

So instead of solving from scratch every time, the code factors the matrix once:

    Q + rho I = L L^T

via Cholesky.

Then each iteration becomes two cheap triangular solves:

    L y = rhs
    L^T w = y

This is faster and more stable than explicitly forming an inverse.

So the Cholesky factor is the cached linear-algebra work that makes repeated
ADMM iterations practical.

## 7. Initial conditions and hyperparameters

### Initial weights

The solver starts from uniform weights scaled by the total intrinsic mass:

    w_init = ones(n_orb) * sum(y_Rzphi) / n_orb

This is a neutral starting point with the right overall scale.

### Regularization

The code uses

    reg = lambda_reg / n_orb

so the penalty is effectively a small ridge term on the weights.

This helps conditioning and discourages extreme weight spikes.

### ADMM penalty parameter

The heuristic

    rho = trace(Q) / n_orb

sets `rho` to the average diagonal scale of `Q`.

It is not part of the statistical model; it is a numerical parameter that
affects convergence speed.

### Over-relaxation

The code uses

    alpha = 1.6

Over-relaxation often speeds convergence in ADMM by taking a partially
extrapolated primal variable before projection.

## 8. How the solved weights are used afterward

Once the weights are found, the code reconstructs the actual model maps:

    density_2DXY = A_xy @ weights
    h1_model = ((A_h1 * A_xy) @ weights) / y_xy
    h2_model = ((A_h2 * A_xy) @ weights) / y_xy
    ...

Then it converts `(h1, h2)` into `(V, sigma)` and computes the final likelihood.

So the ADMM block is only the weight-estimation step. The final model evaluation
happens afterward.

## 9. Bootstrap variant

`solve_nnls_admm_bootstrap()` uses exactly the same idea, but for many
bootstrapped observation vectors.

The key optimization is:

- build `U` once
- build `Q` once
- compute the Cholesky factor once
- build many right-hand sides `c_i = -U^T y_i`
- vmap the ADMM scan over all bootstrap realizations

Mathematically, each bootstrap sample solves

    min_{w_i >= 0}  1/2 ||U w_i - y_i||^2 + (reg/2) ||w_i||^2

with the same `U` but different `y_i`.

This is why the bootstrap path is much faster than re-factorizing the system
for every realization.

## 10. Practical caveats

1. The solver is convex only for the linearized weight subproblem.
   The full Schwarzschild model is still more complicated because the orbit
   library itself depends on the outer parameters.

2. The GH terms use the observed projected density in the denominator during
   the solve. That is an approximation to keep the problem linear in `w`.

3. The fitted objective and the final reported likelihood are related but not
   identical in every code path. The weight solve is the inner optimization
   step, not the entire statistical model.

4. The regularization here is an L2 ridge term. It is different from the
   entropy-like regularizer used in some older Schwarzschild formulations.

## 11. Short summary

The ADMM NNLS solver in `model_bar.py` does this:

- convert all orbit-library constraints into one weighted linear system
- solve a convex quadratic problem for non-negative orbital weights
- enforce `w >= 0` by projection in ADMM
- use one cached Cholesky factor to make repeated iterations cheap
- reuse the same factorization across all bootstrap realizations in the
  bootstrap variant

If you remember one formula, it is the `w`-update:

    (Q + rho I) w^{k+1} = rho (z^k - u^k) - c

Everything else is there to enforce non-negativity efficiently.
