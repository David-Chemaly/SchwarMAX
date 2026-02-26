import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnn
import jaxopt
from functools import partial

@partial(jax.jit, static_argnames=('nll',))
def solve_lbfgs_softplus(A, y, sigma, nll, l2=1e-3, maxiter=500, tol=1e-6):
    z0 = jnp.zeros(A.shape[1], A.dtype)
    solver = jaxopt.LBFGS(fun=nll, value_and_grad=True, maxiter=maxiter, tol=tol)
    res = solver.run(z0, A, y, sigma, l2)
    x_hat = jnn.softplus(res.params)
    return x_hat