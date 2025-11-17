import jax
import jax.numpy as jnp
from functools import partial

def _split(w):
    return w[:3], w[3:]

def _merge(r, v):
    return jnp.concatenate([r, v], axis=0)

@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll'))
def integrate_leapfrog_traj(w0, params_halo, params_disk, params_bar, acc_fn, n_steps, dt = 0.010, t0 = 0.0, unroll=True):
    """Leapfrog (KDK) — returns final time and final state only."""

    def step(carry, _):
        t, y = carry
        r, v = _split(y)

        params_bar['t'] = t  # Update time-dependent parameters
        a0 = acc_fn(*r, params_halo, params_disk, params_bar)
        v_half = v + 0.5 * dt * a0
        r_new = r + dt * v_half
        t_new = t + dt

        params_bar['t'] = t_new  # Update time-dependent parameters
        a1 = acc_fn(*r_new, params_halo, params_disk, params_bar)
        v_new = v_half + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return (t_new, y_new), (t_new, y_new)

    (_, _), (tN, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)
    return tN, wN