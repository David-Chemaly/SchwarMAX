import jax
import jax.numpy as jnp
from functools import partial

from constants import EPSILON
from integrants_with_binning import _integrate_barred_vmap


def _split(w):
    return w[:3], w[3:]


def _merge(r, v):
    return jnp.concatenate([r, v], axis=0)


@partial(jax.jit, static_argnames=("acc_fn", "pot_fn", "n_steps", "unroll"))
def delta_ej_orbit(w0, acc_fn, pot_fn, n_steps, dt=0.01, t0=0.0, Omega=0.0, unroll=False):
    """Return delta_EJ = |EJ_end / EJ_start - 1| for one orbit."""

    def step(carry, _):
        t, y = carry
        r, v = _split(y)

        a0 = acc_fn(*r)
        v_half = v + 0.5 * dt * a0

        theta = Omega * dt
        c, s = jnp.cos(theta), jnp.sin(theta)

        x_bar = r[0] + dt * v_half[0]
        y_bar = r[1] + dt * v_half[1]
        x_new = c * x_bar + s * y_bar
        y_new = -s * x_bar + c * y_bar
        z_new = r[2] + dt * v_half[2]

        vx_rot = c * v_half[0] + s * v_half[1]
        vy_rot = -s * v_half[0] + c * v_half[1]
        vz_rot = v_half[2]

        r_new = jnp.array([x_new, y_new, z_new])
        v_rot = jnp.array([vx_rot, vy_rot, vz_rot])
        t_new = t + dt

        a1 = acc_fn(*r_new)
        v_new = v_rot + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return (t_new, y_new), (t_new, y_new)

    (_, _), (_, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

    x, y, z = wN[:, 0], wN[:, 1], wN[:, 2]
    vx, vy, vz = wN[:, 3], wN[:, 4], wN[:, 5]
    Lz = vx * y - vy * x

    E_pot = pot_fn(x, y, z)
    E_kin = 0.5 * (vx * vx + vy * vy + vz * vz)
    E_J = E_pot + E_kin - Omega * Lz
    return jnp.abs(E_J[-1] / E_J[0] - 1.0)


@jax.jit
def acc_fn(x, y, z):
    return -jnp.array([x, y, z])


@jax.jit
def pot_scalar(x, y, z):
    return 0.5 * (x * x + y * y + z * z) + 1.0


pot_fn = jax.vmap(pot_scalar, in_axes=(0, 0, 0))


def print_stats(name, x):
    print(
        f"{name}: min={float(jnp.min(x)):.6e}, max={float(jnp.max(x)):.6e}, "
        f"mean={float(jnp.mean(x)):.6e}, sum={float(jnp.sum(x)):.6e}"
    )


def main():
    # Deterministic configuration that reproduces the shape-dependent mismatch.
    seed = 0
    n_batch = 2000
    n_orb = 10
    n_steps = 100
    scale_pos = 4.0
    scale_vel = 2.0
    dt_base = 0.1
    Omega = -1.6

    key = jax.random.PRNGKey(seed)
    kw0, kdt = jax.random.split(key)
    w0 = jax.random.normal(kw0, (n_batch, n_orb, 6), dtype=jnp.float32)
    w0 = w0.at[..., :3].set(w0[..., :3] * scale_pos)
    w0 = w0.at[..., 3:].set(w0[..., 3:] * scale_vel)
    dt = dt_base + dt_base * jax.random.uniform(kdt, (n_batch, n_orb), dtype=jnp.float32)

    # Minimal binning setup (not physically important here).
    num_Vbin = 8
    nXY = jnp.array([4, 4], dtype=jnp.int32)
    xy_n_tot = int(nXY.prod())
    bin_mapping = jnp.mod(jnp.arange(xy_n_tot, dtype=jnp.int32), num_Vbin)
    num_per_bin = jnp.ones((num_Vbin,), dtype=jnp.float32)
    Rzphi_minmax = jnp.array([[0.0, 20.0], [-20.0, 20.0], [-jnp.pi, jnp.pi]], dtype=jnp.float32)
    XY_minmax = jnp.array([[-20.0, 20.0], [-20.0, 20.0]], dtype=jnp.float32)
    nRzphi = jnp.array([4, 4, 4], dtype=jnp.int32)
    num_segments_Rzphi = int(nRzphi.prod())
    v0 = jnp.zeros((num_Vbin,), dtype=jnp.float32)
    s = jnp.ones((num_Vbin,), dtype=jnp.float32)
    rot = jnp.eye(3, dtype=jnp.float32)

    # Mode A: one flat vmap over all orbits.
    R_a, S_a, h1_a, h2_a, h3_a, h4_a, valid_a = _integrate_barred_vmap(
        w0.reshape(-1, 6),
        acc_fn,
        pot_fn,
        n_steps,
        dt.reshape(-1),
        0.0,
        Omega,
        False,
        num_Vbin,
        bin_mapping,
        num_per_bin,
        Rzphi_minmax,
        XY_minmax,
        nRzphi,
        nXY,
        num_segments_Rzphi,
        v0,
        s,
        rot,
    )

    # Mode B: same function, but evaluated with inner width = n_orb and then flattened.
    R_b, S_b, h1_b, h2_b, h3_b, h4_b, valid_b = jax.vmap(
        lambda ww, dd: _integrate_barred_vmap(
            ww,
            acc_fn,
            pot_fn,
            n_steps,
            dd,
            0.0,
            Omega,
            False,
            num_Vbin,
            bin_mapping,
            num_per_bin,
            Rzphi_minmax,
            XY_minmax,
            nRzphi,
            nXY,
            num_segments_Rzphi,
            v0,
            s,
            rot,
        ),
        in_axes=(0, 0),
    )(w0, dt)

    R_b = R_b.reshape(-1, R_b.shape[-1])
    S_b = S_b.reshape(-1, S_b.shape[-1])
    h1_b = h1_b.reshape(-1, h1_b.shape[-1])
    h2_b = h2_b.reshape(-1, h2_b.shape[-1])
    h3_b = h3_b.reshape(-1, h3_b.shape[-1])
    h4_b = h4_b.reshape(-1, h4_b.shape[-1])
    valid_b = valid_b.reshape(-1)

    valid_diff = jnp.abs(valid_a - valid_b)
    print(f"n_orbits={valid_a.shape[0]}, n_batch={n_batch}, n_orb_per_batch={n_orb}")
    print_stats("valid_abs_diff", valid_diff)
    print(f"valid_mismatch_count={(valid_diff > 0).sum().item()}")

    # Compare delta_EJ directly in both evaluation shapes.
    delta_vmap = jax.vmap(delta_ej_orbit, in_axes=(0, None, None, None, 0, None, None, None))
    delta_nested = jax.vmap(delta_vmap, in_axes=(0, None, None, None, 0, None, None, None))
    delta_a = delta_vmap(w0.reshape(-1, 6), acc_fn, pot_fn, n_steps, dt.reshape(-1), 0.0, Omega, False)
    delta_b = delta_nested(w0, acc_fn, pot_fn, n_steps, dt, 0.0, Omega, False).reshape(-1)
    delta_abs = jnp.abs(delta_a - delta_b)
    print_stats("deltaEJ_abs_diff", delta_abs)

    idx = jnp.where(valid_diff > 0)[0]
    if idx.size == 0:
        print("No valid flips in this run.")
        return

    i = int(idx[0])
    b = i // n_orb
    lo, hi = b * n_orb, (b + 1) * n_orb
    print(f"first_valid_flip_orbit={i}, batch={b}")
    print(
        f"valid(flat,batchwise)=({float(valid_a[i]):.1f}, {float(valid_b[i]):.1f}), "
        f"deltaEJ(flat,batchwise)=({float(delta_a[i]):.9f}, {float(delta_b[i]):.9f})"
    )
    print(
        f"deltaEJ_minus_0.5(flat,batchwise)=({float(delta_a[i]-0.5):.9e}, {float(delta_b[i]-0.5):.9e})"
    )

    # Show amplification after batch aggregation (same algebra as integrate_batch).
    A_R_a = R_a.T / n_steps
    A_xy_a = S_a.T / n_steps
    A_h1_a = h1_a.T

    A_R_b = R_b.T / n_steps
    A_xy_b = S_b.T / n_steps
    A_h1_b = h1_b.T

    v_sum_a = valid_a[lo:hi].sum()
    v_sum_b = valid_b[lo:hi].sum()
    w_a = jnp.ones(n_orb) / (v_sum_a + 0.1)
    w_b = jnp.ones(n_orb) / (v_sum_b + 0.1)

    R_batch_a = A_R_a[:, lo:hi] @ w_a
    R_batch_b = A_R_b[:, lo:hi] @ w_b
    xy_batch_a = A_xy_a[:, lo:hi] @ w_a
    xy_batch_b = A_xy_b[:, lo:hi] @ w_b
    h1_batch_a = ((A_h1_a[:, lo:hi] * A_xy_a[:, lo:hi]) @ w_a) / (xy_batch_a + EPSILON)
    h1_batch_b = ((A_h1_b[:, lo:hi] * A_xy_b[:, lo:hi]) @ w_b) / (xy_batch_b + EPSILON)

    print(f"valid_sum(flat,batchwise)=({float(v_sum_a):.1f}, {float(v_sum_b):.1f})")
    print(
        f"scalar_weight_1/(valid+0.1)=( {float(1.0/(v_sum_a+0.1)):.9f}, {float(1.0/(v_sum_b+0.1)):.9f} )"
    )
    print(
        f"batch_R_diff(sum,max)=({float(jnp.sum(jnp.abs(R_batch_a-R_batch_b))):.6e}, "
        f"{float(jnp.max(jnp.abs(R_batch_a-R_batch_b))):.6e})"
    )
    print(
        f"batch_xy_diff(sum,max)=({float(jnp.sum(jnp.abs(xy_batch_a-xy_batch_b))):.6e}, "
        f"{float(jnp.max(jnp.abs(xy_batch_a-xy_batch_b))):.6e})"
    )
    print(
        f"batch_h1_diff(sum,max)=({float(jnp.sum(jnp.abs(h1_batch_a-h1_batch_b))):.6e}, "
        f"{float(jnp.max(jnp.abs(h1_batch_a-h1_batch_b))):.6e})"
    )

    print("\nConclusion:")
    print("1) Same orbit inputs can produce tiny delta_EJ differences under different vmap shapes.")
    print("2) A hard cutoff valid = (delta_EJ < 0.5) can flip from those tiny differences.")
    print("3) valid flips change normalization 1/(valid_sum+0.1), amplifying batch-level outputs.")


if __name__ == "__main__":
    main()
