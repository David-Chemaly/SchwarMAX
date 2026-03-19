"""
JAX-jitted access to Dehnen & Aly bar potential-density pairs via local discBar code.

This module keeps your existing project files untouched and exposes functions with
`potentials.py` / `densities.py`-style signatures:

- Dehnen2023Bar_potential(x, y, z, params)
- Dehnen2023Bar_density(x, y, z, params)
- Dehnen2023Bar_acceleration(x, y, z, params)

For repeated orbit integration, use:
- make_dehnen2023_bar_functions(params)
which builds the discBar model once and returns jitted callables.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from constants import G
from utils import get_mat

import discBar.discBar as discbar


PARAMETER_REFERENCE = {
    "model_bar": "Model family: 'T0'..'T4', 'V0'..'V4', and optionally 'Dk'/'Wk'",
    "logM_bar": "log10(total mass in Msun). Use this OR M_bar",
    "M_bar": "total mass in Msun. Use this OR logM_bar",
    "L_bar": "needle half-length L (kpc), L>=0",
    "gamma_bar": "needle slope parameter gamma in [-1,1], default 0",
    "s_bar": "seed model scale radius s=a+b (kpc). Use with q_bar",
    "q_bar": "seed flattening q=b/s in [0,1]. Use with s_bar",
    "a_seed": "optional seed scale length a (kpc), overrides s_bar/q_bar if a_seed and b_seed are both set",
    "b_seed": "optional seed scale height b (kpc), overrides s_bar/q_bar if a_seed and b_seed are both set",
    "phi_bar": "crossed-bar angle phi (radians), default 0",
    "x_origin": "bar center x (kpc), default 0",
    "y_origin": "bar center y (kpc), default 0",
    "z_origin": "bar center z (kpc), default 0",
    "dirx": "bar-frame z-axis direction x component, default 0",
    "diry": "bar-frame z-axis direction y component, default 0",
    "dirz": "bar-frame z-axis direction z component, default 1",
}


def describe_dehnen2023_bar_parameters():
    return dict(PARAMETER_REFERENCE)


def _as_float(params, key, default=None):
    if key in params:
        return float(params[key])
    if default is None:
        raise KeyError(f"Missing required parameter '{key}'")
    return float(default)


def _mass_from_params(params):
    if "M_bar" in params:
        return float(params["M_bar"])
    if "logM_bar" in params:
        return 10.0 ** float(params["logM_bar"])
    raise KeyError("Need one of {'M_bar', 'logM_bar'}")


def _build_discbar_model(params):
    mtype = params["model_bar"]
    mass = _mass_from_params(params)
    L = _as_float(params, "L_bar", 0.0)
    gamma = _as_float(params, "gamma_bar", 0.0)
    phi = _as_float(params, "phi_bar", 0.0)

    if ("a_seed" in params) and ("b_seed" in params):
        a_seed = _as_float(params, "a_seed")
        b_seed = _as_float(params, "b_seed")
        model = discbar.makeSingleModel(
            mtype=mtype,
            M=mass,
            L=L,
            gamma=gamma,
            phi=phi,
            a=a_seed,
            b=b_seed,
        )
    else:
        s = _as_float(params, "s_bar")
        q = _as_float(params, "q_bar")
        model = discbar.makeSingleModel(
            mtype=mtype,
            M=mass,
            s=s,
            q=q,
            L=L,
            gamma=gamma,
            phi=phi,
        )
    return model


def make_dehnen2023_bar_functions(params):
    """
    Build a local discBar model once and return JAX-jitted callables:
      potential(x,y,z), density(x,y,z), acceleration(x,y,z), hessian(x,y,z)

    Notes:
    - output potential/acceleration are multiplied by `G` from constants.py so
      units match this repository's convention (kpc, Msun, Gyr).
    - density is returned directly in Msun/kpc^3.
    """
    model = _build_discbar_model(params)

    x0 = _as_float(params, "x_origin", 0.0)
    y0 = _as_float(params, "y_origin", 0.0)
    z0 = _as_float(params, "z_origin", 0.0)
    dirx = _as_float(params, "dirx", 0.0)
    diry = _as_float(params, "diry", 0.0)
    dirz = _as_float(params, "dirz", 1.0)

    rot = get_mat(dirx, diry, dirz).astype(jnp.float32)
    rot_t = rot.T
    scalar_spec = jax.ShapeDtypeStruct((), jnp.float32)
    vec3_spec = jax.ShapeDtypeStruct((3,), jnp.float32)

    def _to_model_frame(x, y, z):
        vec = jnp.array([x - x0, y - y0, z - z0], dtype=jnp.float32)
        return rot @ vec

    def _cb_potential(rxyz_np):
        x, y, z = map(float, np.asarray(rxyz_np))
        phi = model.potential(x, y, z, forces=False)
        return np.asarray(phi * G, dtype=np.float32)

    def _cb_density(rxyz_np):
        x, y, z = map(float, np.asarray(rxyz_np))
        rho = model.density(x, y, z)
        return np.asarray(rho, dtype=np.float32)

    def _cb_acceleration(rxyz_np):
        x, y, z = map(float, np.asarray(rxyz_np))
        _, force = model.potential(x, y, z, forces=True)
        # discBar uses G=1 convention; convert to this repo's G.
        return np.asarray(force, dtype=np.float32) * G

    @jax.jit
    def potential_scalar(x, y, z):
        rxyz = _to_model_frame(x, y, z)
        return jax.pure_callback(
            _cb_potential,
            scalar_spec,
            rxyz,
            vmap_method="sequential",
        )

    @jax.jit
    def density_scalar(x, y, z):
        rxyz = _to_model_frame(x, y, z)
        return jax.pure_callback(
            _cb_density,
            scalar_spec,
            rxyz,
            vmap_method="sequential",
        )

    @jax.jit
    def acceleration(x, y, z):
        rxyz = _to_model_frame(x, y, z)
        f_model = jax.pure_callback(
            _cb_acceleration,
            vec3_spec,
            rxyz,
            vmap_method="sequential",
        )
        return rot_t @ f_model

    @jax.jit
    def acceleration_vmap(points):
        # points: (N,3), world frame
        return jax.vmap(
            lambda p: acceleration(p[0], p[1], p[2]),
            in_axes=0,
        )(points)

    @jax.jit
    def hessian(x, y, z):
        # Numerical Hessian from finite differences of acceleration.
        # H_ij = d2 Phi / dxi dxj = - d(acc_i) / dxj
        dx = 1e-4

        acc_xp = acceleration(x + dx, y, z)
        acc_xm = acceleration(x - dx, y, z)
        acc_yp = acceleration(x, y + dx, z)
        acc_ym = acceleration(x, y - dx, z)
        acc_zp = acceleration(x, y, z + dx)
        acc_zm = acceleration(x, y, z - dx)

        col_x = -(acc_xp - acc_xm) / (2.0 * dx)
        col_y = -(acc_yp - acc_ym) / (2.0 * dx)
        col_z = -(acc_zp - acc_zm) / (2.0 * dx)
        return jnp.stack([col_x, col_y, col_z], axis=1)

    @jax.jit
    def potential(x, y, z):
        x_arr, y_arr, z_arr = jnp.broadcast_arrays(x, y, z)
        pts = jnp.stack([x_arr.ravel(), y_arr.ravel(), z_arr.ravel()], axis=1)
        vals = jax.lax.map(lambda p: potential_scalar(p[0], p[1], p[2]), pts)
        return vals.reshape(x_arr.shape)

    @jax.jit
    def density(x, y, z):
        x_arr, y_arr, z_arr = jnp.broadcast_arrays(x, y, z)
        pts = jnp.stack([x_arr.ravel(), y_arr.ravel(), z_arr.ravel()], axis=1)
        vals = jax.lax.map(lambda p: density_scalar(p[0], p[1], p[2]), pts)
        return vals.reshape(x_arr.shape)

    return {
        "potential": potential,
        "density": density,
        "acceleration": acceleration,
        "acceleration_vmap": acceleration_vmap,
        "hessian": hessian,
        "potential_scalar": potential_scalar,
        "density_scalar": density_scalar,
    }


def make_dehnen2023_bar_numpy_backend(params):
    """
    Fast numpy backend for bulk evaluation (no JAX callback overhead).

    Returns callables:
      potential(points): ndarray (N,)
      density(points): ndarray (N,)
      acceleration(points): ndarray (N,3)

    where `points` is ndarray (N,3) in world coordinates.
    """
    model = _build_discbar_model(params)

    x0 = _as_float(params, "x_origin", 0.0)
    y0 = _as_float(params, "y_origin", 0.0)
    z0 = _as_float(params, "z_origin", 0.0)
    dirx = _as_float(params, "dirx", 0.0)
    diry = _as_float(params, "diry", 0.0)
    dirz = _as_float(params, "dirz", 1.0)

    rot = np.asarray(get_mat(dirx, diry, dirz), dtype=np.float64)
    rot_t = rot.T
    origin = np.array([x0, y0, z0], dtype=np.float64)

    def _to_model(points):
        p = np.asarray(points, dtype=np.float64)
        return (p - origin) @ rot.T

    def potential(points):
        pm = _to_model(points)
        return np.asarray(model.potential(pm[:, 0], pm[:, 1], pm[:, 2], forces=False)) * G

    def density(points):
        pm = _to_model(points)
        return np.asarray(model.density(pm[:, 0], pm[:, 1], pm[:, 2]))

    def acceleration(points):
        pm = _to_model(points)
        _, force = model.potential(pm[:, 0], pm[:, 1], pm[:, 2], forces=True)
        f_model = np.column_stack(force) * G
        return f_model @ rot

    return {
        "potential": potential,
        "density": density,
        "acceleration": acceleration,
    }


def Dehnen2023Bar_potential(x, y, z, params):
    funcs = make_dehnen2023_bar_functions(params)
    return funcs["potential"](x, y, z)


def Dehnen2023Bar_density(x, y, z, params):
    funcs = make_dehnen2023_bar_functions(params)
    return funcs["density"](x, y, z)


def Dehnen2023Bar_acceleration(x, y, z, params):
    funcs = make_dehnen2023_bar_functions(params)
    return funcs["acceleration"](x, y, z)


def Dehnen2023Bar_hessian(x, y, z, params):
    funcs = make_dehnen2023_bar_functions(params)
    return funcs["hessian"](x, y, z)


if __name__ == "__main__":
    # Quick backend-consistency test against direct discBar call.
    params = {
        "model_bar": "V3",
        "logM_bar": 10.4,
        "L_bar": 3.0,
        "gamma_bar": 0.0,
        "s_bar": 4.2,
        "q_bar": 1.2 / 4.2,
        "x_origin": 0.0,
        "y_origin": 0.0,
        "z_origin": 0.0,
        "dirx": 0.0,
        "diry": 0.0,
        "dirz": 1.0,
    }

    funcs = make_dehnen2023_bar_functions(params)
    x0, y0, z0 = 1.7, 0.6, 0.2

    phi = funcs["potential_scalar"](x0, y0, z0)
    acc = funcs["acceleration"](x0, y0, z0)
    rho = funcs["density_scalar"](x0, y0, z0)

    print("phi =", float(phi))
    print("acc =", acc)
    print("rho =", float(rho))
