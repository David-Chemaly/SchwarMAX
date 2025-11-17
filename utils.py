import jax
import jax.numpy as jnp
from functools import partial

from constants import G, EPSILON

@jax.jit
def get_mat(x, y, z):
    v1 = jnp.array([0.0, 0.0, 1.0])
    I3 = jnp.eye(3)

    # Create a fixed-shape vector from inputs
    v2 = jnp.array([x, y, z])
    # Normalize v2 in one step
    v2 = v2 / (jnp.linalg.norm(v2) + EPSILON)

    # Compute the angle using a fused dot and clip operation
    angle = jnp.arccos(jnp.clip(jnp.dot(v1, v2), -1.0, 1.0))

    # Compute normalized rotation axis
    v3 = jnp.cross(v1, v2)
    v3 = v3 / (jnp.linalg.norm(v3) + EPSILON)

    # Build the skew-symmetric matrix K for Rodrigues' formula
    K = jnp.array([
        [0, -v3[2], v3[1]],
        [v3[2], 0, -v3[0]],
        [-v3[1], v3[0], 0]
    ])

    sin_angle = jnp.sin(angle)
    cos_angle = jnp.cos(angle)

    # Compute rotation matrix using Rodrigues' formula
    rot_mat = I3 + sin_angle * K + (1 - cos_angle) * jnp.dot(K, K)
    return rot_mat

@jax.jit
def go_to_bar_ref(xv, angle):
    # Rotate contourclockwise with positive angle
    sina, cosa = jnp.sin(angle), jnp.cos(angle)
    x, y, z, vx, vy, vz = xv
    x_new  = x * cosa - y * sina
    y_new  = x * sina + y * cosa
    vx_new = vx * cosa - vy * sina
    vy_new = vx * sina + vy * cosa

    return xv.at[0].set(x_new).at[1].set(y_new).at[3].set(vx_new).at[4].set(vy_new)

@partial(jax.jit, static_argnames=('xlim', 'ylim', 'zlim', 'dx', 'dy', 'dz'))
def histogram3d(x, xlim=(-10, 10), ylim=(-10, 10), zlim=(-3, 3), dx=1.0, dy=1.0, dz=1.0):
    # Define bin edges for each dimension
    x_bins = jnp.arange(xlim[0], xlim[1] + dx, dx)
    y_bins = jnp.arange(ylim[0], ylim[1] + dy, dy)
    z_bins = jnp.arange(zlim[0], zlim[1] + dz, dz)

    bins, _ = jnp.histogramdd(x, bins=[x_bins, y_bins, z_bins])
    return bins