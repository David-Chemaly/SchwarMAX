import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from densities import DoubleExponentialDisk_density, Hernquist_density


@jax.jit
def model_style_density(x, y, z, params):
    """
    Same composition style used in model.py:
      rho = rho_disc + rho_bulge
    """
    rho_disc = DoubleExponentialDisk_density(x, y, z, params["disc"])
    rho_bulge = Hernquist_density(x, y, z, params["bulge"])
    return rho_disc + rho_bulge


@partial(
    jax.jit,
    static_argnames=("density_fn", "n_x", "n_y", "n_z", "n_samples"),
)
def sample_from_density_grid(
    key,
    density_fn,
    params,
    bounds,
    n_samples,
    n_x=96,
    n_y=96,
    n_z=64,
):
    """
    JAX-jittable generic sampler from a 3D density via grid-based inverse-CDF.

    Works well for cuspy profiles where naive rejection from a wide uniform box
    becomes inefficient.

    Parameters
    ----------
    key : PRNGKey
    density_fn : callable
        Signature density_fn(x, y, z, params) -> rho.
    params : pytree
        Density parameters.
    bounds : array (3,2)
        [[xmin,xmax], [ymin,ymax], [zmin,zmax]].
    n_samples : int
        Number of particles.
    n_x, n_y, n_z : int
        Grid resolution used to discretize rho.
    """
    bounds = jnp.asarray(bounds, dtype=jnp.float32)

    x_edges = jnp.linspace(bounds[0, 0], bounds[0, 1], n_x + 1)
    y_edges = jnp.linspace(bounds[1, 0], bounds[1, 1], n_y + 1)
    z_edges = jnp.linspace(bounds[2, 0], bounds[2, 1], n_z + 1)

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    dz = z_edges[1] - z_edges[0]

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    X, Y, Z = jnp.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    rho = density_fn(X, Y, Z, params)
    rho = jnp.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    rho = jnp.clip(rho, a_min=0.0)

    cell_mass = rho * dx * dy * dz
    flat_mass = cell_mass.reshape((-1,))

    eps = 1e-30
    logp = jnp.log(flat_mass + eps) - jnp.log(jnp.sum(flat_mass) + eps)

    key_idx, key_jit = jax.random.split(key)

    draw_batch = 256
    n_rounds = (n_samples + draw_batch - 1) // draw_batch

    def draw_step(key_in, _):
        key_out, key_cat = jax.random.split(key_in)
        idx_batch = jax.random.categorical(key_cat, logp, shape=(draw_batch,))
        return key_out, idx_batch

    _, idx_batches = jax.lax.scan(draw_step, key_idx, xs=None, length=n_rounds)
    idx = idx_batches.reshape((-1,))[:n_samples]

    ny_nz = n_y * n_z
    ix = idx // ny_nz
    rem = idx % ny_nz
    iy = rem // n_z
    iz = rem % n_z

    u = jax.random.uniform(key_jit, shape=(n_samples, 3), dtype=jnp.float32)
    x = x_edges[ix] + u[:, 0] * dx
    y = y_edges[iy] + u[:, 1] * dy
    z = z_edges[iz] + u[:, 2] * dz

    samples = jnp.stack([x, y, z], axis=1)

    return {
        "samples": samples,
        "cell_mass_grid": cell_mass,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }


@partial(jax.jit, static_argnames=("density_fn", "n_los"))
def projected_density_face_on(density_fn, params, x_centers, y_centers, z_bounds, n_los=512):
    z_los = jnp.linspace(z_bounds[0], z_bounds[1], n_los, dtype=jnp.float32)
    X, Y, Z = jnp.meshgrid(x_centers, y_centers, z_los, indexing="ij")
    rho = density_fn(X, Y, Z, params)
    rho = jnp.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    dz = z_los[1] - z_los[0]
    return jnp.sum(0.5 * (rho[:, :, :-1] + rho[:, :, 1:]), axis=-1) * dz


@partial(jax.jit, static_argnames=("density_fn", "n_los"))
def projected_density_edge_on(density_fn, params, x_centers, z_centers, y_bounds, n_los=512):
    y_los = jnp.linspace(y_bounds[0], y_bounds[1], n_los, dtype=jnp.float32)
    X, Z, Y = jnp.meshgrid(x_centers, z_centers, y_los, indexing="ij")
    rho = density_fn(X, Y, Z, params)
    rho = jnp.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    dy = y_los[1] - y_los[0]
    return jnp.sum(0.5 * (rho[:, :, :-1] + rho[:, :, 1:]), axis=-1) * dy


def _to_prob_per_pixel_from_surface_density(sigma_map, d1, d2):
    mass_map = np.asarray(sigma_map, dtype=np.float64) * float(d1) * float(d2)
    s = np.sum(mass_map)
    if s <= 0:
        return np.zeros_like(mass_map)
    return mass_map / s


def _to_prob_per_pixel_from_counts(count_map):
    count_map = np.asarray(count_map, dtype=np.float64)
    s = np.sum(count_map)
    if s <= 0:
        return np.zeros_like(count_map)
    return count_map / s


def _metrics(sample_prob, truth_prob):
    eps = 1e-16
    s = sample_prob.ravel()
    t = truth_prob.ravel()
    mae = float(np.mean(np.abs(s - t)))
    rmse = float(np.sqrt(np.mean((s - t) ** 2)))
    corr = float(np.corrcoef(s, t)[0, 1])
    kl_st = float(np.sum((s + eps) * np.log((s + eps) / (t + eps))))
    return {"mae": mae, "rmse": rmse, "corr": corr, "kl_sample_truth": kl_st}


def _demo():
    params = {
        "disc": {
            "rho0_disc": 2.0e8,
            "Rd_disc": 2.5,
            "hz_disc": 0.35,
            "x_origin": 0.0,
            "y_origin": 0.0,
            "z_origin": 0.0,
            "dirx": 0.0,
            "diry": 0.0,
            "dirz": 1.0,
        },
        "bulge": {
            "logM_bulge": 10.2,
            "Rs_bulge": 0.7,
            "x_origin": 0.0,
            "y_origin": 0.0,
            "z_origin": 0.0,
            "dirx": 0.0,
            "diry": 0.0,
            "dirz": 1.0,
        },
    }

    bounds = jnp.array(
        [
            [-15.0, 15.0],  # x
            [-15.0, 15.0],  # y
            [-5.0, 5.0],    # z
        ],
        dtype=jnp.float32,
    )

    n_samples = 40_000
    n_x = 48
    n_y = 48
    n_z = 32

    key = jax.random.PRNGKey(0)
    out = sample_from_density_grid(
        key,
        model_style_density,
        params,
        bounds,
        n_samples=n_samples,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
    )
    samples = np.asarray(out["samples"])
    x_edges = np.asarray(out["x_edges"])
    y_edges = np.asarray(out["y_edges"])
    z_edges = np.asarray(out["z_edges"])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dx = float(out["dx"])
    dy = float(out["dy"])
    dz = float(out["dz"])

    # Sampled projected maps (probability per pixel).
    sample_face_counts, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=[x_edges, y_edges]
    )
    sample_edge_counts, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 2], bins=[x_edges, z_edges]
    )
    sample_face_prob = _to_prob_per_pixel_from_counts(sample_face_counts)
    sample_edge_prob = _to_prob_per_pixel_from_counts(sample_edge_counts)

    # Ground-truth projected maps from the input density evaluated on the 3D grid.
    # Integrating this grid over LOS gives the target surface-density form.
    cell_mass_grid = np.asarray(out["cell_mass_grid"])
    sigma_face = np.sum(cell_mass_grid, axis=2) / dz
    sigma_edge = np.sum(cell_mass_grid, axis=1) / dy
    truth_face_prob = _to_prob_per_pixel_from_surface_density(sigma_face, dx, dy)
    truth_edge_prob = _to_prob_per_pixel_from_surface_density(sigma_edge, dx, dz)

    m_face = _metrics(sample_face_prob, truth_face_prob)
    m_edge = _metrics(sample_edge_prob, truth_edge_prob)

    # Plot comparison.
    eps_plot = 1e-12
    fig, ax = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    face_extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    edge_extent = [x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]]

    vmin_face = min(
        np.log10(sample_face_prob + eps_plot).min(),
        np.log10(truth_face_prob + eps_plot).min(),
    )
    vmax_face = max(
        np.log10(sample_face_prob + eps_plot).max(),
        np.log10(truth_face_prob + eps_plot).max(),
    )
    vmin_edge = min(
        np.log10(sample_edge_prob + eps_plot).min(),
        np.log10(truth_edge_prob + eps_plot).min(),
    )
    vmax_edge = max(
        np.log10(sample_edge_prob + eps_plot).max(),
        np.log10(truth_edge_prob + eps_plot).max(),
    )

    im00 = ax[0, 0].imshow(
        np.log10(sample_face_prob.T + eps_plot),
        origin="lower",
        extent=face_extent,
        aspect="auto",
        vmin=vmin_face,
        vmax=vmax_face,
        cmap="magma",
    )
    ax[0, 0].set_title("Face-on: sampled")
    ax[0, 0].set_xlabel("x [kpc]")
    ax[0, 0].set_ylabel("y [kpc]")
    fig.colorbar(im00, ax=ax[0, 0], fraction=0.046)

    im01 = ax[0, 1].imshow(
        np.log10(truth_face_prob.T + eps_plot),
        origin="lower",
        extent=face_extent,
        aspect="auto",
        vmin=vmin_face,
        vmax=vmax_face,
        cmap="magma",
    )
    ax[0, 1].set_title("Face-on: ground truth")
    ax[0, 1].set_xlabel("x [kpc]")
    ax[0, 1].set_ylabel("y [kpc]")
    fig.colorbar(im01, ax=ax[0, 1], fraction=0.046)

    rel_face = (sample_face_prob - truth_face_prob) / (truth_face_prob + eps_plot)
    im02 = ax[0, 2].imshow(
        np.clip(rel_face.T, -1.0, 1.0),
        origin="lower",
        extent=face_extent,
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    ax[0, 2].set_title("Face-on: (sample-truth)/truth")
    ax[0, 2].set_xlabel("x [kpc]")
    ax[0, 2].set_ylabel("y [kpc]")
    fig.colorbar(im02, ax=ax[0, 2], fraction=0.046)

    im10 = ax[1, 0].imshow(
        np.log10(sample_edge_prob.T + eps_plot),
        origin="lower",
        extent=edge_extent,
        aspect="auto",
        vmin=vmin_edge,
        vmax=vmax_edge,
        cmap="magma",
    )
    ax[1, 0].set_title("Edge-on: sampled")
    ax[1, 0].set_xlabel("x [kpc]")
    ax[1, 0].set_ylabel("z [kpc]")
    fig.colorbar(im10, ax=ax[1, 0], fraction=0.046)

    im11 = ax[1, 1].imshow(
        np.log10(truth_edge_prob.T + eps_plot),
        origin="lower",
        extent=edge_extent,
        aspect="auto",
        vmin=vmin_edge,
        vmax=vmax_edge,
        cmap="magma",
    )
    ax[1, 1].set_title("Edge-on: ground truth")
    ax[1, 1].set_xlabel("x [kpc]")
    ax[1, 1].set_ylabel("z [kpc]")
    fig.colorbar(im11, ax=ax[1, 1], fraction=0.046)

    rel_edge = (sample_edge_prob - truth_edge_prob) / (truth_edge_prob + eps_plot)
    im12 = ax[1, 2].imshow(
        np.clip(rel_edge.T, -1.0, 1.0),
        origin="lower",
        extent=edge_extent,
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    ax[1, 2].set_title("Edge-on: (sample-truth)/truth")
    ax[1, 2].set_xlabel("x [kpc]")
    ax[1, 2].set_ylabel("z [kpc]")
    fig.colorbar(im12, ax=ax[1, 2], fraction=0.046)

    out_png = os.path.join(os.path.dirname(__file__), "sampling_validation.png")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    print("=== Sampling validation (grid inverse-CDF) ===")
    print(f"n_samples: {n_samples}")
    print(f"grid: ({n_x}, {n_y}, {n_z})")
    print(f"saved_plot: {out_png}")
    print("")
    print("Face-on metrics:")
    print(f"  corr              = {m_face['corr']:.6f}")
    print(f"  mae(prob/pixel)   = {m_face['mae']:.6e}")
    print(f"  rmse(prob/pixel)  = {m_face['rmse']:.6e}")
    print(f"  KL(sample||truth) = {m_face['kl_sample_truth']:.6e}")
    print("")
    print("Edge-on metrics:")
    print(f"  corr              = {m_edge['corr']:.6f}")
    print(f"  mae(prob/pixel)   = {m_edge['mae']:.6e}")
    print(f"  rmse(prob/pixel)  = {m_edge['rmse']:.6e}")
    print(f"  KL(sample||truth) = {m_edge['kl_sample_truth']:.6e}")


if __name__ == "__main__":
    _demo()
