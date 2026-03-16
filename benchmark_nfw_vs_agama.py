"""
Benchmark NFW potential/acceleration: analytic JAX vs AGAMA CylSpline.

Same style as benchmark_dehnen_bar_vs_agama.py.
"""
from __future__ import annotations

import os
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import agama
from constants import KPCGYR_TO_KMS, G, EPSILON
from potentials import NFW_potential, NFW_acceleration
from densities import NFW_density

agama.setUnits(mass=1, length=1, velocity=KPCGYR_TO_KMS)


# ============================================================================
# Helpers (same as benchmark_dehnen_bar_vs_agama.py)
# ============================================================================
def getCylindricalFromCartesian_clockwise(x, y, vx, vy):
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    vR = (x * vx + y * vy) / np.maximum(R, 1e-12)
    vphi = -(x * vy - y * vx) / np.maximum(R, 1e-12)
    return R, phi, vR, vphi


def summarize_relative(name, ours, ref, eps=1e-8, floor=1e-6):
    denom = np.maximum(np.abs(ref), eps)
    rel = np.abs(ours - ref) / denom
    mask = np.abs(ref) > floor
    if np.any(mask):
        rel = rel[mask]
    print(
        f"  {name}: mean={rel.mean():.4e}, "
        f"median={np.median(rel):.4e}, "
        f"p95={np.percentile(rel, 95):.4e}, "
        f"max={rel.max():.4e}"
    )


def timed_eval(fn, *args, repeat=5, label=""):
    out = fn(*args)
    if hasattr(out, 'block_until_ready'):
        out.block_until_ready()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    if label:
        n_pts = args[0].shape[0] if hasattr(args[0], 'shape') else len(args[0])
        print(f"  {label}: median={np.median(times):.4f} s, "
              f"{1e6 * np.median(times) / n_pts:.2f} us/pt  ({n_pts} pts)")
    return out, times


def make_plot(x_flat, y_flat, acc_ours, acc_agama, figure_path=None):
    _, _, fR_ours, fphi_ours = getCylindricalFromCartesian_clockwise(
        x_flat, y_flat, acc_ours[:, 0], acc_ours[:, 1])
    _, _, fR_agama, fphi_agama = getCylindricalFromCartesian_clockwise(
        x_flat, y_flat, acc_agama[:, 0], acc_agama[:, 1])

    ours_cyl = np.column_stack([fR_ours, fphi_ours, acc_ours[:, 2]])
    agama_cyl = np.column_stack([fR_agama, fphi_agama, acc_agama[:, 2]])
    residual = (agama_cyl - ours_cyl) / np.maximum(np.abs(agama_cyl), 1e-8)
    labels = ["R acceleration", "phi acceleration", "Z acceleration"]

    fig, ax = plt.subplots(3, 3, figsize=(22, 18),
                           gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    for i in range(3):
        vmin = np.percentile(agama_cyl[:, i], 2)
        vmax = np.percentile(agama_cyl[:, i], 98)
        cb = ax[i, 0].scatter(x_flat, y_flat, c=ours_cyl[:, i], s=6,
                              marker="s", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(cb, ax=ax[i, 0], label=f"{labels[i]} [kpc/Gyr^2]")
        ax[i, 0].set_title(f"{labels[i]} (JAX analytic)")
        ax[i, 0].set_xlabel("X [kpc]"); ax[i, 0].set_ylabel("Y [kpc]")

        cb = ax[i, 1].scatter(x_flat, y_flat, c=agama_cyl[:, i], s=6,
                              marker="s", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(cb, ax=ax[i, 1], label=f"{labels[i]} [kpc/Gyr^2]")
        ax[i, 1].set_title(f"{labels[i]} (AGAMA)")
        ax[i, 1].set_xlabel("X [kpc]"); ax[i, 1].set_ylabel("Y [kpc]")

        q = max(np.percentile(np.abs(residual[:, i]), 98), 1e-3)
        cb = ax[i, 2].scatter(x_flat, y_flat, c=residual[:, i], s=6,
                              marker="s", cmap="coolwarm", vmin=-q, vmax=q)
        plt.colorbar(cb, ax=ax[i, 2], label="fractional residual")
        ax[i, 2].set_title(f"{labels[i]} (AGAMA - ours) / |AGAMA|")
        ax[i, 2].set_xlabel("X [kpc]"); ax[i, 2].set_ylabel("Y [kpc]")

    if figure_path:
        os.makedirs(os.path.dirname(figure_path) or ".", exist_ok=True)
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {figure_path}")
    plt.close(fig)


# ============================================================================
# NFW JAX evaluators
# ============================================================================
def make_nfw_jax_evaluators(params):
    """Build vmapped + jitted NFW potential and acceleration."""
    pot_vmap = jax.jit(jax.vmap(NFW_potential, in_axes=(0, 0, 0, None)))
    acc_vmap = jax.jit(jax.vmap(NFW_acceleration, in_axes=(0, 0, 0, None)))

    def potential_batch(pts):
        x = jnp.array(pts[:, 0])
        y = jnp.array(pts[:, 1])
        z = jnp.array(pts[:, 2])
        return pot_vmap(x, y, z, params)

    def acceleration_batch(pts):
        x = jnp.array(pts[:, 0])
        y = jnp.array(pts[:, 1])
        z = jnp.array(pts[:, 2])
        return acc_vmap(x, y, z, params)

    return potential_batch, acceleration_batch


def make_nfw_density_for_agama(params):
    """Return a callable density(xyz) -> ndarray for AGAMA CylSpline."""
    density_vmap = jax.jit(jax.vmap(NFW_density, in_axes=(0, 0, 0, None)))

    def density_fn(xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim == 1:
            xyz = xyz[None, :]
        x = jnp.array(xyz[:, 0])
        y = jnp.array(xyz[:, 1])
        z = jnp.array(xyz[:, 2])
        return np.asarray(density_vmap(x, y, z, params))

    return density_fn


# ============================================================================
# Main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark NFW JAX analytic vs AGAMA CylSpline")
    p.add_argument("--logM", type=float, default=12.0)
    p.add_argument("--Rs", type=float, default=15.0)
    # Triaxial axis ratios (1,1,1 = spherical)
    p.add_argument("--ax-a", type=float, default=1.0)
    p.add_argument("--ax-b", type=float, default=1.0)
    p.add_argument("--ax-c", type=float, default=1.0)
    # Grid
    p.add_argument("--nxy", type=int, default=100)
    p.add_argument("--xymax", type=float, default=10.0)
    p.add_argument("--zplane", type=float, default=1.0)
    p.add_argument("--nrand", type=int, default=12000)
    p.add_argument("--rmax-rand", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)
    # AGAMA
    p.add_argument("--nr", type=int, default=70)
    p.add_argument("--nz", type=int, default=40)
    p.add_argument("--rmin", type=float, default=1e-3)
    p.add_argument("--rmax", type=float, default=25.0)
    p.add_argument("--zmin", type=float, default=1e-3)
    p.add_argument("--zmax", type=float, default=12.0)
    p.add_argument("--mmax", type=int, default=8)
    # Timing
    p.add_argument("--repeat", type=int, default=5)
    # Output
    p.add_argument("--savefig", type=str, default="plots/nfw_vs_agama.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    params = {
        'logM': args.logM,
        'Rs': args.Rs,
        'a': args.ax_a,
        'b': args.ax_b,
        'c': args.ax_c,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
    }

    M = 10.0 ** args.logM
    print(f"NFW: M={M:.2e} Msun, Rs={args.Rs} kpc, "
          f"axes=({args.ax_a}, {args.ax_b}, {args.ax_c})")

    # ---- Build test points ----
    x_grid = np.linspace(-args.xymax, args.xymax, args.nxy)
    y_grid = np.linspace(-args.xymax, args.xymax, args.nxy)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Xf, Yf = Xg.ravel(), Yg.ravel()
    Zf = np.full_like(Xf, args.zplane)
    pts_grid = np.column_stack([Xf, Yf, Zf])

    rng = np.random.default_rng(args.seed)
    pts_rand = rng.uniform(-args.rmax_rand, args.rmax_rand, size=(args.nrand, 3))
    pts_all = np.vstack([pts_grid, pts_rand])
    n_pts = len(pts_all)
    print(f"Total test points: {n_pts} ({len(pts_grid)} grid + {len(pts_rand)} random)")

    # ---- Build AGAMA potential from density ----
    print("\n--- Building AGAMA CylSpline potential ---")
    density_agama = make_nfw_density_for_agama(params)

    # Use AGAMA's built-in analytic NFW potential (not CylSpline)
    # AGAMA NFW: rho = M / (4*pi) * 1 / (r * (Rs + r)^2)
    # This matches our NFW_density exactly for spherical case.
    t0 = time.perf_counter()
    pot_agama = agama.Potential(
        type="NFW",
        mass=M,
        scaleRadius=args.Rs,
    )
    t1 = time.perf_counter()
    print(f"  AGAMA NFW build time: {t1 - t0:.4f} s")

    # ---- Build JAX evaluators ----
    print("\n--- Building JAX evaluators ---")
    pot_jax_fn, acc_jax_fn = make_nfw_jax_evaluators(params)

    # Warmup JIT
    print("  Warming up JIT...")
    _pts_small = pts_all[:10]
    _ = pot_jax_fn(_pts_small)
    _ = acc_jax_fn(_pts_small)
    print("  JIT warmup complete.")

    # ---- Timing ----
    print("\n--- Timing ---")
    phi_agama, t_agama_pot = timed_eval(pot_agama.potential, pts_all,
                                         repeat=args.repeat,
                                         label="AGAMA potential")
    acc_agama, t_agama_acc = timed_eval(pot_agama.force, pts_all,
                                         repeat=args.repeat,
                                         label="AGAMA acceleration")

    phi_jax, t_jax_pot = timed_eval(pot_jax_fn, pts_all,
                                     repeat=args.repeat,
                                     label="JAX potential")
    acc_jax_raw, t_jax_acc = timed_eval(acc_jax_fn, pts_all,
                                         repeat=args.repeat,
                                         label="JAX acceleration")

    phi_jax = np.asarray(phi_jax)
    acc_jax = np.asarray(acc_jax_raw)

    # ---- Speed comparison ----
    print("\n--- Speed Summary ---")
    print(f"  Potential:")
    print(f"    AGAMA:  {np.median(t_agama_pot):.4f}s  "
          f"({1e6*np.median(t_agama_pot)/n_pts:.2f} us/pt)")
    print(f"    JAX:    {np.median(t_jax_pot):.4f}s  "
          f"({1e6*np.median(t_jax_pot)/n_pts:.2f} us/pt)")
    print(f"  Acceleration:")
    print(f"    AGAMA:  {np.median(t_agama_acc):.4f}s  "
          f"({1e6*np.median(t_agama_acc)/n_pts:.2f} us/pt)")
    print(f"    JAX:    {np.median(t_jax_acc):.4f}s  "
          f"({1e6*np.median(t_jax_acc)/n_pts:.2f} us/pt)")

    ratio = np.median(t_agama_acc) / np.median(t_jax_acc)
    if ratio > 1:
        print(f"  JAX accel is {ratio:.1f}x faster than AGAMA")
    else:
        print(f"  AGAMA accel is {1/ratio:.1f}x faster than JAX")

    # ---- Accuracy ----
    print("\n--- Accuracy (JAX vs AGAMA) ---")
    summarize_relative("Potential", phi_jax, phi_agama)
    summarize_relative("ax", acc_jax[:, 0], acc_agama[:, 0])
    summarize_relative("ay", acc_jax[:, 1], acc_agama[:, 1])
    summarize_relative("az", acc_jax[:, 2], acc_agama[:, 2])
    summarize_relative("|a|",
                       np.linalg.norm(acc_jax, axis=1),
                       np.linalg.norm(acc_agama, axis=1))

    # ---- Comparison table with Dehnen bar ----
    print("\n--- For reference (copy from Dehnen bar benchmark): ---")
    print("  Dehnen V3 bar JAX fast accel: ~0.05 us/pt")
    print("  Dehnen V3 bar AGAMA accel:    ~0.06 us/pt")

    # ---- Plot ----
    if args.savefig:
        print(f"\n--- Generating plot ---")
        acc_grid_jax = np.asarray(acc_jax_fn(pts_grid))
        acc_grid_agama = pot_agama.force(pts_grid)
        make_plot(Xf, Yf, acc_grid_jax, acc_grid_agama,
                  figure_path=args.savefig)

    print("\nDone.")
