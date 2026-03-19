"""
Benchmark and compare dehnen_bar.py (pure JAX) vs AGAMA for Dehnen & Aly (2022)
barred disc potential/acceleration.

AGAMA takes a density function and computes the potential via CylSpline expansion.
Our dehnen_bar.py evaluates the analytic potential directly via JAX.

This script:
1. Builds the AGAMA potential from our density function
2. Compares accuracy of potential & acceleration at grid + random points
3. Times both evaluations
4. Produces a 3x3 residual figure (like test_dehnen2023_vs_agama.py)
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
from constants import KPCGYR_TO_KMS, G
from dehnen_bar import (
    make_params, DehnenBar_potential, DehnenBar_density,
    DehnenBar_acceleration, make_dehnen_bar_fns,
)

agama.setUnits(mass=1, length=1, velocity=KPCGYR_TO_KMS)


# ============================================================================
# Helpers
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
    """Time a function, return (result, times_array)."""
    out = fn(*args)  # warmup
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


# ============================================================================
# Make our density evaluator for AGAMA (needs numpy-compatible callable)
# ============================================================================
def make_density_for_agama(params):
    """Return a callable density(xyz) -> ndarray for AGAMA CylSpline."""
    density_vmap = jax.vmap(DehnenBar_density, in_axes=(0, 0, 0, None))

    def density_fn(xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim == 1:
            xyz = xyz[None, :]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        return np.asarray(density_vmap(jnp.array(x), jnp.array(y), jnp.array(z), params))

    return density_fn


# ============================================================================
# Build vectorized JAX evaluators
# ============================================================================
def make_jax_evaluators(params):
    """Build vmapped + jitted potential and acceleration evaluators."""
    pot_vmap = jax.jit(jax.vmap(DehnenBar_potential, in_axes=(0, 0, 0, None)))
    acc_vmap = jax.jit(jax.vmap(DehnenBar_acceleration, in_axes=(0, 0, 0, None)))

    def potential_batch(pts):
        x, y, z = jnp.array(pts[:, 0]), jnp.array(pts[:, 1]), jnp.array(pts[:, 2])
        return pot_vmap(x, y, z, params)

    def acceleration_batch(pts):
        x, y, z = jnp.array(pts[:, 0]), jnp.array(pts[:, 1]), jnp.array(pts[:, 2])
        return acc_vmap(x, y, z, params)

    return potential_batch, acceleration_batch


# ============================================================================
# Plotting
# ============================================================================
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
        ax[i, 0].set_title(f"{labels[i]} (dehnen_bar.py)")
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
# Main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark dehnen_bar.py (JAX) vs AGAMA CylSpline")
    # Model parameters
    p.add_argument("--model", type=str, default="T3")
    p.add_argument("--logM", type=float, default=10.4)
    p.add_argument("--L", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--s", type=float, default=4.2)
    p.add_argument("--q", type=float, default=1.2 / 4.2)
    p.add_argument("--phi", type=float, default=0.0)
    # Grid parameters
    p.add_argument("--nxy", type=int, default=100)
    p.add_argument("--xymax", type=float, default=10.0)
    p.add_argument("--zplane", type=float, default=1.0)
    p.add_argument("--nrand", type=int, default=12000)
    p.add_argument("--rmax-rand", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)
    # AGAMA CylSpline parameters
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
    p.add_argument("--savefig", type=str,
                   default="plots/dehnen_bar_vs_agama.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    M = 10.0 ** args.logM
    params = make_params(M=M, s=args.s, q=args.q, L=args.L,
                         gamma=args.gamma, mtype=args.model, phi=args.phi)

    print(f"Model: {args.model}, M={M:.2e} Msun, s={args.s}, q={args.q:.3f}, "
          f"L={args.L}, gamma={args.gamma}, phi={args.phi}")

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

    # ---- Build AGAMA potential from our density ----
    print("\n--- Building AGAMA CylSpline potential ---")
    density_agama = make_density_for_agama(params)

    t0 = time.perf_counter()
    pot_agama = agama.Potential(
        type="CylSpline",
        density=density_agama,
        symmetry="Triaxial",
        mmax=int(args.mmax),
        gridSizeR=int(args.nr),
        gridSizeZ=int(args.nz + 1),
        Rmin=float(args.rmin),
        Rmax=float(args.rmax),
        zmin=float(args.zmin),
        zmax=float(args.zmax),
        fixOrder=True,
    )
    t1 = time.perf_counter()
    print(f"  AGAMA CylSpline build time: {t1 - t0:.2f} s")

    # ---- Build our JAX evaluators ----
    print("\n--- Building JAX evaluators (generic API) ---")
    pot_jax_fn, acc_jax_fn = make_jax_evaluators(params)

    # ---- Build specialized (fast) JAX evaluators ----
    print("--- Building JAX evaluators (specialized/fast) ---")
    fns = make_dehnen_bar_fns(M=M, s=args.s, q=args.q, L=args.L,
                               gamma=args.gamma, mtype=args.model, phi=args.phi)
    pot_fast_fn = fns['potential_batch']
    acc_fast_fn = fns['acceleration_batch']

    def pot_fast_pts(pts):
        return pot_fast_fn(jnp.array(pts[:, 0]), jnp.array(pts[:, 1]),
                           jnp.array(pts[:, 2]))

    def acc_fast_pts(pts):
        return acc_fast_fn(jnp.array(pts[:, 0]), jnp.array(pts[:, 1]),
                           jnp.array(pts[:, 2]))

    # Warmup JIT
    print("  Warming up JIT (first call triggers compilation)...")
    _pts_small = pts_all[:10]
    _ = pot_jax_fn(_pts_small)
    _ = acc_jax_fn(_pts_small)
    _ = pot_fast_pts(_pts_small)
    _ = acc_fast_pts(_pts_small)
    print("  JIT warmup complete.")

    # ---- Timing: AGAMA ----
    print("\n--- Timing ---")
    phi_agama, t_agama_pot = timed_eval(pot_agama.potential, pts_all,
                                         repeat=args.repeat,
                                         label="AGAMA potential")
    acc_agama, t_agama_acc = timed_eval(pot_agama.force, pts_all,
                                         repeat=args.repeat,
                                         label="AGAMA acceleration")

    # ---- Timing: Generic JAX ----
    phi_jax, t_jax_pot = timed_eval(pot_jax_fn, pts_all,
                                     repeat=args.repeat,
                                     label="JAX generic potential")
    acc_jax_raw, t_jax_acc = timed_eval(acc_jax_fn, pts_all,
                                         repeat=args.repeat,
                                         label="JAX generic acceleration")

    # ---- Timing: Specialized (fast) JAX ----
    phi_fast, t_fast_pot = timed_eval(pot_fast_pts, pts_all,
                                       repeat=args.repeat,
                                       label="JAX fast potential")
    acc_fast_raw, t_fast_acc = timed_eval(acc_fast_pts, pts_all,
                                           repeat=args.repeat,
                                           label="JAX fast acceleration")

    phi_jax = np.asarray(phi_fast)
    acc_jax = np.asarray(acc_fast_raw)

    # ---- Speed comparison ----
    print("\n--- Speed Summary ---")
    def speed_str(t1, t2, name1, name2):
        r = np.median(t1) / np.median(t2)
        if r > 1:
            return f"{name2} {r:.1f}x faster"
        else:
            return f"{name1} {1/r:.1f}x faster"

    print(f"  Potential:")
    print(f"    AGAMA:        {np.median(t_agama_pot):.4f}s  ({1e6*np.median(t_agama_pot)/n_pts:.2f} us/pt)")
    print(f"    JAX generic:  {np.median(t_jax_pot):.4f}s  ({1e6*np.median(t_jax_pot)/n_pts:.2f} us/pt)")
    print(f"    JAX fast:     {np.median(t_fast_pot):.4f}s  ({1e6*np.median(t_fast_pot)/n_pts:.2f} us/pt)")
    print(f"  Acceleration:")
    print(f"    AGAMA:        {np.median(t_agama_acc):.4f}s  ({1e6*np.median(t_agama_acc)/n_pts:.2f} us/pt)")
    print(f"    JAX generic:  {np.median(t_jax_acc):.4f}s  ({1e6*np.median(t_jax_acc)/n_pts:.2f} us/pt)")
    print(f"    JAX fast:     {np.median(t_fast_acc):.4f}s  ({1e6*np.median(t_fast_acc)/n_pts:.2f} us/pt)")
    print(f"  Speedup fast vs generic accel: {np.median(t_jax_acc)/np.median(t_fast_acc):.1f}x")
    print(f"  Speedup fast vs AGAMA accel:   {speed_str(t_agama_acc, t_fast_acc, 'AGAMA', 'JAX fast')}")

    # ---- Accuracy comparison ----
    print("\n--- Accuracy (JAX vs AGAMA) ---")
    print("  Note: AGAMA uses a CylSpline approximation of the density,")
    print("  so residuals reflect AGAMA's approximation error, not ours.")
    summarize_relative("Potential", phi_jax, phi_agama)
    summarize_relative("ax", acc_jax[:, 0], acc_agama[:, 0])
    summarize_relative("ay", acc_jax[:, 1], acc_agama[:, 1])
    summarize_relative("az", acc_jax[:, 2], acc_agama[:, 2])
    summarize_relative("|a|",
                       np.linalg.norm(acc_jax, axis=1),
                       np.linalg.norm(acc_agama, axis=1))

    # ---- Plot ----
    if args.savefig:
        print(f"\n--- Generating plot ---")
        acc_grid_jax = np.asarray(acc_fast_pts(pts_grid))
        acc_grid_agama = pot_agama.force(pts_grid)
        make_plot(Xf, Yf, acc_grid_jax, acc_grid_agama,
                  figure_path=args.savefig)

    print("\nDone.")
