"""
Compare Dehnen+2023 bar potential/forces against AGAMA.

Style and outputs are intentionally similar to `test_potential.py`:
- clear timing blocks
- component-wise error summaries
- 3x3 visualization panel (ours / AGAMA / residual for R, phi, z force)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import agama

from constants import KPCGYR_TO_KMS
from dehnen_2023_bar_pairs import (
    make_dehnen2023_bar_functions,
    make_dehnen2023_bar_numpy_backend,
)


agama.setUnits(mass=1, length=1, velocity=KPCGYR_TO_KMS)


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
        f"{name}: mean={rel.mean():.4e}, "
        f"median={np.median(rel):.4e}, "
        f"p95={np.percentile(rel,95):.4e}, "
        f"max={rel.max():.4e}"
    )


def timed_eval(fn, arr, repeat=3):
    out = fn(arr)  # warmup
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(arr)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return out, np.array(times, dtype=float)


def build_params(args):
    return {
        "model_bar": args.model,
        "logM_bar": args.logM,
        "L_bar": args.L,
        "gamma_bar": args.gamma,
        "s_bar": args.s,
        "q_bar": args.q,
        "phi_bar": args.phi,
        "x_origin": args.x0,
        "y_origin": args.y0,
        "z_origin": args.z0,
        "dirx": args.dirx,
        "diry": args.diry,
        "dirz": args.dirz,
    }


def density_for_agama(backend):
    def fn(x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
            return backend["density"](arr)[0]
        return backend["density"](arr)

    return fn


def make_plot(
    x_flat,
    y_flat,
    acc_ours,
    acc_agama,
    figure_path: str | None = None,
):
    _, _, fR_ours, fphi_ours = getCylindricalFromCartesian_clockwise(
        x_flat, y_flat, acc_ours[:, 0], acc_ours[:, 1]
    )
    _, _, fR_agama, fphi_agama = getCylindricalFromCartesian_clockwise(
        x_flat, y_flat, acc_agama[:, 0], acc_agama[:, 1]
    )

    ours_cyl = np.column_stack([fR_ours, fphi_ours, acc_ours[:, 2]])
    agama_cyl = np.column_stack([fR_agama, fphi_agama, acc_agama[:, 2]])

    residual = (agama_cyl - ours_cyl) / np.maximum(np.abs(agama_cyl), 1e-8)
    labels = ["R acceleration", "phi acceleration", "Z acceleration"]

    fig, ax = plt.subplots(
        3, 3, figsize=(22, 18), gridspec_kw={"hspace": 0.35, "wspace": 0.25}
    )

    for i in range(3):
        vmin = np.percentile(agama_cyl[:, i], 2)
        vmax = np.percentile(agama_cyl[:, i], 98)
        cb = ax[i, 0].scatter(
            x_flat, y_flat, c=ours_cyl[:, i], s=6, marker="s", cmap="viridis", vmin=vmin, vmax=vmax
        )
        plt.colorbar(cb, ax=ax[i, 0], label=f"{labels[i]} [kpc/Gyr^2]")
        ax[i, 0].set_title(f"{labels[i]} (ours)")
        ax[i, 0].set_xlabel("X [kpc]")
        ax[i, 0].set_ylabel("Y [kpc]")

        cb = ax[i, 1].scatter(
            x_flat, y_flat, c=agama_cyl[:, i], s=6, marker="s", cmap="viridis", vmin=vmin, vmax=vmax
        )
        plt.colorbar(cb, ax=ax[i, 1], label=f"{labels[i]} [kpc/Gyr^2]")
        ax[i, 1].set_title(f"{labels[i]} (AGAMA)")
        ax[i, 1].set_xlabel("X [kpc]")
        ax[i, 1].set_ylabel("Y [kpc]")

        q = np.percentile(np.abs(residual[:, i]), 98)
        q = max(q, 1e-3)
        cb = ax[i, 2].scatter(
            x_flat,
            y_flat,
            c=residual[:, i],
            s=6,
            marker="s",
            cmap="coolwarm",
            vmin=-q,
            vmax=q,
        )
        plt.colorbar(cb, ax=ax[i, 2], label="fractional residual")
        ax[i, 2].set_title(f"{labels[i]} (AGAMA - ours) / |AGAMA|")
        ax[i, 2].set_xlabel("X [kpc]")
        ax[i, 2].set_ylabel("Y [kpc]")

    if figure_path:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {figure_path}")
    else:
        plt.show()
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Validate Dehnen+2023 bar against AGAMA.")
    p.add_argument("--model", type=str, default="V3")
    p.add_argument("--logM", type=float, default=10.4)
    p.add_argument("--L", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--s", type=float, default=4.2)
    p.add_argument("--q", type=float, default=1.2 / 4.2)
    p.add_argument("--phi", type=float, default=0.0)
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--z0", type=float, default=0.0)
    p.add_argument("--dirx", type=float, default=0.0)
    p.add_argument("--diry", type=float, default=0.0)
    p.add_argument("--dirz", type=float, default=1.0)

    p.add_argument("--nxy", type=int, default=100)
    p.add_argument("--xymax", type=float, default=10.0)
    p.add_argument("--zplane", type=float, default=1.0)

    p.add_argument("--nrand", type=int, default=12000)
    p.add_argument("--rmax-rand", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--nr", type=int, default=70)
    p.add_argument("--nz", type=int, default=40)
    p.add_argument("--rmin", type=float, default=1e-3)
    p.add_argument("--rmax", type=float, default=25.0)
    p.add_argument("--zmin", type=float, default=1e-3)
    p.add_argument("--zmax", type=float, default=12.0)
    p.add_argument("--mmax", type=int, default=8)
    p.add_argument("--repeat", type=int, default=3)

    p.add_argument("--plot", action="store_true")
    p.add_argument("--savefig", type=str, default="")

    p.add_argument("--check-vmap", action="store_true")
    p.add_argument("--vmap-npts", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = build_params(args)

    backend = make_dehnen2023_bar_numpy_backend(params)

    x_grid = np.linspace(-args.xymax, args.xymax, args.nxy)
    y_grid = np.linspace(-args.xymax, args.xymax, args.nxy)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    Xf, Yf = Xg.ravel(), Yg.ravel()
    Zf = np.ones_like(Xf) * args.zplane
    pts_grid = np.column_stack([Xf, Yf, Zf])

    rng = np.random.default_rng(args.seed)
    pts_rand = rng.uniform(-args.rmax_rand, args.rmax_rand, size=(args.nrand, 3))
    pts_all = np.vstack([pts_grid, pts_rand])

    t0 = time.perf_counter()
    pot_agama = agama.Potential(
        type="CylSpline",
        density=density_for_agama(backend),
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
    print(f"Time to build AGAMA potential: {t1 - t0:.3f} s")

    phi_ours, t_phi_ours = timed_eval(backend["potential"], pts_all, repeat=args.repeat)
    time_start = time.time()
    acc_ours, t_acc_ours = timed_eval(backend["acceleration"], pts_all, repeat=args.repeat)
    time_end = time.time()
    print(f"Time taken to compute acceleration (ours): {time_end - time_start:.4f} s")
    phi_agama, t_phi_agama = timed_eval(pot_agama.potential, pts_all, repeat=args.repeat)
    time_start = time.time()
    acc_agama, t_acc_agama = timed_eval(pot_agama.force, pts_all, repeat=args.repeat)
    time_end = time.time()
    print(f"Time taken to compute acceleration (AGAMA): {time_end - time_start:.4f} s")

    n_pts = len(pts_all)
    print(f"Total points: {n_pts}")
    print(
        f"Ours potential time: median={np.median(t_phi_ours):.4f} s, "
        f"{1e6*np.median(t_phi_ours)/n_pts:.2f} us/pt"
    )
    print(
        f"Ours accel time: median={np.median(t_acc_ours):.4f} s, "
        f"{1e6*np.median(t_acc_ours)/n_pts:.2f} us/pt"
    )
    print(
        f"AGAMA potential time: median={np.median(t_phi_agama):.4f} s, "
        f"{1e6*np.median(t_phi_agama)/n_pts:.2f} us/pt"
    )
    print(
        f"AGAMA accel time: median={np.median(t_acc_agama):.4f} s, "
        f"{1e6*np.median(t_acc_agama)/n_pts:.2f} us/pt"
    )

    summarize_relative("Potential fractional diff", phi_ours, phi_agama)
    summarize_relative("ax fractional diff", acc_ours[:, 0], acc_agama[:, 0])
    summarize_relative("ay fractional diff", acc_ours[:, 1], acc_agama[:, 1])
    summarize_relative("az fractional diff", acc_ours[:, 2], acc_agama[:, 2])
    summarize_relative(
        "|a| fractional diff",
        np.linalg.norm(acc_ours, axis=1),
        np.linalg.norm(acc_agama, axis=1),
    )

    if args.plot or args.savefig:
        acc_grid_ours = backend["acceleration"](pts_grid)
        acc_grid_agama = pot_agama.force(pts_grid)
        make_plot(
            Xf,
            Yf,
            acc_grid_ours,
            acc_grid_agama,
            figure_path=(args.savefig if args.savefig else None),
        )

    if args.check_vmap:
        funcs = make_dehnen2023_bar_functions(params)
        n_check = min(args.vmap_npts, n_pts)
        pts_check = jnp.asarray(pts_all[:n_check], dtype=jnp.float32)

        # Warmup + timing for vmapped JAX wrapper
        _ = funcs["acceleration_vmap"](pts_check).block_until_ready()
        t0 = time.perf_counter()
        acc_vmap = np.array(funcs["acceleration_vmap"](pts_check))
        t1 = time.perf_counter()
        print(
            f"JAX vmapped accel time: {(t1 - t0):.4f} s for {n_check} pts, "
            f"{1e6*(t1 - t0)/n_check:.2f} us/pt"
        )

        acc_fast = backend["acceleration"](np.array(pts_check))
        summarize_relative("vmap-vs-fast ax diff", acc_vmap[:, 0], acc_fast[:, 0], floor=0.0)
        summarize_relative("vmap-vs-fast ay diff", acc_vmap[:, 1], acc_fast[:, 1], floor=0.0)
        summarize_relative("vmap-vs-fast az diff", acc_vmap[:, 2], acc_fast[:, 2], floor=0.0)
