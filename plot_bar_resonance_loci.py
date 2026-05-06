"""
Plot expected planar bar-resonance loci in E-Lz and Jr-Lz space.

The key idea is:
1. Use the axisymmetrized background potential H0, not the full barred potential,
   to define actions and frequencies.
2. Sample a family of planar orbits in H0.
3. Measure (Omega_r, Omega_phi, Jr, E, Lz) for each orbit.
4. Plot the zero-contours of

       l_r * Omega_r + l_phi * Omega_phi - l_b * Omega_b = 0

   in both E-Lz and Jr-Lz.

This gives the expected resonance loci.  To check whether the full barred model is
actually trapping orbits around a given resonance, the next diagnostic is the
resonant angle

    theta_res = l_r theta_r + l_phi theta_phi - l_b Omega_b t,

which should librate for trapped orbits.

Notes
-----
- This script is planar: z = vz = 0, so it is a Jz ~ 0 slice.
- The orbits are integrated in the axisymmetrized background potential.
- In this repository's sign convention, prograde orbits have positive Lz and
  positive Omega_phi, so Omega_phi is defined with an extra minus sign relative
  to the raw atan2 phase drift.
"""

from __future__ import annotations

import argparse
import pickle
from collections import OrderedDict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from constants import G
from dehnen_bar import T3_potential, V4_potential
from integrants_with_binning import integrate_leapfrog_traj
from potentials import MiyamotoNagai_potential, NFW_potential
from utils import get_rotation_curve, logMenc_logc_to_logM_logRs


COMMON_RESONANCES = OrderedDict(
    [
        ("ILR", (-1, 2, 2, "tab:blue")),
        ("CR", (0, 2, 2, "tab:green")),
        ("OLR", (1, 2, 2, "tab:red")),
        ("IUHR", (-1, 4, 4, "tab:purple")),
        ("OUHR", (1, 4, 4, "tab:orange")),
    ]
)

# Keep these aligned with the current orbital-structure script.
GAMMA_BAR = 1.0
V4_A = 0.5
V4_B = 0.5
V4_GAMMA = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-file",
        default="posteriors/mcmc_checkpoint_0415_beta25_gamma140_D50_gal2.pkl",
        help="Posterior checkpoint used to build the best-fit potential.",
    )
    parser.add_argument(
        "--output-file",
        default="bar_resonance_loci.png",
        help="Where to save the final figure.",
    )
    parser.add_argument(
        "--discard",
        type=int,
        default=400,
        help="Discard this many initial MCMC steps before summarising the posterior.",
    )
    parser.add_argument(
        "--logprob-window",
        type=float,
        default=100.0,
        help="Keep walkers with final logprob within this value of the maximum.",
    )
    parser.add_argument("--n-r", type=int, default=36, help="Number of launch radii.")
    parser.add_argument(
        "--n-eta",
        type=int,
        default=28,
        help="Number of launch tangential-velocity fractions.",
    )
    parser.add_argument("--r-min", type=float, default=0.4, help="Minimum launch radius [kpc].")
    parser.add_argument("--r-max", type=float, default=12.0, help="Maximum launch radius [kpc].")
    parser.add_argument(
        "--eta-min",
        type=float,
        default=0.55,
        help="Minimum v_phi / v_circ launch fraction.",
    )
    parser.add_argument(
        "--eta-max",
        type=float,
        default=1.15,
        help="Maximum v_phi / v_circ launch fraction.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=4096,
        help="Number of leapfrog steps per orbit.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0025,
        help="Leapfrog timestep [Gyr].",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Integrate this many orbits at once.",
    )
    parser.add_argument(
        "--tol-frac",
        type=float,
        default=0.02,
        help="Highlight approximate near-resonant points satisfying |F| < tol_frac * Omega_b.",
    )
    parser.add_argument(
        "--resonances",
        nargs="+",
        default=["ILR", "CR", "OLR"],
        choices=list(COMMON_RESONANCES.keys()),
        help="Resonances to overlay.",
    )
    return parser.parse_args()


def plot_prettier(dpi: int = 180, fontsize: int = 15) -> None:
    plt.rcParams["figure.dpi"] = dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc("font", size=fontsize)
    plt.rc("xtick", direction="in")
    plt.rc("ytick", direction="in")
    plt.rc("text", usetex=False)
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def load_best_fit_vector(
    checkpoint_file: str,
    discard: int = 400,
    logprob_window: float = 100.0,
) -> np.ndarray:
    with open(checkpoint_file, "rb") as f:
        checkpoint = pickle.load(f)

    posterior = np.stack(checkpoint["all_samples"], axis=0)
    logprob = np.stack(checkpoint["all_logprob"], axis=0)

    good = logprob[-1, :] > np.amax(logprob[-1, :]) - logprob_window
    posterior = posterior[:, good, :]
    posterior = posterior[discard:, :, :]
    posterior = posterior.reshape(-1, posterior.shape[-1])
    return np.percentile(posterior, 50, axis=0)


def load_potential_dict(params: np.ndarray) -> tuple[dict, dict]:
    logM_enc = params[0]
    log_c = params[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(
        logM_enc,
        log_c,
        r_enc=10.0,
        Delta=200.0,
        rho_crit=277.54,
    )

    logM_disc = params[1]
    logM_bar = params[2]
    logRs_disk = params[4]
    logHs_disk = params[5]
    logL_bar = params[6]
    log_light_to_mass_ratio = params[10]
    log_Omega_bar = params[11]

    L_bar = 10.0**logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0**logHs_disk
    b_bar = Hs_disc

    params_halo = {
        "logM": logM_halo,
        "Rs": 10.0**logRs_halo,
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
        "x_origin": 0.0,
        "y_origin": 0.0,
        "z_origin": 0.0,
        "dirx": 0.0,
        "diry": 0.0,
        "dirz": 1.0,
    }

    params_baryon = {
        "logM_disc": logM_disc,
        "Rs_disc": 10.0**logRs_disk,
        "Hs_disc": Hs_disc,
        "logM_bar": logM_bar,
        "L_bar": L_bar,
        "a_bar": a_bar,
        "b_bar": b_bar,
        "light_to_mass_ratio": 10.0**log_light_to_mass_ratio,
        "Omega_bar": 10.0**log_Omega_bar,
        "x_origin": 0.0,
        "y_origin": 0.0,
        "z_origin": 0.0,
        "dirx": 0.0,
        "diry": 0.0,
        "dirz": 1.0,
    }
    return params_baryon, params_halo


@jax.jit
def axisymmetric_background_potential(x, y, z, params_baryon, params_halo):
    """Axisymmetrized H0 used to define actions and frequencies."""
    phi_halo = NFW_potential(x, y, z, params_halo)

    mn_params = {
        "logM_disc": params_baryon["logM_disc"],
        "Rs_disc": params_baryon["Rs_disc"],
        "Hs_disc": params_baryon["Hs_disc"],
        "x_origin": params_baryon["x_origin"],
        "y_origin": params_baryon["y_origin"],
        "z_origin": params_baryon["z_origin"],
        "dirx": params_baryon["dirx"],
        "diry": params_baryon["diry"],
        "dirz": params_baryon["dirz"],
    }
    phi_disc = MiyamotoNagai_potential(x, y, z, mn_params)

    M_bar = 10.0 ** params_baryon["logM_bar"]
    a_bar = params_baryon["a_bar"]
    b_bar = params_baryon["b_bar"]

    # Turn the bar harmonics into their axisymmetric (L=0) counterparts.
    phi_t3_axi = T3_potential(x, y, z, M_bar, a_bar, b_bar, 0.0, GAMMA_BAR)
    phi_v4_axi = V4_potential(x, y, z, M_bar, V4_A, V4_B, 0.0, V4_GAMMA)
    return phi_halo + phi_disc + phi_t3_axi + phi_v4_axi


@jax.jit
def axisymmetric_background_acceleration(x, y, z, params_baryon, params_halo):
    def _pot(pos):
        return axisymmetric_background_potential(pos[0], pos[1], pos[2], params_baryon, params_halo)

    return -jax.grad(_pot)(jnp.array([x, y, z]))


def make_initial_conditions(
    params_baryon: dict,
    params_halo: dict,
    n_r: int,
    n_eta: int,
    r_min: float,
    r_max: float,
    eta_min: float,
    eta_max: float,
) -> np.ndarray:
    r_grid = np.geomspace(r_min, r_max, n_r)
    eta_grid = np.linspace(eta_min, eta_max, n_eta)

    vcirc = np.asarray(
        jax.vmap(
            lambda rr: get_rotation_curve(
                rr,
                axisymmetric_background_potential,
                (params_baryon, params_halo),
                0.0,
            )
        )(jnp.asarray(r_grid))
    )

    rr, ee = np.meshgrid(r_grid, eta_grid, indexing="ij")
    vv = ee * vcirc[:, None]

    # Repository convention: prograde launch at x=R, y=0 has vy < 0 and Lz > 0.
    w0 = np.zeros((rr.size, 6), dtype=float)
    w0[:, 0] = rr.ravel()
    w0[:, 4] = -vv.ravel()
    return w0


def integrate_orbits(
    w0_all: np.ndarray,
    params_baryon: dict,
    params_halo: dict,
    n_steps: int,
    dt: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    def acc_fn(x, y, z):
        return axisymmetric_background_acceleration(x, y, z, params_baryon, params_halo)

    def one_orbit(w0):
        _, w = integrate_leapfrog_traj(w0, acc_fn, n_steps, dt=dt, t0=0.0, unroll=False)
        return w

    integrate_batch = jax.jit(jax.vmap(one_orbit, in_axes=0))

    t = dt * np.arange(n_steps + 1)
    all_traj = []

    for start in range(0, len(w0_all), batch_size):
        stop = min(start + batch_size, len(w0_all))
        batch = jnp.asarray(w0_all[start:stop])
        traj = np.asarray(integrate_batch(batch))
        traj = np.concatenate([np.asarray(batch)[:, None, :], traj], axis=1)
        all_traj.append(traj)

    return t, np.concatenate(all_traj, axis=0)


def _dedupe_indices(idx: np.ndarray, min_sep: int = 8) -> np.ndarray:
    if len(idx) == 0:
        return idx
    keep = [int(idx[0])]
    for val in idx[1:]:
        if int(val) - keep[-1] >= min_sep:
            keep.append(int(val))
    return np.asarray(keep, dtype=int)


def measure_planar_orbit(
    t: np.ndarray,
    orbit: np.ndarray,
    params_baryon: dict,
    params_halo: dict,
) -> dict | None:
    x = orbit[:, 0]
    y = orbit[:, 1]
    vx = orbit[:, 3]
    vy = orbit[:, 4]

    r = np.hypot(x, y)
    valid_r = r > 1e-10
    if np.count_nonzero(valid_r) < 10:
        return None

    v_r = np.zeros_like(r)
    v_r[valid_r] = (x[valid_r] * vx[valid_r] + y[valid_r] * vy[valid_r]) / r[valid_r]

    phi = np.unwrap(np.arctan2(y, x))
    idx_apo = _dedupe_indices(np.where((v_r[:-1] > 0.0) & (v_r[1:] <= 0.0))[0] + 1)
    idx_peri = _dedupe_indices(np.where((v_r[:-1] < 0.0) & (v_r[1:] >= 0.0))[0] + 1)
    turns = idx_apo if len(idx_apo) >= len(idx_peri) else idx_peri

    if len(turns) < 3:
        return None

    periods = np.diff(t[turns])
    periods = periods[periods > 0.0]
    if len(periods) == 0:
        return None

    omega_r = 2.0 * np.pi / np.median(periods)
    if not np.isfinite(omega_r) or omega_r <= 0.0:
        return None

    if len(turns) >= 4:
        i0 = int(turns[1])
        i1 = int(turns[-2])
    else:
        i0 = int(turns[0])
        i1 = int(turns[-1])
    if i1 <= i0 + 8:
        return None

    dt_span = t[i1] - t[i0]
    # Minus sign matches the repository's positive-prograde convention.
    omega_phi = -(phi[i1] - phi[i0]) / dt_span
    if not np.isfinite(omega_phi) or omega_phi <= 0.0:
        return None

    jr = np.trapz(v_r[i0 : i1 + 1] ** 2, t[i0 : i1 + 1]) / (omega_r * dt_span)
    if not np.isfinite(jr) or jr < 0.0:
        return None

    x0, y0, z0, vx0, vy0, vz0 = orbit[0]
    phi0 = float(axisymmetric_background_potential(x0, y0, z0, params_baryon, params_halo))
    energy = 0.5 * (vx0 * vx0 + vy0 * vy0 + vz0 * vz0) + phi0
    # Positive for prograde orbits in this repository's sign convention.
    lz = y0 * vx0 - x0 * vy0

    return {
        "E": energy,
        "Lz": lz,
        "Jr": jr,
        "Omega_r": omega_r,
        "Omega_phi": omega_phi,
    }


def measure_orbit_family(
    t: np.ndarray,
    traj: np.ndarray,
    params_baryon: dict,
    params_halo: dict,
) -> dict[str, np.ndarray]:
    rows = []
    for orbit in traj:
        result = measure_planar_orbit(t, orbit, params_baryon, params_halo)
        if result is not None:
            rows.append(result)

    if not rows:
        raise RuntimeError("No valid planar orbits were recovered; widen the launch grid.")

    return {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in ["E", "Lz", "Jr", "Omega_r", "Omega_phi"]
    }


def build_contour_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    n_x: int = 250,
    n_y: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), n_x)
    y_grid = np.linspace(np.nanpercentile(y, 1), np.nanpercentile(y, 99), n_y)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")
    zz = griddata(np.column_stack([x, y]), values, (xx, yy), method="linear")
    return xx, yy, zz


def add_resonance_overlay(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    omega_r: np.ndarray,
    omega_phi: np.ndarray,
    omega_bar: float,
    resonance_names: list[str],
    tol_frac: float,
    x_label: str,
) -> None:
    ax.hexbin(x, y, gridsize=85, bins="log", mincnt=1, cmap="Greys")

    handles = []
    tol = tol_frac * omega_bar
    for name in resonance_names:
        lr, lphi, lb, color = COMMON_RESONANCES[name]
        residual = lr * omega_r + lphi * omega_phi - lb * omega_bar
        xx, yy, zz = build_contour_grid(x, y, residual)

        if np.isfinite(zz).sum() > 10:
            ax.contour(xx, yy, zz, levels=[0.0], colors=[color], linewidths=2.0)

        near = np.abs(residual) < tol
        if np.any(near):
            ax.scatter(
                x[near],
                y[near],
                s=6,
                c=color,
                alpha=0.25,
                linewidths=0.0,
            )

        handle = plt.Line2D([], [], color=color, lw=2, label=f"{name}: ({lr}, {lphi}, {lb})")
        handles.append(handle)

    ax.legend(handles=handles, loc="best", frameon=True, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$L_z \ [{\rm kpc}^2\,{\rm Gyr}^{-1}]$")
    ax.grid(alpha=0.2, lw=0.5)


def make_figure(
    results: dict[str, np.ndarray],
    omega_bar: float,
    resonance_names: list[str],
    tol_frac: float,
    output_file: str,
) -> None:
    plot_prettier()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    add_resonance_overlay(
        axes[0],
        results["E"],
        results["Lz"],
        results["Omega_r"],
        results["Omega_phi"],
        omega_bar,
        resonance_names,
        tol_frac,
        r"$E \ [{\rm kpc}^2\,{\rm Gyr}^{-2}]$",
    )
    axes[0].set_title(r"$E-L_z$ resonance loci")

    add_resonance_overlay(
        axes[1],
        results["Jr"],
        results["Lz"],
        results["Omega_r"],
        results["Omega_phi"],
        omega_bar,
        resonance_names,
        tol_frac,
        r"$J_r \ [{\rm kpc}^2\,{\rm Gyr}^{-1}]$",
    )
    axes[1].set_title(r"$J_r-L_z$ resonance loci")

    fig.suptitle(
        rf"Axisymmetrized background resonances, $\Omega_b = {omega_bar:.2f}\ {{\rm Gyr}}^{{-1}}$",
        y=1.02,
    )
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved figure to {output_file}")


def main() -> None:
    args = parse_args()

    best_fit = load_best_fit_vector(
        checkpoint_file=args.checkpoint_file,
        discard=args.discard,
        logprob_window=args.logprob_window,
    )
    params_baryon, params_halo = load_potential_dict(best_fit)
    omega_bar = params_baryon["Omega_bar"]

    print(f"Using Omega_bar = {omega_bar:.4f} Gyr^-1")
    print("Sampling launch conditions...")
    w0_all = make_initial_conditions(
        params_baryon=params_baryon,
        params_halo=params_halo,
        n_r=args.n_r,
        n_eta=args.n_eta,
        r_min=args.r_min,
        r_max=args.r_max,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
    )

    print(f"Integrating {len(w0_all)} planar orbits...")
    t, traj = integrate_orbits(
        w0_all=w0_all,
        params_baryon=params_baryon,
        params_halo=params_halo,
        n_steps=args.n_steps,
        dt=args.dt,
        batch_size=args.batch_size,
    )

    print("Measuring (E, Lz, Jr, Omega_r, Omega_phi)...")
    results = measure_orbit_family(t, traj, params_baryon, params_halo)
    print(f"Recovered {len(results['E'])} usable orbits.")

    make_figure(
        results=results,
        omega_bar=omega_bar,
        resonance_names=args.resonances,
        tol_frac=args.tol_frac,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
