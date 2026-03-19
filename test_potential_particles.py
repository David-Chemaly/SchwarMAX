"""
Benchmark the AGAMA-assisted CylindricalSpline potential (from particles)
against direct AGAMA evaluation.

Loads the pre-computed dict_phi from dict_phi_stellar_t_t0_4.pkl,
rebuilds the AGAMA reference potential from the same snapshot,
and compares acceleration on an XY grid at z=1 kpc.

Produces a 3x3 figure (R, phi, z components) like test_potential.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import jax
import jax.numpy as jnp
import agama
from astropy.constants import G as G_astropy
import astropy.units as u

from constants import KPCGYR_TO_KMS
from CylindricalSpline import get_acc, evaluate_phi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def getCylindricalFromCartesian_clockwise(x, y, vx, vy):
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    vR = (x*vx + y*vy) / R
    vphi = -(x*vy - y*vx) / R
    return R, phi, vR, vphi


def bar_angle_bar_strength(x, y, R_anulus=np.arange(1, 5, 0.25)):
    R0 = np.sqrt(x**2 + y**2)
    phi0 = np.arctan2(y, x)
    bar_angles, bar_strengths = [], []
    for i in range(len(R_anulus) - 1):
        sel = (R0 > R_anulus[i]) & (R0 < R_anulus[i + 1])
        phi_bin = phi0[sel]
        A2 = np.sum(np.cos(2 * phi_bin)) / len(phi_bin)
        B2 = np.sum(np.sin(2 * phi_bin)) / len(phi_bin)
        bar_angles.append(0.5 * np.arctan2(B2, A2))
        bar_strengths.append(np.sqrt(A2**2 + B2**2))
    R_mid = (R_anulus[:-1] + R_anulus[1:]) / 2
    return R_mid, np.array(bar_angles), np.array(bar_strengths)


def rotate(posvel, angle):
    x, y, z, vx, vy, vz = posvel.T
    s, c = np.sin(angle), np.cos(angle)
    return np.array([x*c - y*s, x*s + y*c, z,
                     vx*c - vy*s, vx*s + vy*c, vz]).T


if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # 1. Load pre-computed potential
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("1. LOADING PRE-COMPUTED POTENTIAL")
    print("=" * 70)

    phi_path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/Bar_model_TG21/dict_phi_stellar_t_t0_4.pkl'
    with open(phi_path, 'rb') as f:
        d = pickle.load(f)
    metadata = d.pop('_metadata', {})
    dict_phi = {k: jnp.array(v) for k, v in d.items()}
    print(f"  Loaded from: {phi_path}")
    print(f"  Grid: NR={metadata.get('NR')}, NZ={metadata.get('NZ')}, Mmax={metadata.get('Mmax')}")
    print(f"  Rmax={metadata.get('Rmax')}, Zmax={metadata.get('Zmax')}")

    # ---------------------------------------------------------------------------
    # 2. Load snapshot and build AGAMA reference
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. BUILDING AGAMA REFERENCE POTENTIAL")
    print("=" * 70)

    snapshot_path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/Bar_model_TG21/model/t_t0_4'

    # Read in GADGET code units
    agama.setUnits(length=1, velocity=1, mass=1)
    mass_unit = 1 / ((G_astropy * u.Msun).to(u.kpc * (u.km / u.s)**2)).value

    w0_data, mass_raw = agama.readSnapshot(snapshot_path)
    mass_data = mass_raw * mass_unit

    # Remove DM
    unique_masses, counts = np.unique(mass_data, return_counts=True)
    mask_stellar = mass_data != unique_masses[-1]
    w0_data = w0_data[mask_stellar]
    mass_data = mass_data[mask_stellar]
    print(f"  Stellar particles: {len(mass_data)}")

    # Center on COM
    R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
    mask_inner = R < 5
    for i in range(6):
        w0_data[:, i] -= np.sum(w0_data[mask_inner, i] * mass_data[mask_inner]) / np.sum(mass_data[mask_inner])

    # Flip x-axis
    w0_data[:, 0] *= -1
    w0_data[:, 3] *= -1

    # Rotate bar to x-axis
    R_mid, bar_angles, _ = bar_angle_bar_strength(w0_data[:, 0], w0_data[:, 1])
    bar_angle0 = np.mean(bar_angles[R_mid < 4])
    w0_data = rotate(w0_data, -bar_angle0)
    print(f"  Bar angle: {np.degrees(bar_angle0):.1f} deg")

    positions = w0_data[:, :3]
    masses = mass_data

    # Build AGAMA potential with same units as dict_phi (kpc/Gyr)^2
    agama.setUnits(mass=1, length=1, velocity=KPCGYR_TO_KMS)
    t0 = time.time()
    pot_agama = agama.Potential(
        type='CylSpline',
        particles=(positions, masses),
        symmetry='Triaxial',
        mmax=int(metadata.get('Mmax', 8)),
        gridSizeR=int(metadata.get('agama_gridSizeR', 50)),
        gridSizeZ=int(metadata.get('agama_gridSizeZ', 30)),
    )
    t_agama_build = time.time() - t0
    print(f"  AGAMA build time: {t_agama_build:.2f} s")

    # ---------------------------------------------------------------------------
    # 3. Evaluate acceleration on XY grid
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. EVALUATING ACCELERATION")
    print("=" * 70)

    x_c, y_c = np.linspace(-10, 10, 100), np.linspace(-10, 10, 100)
    X_c, Y_c = np.meshgrid(x_c, y_c)
    X_c, Y_c = X_c.flatten(), Y_c.flatten()
    Z_c = np.ones_like(X_c) * 1.0  # z = 1 kpc

    pts = np.array([X_c, Y_c, Z_c]).T

    # --- Our CylindricalSpline ---
    get_acc_vec = jax.jit(jax.vmap(get_acc, in_axes=(0, 0, 0, None)))

    # JIT warm-up
    _ = get_acc_vec(jnp.array(pts[:, 0] * 0.99), jnp.array(pts[:, 1] * 0.99),
                    jnp.array(pts[:, 2] * 0.99), dict_phi)
    _.block_until_ready()

    t0 = time.time()
    acc_pts_us = get_acc_vec(jnp.array(pts[:, 0]), jnp.array(pts[:, 1]),
                             jnp.array(pts[:, 2]), dict_phi)
    acc_pts_us.block_until_ready()
    t_us = time.time() - t0
    print(f"  Our acceleration ({len(pts)} points): {t_us*1000:.1f} ms")

    # Convert to cylindrical (R, phi, z)
    R_us, phi_us, FR_us, Fphi_us = getCylindricalFromCartesian_clockwise(
        pts[:, 0], pts[:, 1], np.array(acc_pts_us[:, 0]), np.array(acc_pts_us[:, 1]))
    acc_cyl_us = np.column_stack([FR_us, Fphi_us, np.array(acc_pts_us[:, 2])])

    # --- AGAMA reference ---
    t0 = time.time()
    acc_pts_agama = pot_agama.force(pts)
    t_agama = time.time() - t0
    print(f"  AGAMA acceleration ({len(pts)} points): {t_agama*1000:.1f} ms")
    print(f"  Speed ratio (ours / AGAMA): {t_us/t_agama:.2f}x")

    R_ag, phi_ag, FR_ag, Fphi_ag = getCylindricalFromCartesian_clockwise(
        pts[:, 0], pts[:, 1], acc_pts_agama[:, 0], acc_pts_agama[:, 1])
    acc_cyl_agama = np.column_stack([FR_ag, Fphi_ag, acc_pts_agama[:, 2]])

    # ---------------------------------------------------------------------------
    # 4. Speed benchmark (repeated evaluation)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. SPEED BENCHMARK (10 evaluations)")
    print("=" * 70)

    N_repeat = 10
    t0 = time.time()
    for _ in range(N_repeat):
        a = get_acc_vec(jnp.array(pts[:, 0]), jnp.array(pts[:, 1]),
                        jnp.array(pts[:, 2]), dict_phi)
        a.block_until_ready()
    t_us_avg = (time.time() - t0) / N_repeat

    t0 = time.time()
    for _ in range(N_repeat):
        pot_agama.force(pts)
    t_agama_avg = (time.time() - t0) / N_repeat

    print(f"  Our average:   {t_us_avg*1000:.2f} ms / {len(pts)} pts")
    print(f"  AGAMA average: {t_agama_avg*1000:.2f} ms / {len(pts)} pts")
    print(f"  Ratio: {t_us_avg/t_agama_avg:.2f}x")

    # ---------------------------------------------------------------------------
    # 5. Accuracy statistics
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. ACCURACY STATISTICS")
    print("=" * 70)

    eps_rel = 1e-8
    denom = np.fabs(acc_cyl_agama)
    rel = np.fabs(acc_cyl_agama - acc_cyl_us) / np.maximum(denom, eps_rel)

    comp_names = ['R', 'phi', 'z']
    for i, name in enumerate(comp_names):
        mask = denom[:, i] > 1e-6
        comp_rel = rel[mask, i]
        print(f"  F_{name}: mean={comp_rel.mean():.4e}, median={np.median(comp_rel):.4e}, "
              f"p95={np.percentile(comp_rel, 95):.4e}, max={comp_rel.max():.4e}")

    # ---------------------------------------------------------------------------
    # 6. Figure: 3x3 acceleration comparison
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. PLOTTING")
    print("=" * 70)

    fig, ax = plt.subplots(3, 3, figsize=(20, 16),
                           gridspec_kw={'hspace': 0.35, 'wspace': 0.35})

    comp_labels = [r'$F_R$', r'$F_\phi$', r'$F_z$']
    diff_vlims = [0.05, 0.05, 0.05]

    for row, (label, vlim) in enumerate(zip(comp_labels, diff_vlims)):
        # Our acceleration
        cb = ax[row, 0].scatter(X_c, Y_c, c=acc_cyl_us[:, row], s=3, marker='s', cmap='viridis')
        plt.colorbar(cb, ax=ax[row, 0])
        ax[row, 0].set_title(f'{label} (CylSpline from particles)')
        ax[row, 0].set_xlabel('X [kpc]')
        ax[row, 0].set_ylabel('Y [kpc]')

        # AGAMA acceleration
        cb = ax[row, 1].scatter(X_c, Y_c, c=acc_cyl_agama[:, row], s=3, marker='s', cmap='viridis')
        plt.colorbar(cb, ax=ax[row, 1])
        ax[row, 1].set_title(f'{label} (AGAMA direct)')
        ax[row, 1].set_xlabel('X [kpc]')
        ax[row, 1].set_ylabel('Y [kpc]')

        # Fractional difference
        res = (acc_cyl_agama[:, row] - acc_cyl_us[:, row]) / np.maximum(np.fabs(acc_cyl_agama[:, row]), 1e-8)
        cb = ax[row, 2].scatter(X_c, Y_c, c=res, s=3, marker='s', cmap='coolwarm',
                                vmin=-vlim, vmax=vlim)
        plt.colorbar(cb, ax=ax[row, 2], label='fractional diff')
        ax[row, 2].set_title(f'{label} fractional diff (AGAMA - ours)/AGAMA')
        ax[row, 2].set_xlabel('X [kpc]')
        ax[row, 2].set_ylabel('Y [kpc]')

    fig.suptitle('Acceleration benchmark: AGAMA-assisted CylSpline vs direct AGAMA\n'
                 f'(z = 1 kpc, {len(pts)} test points, Mmax={metadata.get("Mmax", "?")})',
                 fontsize=14, y=0.98)

    out_fig = 'test_potential_particles.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    print(f"  Saved figure to: {out_fig}")
    plt.show()
