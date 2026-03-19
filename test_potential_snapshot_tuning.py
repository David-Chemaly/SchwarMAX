"""
Tuning test: find the best grid/smoothing parameters for CylindricalSpline_particles
to minimize acceleration error vs AGAMA on the t_t0_4 snapshot.

Also test: how much does AGAMA's own potential change with grid resolution?
(To establish the "noise floor" from particle discreteness.)
"""

import numpy as np
import time
import agama
from astropy.constants import G as G_astropy
import astropy.units as u
from constants import KPCGYR_TO_KMS

mass_unit = 1 / ((G_astropy * u.Msun).to(u.kpc * (u.km / u.s)**2)).value
agama.setUnits(length=1, velocity=1, mass=1)


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


def to_cyl_acc(x, y, ax, ay):
    R = np.maximum(np.sqrt(x**2 + y**2), 1e-12)
    aR = (x * ax + y * ay) / R
    aphi = (-y * ax + x * ay) / R
    return aR, aphi


# ---- Load snapshot ----
snapshot_path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/Bar_model_TG21/model/t_t0_4'
w0_data, mass_raw = agama.readSnapshot(snapshot_path)
mass_data = mass_raw * mass_unit

unique_masses = np.unique(mass_data)
mask_stellar = mass_data != unique_masses[-1]
w0_data = w0_data[mask_stellar]
mass_data = mass_data[mask_stellar]

R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
mask_inner = R < 5
for i in range(6):
    w0_data[:, i] -= np.sum(w0_data[mask_inner, i] * mass_data[mask_inner]) / np.sum(mass_data[mask_inner])
w0_data[:, 0] *= -1
w0_data[:, 3] *= -1

R_mid, bar_angles, _ = bar_angle_bar_strength(w0_data[:, 0], w0_data[:, 1])
bar_angle0 = np.mean(bar_angles[R_mid < 4])
w0_data = rotate(w0_data, -bar_angle0)

positions = w0_data[:, :3]
masses = mass_data
print(f"Stellar particles: {len(masses)}, total mass: {masses.sum():.3e} Msun")

# Switch to (Msun, kpc, kpc/Gyr)
agama.setUnits(mass=1, length=1, velocity=KPCGYR_TO_KMS)

# ---- Test points: random 3D (the hardest case) ----
np.random.seed(42)
N_test = 10000
R_rand = np.random.uniform(0.5, 12, N_test)
phi_rand = np.random.uniform(0, 2*np.pi, N_test)
z_rand = np.random.uniform(-3, 3, N_test)
x_test = R_rand * np.cos(phi_rand)
y_test = R_rand * np.sin(phi_rand)
pts_test = np.column_stack([x_test, y_test, z_rand])


def eval_error_vs_ref(force_test, force_ref, x, y, label=""):
    """Compute acceleration errors normalized by |a_tot| of reference."""
    aR_t, aphi_t = to_cyl_acc(x, y, force_test[:, 0], force_test[:, 1])
    aR_r, aphi_r = to_cyl_acc(x, y, force_ref[:, 0], force_ref[:, 1])
    az_t, az_r = force_test[:, 2], force_ref[:, 2]
    a_tot_r = np.sqrt(aR_r**2 + aphi_r**2 + az_r**2)
    mask = a_tot_r > 0.01 * a_tot_r.max()

    results = {}
    for name, t, r in [('a_R', aR_t, aR_r), ('a_phi', aphi_t, aphi_r),
                        ('a_z', az_t, az_r)]:
        abs_err = np.abs(t[mask] - r[mask])
        rel = abs_err / a_tot_r[mask]
        results[name] = (np.median(rel), np.percentile(rel, 95))

    a_tot_t = np.sqrt(aR_t**2 + aphi_t**2 + az_t**2)
    rel_tot = np.abs(a_tot_t[mask] - a_tot_r[mask]) / a_tot_r[mask]
    results['|a|'] = (np.median(rel_tot), np.percentile(rel_tot, 95))
    return results


# =====================================================================
# Part 1: AGAMA self-consistency (grid resolution study)
# =====================================================================
print("\n" + "=" * 70)
print("PART 1: AGAMA SELF-CONSISTENCY (how stable is AGAMA with grid size?)")
print("=" * 70)
print("  Testing how much AGAMA's potential changes with different NR, NZ.")
print("  This establishes the 'particle noise floor'.\n")

Mmax = 8

# High-res reference
t0 = time.time()
pot_ref = agama.Potential(
    type='CylSpline', particles=(positions, masses),
    symmetry='Triaxial', mmax=Mmax, gridSizeR=80, gridSizeZ=50,
)
print(f"  AGAMA reference (NR=80,NZ=50): {time.time()-t0:.1f}s")
force_ref = pot_ref.force(pts_test)

print(f"\n  {'NR':>4s} {'NZ':>4s} {'time':>8s}   {'a_R med':>8s} {'a_R p95':>8s} {'a_phi med':>9s} {'a_phi p95':>9s} {'a_z med':>8s} {'a_z p95':>8s} {'|a| med':>8s} {'|a| p95':>8s}")
for NR, NZ in [(20, 15), (30, 20), (40, 25), (50, 30), (60, 40)]:
    t0 = time.time()
    pot = agama.Potential(
        type='CylSpline', particles=(positions, masses),
        symmetry='Triaxial', mmax=Mmax, gridSizeR=NR, gridSizeZ=NZ,
    )
    dt = time.time() - t0
    force = pot.force(pts_test)
    res = eval_error_vs_ref(force, force_ref, x_test, y_test)
    print(f"  {NR:4d} {NZ:4d} {dt:7.1f}s   "
          f"{res['a_R'][0]:7.2%} {res['a_R'][1]:7.2%}  "
          f"{res['a_phi'][0]:8.2%} {res['a_phi'][1]:8.2%}  "
          f"{res['a_z'][0]:7.2%} {res['a_z'][1]:7.2%}  "
          f"{res['|a|'][0]:7.2%} {res['|a|'][1]:7.2%}")


# =====================================================================
# Part 2: Our method — parameter sweep
# =====================================================================
print("\n" + "=" * 70)
print("PART 2: OUR METHOD — PARAMETER SWEEP")
print("=" * 70)
print("  Comparing against AGAMA reference (NR=80, NZ=50, Mmax=8).\n")

import jax
import jax.numpy as jnp
from CylindricalSpline_particles import get_phi_m_from_particles, get_acc

get_acc_vec = jax.jit(jax.vmap(get_acc, in_axes=(0, 0, 0, None)))
x_j, y_j, z_j = jnp.array(x_test), jnp.array(y_test), jnp.array(z_rand)

R_all = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)

print(f"  {'NR':>4s} {'NZ':>4s} {'Rmax':>6s} {'Zmax':>6s} {'smooth':>7s} {'time':>7s}   "
      f"{'a_R med':>8s} {'a_R p95':>8s} {'a_phi med':>9s} {'a_phi p95':>9s} {'a_z med':>8s} {'a_z p95':>8s} {'|a| med':>8s} {'|a| p95':>8s}")

configs = [
    # (NR, NZ, Rmax, Zmax, smooth_pixels)
    (40, 25, 23, 8, 0.7),    # baseline
    (40, 25, 23, 8, 1.0),    # more smoothing
    (40, 25, 23, 8, 1.5),    # even more smoothing
    (40, 25, 23, 8, 2.0),    # heavy smoothing
    (60, 35, 23, 8, 0.7),    # higher res
    (60, 35, 23, 8, 1.0),    # higher res + smoothing
    (60, 35, 23, 8, 1.5),    # higher res + more smoothing
    (80, 50, 23, 8, 0.7),    # very high res
    (80, 50, 23, 8, 1.0),    # very high res + smoothing
    (80, 50, 23, 8, 1.5),    # very high res + more smoothing
    (80, 50, 30, 10, 1.0),   # very high res + bigger grid
]

for NR, NZ, Rmax, Zmax, smooth in configs:
    t0 = time.time()
    dict_phi = get_phi_m_from_particles(
        positions, masses,
        NR=NR, NZ=NZ,
        Rmin=0.01, Rmax=Rmax, Zmin=0.01, Zmax=Zmax,
        Mmax=Mmax, N_int=10000, smooth_pixels=smooth,
    )
    dt = time.time() - t0

    # Warm up + evaluate
    _ = get_acc_vec(x_j * 0.99, y_j * 0.99, z_j + 0.01, dict_phi)
    _.block_until_ready()
    acc = np.array(get_acc_vec(x_j, y_j, z_j, dict_phi))

    res = eval_error_vs_ref(acc, force_ref, x_test, y_test)
    print(f"  {NR:4d} {NZ:4d} {Rmax:5.0f} {Zmax:5.0f} {smooth:6.1f} {dt:6.1f}s   "
          f"{res['a_R'][0]:7.2%} {res['a_R'][1]:7.2%}  "
          f"{res['a_phi'][0]:8.2%} {res['a_phi'][1]:8.2%}  "
          f"{res['a_z'][0]:7.2%} {res['a_z'][1]:7.2%}  "
          f"{res['|a|'][0]:7.2%} {res['|a|'][1]:7.2%}")


# =====================================================================
# Part 3: AGAMA (NR=40, NZ=25) vs AGAMA reference — baseline comparison
# =====================================================================
print("\n" + "=" * 70)
print("PART 3: AGAMA (NR=40, NZ=25) vs AGAMA reference")
print("=" * 70)
print("  This is the error of AGAMA at the SAME grid resolution we use.\n")

pot_agama_40 = agama.Potential(
    type='CylSpline', particles=(positions, masses),
    symmetry='Triaxial', mmax=Mmax, gridSizeR=40, gridSizeZ=25,
)
force_agama_40 = pot_agama_40.force(pts_test)
res = eval_error_vs_ref(force_agama_40, force_ref, x_test, y_test)
print(f"  a_R:   median={res['a_R'][0]:.2%}, p95={res['a_R'][1]:.2%}")
print(f"  a_phi: median={res['a_phi'][0]:.2%}, p95={res['a_phi'][1]:.2%}")
print(f"  a_z:   median={res['a_z'][0]:.2%}, p95={res['a_z'][1]:.2%}")
print(f"  |a|:   median={res['|a|'][0]:.2%}, p95={res['|a|'][1]:.2%}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
