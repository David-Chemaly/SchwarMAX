"""
Benchmark: CylindricalSpline_particles vs AGAMA on the t_t0_4 barred galaxy snapshot.

Steps:
  1. Load snapshot, remove DM (heaviest species), center on COM, orient bar to x-axis
  2. Compute stellar potential via AGAMA CylSpline (from particles)
  3. Compute stellar potential via our CylindricalSpline_particles
  4. Compare potential & acceleration on a grid (cylindrical R, phi, z components)
"""

import numpy as np
import time

import agama
from astropy.constants import G as G_astropy
import astropy.units as u

from constants import KPCGYR_TO_KMS

# ---------------------------------------------------------------------------
# Unit setup
# ---------------------------------------------------------------------------
mass_unit = 1 / ((G_astropy * u.Msun).to(u.kpc * (u.km / u.s)**2)).value
agama.setUnits(length=1, velocity=1, mass=1)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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
    """Convert Cartesian acceleration to cylindrical (aR, aphi)."""
    R = np.sqrt(x**2 + y**2)
    R = np.maximum(R, 1e-12)
    aR = (x * ax + y * ay) / R
    aphi = (-y * ax + x * ay) / R
    return aR, aphi


# ---------------------------------------------------------------------------
# 1. Load snapshot, center, orient bar
# ---------------------------------------------------------------------------

snapshot_path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/Bar_model_TG21/model/t_t0_4'

print("=" * 70)
print("1. LOADING SNAPSHOT")
print("=" * 70)

t0 = time.time()
w0_data, mass_raw = agama.readSnapshot(snapshot_path)
mass_data = mass_raw * mass_unit
print(f"  Loaded {len(mass_data)} particles in {time.time() - t0:.2f} s")

unique_masses, counts = np.unique(mass_data, return_counts=True)
print(f"  Particle species: {len(unique_masses)}")
for m, c in zip(unique_masses, counts):
    print(f"    mass = {m:.4e} Msun, count = {c}")

mask_stellar = mass_data != unique_masses[-1]
w0_data = w0_data[mask_stellar]
mass_data = mass_data[mask_stellar]
print(f"  Stellar particles: {len(mass_data)}")
print(f"  Total stellar mass: {mass_data.sum():.4e} Msun")

R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
mask_inner = R < 5
for i in range(6):
    w0_data[:, i] -= np.sum(w0_data[mask_inner, i] * mass_data[mask_inner]) / np.sum(mass_data[mask_inner])

w0_data[:, 0] *= -1
w0_data[:, 3] *= -1

R_mid, bar_angles, bar_strength = bar_angle_bar_strength(w0_data[:, 0], w0_data[:, 1])
bar_angle0 = np.mean(bar_angles[R_mid < 4])
w0_data = rotate(w0_data, -bar_angle0)
print(f"  Bar angle (before rotation): {np.degrees(bar_angle0):.1f} deg")

R_mid2, ba2, bs2 = bar_angle_bar_strength(w0_data[:, 0], w0_data[:, 1])
print(f"  Bar angle (after rotation):  {np.degrees(np.mean(ba2[R_mid2 < 4])):.1f} deg")

positions = w0_data[:, :3]
masses = mass_data

print(f"  Position range: R=[{np.sqrt((positions**2).sum(1)).min():.3f}, {np.sqrt((positions**2).sum(1)).max():.1f}] kpc")
print(f"  |z| range: [{np.abs(positions[:,2]).min():.4f}, {np.abs(positions[:,2]).max():.1f}] kpc")


# ---------------------------------------------------------------------------
# 2. AGAMA CylSpline from stellar particles
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("2. AGAMA CylSpline from STELLAR PARTICLES")
print("=" * 70)

agama.setUnits(mass=1, length=1, velocity=KPCGYR_TO_KMS)

NR, NZ = 40, 25
Mmax = 8

t0 = time.time()
pot_agama = agama.Potential(
    type='CylSpline',
    particles=(positions, masses),
    symmetry='Triaxial',
    mmax=Mmax,
    gridSizeR=NR,
    gridSizeZ=NZ,
)
t_agama = time.time() - t0
print(f"  AGAMA CylSpline (Mmax={Mmax}): {t_agama:.3f} s")


# ---------------------------------------------------------------------------
# 3. Our CylindricalSpline from stellar particles
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("3. OUR CylindricalSpline from STELLAR PARTICLES")
print("=" * 70)

from CylindricalSpline_particles import get_phi_m_from_particles, evaluate_phi, get_acc

R_all = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
Rmax = np.percentile(R_all, 99.9)
Zmax = np.percentile(np.abs(positions[:, 2]), 99.9)
# Ensure grid covers test region (R<15, |z|<5)
Rmax = max(Rmax, 20.0)
Zmax = max(Zmax, 8.0)
print(f"  Grid: Rmax={Rmax:.1f} kpc, Zmax={Zmax:.1f} kpc")

t0 = time.time()
dict_phi = get_phi_m_from_particles(
    positions, masses,
    NR=NR, NZ=NZ,
    Rmin=0.01, Rmax=Rmax,
    Zmin=0.01, Zmax=Zmax,
    Mmax=Mmax,
    N_int=10000,
    smooth_pixels=0.7,
    verbose=True,
)
t_ours_build = time.time() - t0
print(f"  Our CylSpline (Mmax={Mmax}): {t_ours_build:.3f} s")


# ---------------------------------------------------------------------------
# 4. Compare potential & acceleration on multiple grids
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp

get_acc_vec = jax.jit(jax.vmap(get_acc, in_axes=(0, 0, 0, None)))

conv = KPCGYR_TO_KMS**2  # (kpc/Gyr)^2 → (km/s)^2


def run_comparison(X_flat, Y_flat, Z_flat, label):
    """Run full potential + acceleration comparison on a set of test points."""
    pts = np.column_stack([X_flat, Y_flat, Z_flat])
    x_j, y_j, z_j = jnp.array(X_flat), jnp.array(Y_flat), jnp.array(Z_flat)
    N = len(pts)

    # --- Potential ---
    phi_agama = pot_agama.potential(pts)
    phi_ours = np.array(evaluate_phi(x_j, y_j, z_j, dict_phi))

    mask_valid = np.abs(phi_agama) > 0.1 * np.abs(phi_agama).max()
    if mask_valid.sum() > 0:
        rel_phi = np.abs((phi_ours[mask_valid] - phi_agama[mask_valid]) / phi_agama[mask_valid])
        print(f"\n  Potential ({label}, {N} pts):")
        print(f"    median={np.median(rel_phi):.2%}, mean={rel_phi.mean():.2%}, "
              f"p95={np.percentile(rel_phi, 95):.2%}, max={rel_phi.max():.2%}")

    # --- Force: AGAMA ---
    force_agama = pot_agama.force(pts)

    # --- Force: Ours (warm-up JIT) ---
    _ = get_acc_vec(x_j * 0.99, y_j * 0.99, z_j + 0.01, dict_phi)
    _.block_until_ready()

    # Timed runs (multiple iterations for stable timing)
    N_iter = 20

    t0 = time.time()
    for _ in range(N_iter):
        fa = pot_agama.force(pts)
    t_agama_eval = (time.time() - t0) / N_iter

    t0 = time.time()
    for _ in range(N_iter):
        ao = get_acc_vec(x_j, y_j, z_j, dict_phi)
        ao.block_until_ready()
    t_ours_eval = (time.time() - t0) / N_iter

    acc_ours = np.array(get_acc_vec(x_j, y_j, z_j, dict_phi))

    # --- Convert to cylindrical components ---
    x, y = pts[:, 0], pts[:, 1]

    aR_agama, aphi_agama = to_cyl_acc(x, y, force_agama[:, 0], force_agama[:, 1])
    az_agama = force_agama[:, 2]

    aR_ours, aphi_ours = to_cyl_acc(x, y, acc_ours[:, 0], acc_ours[:, 1])
    az_ours = acc_ours[:, 2]

    # --- Error stats per component ---
    # Use total acceleration magnitude as normalization for all components
    a_tot_agama = np.sqrt(aR_agama**2 + aphi_agama**2 + az_agama**2)
    a_tot_ours = np.sqrt(aR_ours**2 + aphi_ours**2 + az_ours**2)
    mask_good = a_tot_agama > 0.01 * a_tot_agama.max()

    print(f"\n  Fractional acceleration error ({label}, {mask_good.sum()}/{N} valid pts):")
    print(f"  {'comp':>8s} {'median':>10s} {'mean':>10s} {'p5':>10s} {'p95':>10s} {'max':>10s}")

    comps = [('a_R', aR_ours, aR_agama),
             ('a_phi', aphi_ours, aphi_agama),
             ('a_z', az_ours, az_agama),
             ('|a_tot|', a_tot_ours, a_tot_agama)]
    for name, ours, ref in comps:
        if not mask_good.any():
            print(f"  {name:>8s}   (no valid pts)")
            continue
        # Normalize by total acceleration magnitude (avoids blow-up for weak components)
        abs_err = np.abs(ours[mask_good] - ref[mask_good])
        rel = abs_err / a_tot_agama[mask_good]
        print(f"  {name:>8s} {np.median(rel):10.2%} {rel.mean():10.2%} "
              f"{np.percentile(rel, 5):10.2%} {np.percentile(rel, 95):10.2%} {rel.max():10.2%}")

    # Also show component-relative error for a_R (the dominant component)
    mask_aR = np.abs(aR_agama) > 0.05 * np.abs(aR_agama).max()
    if mask_aR.sum() > 0:
        rel_aR = np.abs((aR_ours[mask_aR] - aR_agama[mask_aR]) / aR_agama[mask_aR])
        print(f"\n  a_R self-relative error (|a_R| > 5% max): "
              f"median={np.median(rel_aR):.2%}, mean={rel_aR.mean():.2%}, p95={np.percentile(rel_aR, 95):.2%}")

    print(f"\n  Eval speed ({N} pts, avg over {N_iter} calls):")
    print(f"    AGAMA: {t_agama_eval*1e3:.3f} ms")
    print(f"    Ours:  {t_ours_eval*1e3:.3f} ms")
    print(f"    Ratio: {t_ours_eval/t_agama_eval:.2f}x")

    return phi_agama, phi_ours, force_agama, acc_ours


# ===== Grid 1: z=0 plane (XY grid) =====
print("\n" + "=" * 70)
print("4a. COMPARISON: z=0 plane (80x80 grid)")
print("=" * 70)

x_grid = np.linspace(-10, 10, 80)
y_grid = np.linspace(-10, 10, 80)
X, Y = np.meshgrid(x_grid, y_grid)
run_comparison(X.ravel(), Y.ravel(), np.zeros(X.size), "z=0 plane")

# ===== Grid 2: z=1 kpc plane =====
print("\n" + "=" * 70)
print("4b. COMPARISON: z=1 kpc plane (80x80 grid)")
print("=" * 70)

run_comparison(X.ravel(), Y.ravel(), np.ones(X.size), "z=1 kpc plane")

# ===== Grid 3: z=0.3 kpc (typical disk midplane) =====
print("\n" + "=" * 70)
print("4c. COMPARISON: z=0.3 kpc plane (80x80 grid)")
print("=" * 70)

run_comparison(X.ravel(), Y.ravel(), np.full(X.size, 0.3), "z=0.3 kpc plane")

# ===== Grid 4: R-z plane at phi=0 (x-z plane, along bar) =====
print("\n" + "=" * 70)
print("4d. COMPARISON: x-z plane (along bar, 60x40 grid)")
print("=" * 70)

x_rz = np.linspace(0.3, 15, 60)
z_rz = np.linspace(-5, 5, 40)
X_rz, Z_rz = np.meshgrid(x_rz, z_rz)
run_comparison(X_rz.ravel(), np.zeros(X_rz.size), Z_rz.ravel(), "x-z plane (bar)")

# ===== Grid 5: Random 3D points (realistic orbit positions) =====
print("\n" + "=" * 70)
print("4e. COMPARISON: random 3D points (10000 pts, R<15, |z|<5)")
print("=" * 70)

np.random.seed(42)
R_rand = np.random.uniform(0.3, 15, 10000)
phi_rand = np.random.uniform(0, 2*np.pi, 10000)
z_rand = np.random.uniform(-5, 5, 10000)
x_rand = R_rand * np.cos(phi_rand)
y_rand = R_rand * np.sin(phi_rand)
run_comparison(x_rand, y_rand, z_rand, "random 3D")


# ===== Scaling test: evaluation speed vs number of points =====
print("\n" + "=" * 70)
print("5. EVALUATION SPEED SCALING")
print("=" * 70)

for N_test in [100, 1000, 10000, 100000]:
    np.random.seed(123)
    R_t = np.random.uniform(0.5, 12, N_test)
    phi_t = np.random.uniform(0, 2*np.pi, N_test)
    z_t = np.random.uniform(-3, 3, N_test)
    x_t = R_t * np.cos(phi_t)
    y_t = R_t * np.sin(phi_t)
    pts_t = np.column_stack([x_t, y_t, z_t])
    xj, yj, zj = jnp.array(x_t), jnp.array(y_t), jnp.array(z_t)

    # Warm up
    _ = get_acc_vec(xj, yj, zj, dict_phi)
    _.block_until_ready()

    N_iter = max(3, 50000 // N_test)

    t0 = time.time()
    for _ in range(N_iter):
        pot_agama.force(pts_t)
    t_ag = (time.time() - t0) / N_iter

    t0 = time.time()
    for _ in range(N_iter):
        ao = get_acc_vec(xj, yj, zj, dict_phi)
        ao.block_until_ready()
    t_ou = (time.time() - t0) / N_iter

    print(f"  N={N_test:>6d}: AGAMA={t_ag*1e3:8.3f} ms, Ours={t_ou*1e3:8.3f} ms, ratio={t_ou/t_ag:.2f}x")


# ===== Radial profiles =====
print("\n" + "=" * 70)
print("6. POTENTIAL RADIAL PROFILES")
print("=" * 70)

test_R = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
zeros = np.zeros_like(test_R)

print(f"\n  Along x-axis (bar major axis):")
print(f"  {'R':>5s} {'AGAMA (km/s)^2':>16s} {'Ours (km/s)^2':>16s} {'err':>8s}")
test_pts = np.column_stack([test_R, zeros, zeros])
phi_ag = pot_agama.potential(test_pts)
phi_ou = np.array(evaluate_phi(jnp.array(test_R), jnp.zeros(len(test_R)),
                                jnp.zeros(len(test_R)), dict_phi))
for i in range(len(test_R)):
    e = abs((phi_ou[i] - phi_ag[i]) / (phi_ag[i] + 1e-30))
    print(f"  {test_R[i]:5.1f} {phi_ag[i]*conv:16.1f} {phi_ou[i]*conv:16.1f} {e:7.2%}")

print(f"\n  Along y-axis (bar minor axis):")
print(f"  {'R':>5s} {'AGAMA (km/s)^2':>16s} {'Ours (km/s)^2':>16s} {'err':>8s}")
test_pts_y = np.column_stack([zeros, test_R, zeros])
phi_ag_y = pot_agama.potential(test_pts_y)
phi_ou_y = np.array(evaluate_phi(jnp.zeros(len(test_R)), jnp.array(test_R),
                                  jnp.zeros(len(test_R)), dict_phi))
for i in range(len(test_R)):
    e = abs((phi_ou_y[i] - phi_ag_y[i]) / (phi_ag_y[i] + 1e-30))
    print(f"  {test_R[i]:5.1f} {phi_ag_y[i]*conv:16.1f} {phi_ou_y[i]*conv:16.1f} {e:7.2%}")

print(f"\n  Along z-axis (x=y=0, vertical):")
print(f"  {'z':>5s} {'AGAMA (km/s)^2':>16s} {'Ours (km/s)^2':>16s} {'err':>8s}")
test_z = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
test_pts_z = np.column_stack([zeros[:len(test_z)], zeros[:len(test_z)], test_z])
phi_ag_z = pot_agama.potential(test_pts_z)
phi_ou_z = np.array(evaluate_phi(jnp.zeros(len(test_z)), jnp.zeros(len(test_z)),
                                  jnp.array(test_z), dict_phi))
for i in range(len(test_z)):
    e = abs((phi_ou_z[i] - phi_ag_z[i]) / (phi_ag_z[i] + 1e-30))
    print(f"  {test_z[i]:5.1f} {phi_ag_z[i]*conv:16.1f} {phi_ou_z[i]*conv:16.1f} {e:7.2%}")


# ===== Summary =====
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  AGAMA build time: {t_agama:.1f} s")
print(f"  Our build time:   {t_ours_build:.1f} s  ({t_agama/t_ours_build:.0f}x faster)")
print(f"  Mmax = {Mmax}")
print(f"  Grid: NR={NR}, NZ={NZ}, Rmax={Rmax:.1f} kpc, Zmax={Zmax:.1f} kpc")
print("=" * 70)
