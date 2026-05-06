"""
Compute the CylindricalSpline potential for the t_t0_4 barred galaxy snapshot
using AGAMA-assisted pipeline, and save as a pickle file.

The saved dict_phi is directly compatible with:
  - evaluate_phi(x, y, z, dict_phi)
  - get_acc(x, y, z, dict_phi)
  - evaluate_phi_axisymmetric(x, y, z, dict_phi)
  - get_hessian(x, y, z, dict_phi)
from CylindricalSpline.py or CylindricalSpline_particles.py.

Usage:
  python compute_snapshot_potential.py
"""

import numpy as np
import pickle
import time
import jax.numpy as jnp

import agama
from astropy.constants import G as G_astropy
import astropy.units as u

from constants import KPCGYR_TO_KMS
from CylindricalSpline_particles import get_phi_m_from_agama


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
snapshot_path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/Bar_model_TG21/model/t_t0_7'
output_path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/Bar_model_TG21/dict_phi_stellar_t_t0_7_axi.pkl'

# Grid parameters
NR, NZ = 50, 30
Rmin, Rmax = 0.01, 25.0
Zmin, Zmax = 0.01, 10.0
Mmax = 8
Nphi = 64

# AGAMA internal grid (higher = smoother, but slower build)
agama_gridSizeR = 50
agama_gridSizeZ = 30


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


# ---------------------------------------------------------------------------
# 1. Load snapshot
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. LOADING SNAPSHOT")
print("=" * 70)

# Read in GADGET code units (G=1 in kpc, km/s)
agama.setUnits(length=1, velocity=1, mass=1)
mass_unit = 1 / ((G_astropy * u.Msun).to(u.kpc * (u.km / u.s)**2)).value

t0 = time.time()
w0_data, mass_raw = agama.readSnapshot(snapshot_path)
mass_data = mass_raw * mass_unit  # Convert to Msun
print(f"  Loaded {len(mass_data)} particles in {time.time() - t0:.2f} s")

# # Center on COM (inner R < 5 kpc)
# R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
# mask_inner = R < 5
# for i in range(6):
#     w0_data[:, i] -= np.sum(w0_data[mask_inner, i] * mass_data[mask_inner]) / np.sum(mass_data[mask_inner])

# # Flip x-axis (convention)
# w0_data[:, 0] *= -1
# w0_data[:, 3] *= -1

unique_masses = np.unique(mass_data)
mask_halo = mass_data == unique_masses[-1]
mask_disc = ~mask_halo

# Iterative centering on disc particles (shrinking aperture)
for r_ap in [10.0, 5.0, 3.0, 2.0]:
    R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
    mask_center = mask_disc & (R < r_ap)
    m_c = mass_data[mask_center]
    for col in range(6):
        w0_data[:, col] -= np.sum(w0_data[mask_center, col] * m_c) / np.sum(m_c)

w0_data[:, 0] = -w0_data[:, 0]
w0_data[:, 3] = -w0_data[:, 3]


# Remove DM (heaviest species)
unique_masses, counts = np.unique(mass_data, return_counts=True)
print(f"  Particle species: {len(unique_masses)}")
for m, c in zip(unique_masses, counts):
    print(f"    mass = {m:.4e} Msun, count = {c}")

mask_stellar = mass_data != unique_masses[-1]
w0_data = w0_data[mask_stellar]
mass_data = mass_data[mask_stellar]
print(f"  Stellar particles: {len(mass_data)}")
print(f"  Total stellar mass: {mass_data.sum():.4e} Msun")


# ---------------------------------------------------------------------------
# 2. Center and orient bar
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. CENTERING & ORIENTING BAR")
print("=" * 70)

# Rotate bar to x-axis
R_mid, bar_angles, bar_strength = bar_angle_bar_strength(w0_data[:, 0], w0_data[:, 1])
bar_angle0 = np.mean(bar_angles[R_mid < 4])
w0_data = rotate(w0_data, -bar_angle0)
print(f"  Bar angle (before rotation): {np.degrees(bar_angle0):.1f} deg")

R_mid2, ba2, bs2 = bar_angle_bar_strength(w0_data[:, 0], w0_data[:, 1])
print(f"  Bar angle (after rotation):  {np.degrees(np.mean(ba2[R_mid2 < 4])):.1f} deg")

R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
phi_random = np.random.uniform(0, 2 * np.pi, len(R))
x_random = R * np.cos(phi_random)
y_random = R * np.sin(phi_random)

mask_R6 = R > 6
w0_data[mask_R6, 0] = x_random[mask_R6]
w0_data[mask_R6, 1] = y_random[mask_R6]

positions = w0_data[:, :3]
masses = mass_data


# ---------------------------------------------------------------------------
# 3. Compute potential via AGAMA-assisted pipeline
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. COMPUTING POTENTIAL (AGAMA-assisted)")
print("=" * 70)
print(f"  Grid: NR={NR}, NZ={NZ}, Rmax={Rmax}, Zmax={Zmax}, Mmax={Mmax}")
print(f"  AGAMA grid: NR={agama_gridSizeR}, NZ={agama_gridSizeZ}")

dict_phi = get_phi_m_from_agama(
    positions, masses,
    NR=NR, NZ=NZ,
    Rmin=Rmin, Rmax=Rmax,
    Zmin=Zmin, Zmax=Zmax,
    Mmax=Mmax, Nphi=Nphi,
    agama_gridSizeR=agama_gridSizeR,
    agama_gridSizeZ=agama_gridSizeZ,
    verbose=True,
)


# ---------------------------------------------------------------------------
# 4. Quick sanity check
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. SANITY CHECK")
print("=" * 70)

from CylindricalSpline_particles import evaluate_phi, get_acc
import jax

test_R = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
zeros = np.zeros_like(test_R)
x_t, y_t, z_t = jnp.array(test_R), jnp.zeros(len(test_R)), jnp.zeros(len(test_R))

phi_vals = np.array(evaluate_phi(x_t, y_t, z_t, dict_phi))
conv = KPCGYR_TO_KMS**2

print(f"  Potential along x-axis (bar major axis):")
print(f"  {'R (kpc)':>8s} {'Phi (km/s)^2':>14s}")
for i in range(len(test_R)):
    print(f"  {test_R[i]:8.1f} {phi_vals[i]*conv:14.1f}")

# Test acceleration
get_acc_single = jax.jit(get_acc)
acc = get_acc_single(jnp.array(3.0), jnp.array(0.0), jnp.array(0.5), dict_phi)
print(f"\n  Acceleration at (3, 0, 0.5) kpc: ({float(acc[0])*conv:.1f}, {float(acc[1])*conv:.1f}, {float(acc[2])*conv:.1f}) (km/s)^2/kpc")


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. SAVING")
print("=" * 70)

# Convert JAX arrays to numpy for pickling
dict_phi_save = {k: np.array(v) for k, v in dict_phi.items()}

# Also save metadata
dict_phi_save['_metadata'] = {
    'snapshot': snapshot_path,
    'total_stellar_mass_Msun': float(masses.sum()),
    'n_stellar_particles': len(masses),
    'bar_angle_deg': float(np.degrees(bar_angle0)),
    'NR': NR, 'NZ': NZ,
    'Rmin': Rmin, 'Rmax': Rmax,
    'Zmin': Zmin, 'Zmax': Zmax,
    'Mmax': Mmax, 'Nphi': Nphi,
    'agama_gridSizeR': agama_gridSizeR,
    'agama_gridSizeZ': agama_gridSizeZ,
    'units': 'potential in (kpc/Gyr)^2, positions in kpc, masses in Msun',
    'usage': 'from CylindricalSpline import evaluate_phi, get_acc; phi = evaluate_phi(x, y, z, dict_phi)',
}

with open(output_path, 'wb') as f:
    pickle.dump(dict_phi_save, f)

import os
fsize = os.path.getsize(output_path) / 1024
print(f"  Saved to: {output_path}")
print(f"  File size: {fsize:.1f} KB")

print("\n  To load and use:")
print("    import pickle, jax.numpy as jnp")
print("    with open(path, 'rb') as f:")
print("        d = pickle.load(f)")
print("    dict_phi = {k: jnp.array(v) for k, v in d.items() if k != '_metadata'}")
print("    from CylindricalSpline import evaluate_phi, get_acc")
print("    phi = evaluate_phi(x, y, z, dict_phi)")
print("    acc = get_acc(x, y, z, dict_phi)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
