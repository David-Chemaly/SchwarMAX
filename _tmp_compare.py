"""
Compare Φ and ρ between:
  - AGAMA's own CylSpline (evaluated via AGAMA routines)   [pot_agama]
  - Our reconstructed JAX dict (used in model.py)          [dict_phi]
  - Ground truth analytic density                          [ρ_truth]

Mock: exponential disk with sech^2/Gaussian z-profile — analytic ground truth.
Slices: z = 0, 1, 2, 3 kpc. Lines vs R.
"""
import numpy as np
import jax.numpy as jnp
import agama
import matplotlib.pyplot as plt

from constants import KPCGYR_TO_KMS, G
from CylindricalSpline_particles import get_phi_m_from_agama
from CylindricalSpline import evaluate_phi


# ---------- Ground truth mock (isothermal disk, sech^2 vertical) ----------
Rd, hz = 3.0, 0.8   # thicker disk: well-sampled up to z~3 kpc
Sigma0 = 5e8        # Msun/kpc^2 central surface density
Mtot = 2 * np.pi * Sigma0 * Rd**2

def rho_truth(R, z):
    """Exponential R × sech^2(z/(2hz)) vertical profile. Msun/kpc^3.
    Normalized so int rho dz = Sigma(R) = Sigma0 exp(-R/Rd)."""
    Sigma_R = Sigma0 * np.exp(-R / Rd)
    vert = 1.0 / (4.0 * hz * np.cosh(z / (2 * hz))**2)
    return Sigma_R * vert


# Sample particles
np.random.seed(42)
N = 1_000_000
R_s = np.random.gamma(2.0, Rd, N)  # exponential disk: Sigma(R) = Sigma0 exp(-R/Rd)
phi_s = np.random.uniform(0, 2 * np.pi, N)
# Inverse CDF for sech^2(z/2h): CDF = (tanh(z/2h)+1)/2
u = np.random.uniform(-0.9999, 0.9999, N)
z_s = 2 * hz * np.arctanh(u)
positions = np.column_stack([R_s * np.cos(phi_s), R_s * np.sin(phi_s), z_s])
masses = np.full(N, Mtot / N)

# ---------- AGAMA CylSpline (reference) ----------
agama.setUnits(mass=1, length=1, velocity=KPCGYR_TO_KMS)
pot_agama = agama.Potential(
    type='CylSpline', particles=(positions, masses),
    symmetry='a', mmax=0,
    gridSizeR=50, gridSizeZ=30,
)

# ---------- Our JAX dict (same as model.py uses) ----------
dict_phi = get_phi_m_from_agama(
    positions, masses,
    NR=50, NZ=30, Rmin=0.01, Rmax=25.0, Zmin=0.01, Zmax=10.0,
    Mmax=0, Nphi=64,
    agama_gridSizeR=50, agama_gridSizeZ=30,
    verbose=False,
)


# ---------- density_from_potential (exactly as in model.py:69) ----------
def density_from_potential(x, y, z, dp, d=0.01):
    phi_c = evaluate_phi(x, y, z, dp)
    lap = (
        (evaluate_phi(x + d, y, z, dp) - 2 * phi_c + evaluate_phi(x - d, y, z, dp)) / d**2
      + (evaluate_phi(x, y + d, z, dp) - 2 * phi_c + evaluate_phi(x, y - d, z, dp)) / d**2
      + (evaluate_phi(x, y, z + d, dp) - 2 * phi_c + evaluate_phi(x, y, z - d, dp)) / d**2
    )
    return lap / (4 * np.pi * G)


# ---------- Evaluate on a dense grid ----------
R_arr = np.geomspace(0.1, 15, 60)
z_slices = [0.0, 1.0, 2.5, 4.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

# Panel A: fractional error of potential
ax = axes[0, 0]
for z0 in z_slices:
    pts = np.column_stack([R_arr, np.zeros_like(R_arr), np.full_like(R_arr, z0)])
    phi_ag = pot_agama.potential(pts)
    phi_ours = np.array(evaluate_phi(jnp.array(R_arr), jnp.zeros(len(R_arr)),
                                     jnp.full(len(R_arr), z0), dict_phi))
    frac = (phi_ours - phi_ag) / phi_ag
    ax.plot(R_arr, frac * 100, label=f'z = {z0} kpc')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel(r'$(\Phi_\mathrm{ours} - \Phi_\mathrm{AGAMA})/\Phi_\mathrm{AGAMA}$  [%]')
ax.set_title(r'Potential: ours vs AGAMA')
ax.set_xscale('log')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: AGAMA density vs truth
ax = axes[0, 1]
for z0 in z_slices:
    pts = np.column_stack([R_arr, np.zeros_like(R_arr), np.full_like(R_arr, z0)])
    rho_ag = pot_agama.density(pts)
    rho_tr = rho_truth(R_arr, z0)
    frac = (rho_ag - rho_tr) / rho_tr
    ax.plot(R_arr, frac * 100, label=f'z = {z0} kpc')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel(r'$(\rho_\mathrm{AGAMA} - \rho_\mathrm{truth})/\rho_\mathrm{truth}$  [%]')
ax.set_title(r'Density: AGAMA vs ground truth')
ax.set_xscale('log')
ax.set_ylim(-50, 50)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel C: our density_from_potential vs truth
ax = axes[1, 0]
for z0 in z_slices:
    x_j = jnp.array(R_arr); y_j = jnp.zeros(len(R_arr)); z_j = jnp.full(len(R_arr), z0)
    rho_ours = np.array([float(density_from_potential(jnp.array(r), jnp.array(0.0), jnp.array(z0),
                                                      dict_phi)) for r in R_arr])
    rho_tr = rho_truth(R_arr, z0)
    frac = (rho_ours - rho_tr) / rho_tr
    ax.plot(R_arr, frac * 100, label=f'z = {z0} kpc')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel(r'$(\rho_\mathrm{ours} - \rho_\mathrm{truth})/\rho_\mathrm{truth}$  [%]')
ax.set_title(r'Density: density_from_potential (model.py) vs ground truth')
ax.set_xlabel('R [kpc]')
ax.set_xscale('log')
ax.set_ylim(-150, 150)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel D: our density vs AGAMA density
ax = axes[1, 1]
for z0 in z_slices:
    pts = np.column_stack([R_arr, np.zeros_like(R_arr), np.full_like(R_arr, z0)])
    rho_ag = pot_agama.density(pts)
    rho_ours = np.array([float(density_from_potential(jnp.array(r), jnp.array(0.0), jnp.array(z0),
                                                      dict_phi)) for r in R_arr])
    frac = (rho_ours - rho_ag) / rho_ag
    ax.plot(R_arr, frac * 100, label=f'z = {z0} kpc')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel(r'$(\rho_\mathrm{ours} - \rho_\mathrm{AGAMA})/\rho_\mathrm{AGAMA}$  [%]')
ax.set_title(r'Density: density_from_potential vs AGAMA')
ax.set_xlabel('R [kpc]')
ax.set_xscale('log')
ax.set_ylim(-150, 150)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle(f'Isothermal disk mock (Rd={Rd} kpc, hz={hz} kpc, sech²; N={N:,}); Mmax=0, grid 50x30',
             fontsize=11)
plt.tight_layout()
out = '_tmp_compare.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'saved {out}')

# Print numbers for record
print("\nFractional errors summary (median |err| %):")
for z0 in z_slices:
    pts = np.column_stack([R_arr, np.zeros_like(R_arr), np.full_like(R_arr, z0)])
    phi_ag = pot_agama.potential(pts)
    phi_ours = np.array(evaluate_phi(jnp.array(R_arr), jnp.zeros(len(R_arr)),
                                     jnp.full(len(R_arr), z0), dict_phi))
    rho_ag = pot_agama.density(pts)
    rho_tr = rho_truth(R_arr, z0)
    rho_ours = np.array([float(density_from_potential(jnp.array(r), jnp.array(0.0), jnp.array(z0),
                                                      dict_phi)) for r in R_arr])
    print(f"z={z0}:  |dPhi/Phi| med={np.median(np.abs((phi_ours-phi_ag)/phi_ag))*100:.3f}%  "
          f"|rhoAG-tr/tr| med={np.median(np.abs((rho_ag-rho_tr)/rho_tr))*100:.2f}%  "
          f"|rho_ours-tr/tr| med={np.median(np.abs((rho_ours-rho_tr)/rho_tr))*100:.2f}%")
