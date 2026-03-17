"""
Fit 3D density (DoubleExponentialDisk + T3 long bar + V4 B/P bulge) to an
N-body barred galaxy snapshot using scipy L-BFGS-B with multiple restarts.

The idea: T3 captures the elongated bar, V4 captures the compact boxy/peanut
bulge in the center. Both share the same bar orientation (aligned with x-axis).

Usage:
    python generate_mock_data/fit_3d_density_scipy_2bar.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.stats import qmc
import matplotlib.pyplot as plt
import agama

from astropy.constants import G
import astropy.units as u

from densities import DoubleExponentialDisk_density
from dehnen_bar import T3_density, V4_density

agama.setUnits(length=1, velocity=1, mass=1)
plt.rcParams['font.size'] = 12

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def bar_angle_bar_strength(x, y, R_anulus=np.arange(1, 5, 0.25)):
    R0 = np.sqrt(x**2 + y**2)
    phi0 = np.arctan2(y, x)
    bar_angles0, bar_strength0 = [], []
    for i in range(len(R_anulus) - 1):
        sel = (R0 > R_anulus[i]) & (R0 < R_anulus[i + 1])
        phi_bin = phi0[sel]
        A2 = np.sum(np.cos(2 * phi_bin)) / len(phi_bin)
        B2 = np.sum(np.sin(2 * phi_bin)) / len(phi_bin)
        bar_angles0.append(0.5 * np.arctan2(B2, A2))
        bar_strength0.append(np.sqrt(A2**2 + B2**2))
    R_mid = (R_anulus[:-1] + R_anulus[1:]) / 2
    return R_mid, np.array(bar_angles0), np.array(bar_strength0)


def rotate(posvel, angle):
    x, y, z, vx, vy, vz = posvel.T
    s, c = np.sin(angle), np.cos(angle)
    return np.array([x*c - y*s, x*s + y*c, z, vx*c - vy*s, vx*s + vy*c, vz]).T


def compute_data_A2(posvel, mass, R_all, R_edges):
    """Compute A2/A0 from N-body particles in radial annuli."""
    R_mid = 0.5 * (R_edges[:-1] + R_edges[1:])
    phi_all = np.arctan2(posvel[:, 1], posvel[:, 0])
    A2 = np.zeros(len(R_mid))
    for j in range(len(R_mid)):
        sel = (R_all > R_edges[j]) & (R_all < R_edges[j + 1])
        m_sel, phi_sel = mass[sel], phi_all[sel]
        if len(m_sel) == 0:
            continue
        A0 = np.sum(m_sel)
        A2[j] = np.sqrt(np.sum(m_sel * np.cos(2 * phi_sel))**2 +
                         np.sum(m_sel * np.sin(2 * phi_sel))**2) / A0
    return R_mid, A2


# ──────────────────────────────────────────────────────────────────────────────
# Density model: Disk + T3 (long bar) + V4 (B/P bulge)
# ──────────────────────────────────────────────────────────────────────────────
# Both T3 and V4 have signature: density(x, y, z, M, a, b, L, gamma)

    # Fixed V4 B/P bulge parameters
V4_A = 0.5     # scale length [kpc]
V4_B = 0.5     # scale height [kpc]
V4_L = 0.1     # half-length [kpc] — nearly axisymmetric
V4_GAMMA = 0.0 # needle slope

@jax.jit
def density_model(x, y, z, params):
    rho_disk = DoubleExponentialDisk_density(x, y, z, params)
    M_bar = 10.0 ** params['logM_bar']
    rho_bar = T3_density(x, y, z,
                         M_bar,
                         params['a_bar'],
                         params['b_bar'],
                         params['L_bar'],
                         params['gamma_bar'])
    # V4 bulge: same mass as T3, fixed shape
    rho_bulge = V4_density(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)
    return rho_disk + rho_bar + rho_bulge


# ──────────────────────────────────────────────────────────────────────────────
# Chi-squared objective
# ──────────────────────────────────────────────────────────────────────────────
# theta layout (8D) — same count as single T3:
#   0: log10(Rd)         disk
#   1: log10(hz)         disk
#   2: log10(rho0)       disk
#   3: log10(M_bar)      T3 bar mass (= V4 bulge mass)
#   4: log10(a_bar)      T3 bar
#   5: log10(b_bar)      T3 bar
#   6: L_bar             T3 bar
#   7: gamma_bar         T3 bar

NPARAM = 8

def make_chi2_fn(x, y, z, density_data, density_err,
                 A2_data, A2_eval_coords, A2_shape, A2_dz, A2_phi,
                 lambda_A2=0.1):
    x, y, z = jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)
    density_data = jnp.asarray(density_data)
    density_err = jnp.asarray(density_err)
    A2_data_j = jnp.asarray(A2_data)
    x_a2, y_a2, z_a2 = (jnp.asarray(c) for c in A2_eval_coords)
    A2_phi_j = jnp.asarray(A2_phi)
    n_R, n_phi, n_z = A2_shape
    ndof = len(density_data)
    A2_weight = ndof / n_R

    @jax.jit
    def _theta_to_params(theta):
        return {
            'rho0_disc': 10.0 ** theta[2],
            'Rd_disc':   10.0 ** theta[0],
            'hz_disc':   10.0 ** theta[1],
            'logM_bar':  theta[3],
            'a_bar':     10.0 ** theta[4],
            'b_bar':     10.0 ** theta[5],
            'L_bar':     theta[6],
            'gamma_bar': theta[7],
            'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
            'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
        }

    @jax.jit
    def chi2(theta):
        params = _theta_to_params(theta)

        # --- density chi2 ---
        model = density_model(x, y, z, params)
        residual = (model - density_data) / (density_err + 1e-12)
        chi2_dens = jnp.sum(residual**2)

        # --- A2 chi2 (bar region only) ---
        rho_a2 = density_model(x_a2, y_a2, z_a2, params)
        rho_a2 = rho_a2.reshape(n_R, n_phi, n_z)
        Sigma = jnp.sum(rho_a2, axis=2) * A2_dz
        a0 = jnp.mean(Sigma, axis=1)
        a2_cos = jnp.mean(Sigma * jnp.cos(2 * A2_phi_j[None, :]), axis=1)
        a2_sin = jnp.mean(Sigma * jnp.sin(2 * A2_phi_j[None, :]), axis=1)
        A2_model = jnp.sqrt(a2_cos**2 + a2_sin**2) / (a0 + 1e-30)
        A2_err = jnp.maximum(A2_data_j * 0.1, 0.01)
        chi2_a2 = jnp.sum(((A2_model - A2_data_j) / A2_err)**2)

        return chi2_dens + lambda_A2 * A2_weight * chi2_a2

    chi2_and_grad = jax.jit(jax.value_and_grad(chi2))
    return chi2, chi2_and_grad


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ---------- 1. Load & preprocess snapshot ----------
    snapshot_path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/Bar_model_TG21/model/t_t0_4'
    print(f"Loading snapshot: {snapshot_path}")
    posvel, mass = agama.readSnapshot(snapshot_path)

    mask = mass != np.unique(mass)[-1]
    posvel, mass = posvel[mask], mass[mask]
    posvel -= posvel.mean(axis=0)
    posvel[:, 0] *= -1
    posvel[:, 3] *= -1

    R_mid, bar_angles, bar_strength = bar_angle_bar_strength(posvel[:, 0], posvel[:, 1])
    bar_angle = np.mean(bar_angles[R_mid < 4])
    posvel = rotate(posvel, -bar_angle)
    print(f"Bar angle: {np.degrees(bar_angle):.1f} deg")

    R = np.sqrt(posvel[:, 0]**2 + posvel[:, 1]**2)
    z_coord = posvel[:, 2]
    Rmax, zmax = 15.0, 5.0
    xmax = Rmax / np.sqrt(2)

    spatial_mask = (R < Rmax) & (np.abs(z_coord) < zmax)
    posvel, mass = posvel[spatial_mask], mass[spatial_mask]
    R, z_coord = R[spatial_mask], z_coord[spatial_mask]

    mass_factor = 1.0 / (G * u.Msun).to(u.kpc * (u.km / u.s)**2).value
    mass = mass * mass_factor

    # ---------- 2. Build 3D density histogram ----------
    nbins_xy, nbins_z = 100, 50
    mass_cell, (xe, ye, ze) = np.histogramdd(
        (posvel[:, 0], posvel[:, 1], posvel[:, 2]),
        bins=(nbins_xy, nbins_xy, nbins_z),
        range=[[-xmax, xmax], [-xmax, xmax], [-zmax, zmax]],
        weights=mass,
    )
    mass2_cell, _ = np.histogramdd(
        (posvel[:, 0], posvel[:, 1], posvel[:, 2]),
        bins=(nbins_xy, nbins_xy, nbins_z),
        range=[[-xmax, xmax], [-xmax, xmax], [-zmax, zmax]],
        weights=mass**2,
    )
    dx, dy, dz = xe[1] - xe[0], ye[1] - ye[0], ze[1] - ze[0]
    cell_vol = dx * dy * dz
    density_cell = mass_cell / cell_vol
    density_err_cell = np.sqrt(mass2_cell) / cell_vol

    xmid = 0.5 * (xe[:-1] + xe[1:])
    ymid = 0.5 * (ye[:-1] + ye[1:])
    zmid = 0.5 * (ze[:-1] + ze[1:])
    xg, yg, zg = np.meshgrid(xmid, ymid, zmid, indexing='ij')

    pos_flat = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
    dens_flat = density_cell.ravel()
    dens_err_flat = density_err_cell.ravel()
    occ = dens_flat > 0
    pos_flat, dens_flat, dens_err_flat = pos_flat[occ], dens_flat[occ], dens_err_flat[occ]

    err_floor = 0.1 * dens_flat
    dens_err_flat = np.sqrt(dens_err_flat**2 + err_floor**2)

    ndof = len(dens_flat) - NPARAM
    print(f"Non-empty cells: {len(dens_flat)}, ndof: {ndof}")

    # ---------- 2b. Compute data A2(R) — bar region only (R < 5 kpc) ----------
    A2_R_edges = np.arange(0.25, 5.25, 0.25)  # only fit A2 inside the bar
    R_mid_a2, A2_data = compute_data_A2(posvel, mass, R, A2_R_edges)
    print(f"A2 bins: {len(R_mid_a2)} (R < 5 kpc), peak A2 = {A2_data.max():.3f} at R = {R_mid_a2[A2_data.argmax()]:.2f} kpc")

    n_phi_a2, n_z_a2 = 64, 30
    phi_a2 = np.linspace(0, 2 * np.pi, n_phi_a2, endpoint=False)
    z_a2 = np.linspace(-zmax, zmax, n_z_a2)
    dz_a2 = z_a2[1] - z_a2[0]

    R_3d, PHI_3d, Z_3d = np.meshgrid(R_mid_a2, phi_a2, z_a2, indexing='ij')
    x_a2_flat = (R_3d * np.cos(PHI_3d)).ravel()
    y_a2_flat = (R_3d * np.sin(PHI_3d)).ravel()
    z_a2_flat = Z_3d.ravel()
    A2_shape = (len(R_mid_a2), n_phi_a2, n_z_a2)

    # ---------- 3. Build chi2 function ----------
    chi2_fn, chi2_and_grad_fn = make_chi2_fn(
        pos_flat[:, 0], pos_flat[:, 1], pos_flat[:, 2],
        dens_flat, dens_err_flat,
        A2_data, (x_a2_flat, y_a2_flat, z_a2_flat), A2_shape, dz_a2, phi_a2,
        lambda_A2=0.1)

    # Warm up JIT  (8 params)
    _test = jnp.array([0.7, -0.3, 8.5, 10.0, 0.0, -0.3, 3.0, 0.0])
    print("JIT compiling (this may take a minute)...")
    _ = chi2_and_grad_fn(_test)
    print("JIT done.")

    def objective(theta):
        theta_jax = jnp.array(theta)
        val, grad = chi2_and_grad_fn(theta_jax)
        return float(val), np.array(grad, dtype=np.float64)

    # ---------- 4. L-BFGS-B with multiple restarts ----------
    #   0: log10(Rd)       1: log10(hz)      2: log10(rho0)
    #   3: log10(M_bar)    4: log10(a_bar)   5: log10(b_bar)   6: L_bar   7: gamma_bar
    #   8: log10(M_bulge)  9: log10(a_bulge) 10: log10(b_bulge) 11: L_bulge  12: gamma_bulge
    bounds = [
        (-1, 2),      # 0  log10(Rd)
        (-1, 1),      # 1  log10(hz)
        (7, 11),      # 2  log10(rho0)
        (8, 12),      # 3  log10(M_bar)       T3 bar (= V4 bulge mass)
        (-1, 1.5),    # 4  log10(a_bar)
        (-1, 1),      # 5  log10(b_bar)
        (0.5, 8.0),   # 6  L_bar              bar half-length
        (-1.0, 1.0),  # 7  gamma_bar
    ]

    n_restarts = 15
    sampler = qmc.LatinHypercube(d=NPARAM, seed=42)
    lh_samples = sampler.random(n=n_restarts)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    initial_points = qmc.scale(lh_samples, lb, ub)

    # Hand-crafted starting points (8D)
    #                  Rd    hz   rho0  Mbar  abar  bbar  Lbar  gbar
    initial_points = np.vstack([
        [0.75, -0.25, 8.2,  10.2, -0.2, -0.2, 4.0, 0.5],
        [0.75, -0.30, 8.1,  10.5, 0.0,  -0.3, 3.5, 0.0],
        [0.70, -0.20, 8.2,  10.3, -0.1, -0.2, 4.5, -0.5],
        [0.80, -0.25, 8.0,  10.0, 0.0,  -0.3, 3.0, 0.8],
        [0.75, -0.25, 8.2,   9.8, 0.2,  -0.2, 2.5, 0.0],
        initial_points,
    ])

    best_result = None
    print(f"\nRunning {len(initial_points)} L-BFGS-B restarts...")
    for i, x0 in enumerate(initial_points):
        res = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                       options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
        chi2_val = res.fun
        chi2_dof = chi2_val / ndof
        status = "converged" if res.success else "not converged"
        print(f"  Restart {i:2d}: chi2/dof = {chi2_dof:.4f}  ({status})")
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    # ---------- 5. Print results ----------
    theta_best = best_result.x
    chi2_best = best_result.fun
    param_names = [
        'log10(Rd)', 'log10(hz)', 'log10(rho0)',
        'log10(M_bar)', 'log10(a_bar)', 'log10(b_bar)', 'L_bar', 'gamma_bar',
    ]

    print("\n" + "=" * 60)
    print("Best-fit parameters (Disk + T3 bar + V4 bulge [fixed]):")
    print("=" * 60)
    for name, val in zip(param_names, theta_best):
        print(f"  {name:15s} = {val:.6f}")
    print(f"\n  chi2     = {chi2_best:.1f}")
    print(f"  ndof     = {ndof}")
    print(f"  chi2/dof = {chi2_best / ndof:.4f}")
    print("=" * 60)

    M_bar_phys = 10**theta_best[3]
    print("\nPhysical parameters:")
    print(f"  --- Disk ---")
    print(f"  Rd          = {10**theta_best[0]:.4f} kpc")
    print(f"  hz          = {10**theta_best[1]:.4f} kpc")
    print(f"  rho0        = {10**theta_best[2]:.4e} Msun/kpc^3")
    print(f"  --- T3 long bar ---")
    print(f"  M_bar       = {M_bar_phys:.4e} Msun")
    print(f"  a_bar       = {10**theta_best[4]:.4f} kpc")
    print(f"  b_bar       = {10**theta_best[5]:.4f} kpc")
    print(f"  L_bar       = {theta_best[6]:.4f} kpc")
    print(f"  gamma_bar   = {theta_best[7]:.4f}")
    print(f"  --- V4 B/P bulge (fixed shape, M = M_bar) ---")
    print(f"  M_bulge     = {M_bar_phys:.4e} Msun  (= M_bar)")
    print(f"  a_bulge     = {V4_A} kpc  (fixed)")
    print(f"  b_bulge     = {V4_B} kpc  (fixed)")
    print(f"  L_bulge     = {V4_L} kpc  (fixed)")
    print(f"  gamma_bulge = {V4_GAMMA}  (fixed)")

    # ---------- 6. Evaluate best-fit model on the grid ----------
    params_best = {
        'rho0_disc': 10.0 ** theta_best[2],
        'Rd_disc':   10.0 ** theta_best[0],
        'hz_disc':   10.0 ** theta_best[1],
        'logM_bar':  theta_best[3],
        'a_bar':     10.0 ** theta_best[4],
        'b_bar':     10.0 ** theta_best[5],
        'L_bar':     theta_best[6],
        'gamma_bar': theta_best[7],
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    }

    model_density_full = np.array(density_model(
        jnp.asarray(xg.ravel()), jnp.asarray(yg.ravel()), jnp.asarray(zg.ravel()),
        params_best,
    )).reshape(density_cell.shape)
    model_mass_cell = model_density_full * cell_vol

    # ---------- 7. Diagnostic plots ----------
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 7a. Projections ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'wspace': 0.4, 'hspace': 0.35})

    data_xy = mass_cell.sum(axis=2)
    model_xy = model_mass_cell.sum(axis=2)
    vmin_xy, vmax_xy = 1e6, 1e9

    cb = axes[0, 0].imshow(data_xy.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                           norm='log', vmin=vmin_xy, vmax=vmax_xy)
    axes[0, 0].contour(xmid, ymid, np.log10(np.clip(data_xy, 1, None)).T,
                       levels=np.arange(6, 9.5, 0.5), colors='white', linewidths=0.5)
    axes[0, 0].set_title('Data (XY)'); axes[0, 0].set_xlabel('x [kpc]'); axes[0, 0].set_ylabel('y [kpc]')
    plt.colorbar(cb, ax=axes[0, 0], label='Mass')

    cb = axes[0, 1].imshow(model_xy.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                           norm='log', vmin=vmin_xy, vmax=vmax_xy)
    axes[0, 1].contour(xmid, ymid, np.log10(np.clip(model_xy, 1, None)).T,
                       levels=np.arange(6, 9.5, 0.5), colors='white', linewidths=0.5)
    axes[0, 1].set_title('Model (XY)'); axes[0, 1].set_xlabel('x [kpc]'); axes[0, 1].set_ylabel('y [kpc]')
    plt.colorbar(cb, ax=axes[0, 1], label='Mass')

    rel_res_xy = (data_xy - model_xy) / np.where(data_xy > 0, data_xy, 1)
    cb = axes[0, 2].pcolormesh(xe, ye, rel_res_xy.T, vmin=-0.5, vmax=0.5, cmap='coolwarm')
    axes[0, 2].set_title('Residual (XY)'); axes[0, 2].set_xlabel('x [kpc]'); axes[0, 2].set_ylabel('y [kpc]')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(cb, ax=axes[0, 2], label='(Data-Model)/Data')

    data_xz = mass_cell.sum(axis=1)
    model_xz = model_mass_cell.sum(axis=1)
    vmin_xz, vmax_xz = 5e5, 5e8

    cb = axes[1, 0].imshow(data_xz.T, origin='lower', extent=[xe[0], xe[-1], ze[0], ze[-1]],
                           norm='log', vmin=vmin_xz, vmax=vmax_xz)
    axes[1, 0].set_title('Data (XZ)'); axes[1, 0].set_xlabel('x [kpc]'); axes[1, 0].set_ylabel('z [kpc]')
    plt.colorbar(cb, ax=axes[1, 0], label='Mass')

    cb = axes[1, 1].imshow(model_xz.T, origin='lower', extent=[xe[0], xe[-1], ze[0], ze[-1]],
                           norm='log', vmin=vmin_xz, vmax=vmax_xz)
    axes[1, 1].set_title('Model (XZ)'); axes[1, 1].set_xlabel('x [kpc]'); axes[1, 1].set_ylabel('z [kpc]')
    plt.colorbar(cb, ax=axes[1, 1], label='Mass')

    rel_res_xz = (data_xz - model_xz) / np.where(data_xz > 0, data_xz, 1)
    cb = axes[1, 2].pcolormesh(xe, ze, rel_res_xz.T, vmin=-0.5, vmax=0.5, cmap='coolwarm')
    axes[1, 2].set_title('Residual (XZ)'); axes[1, 2].set_xlabel('x [kpc]'); axes[1, 2].set_ylabel('z [kpc]')
    axes[1, 2].set_aspect('equal')
    plt.colorbar(cb, ax=axes[1, 2], label='(Data-Model)/Data')

    fig.suptitle(f'Disk + T3 bar + V4 bulge (fixed) — chi2/dof = {chi2_best/ndof:.4f}', fontsize=14)
    fig.savefig(os.path.join(output_dir, 'fit_3d_density_2bar_projections.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved projection plot")

    # --- 7b. Radial and vertical profiles ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    R_all = np.sqrt(posvel[:, 0]**2 + posvel[:, 1]**2)
    z_all = posvel[:, 2]
    xg_f, yg_f, zg_f = xg.ravel(), yg.ravel(), zg.ravel()
    model_mass_flat = model_mass_cell.ravel()
    R_grid = np.sqrt(xg_f**2 + yg_f**2)

    z_bins = [0, 1, 2, 3]
    colors = ['blue', 'orange', 'green', 'red']
    for i in range(len(z_bins) - 1):
        m1 = (np.abs(z_all) > z_bins[i]) & (np.abs(z_all) < z_bins[i + 1])
        H_d, redge = np.histogram(R_all[m1], bins=100, range=[0, Rmax], weights=mass[m1])
        dr = redge[1] - redge[0]
        rmid = 0.5 * (redge[:-1] + redge[1:])
        axes[0].plot(rmid, H_d / (2 * np.pi * rmid * dr), color=colors[i],
                     label=f'{z_bins[i]}<|z|<{z_bins[i+1]} data')
        m2 = (np.abs(zg_f) > z_bins[i]) & (np.abs(zg_f) < z_bins[i + 1])
        H_m, _ = np.histogram(R_grid[m2], bins=100, range=[0, Rmax], weights=model_mass_flat[m2])
        axes[0].plot(rmid, H_m / (2 * np.pi * rmid * dr), color=colors[i], ls='--',
                     label=f'{z_bins[i]}<|z|<{z_bins[i+1]} model')
    axes[0].set_xlabel('R [kpc]'); axes[0].set_ylabel('Surface density')
    axes[0].set_yscale('log'); axes[0].legend(fontsize=8, ncol=2); axes[0].set_title('Radial profiles')

    R_bins = [0, 3, 6, 9, 12]
    colors2 = ['blue', 'orange', 'green', 'red', 'purple']
    for i in range(len(R_bins) - 1):
        m1 = (R_all > R_bins[i]) & (R_all < R_bins[i + 1])
        H_d, zedge = np.histogram(z_all[m1], bins=50, range=[-zmax, zmax], weights=mass[m1])
        zmid_h = 0.5 * (zedge[:-1] + zedge[1:])
        axes[1].plot(zmid_h, H_d, color=colors2[i], label=f'{R_bins[i]}<R<{R_bins[i+1]} data')
        m2 = (R_grid > R_bins[i]) & (R_grid < R_bins[i + 1])
        H_m, _ = np.histogram(zg_f[m2], bins=50, range=[-zmax, zmax], weights=model_mass_flat[m2])
        axes[1].plot(zmid_h, H_m, color=colors2[i], ls='--', label=f'{R_bins[i]}<R<{R_bins[i+1]} model')
    axes[1].set_xlabel('z [kpc]'); axes[1].set_ylabel('Mass')
    axes[1].set_yscale('log'); axes[1].legend(fontsize=8, ncol=2); axes[1].set_title('Vertical profiles')

    fig.savefig(os.path.join(output_dir, 'fit_3d_density_2bar_profiles.png'), dpi=150, bbox_inches='tight')
    print(f"Saved profile plot")

    # --- 7c. A2(R) ---
    A2_plot_edges = np.arange(0.25, 12.0, 0.25)
    R_mid_plot, A2_data_plot = compute_data_A2(posvel, mass, R, A2_plot_edges)

    n_phi_plot, n_z_plot = 128, 50
    phi_plot = np.linspace(0, 2 * np.pi, n_phi_plot, endpoint=False)
    z_plot = np.linspace(-zmax, zmax, n_z_plot)
    dz_plot = z_plot[1] - z_plot[0]

    A2_model_plot = np.zeros(len(R_mid_plot))
    for j in range(len(R_mid_plot)):
        Rj = R_mid_plot[j]
        PHI, Z_p = np.meshgrid(phi_plot, z_plot, indexing='ij')
        rho_eval = np.array(density_model(
            jnp.asarray((Rj * np.cos(PHI)).ravel()),
            jnp.asarray((Rj * np.sin(PHI)).ravel()),
            jnp.asarray(Z_p.ravel()),
            params_best,
        )).reshape(n_phi_plot, n_z_plot)
        Sigma = np.sum(rho_eval, axis=1) * dz_plot
        a0 = np.mean(Sigma)
        a2c = np.mean(Sigma * np.cos(2 * phi_plot))
        a2s = np.mean(Sigma * np.sin(2 * phi_plot))
        if a0 > 0:
            A2_model_plot[j] = np.sqrt(a2c**2 + a2s**2) / a0

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(R_mid_plot, A2_data_plot, 'k-', lw=2, label='N-body data')
    ax.plot(R_mid_plot, A2_model_plot, 'r--', lw=2, label='T3 bar + V4 bulge')
    ax.set_xlabel('R [kpc]'); ax.set_ylabel(r'$A_2 / A_0$')
    ax.set_title('Bar quadrupole strength')
    ax.legend(); ax.set_xlim(0, 12); ax.set_ylim(0, None)

    fig.savefig(os.path.join(output_dir, 'fit_3d_density_2bar_A2.png'), dpi=150, bbox_inches='tight')
    print(f"Saved A2 plot")

    plt.show()
