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

from densities import MiyamotoNagai_density
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
# Density model: Miyamoto-Nagai disc + T3 (long bar) + V4 (B/P bulge)
# ──────────────────────────────────────────────────────────────────────────────
# Both T3 and V4 have signature: density(x, y, z, M, a, b, L, gamma)

# Fixed V4 B/P bulge parameters
V4_A = 0.5     # scale length [kpc]
V4_B = 0.5     # scale height [kpc]
V4_L = 0.1     # half-length [kpc] — nearly axisymmetric
V4_GAMMA = 0.0 # needle slope

@jax.jit
def density_model(x, y, z, params):
    rho_disk = MiyamotoNagai_density(x, y, z, params)
    M_bar = 10.0 ** params['logM_bar']
    rho_bar = T3_density(x, y, z,
                         M_bar,
                         params['a_bar'],
                        # params['L_bar'] / 5, 
                         params['b_bar'],
                         params['L_bar'],
                         params['gamma_bar'])
    # V4 bulge: same mass as T3, fixed shape
    rho_bulge = V4_density(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)
    return rho_disk + rho_bar + rho_bulge

def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        ckpt = pickle.load(f)

    all_samples = ckpt['all_samples']   # list of (N_CHAINS, NDIM) arrays
    all_logprob = ckpt['all_logprob']   # list of (N_CHAINS,) arrays
    step = ckpt['step']


    # Stack into (N_STEPS, N_CHAINS, NDIM) and (N_STEPS, N_CHAINS)
    posterior = np.stack(all_samples, axis=0)
    logprob = np.stack(all_logprob, axis=0)
    
    print(f"Loaded checkpoint: {step} steps, "
          f"{posterior.shape[1]} chains, {posterior.shape[2]} params")
    print(f"Chain shape: {posterior.shape}")
    return posterior, logprob, step

if __name__ == "__main__":

    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

    figname = data_folder+'/plots/rotation_curve_data_vs_model.png'
    # CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_gal2_0406.pkl'
    CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_0415_beta25_gamma140_D50_gal2.pkl'
    
    DISCARD=400
    THIN=1
    posterior, logprob, step = load_checkpoint(CHECKPOINT_FILE)
    posterior = posterior[:, logprob[-1, :]>np.amax(logprob[-1, :])-100, :]
    posterior = posterior[DISCARD::THIN, :, :]
    posterior = posterior.reshape(-1, posterior.shape[-1])

    best_params = np.percentile(posterior, 50, axis=0)

    print('shape of posterior', posterior.shape)
    param_names = ['logM_halo', 'logM_disk', 'logM_bar', 'logRs_halo', 'logRs_disk', 'logHs_disk', 'logRs_bar', 
               r'$\alpha$', r'$\beta$', r'$\gamma$', 
               r'$\log_{10}(L/M)$', r'$\log_{10}(\Omega)$', r'$\log_{10}(\sigma amplifier)$']
    print("Best-fit parameters:")
    for i, name in enumerate(param_names):
        print(f"{name}: {best_params[i]:.4f}")

    # ---------- 1. Load & preprocess snapshot ----------
    snapshot_path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/Bar_model_TG21/model/t_t0_7'
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
    params_best = {
        'logM_disc': theta_best[0],
        'Rs_disc':   10.0 ** theta_best[1],
        'Hs_disc':   10.0 ** theta_best[2],
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
    data_xy = mass_cell.sum(axis=2)
    model_xy = model_mass_cell.sum(axis=2)
    vmin_xy, vmax_xy = 1e6, 1e9