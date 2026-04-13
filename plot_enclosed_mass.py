"""
Plot enclosed mass M(<r) as a function of spherical radius for disc (baryonic)
and halo components, comparing model posteriors to N-body data.

Usage:
    python plot_enclosed_mass.py
"""

import agama
agama.setUnits(mass=1, length=1, velocity=1)
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from utils import logMenc_logc_to_logM_logRs
from densities import MiyamotoNagai_density
from dehnen_bar import T3_density, V4_density
from tqdm import tqdm
import pickle

from astropy import units as u
from astropy.constants import G


def plot_prettier(dpi=200, fontsize=12, usetex=False):
    plt.rcParams['figure.dpi'] = dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern=[2., 2.])
    plt.rc('text', usetex=usetex)
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plot_prettier(usetex=False)

# ── Fixed bar/bulge shape parameters ──
V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
GAMMA_BAR = 1.0


# ── Density functions (disc + bar + bulge) ──
@jax.jit
def density_baryon(x, y, z, params):
    """Total baryonic density: MN disc + T3 bar + V4 bulge."""
    mn_params = {
        'logM_disc': params['logM_disc'],
        'Rs_disc': params['Rs_disc'],
        'Hs_disc': params['Hs_disc'],
        'x_origin': params['x_origin'],
        'y_origin': params['y_origin'],
        'z_origin': params['z_origin'],
        'dirx': params['dirx'],
        'diry': params['diry'],
        'dirz': params['dirz'],
    }
    rho_mn = MiyamotoNagai_density(x, y, z, mn_params)
    M_bar = 10.0 ** params['logM_bar']
    rho_t3 = T3_density(x, y, z, M_bar, params['a_bar'], params['b_bar'],
                        params['L_bar'], GAMMA_BAR)
    rho_v4 = V4_density(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)
    return rho_mn + rho_t3 + rho_v4


# ── NFW enclosed mass (analytic) ──
def nfw_enclosed_mass(r, logM, Rs):
    """NFW enclosed mass within spherical radius r.
    M_enc(r) = M * [ln(1 + r/Rs) - (r/Rs) / (1 + r/Rs)]
    """
    M = 10.0 ** logM
    x = r / Rs
    return M * (np.log(1.0 + x) - x / (1.0 + x))


# ── Baryonic enclosed mass (numerical integration on cylindrical grid) ──
def baryon_enclosed_mass(r_targets, params_baryon, n_R=200, n_z=200, n_phi=64,
                         R_max=30.0, z_max=15.0):
    """
    Compute baryonic enclosed mass M(<r) by integrating density on a
    cylindrical (R, z, phi) grid and accumulating into spherical shells.

    Parameters
    ----------
    r_targets : array
        Spherical radii at which to evaluate M_enc.
    params_baryon : dict
        Baryonic density parameters.

    Returns
    -------
    M_enc : array, same shape as r_targets
    """
    # Build cylindrical grid
    R_edges = np.linspace(0, R_max, n_R + 1)
    z_edges = np.linspace(-z_max, z_max, n_z + 1)
    phi_edges = np.linspace(0, 2 * np.pi, n_phi + 1)

    R_mid = 0.5 * (R_edges[:-1] + R_edges[1:])
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    phi_mid = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    dR = R_edges[1] - R_edges[0]
    dz = z_edges[1] - z_edges[0]
    dphi = phi_edges[1] - phi_edges[0]

    # 3D grid: (n_R, n_phi, n_z)
    R_3d, PHI_3d, Z_3d = np.meshgrid(R_mid, phi_mid, z_mid, indexing='ij')
    x_flat = jnp.array((R_3d * np.cos(PHI_3d)).ravel())
    y_flat = jnp.array((R_3d * np.sin(PHI_3d)).ravel())
    z_flat = jnp.array(Z_3d.ravel())
    R_flat = R_3d.ravel()

    # Evaluate density
    rho = np.array(density_baryon(x_flat, y_flat, z_flat, params_baryon))

    # Cell volume in cylindrical coords: R * dR * dphi * dz
    cell_mass = rho * np.array(R_flat) * dR * dphi * dz

    # Spherical radius of each cell center
    r_sph = np.sqrt(np.array(R_flat)**2 + np.array(Z_3d.ravel())**2)

    # Sort by spherical radius and cumulative sum
    order = np.argsort(r_sph)
    r_sorted = r_sph[order]
    mass_sorted = cell_mass[order]
    cumulative_mass = np.cumsum(mass_sorted)

    # Interpolate to target radii
    M_enc = np.interp(r_targets, r_sorted, cumulative_mass)
    return M_enc


# ── Data loading ──
def rotate(posvel, angle):
    x, y, z, vx, vy, vz = posvel.T
    sina, cosa = np.sin(angle), np.cos(angle)
    return np.array([x*cosa - y*sina, x*sina + y*cosa, z,
                     vx*cosa - vy*sina, vx*sina + vy*cosa, vz]).T


def bar_angle_bar_strength(x, y, R_anulus=np.arange(1, 5, 0.25)):
    R0, phi0 = np.sqrt(x**2 + y**2), np.arctan2(y, x)
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


def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        ckpt = pickle.load(f)
    all_samples = ckpt['all_samples']
    all_logprob = ckpt['all_logprob']
    step = ckpt['step']
    posterior = np.stack(all_samples, axis=0)
    logprob = np.stack(all_logprob, axis=0)
    print(f"Loaded checkpoint: {step} steps, "
          f"{posterior.shape[1]} chains, {posterior.shape[2]} params")
    return posterior, logprob, step


def load_density_dict(params):
    logM_enc = params[0]
    log_c = params[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(
        logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54)

    logM_disc = params[1]
    logM_bar = params[2]
    logRs_disk = params[4]
    logHs_disk = params[5]
    logL_bar = params[6]

    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc

    params_halo = {
        'logM': float(logM_halo),
        'Rs': float(10 ** logRs_halo),
    }

    params_baryon = {
        'logM_disc': logM_disc,
        'Rs_disc': 10 ** logRs_disk,
        'Hs_disc': Hs_disc,
        'logM_bar': logM_bar,
        'L_bar': L_bar,
        'a_bar': a_bar,
        'b_bar': b_bar,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
    }

    return params_baryon, params_halo


# ── N-body enclosed mass by component ──
def nbody_enclosed_mass(posvel, mass, r_targets):
    """Compute cumulative enclosed mass from particle data."""
    r = np.sqrt(posvel[:, 0]**2 + posvel[:, 1]**2 + posvel[:, 2]**2)
    order = np.argsort(r)
    r_sorted = r[order]
    mass_sorted = mass[order]
    cumsum = np.cumsum(mass_sorted)
    return np.interp(r_targets, r_sorted, cumsum)


if __name__ == '__main__':

    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    figname = data_folder + '/plots/enclosed_mass_data_vs_model.png'
    CHECKPOINT_FILE = data_folder + '/ensemble_checkpoint_gal2_0406.pkl'

    DISCARD = 500
    THIN = 100

    # ── Load posterior ──
    posterior, logprob, step = load_checkpoint(CHECKPOINT_FILE)
    posterior = posterior[:, logprob[-1, :] > np.amax(logprob[-1, :]) - 100, :]
    posterior = posterior[DISCARD::THIN, :, :]
    posterior = posterior.reshape(-1, posterior.shape[-1])
    print(f'Posterior shape: {posterior.shape}')

    # ── Load N-body snapshot ──
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s)**2))
    w0_data, mass_data = agama.readSnapshot(
        data_folder + '/Bar_model_TG21/model/t_t0_4')
    mass_data = mass_data * mass_unit.value

    # Separate components by particle mass
    unique_masses = np.unique(mass_data)
    print(f"Particle types: {len(unique_masses)}, masses: {unique_masses}")

    # Heaviest particles = dark matter halo
    mask_halo = mass_data == unique_masses[-1]
    mask_disc = ~mask_halo

    # Center on stellar component
    R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
    mask_center = R < 5
    for col in range(6):
        w0_data[:, col] -= np.sum(w0_data[mask_center, col] * mass_data[mask_center]) / np.sum(mass_data[mask_center])

    # Flip x-axis convention
    w0_data[:, 0] = -w0_data[:, 0]
    w0_data[:, 3] = -w0_data[:, 3]

    # Rotate to align bar with x-axis
    R_mid, bar_angles0, _ = bar_angle_bar_strength(
        w0_data[:, 0], w0_data[:, 1], R_anulus=np.arange(1, 5, 0.1))
    bar_angle0 = np.mean(bar_angles0[R_mid < 4])
    w0_data = rotate(w0_data, -bar_angle0)

    # ── Compute N-body enclosed masses ──
    r_plot = np.linspace(0.1, 20, 200)

    M_enc_disc_data = nbody_enclosed_mass(w0_data[mask_disc], mass_data[mask_disc], r_plot)
    M_enc_halo_data = nbody_enclosed_mass(w0_data[mask_halo], mass_data[mask_halo], r_plot)
    M_enc_total_data = M_enc_disc_data + M_enc_halo_data

    # ── Compute model enclosed masses from posterior ──
    M_enc_disc_all = []
    M_enc_halo_all = []

    for i in tqdm(range(posterior.shape[0]), desc="Computing M_enc"):
        params_baryon, params_halo = load_density_dict(posterior[i])

        # Halo: analytic NFW
        M_halo_i = nfw_enclosed_mass(r_plot, params_halo['logM'], params_halo['Rs'])

        # Baryons: numerical integration
        M_disc_i = baryon_enclosed_mass(r_plot, params_baryon)

        M_enc_halo_all.append(M_halo_i)
        M_enc_disc_all.append(M_disc_i)

    M_enc_halo_all = np.array(M_enc_halo_all)
    M_enc_disc_all = np.array(M_enc_disc_all)
    M_enc_total_all = M_enc_halo_all + M_enc_disc_all

    # Percentiles
    halo_16, halo_50, halo_84 = np.percentile(M_enc_halo_all, [16, 50, 84], axis=0)
    disc_16, disc_50, disc_84 = np.percentile(M_enc_disc_all, [16, 50, 84], axis=0)
    total_16, total_50, total_84 = np.percentile(M_enc_total_all, [16, 50, 84], axis=0)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Halo
    ax = axes[0]
    ax.plot(r_plot, M_enc_halo_data, lw=3, color='royalblue', label='N-body')
    ax.fill_between(r_plot, halo_16, halo_84, color='tomato', alpha=0.2, label=r'Model $1\sigma$')
    ax.plot(r_plot, halo_50, lw=2, ls='--', color='tomato', label='Model median')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel(r'$M(<r)$ [$M_\odot$]')
    ax.set_title('Dark Matter Halo')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlim = (0.1, 10))
    ax.legend(frameon=False, fontsize=10)

    # Disc (baryonic)
    ax = axes[1]
    ax.plot(r_plot, M_enc_disc_data, lw=3, color='royalblue', label='N-body')
    ax.fill_between(r_plot, disc_16, disc_84, color='tomato', alpha=0.2, label=r'Model $1\sigma$')
    ax.plot(r_plot, disc_50, lw=2, ls='--', color='tomato', label='Model median')
    ax.set_xlabel('r [kpc]')
    ax.set_title('Baryonic (Disc + Bar + Bulge)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlim = (0.1, 10))
    ax.legend(frameon=False, fontsize=10)

    # Total
    ax = axes[2]
    ax.plot(r_plot, M_enc_total_data, lw=3, color='royalblue', label='N-body')
    ax.fill_between(r_plot, total_16, total_84, color='tomato', alpha=0.2, label=r'Model $1\sigma$')
    ax.plot(r_plot, total_50, lw=2, ls='--', color='tomato', label='Model median')
    ax.set_xlabel('r [kpc]')
    ax.set_title('Total')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlim = (0.1, 10))
    ax.legend(frameon=False, fontsize=10)

    fig.suptitle('Enclosed Mass Profile: Model vs N-body', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    print(f"Saved to {figname}")
    plt.show()
