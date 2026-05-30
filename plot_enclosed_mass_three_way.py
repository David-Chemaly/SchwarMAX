"""
Plot enclosed mass M(<r) comparing three things on the same panel:
    1. Original N-body mock data (per component, by particle mass)
    2. Analytic density model evaluated at the posterior median
    3. Best-fit orbital library (time-averaged trajectories with NNLS weights)

Useful for diagnosing the light-to-mass ratio: if the orbital library puts
more mass than the N-body data inside a given radius (while the density
model agrees with the N-body), the L/M ratio is being overestimated.

Usage:
    python plot_enclosed_mass_three_way.py
"""

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from tqdm import tqdm
from astropy import units as u
from astropy.constants import G

from utils import logMenc_logc_to_logM_logRs
from densities import MiyamotoNagai_density
from dehnen_bar import T3_density, V4_density


def plot_prettier(dpi=200, fontsize=12, usetex=False):
    plt.rcParams['figure.dpi'] = dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5); plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5); plt.rc('ytick.minor', pad=5)
    plt.rc('text', usetex=usetex)
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plot_prettier(usetex=False)

V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
GAMMA_BAR = 1.0


# ═══════════════════════════════════════════════════════════════════
#  Density model helpers (copied from plot_enclosed_mass.py)
# ═══════════════════════════════════════════════════════════════════

@jax.jit
def density_baryon(x, y, z, params):
    mn_params = {
        'logM_disc': params['logM_disc'],
        'Rs_disc':   params['Rs_disc'],
        'Hs_disc':   params['Hs_disc'],
        'x_origin':  params['x_origin'],
        'y_origin':  params['y_origin'],
        'z_origin':  params['z_origin'],
        'dirx':      params['dirx'],
        'diry':      params['diry'],
        'dirz':      params['dirz'],
    }
    rho_mn = MiyamotoNagai_density(x, y, z, mn_params)
    M_bar = 10.0 ** params['logM_bar']
    rho_t3 = T3_density(x, y, z, M_bar, params['a_bar'], params['b_bar'],
                        params['L_bar'], GAMMA_BAR)
    rho_v4 = V4_density(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)
    return rho_mn + rho_t3 + rho_v4


def nfw_enclosed_mass(r, logM, Rs):
    M = 10.0 ** logM
    x = r / Rs
    return M * (np.log(1.0 + x) - x / (1.0 + x))


def baryon_enclosed_mass(r_targets, params_baryon,
                         n_r=400, n_theta=64, n_phi=64,
                         r_min=0.01, r_max=40.0):
    """Enclosed M(<r) of the baryonic density.

    Uses a SPHERICAL (r, theta, phi) grid with log-spaced r so each radial
    shell maps directly to one r bin. The previous cylindrical grid
    produced visible step-artefacts at small r because all phi-cells at
    a given (R, z) shared the same r_sph and np.interp couldn't fill
    between them.
    """
    r_targets = np.atleast_1d(r_targets).astype(float)

    # Log-spaced radial edges -> fine resolution near origin
    r_edges  = np.geomspace(r_min, r_max, n_r + 1)
    r_mid    = np.sqrt(r_edges[:-1] * r_edges[1:])
    dr       = np.diff(r_edges)

    theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
    theta_mid   = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    dtheta      = np.diff(theta_edges)

    phi_edges = np.linspace(0.0, 2 * np.pi, n_phi + 1)
    phi_mid   = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    dphi      = phi_edges[1] - phi_edges[0]

    R, T, P = np.meshgrid(r_mid, theta_mid, phi_mid, indexing='ij')
    sinT = np.sin(T)
    x = R * sinT * np.cos(P)
    y = R * sinT * np.sin(P)
    z = R * np.cos(T)

    rho = np.array(density_baryon(jnp.array(x.ravel()),
                                  jnp.array(y.ravel()),
                                  jnp.array(z.ravel()),
                                  params_baryon)).reshape(R.shape)

    # Spherical volume element: r^2 sin(theta) dr dtheta dphi
    dV = R**2 * sinT \
         * dr[:, None, None] \
         * dtheta[None, :, None] \
         * dphi
    cell_mass = rho * dV

    # Mass in each radial shell (sum over theta, phi)
    shell_mass = cell_mass.sum(axis=(1, 2))
    cumulative_mass = np.cumsum(shell_mass)

    return np.interp(r_targets, r_mid, cumulative_mass,
                     left=cumulative_mass[0], right=cumulative_mass[-1])


# ═══════════════════════════════════════════════════════════════════
#  Particle / trajectory helpers
# ═══════════════════════════════════════════════════════════════════

def particles_enclosed_mass(positions_xyz, masses, r_targets):
    """Cumulative M(<r) from a particle list (positions in kpc, masses in Msun)."""
    r = np.sqrt(positions_xyz[:, 0]**2 + positions_xyz[:, 1]**2 + positions_xyz[:, 2]**2)
    order = np.argsort(r)
    cum = np.cumsum(masses[order])
    return np.interp(r_targets, r[order], cum)


def rotate(posvel, angle):
    x, y, z, vx, vy, vz = posvel.T
    sina, cosa = np.sin(angle), np.cos(angle)
    return np.array([x*cosa - y*sina, x*sina + y*cosa, z,
                     vx*cosa - vy*sina, vx*sina + vy*cosa, vz]).T


def bar_angle(x, y, R_anulus=np.arange(1, 5, 0.1)):
    R0, phi0 = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    angles = []
    for i in range(len(R_anulus) - 1):
        sel = (R0 > R_anulus[i]) & (R0 < R_anulus[i + 1])
        phi_bin = phi0[sel]
        A2 = np.sum(np.cos(2 * phi_bin)) / len(phi_bin)
        B2 = np.sum(np.sin(2 * phi_bin)) / len(phi_bin)
        angles.append(0.5 * np.arctan2(B2, A2))
    R_mid = (R_anulus[:-1] + R_anulus[1:]) / 2
    return R_mid, np.array(angles)


def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        ckpt = pickle.load(f)
    posterior = np.stack(ckpt['all_samples'], axis=0)
    logprob   = np.stack(ckpt['all_logprob'], axis=0)
    print(f"Loaded checkpoint: {ckpt['step']} steps, "
          f"{posterior.shape[1]} chains, {posterior.shape[2]} params")
    return posterior, logprob


def load_density_dict(params):
    """Unpack a 13-D parameter vector into halo / baryon dicts."""
    logM_enc, log_c = params[0], params[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(
        logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54)

    logM_disc  = params[1]
    logM_bar   = params[2]
    logRs_disk = params[4]
    logHs_disk = params[5]
    logL_bar   = params[6]

    L_bar   = 10.0 ** logL_bar
    a_bar   = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar   = Hs_disc

    params_halo = {'logM': float(logM_halo), 'Rs': float(10**logRs_halo)}
    params_baryon = {
        'logM_disc': logM_disc,
        'Rs_disc':   10**logRs_disk,
        'Hs_disc':   Hs_disc,
        'logM_bar':  logM_bar,
        'L_bar':     L_bar,
        'a_bar':     a_bar,
        'b_bar':     b_bar,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    }
    return params_baryon, params_halo


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    figname     = data_folder + '/plots/enclosed_mass_three_way.png'
    figname_paper = data_folder + '/figs_paper/enclosed_mass_three_way.pdf'
    # CHECKPOINT_FILE = data_folder + '/ensemble_checkpoint_0422_beta25_gamma140_D50_gal2.pkl'
    CHECKPOINT_FILE = data_folder+'/mcmc_checkpoint_0422_beta25_gamma140_D50_gal2.pkl'
    ORBLIB_FILE     = data_folder + '/best_fit_orbital_library_0415_beta25_gamma140_D50_gal2.pkl'

    DISCARD = 500
    THIN    = 30

    # r_plot = np.linspace(0.1, 30, 200)
    r_plot = np.logspace(np.log10(0.3), np.log10(30), 500)

    # ─────────────────────────── N-body data ───────────────────────────
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s)**2))
    w0_data, mass_data = agama.readSnapshot(
        data_folder + '/Bar_model_TG21/model/t_t0_7')
    mass_data = mass_data * mass_unit.value

    unique_masses = np.unique(mass_data)
    mask_halo = mass_data == unique_masses[-1]
    mask_disc = ~mask_halo

    # Iterative shrinking-aperture COM centering on disc particles.
    # One iteration with R<5 kpc isn't enough -- the disc COM hasn't
    # converged and the resulting offset depresses M(<r) at small r.
    for r_ap in [5.0, 5.0]:
        R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
        mask_center = mask_disc & (R < r_ap)
        # mask_center = mask_halo & (R < r_ap)
        m_c = mass_data[mask_center]
        for col in range(6):
            w0_data[:, col] -= np.sum(w0_data[mask_center, col] * m_c) / np.sum(m_c)


    # Flip x-axis convention to match model
    w0_data[:, 0] = -w0_data[:, 0]
    w0_data[:, 3] = -w0_data[:, 3]

    # Align bar with x-axis
    R_mid_ba, bar_angles0 = bar_angle(w0_data[:, 0], w0_data[:, 1])
    bar_angle0 = np.mean(bar_angles0[R_mid_ba < 4])
    w0_data = rotate(w0_data, -bar_angle0)

    M_disc_data  = particles_enclosed_mass(w0_data[mask_disc], mass_data[mask_disc], r_plot)
    M_halo_data  = particles_enclosed_mass(w0_data[mask_halo], mass_data[mask_halo], r_plot)
    M_total_data = M_disc_data + M_halo_data

    # ─────────────────────── Density model (posterior median) ───────────
    posterior, logprob = load_checkpoint(CHECKPOINT_FILE)
    posterior = posterior[:, logprob[-1, :] > np.amax(logprob[-1, :]) - 100, :]
    posterior = posterior[DISCARD::THIN, :, :].reshape(-1, posterior.shape[-1])
    print(f'Posterior shape after thinning: {posterior.shape}')

    M_disc_model = []
    M_halo_model = []
    for i in tqdm(range(posterior.shape[0]), desc="Density model M(<r)"):
        params_baryon, params_halo = load_density_dict(posterior[i])
        M_halo_model.append(nfw_enclosed_mass(r_plot, params_halo['logM'], params_halo['Rs']))
        M_disc_model.append(baryon_enclosed_mass(r_plot, params_baryon))
    M_disc_model = np.array(M_disc_model)
    M_halo_model = np.array(M_halo_model)

    halo_16, halo_50, halo_84 = np.percentile(M_halo_model, [16, 50, 84], axis=0)
    disc_16, disc_50, disc_84 = np.percentile(M_disc_model, [16, 50, 84], axis=0)
    M_total_model = M_halo_model + M_disc_model
    total_16, total_50, total_84 = np.percentile(M_total_model, [16, 50, 84], axis=0)

    # ─────────────────────── Best-fit orbital library ───────────────────
    # NOTE on the saved file (see get_best_orbital_library.py):
    #   - x_orb / y_orb / z_orb are *lists* of (n_time,) arrays, length
    #     n_orb_total = n_orbits_nnls * 4 (the 4 bar-symmetry copies are
    #     already expanded into separate orbits).
    #   - `weights` is also of length n_orb_total (np.repeat by 4).
    #   - Trajectories are interpolated onto a uniform time grid (1000
    #     steps each).
    #   - `mean_mass_per_orbit` saved here is `MMPO_orig / n_time`, i.e.
    #     it is already the *per-sample* mass; no extra 1/n_time needed.
    print(f"Loading orbital library from {ORBLIB_FILE}")
    with open(ORBLIB_FILE, 'rb') as f:
        lib = pickle.load(f)

    w_orb = np.asarray(lib['weights'])              # (n_orb_total,)
    x_orb_list = lib['x_orb']                       # list of (n_time,)
    y_orb_list = lib['y_orb']
    z_orb_list = lib['z_orb']
    mass_per_sample = float(np.asarray(lib['mean_mass_per_orbit']).item()
                            if np.ndim(lib['mean_mass_per_orbit']) else
                            float(lib['mean_mass_per_orbit']))

    n_orb_total = len(x_orb_list)
    n_time = len(x_orb_list[0])
    assert w_orb.shape[0] == n_orb_total, \
        f"weights {w_orb.shape} vs orbits {n_orb_total}"

    print(f"Orbital library: {n_orb_total} orbits (incl. 4× sym) × {n_time} timesteps, "
          f"per-sample mass = {mass_per_sample:.3e} Msun")

    x_flat = np.concatenate([np.asarray(a, dtype=np.float32) for a in x_orb_list])
    y_flat = np.concatenate([np.asarray(a, dtype=np.float32) for a in y_orb_list])
    z_flat = np.concatenate([np.asarray(a, dtype=np.float32) for a in z_orb_list])
    pos_flat = np.column_stack([x_flat, y_flat, z_flat])

    # Each sample carries mass = weight_i * mass_per_sample
    mass_flat = (np.repeat(w_orb, n_time) * mass_per_sample).astype(np.float64)

    print(f"Total mass in orbital library:  {mass_flat.sum():.3e} Msun")
    print(f"Total disc mass in N-body data: {mass_data[mask_disc].sum():.3e} Msun")
    print(f"Density-model baryonic mass at r=30 kpc (median): {disc_50[-1]:.3e} Msun")

    M_disc_orblib  = particles_enclosed_mass(pos_flat, mass_flat, r_plot)
    M_total_orblib = halo_50 + M_disc_orblib   # halo from density model

    # ─────────────────────── Plot ───────────────────────
    # Two rows: top = M(<r) absolute; bottom = log10(M_data) - log10(M_model)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.7),
                             sharex='col',
                             gridspec_kw={'height_ratios': [3, 1.4]})
    xlim = (r_plot[0], r_plot[-1])
    ylim = (1e8, 1e12)

    def log_resid(M_data, M_model):
        """log10(data) - log10(model), masked where either is non-positive."""
        ok = (M_data > 0) & (M_model > 0)
        out = np.full_like(M_data, np.nan, dtype=float)
        out[ok] = np.log10(M_data[ok]) - np.log10(M_model[ok])
        return out

    # ── Top row: absolute M(<r) ──
    # Halo
    ax = axes[0, 0]
    ax.plot(r_plot, M_halo_data, lw=3, color='royalblue', label='N-body (ground truth)')
    ax.fill_between(r_plot, halo_16, halo_84, color='tomato', alpha=0.25,
                    label=r'Density model 1$\sigma$')
    ax.plot(r_plot, halo_50, lw=2, ls='--', color='tomato', label='Density model median')
    ax.set_ylabel(r'$M(<r)$ [$M_\odot$]')
    ax.set_title('Dark matter halo')
    ax.set_yscale('log')
    ax.axvline(10., color='grey', ls='--', lw=1, alpha=0.7, label='Data extent')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # ax.legend(frameon=False, fontsize=10)

    # Baryonic
    ax = axes[0, 1]
    ax.plot(r_plot, M_disc_data, lw=3, color='royalblue', label='N-body (ground truth)')
    ax.fill_between(r_plot, disc_16, disc_84, color='tomato', alpha=0.25,
                    label=r'Density model 1$\sigma$')
    ax.plot(r_plot, disc_50, lw=2, ls='--', color='tomato', label='Density model median')
    ax.plot(r_plot, M_disc_orblib, lw=4, ls=':', color='seagreen', label='Orbital library')
    ax.set_title('Baryonic (Disc + Bar + Bulge)')
    ax.set_yscale('log')
    ax.axvline(10., color='grey', ls='--', lw=1, alpha=0.7, label='Data extent')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)   
    ax.legend(frameon=True, fontsize=12)

    # Total
    ax = axes[0, 2]
    ax.plot(r_plot, M_total_data, lw=3, color='royalblue', label='N-body (ground truth)')
    ax.fill_between(r_plot, total_16, total_84, color='tomato', alpha=0.25,
                    label=r'Density model 1$\sigma$')
    ax.plot(r_plot, total_50, lw=2, ls='--', color='tomato', label='Density model median')
    ax.plot(r_plot, M_total_orblib, lw=4, ls=':', color='seagreen',
            label='Halo (model) + Orb. lib.')
    ax.set_title('Total')
    ax.set_yscale('log')
    ax.axvline(10., color='grey', ls='--', lw=1, alpha=0.7, label='Data extent')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # ax.legend(frameon=False, fontsize=10)

    # ── Bottom row: log-residuals (data - model) ──
    # The 1-sigma band on log-resid is (log_resid using p84, log_resid using p16):
    # if model is larger -> residual is smaller (and vice versa), so swap order.
    # Halo
    ax = axes[1, 0]
    ax.axhline(0., color='black', lw=0.8)
    ax.fill_between(r_plot,
                    log_resid(M_halo_data, halo_84),
                    log_resid(M_halo_data, halo_16),
                    color='tomato', alpha=0.25)
    ax.plot(r_plot, log_resid(M_halo_data, halo_50),
            lw=2, ls='--', color='tomato', label='vs density model')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel(r'$\log_{10}\,M_{\rm data} - \log_{10}\,M_{\rm model}$')
    ax.set_xscale('log'); ax.set(xlim=xlim)
    ax.axvline(10., color='grey', ls='--', lw=1, alpha=0.7)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    # Baryonic
    ax = axes[1, 1]
    ax.axhline(0., color='black', lw=0.8)
    ax.fill_between(r_plot,
                    log_resid(M_disc_data, disc_84),
                    log_resid(M_disc_data, disc_16),
                    color='tomato', alpha=0.25)
    ax.plot(r_plot, log_resid(M_disc_data, disc_50),
            lw=2, ls='--', color='tomato', label='vs density model')
    ax.plot(r_plot, log_resid(M_disc_data, M_disc_orblib),
            lw=4, ls=':',  color='seagreen', label='vs orbital library')
    ax.set_xlabel('r [kpc]')
    ax.set_xscale('log'); ax.set(xlim=xlim)
    ax.axvline(10., color='grey', ls='--', lw=1, alpha=0.7)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # ax.legend(frameon=False, fontsize=9)

    # Total
    M_total_orblib_lo = halo_16 + M_disc_orblib
    M_total_orblib_hi = halo_84 + M_disc_orblib
    ax = axes[1, 2]
    ax.axhline(0., color='black', lw=0.8)
    ax.fill_between(r_plot,
                    log_resid(M_total_data, total_84),
                    log_resid(M_total_data, total_16),
                    color='tomato', alpha=0.25)
    ax.plot(r_plot, log_resid(M_total_data, total_50),
            lw=2, ls='--', color='tomato', label='vs density model')
    ax.plot(r_plot, log_resid(M_total_data, M_total_orblib),
            lw=4, ls=':',  color='seagreen',
            label='vs halo+orb.lib.')
    ax.set_xlabel('r [kpc]')
    ax.set_xscale('log'); ax.set(xlim=xlim)
    ax.axvline(10., color='grey', ls='--', lw=1, alpha=0.7)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # ax.legend(frameon=False, fontsize=9)

    # Common y-range for the residual row, symmetric around 0
    all_resid = np.concatenate([
        log_resid(M_halo_data,  halo_50),
        log_resid(M_disc_data,  disc_50),
        log_resid(M_disc_data,  M_disc_orblib),
        log_resid(M_total_data, total_50),
        log_resid(M_total_data, M_total_orblib),
    ])
    res_max = np.nanmax(np.abs(all_resid))
    pad = 0.05
    for ax_ in axes[1, :]:
        ax_.set_ylim([-0.5, 0.5])

    fig.tight_layout()
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    fig.savefig(figname_paper, dpi=300, bbox_inches='tight')
    print(f"Saved to {figname}")
    plt.show()
