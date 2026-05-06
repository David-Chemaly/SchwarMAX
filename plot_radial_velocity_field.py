"""
Plot the mean radial velocity (v_R) field in the face-on (x-y) plane,
comparing model (from best-fit orbital library) to N-body data.

The v_R quadrupole pattern is a classic bar signature visible in face-on views.

Usage:
    python plot_radial_velocity_field.py
"""

import agama
agama.setUnits(mass=1, length=1, velocity=1)
import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy import units as u
from astropy.constants import G
from constants import KPCGYR_TO_KMS


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


# ═══════════════════════════════════════════════════════════════════
#  N-body helpers
# ═══════════════════════════════════════════════════════════════════

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

def bar_angle_bar_strength_w(x, y, w, R_anulus=np.arange(1, 5, 0.25)):
    R0, phi0 = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    bar_angles0, bar_strength0 = [], []
    for i in range(len(R_anulus) - 1):
        sel = (R0 > R_anulus[i]) & (R0 < R_anulus[i + 1])
        phi_bin = phi0[sel]
        w_bin = w[sel]
        A2 = np.sum(w_bin * np.cos(2 * phi_bin)) / np.sum(w_bin)
        B2 = np.sum(w_bin * np.sin(2 * phi_bin)) / np.sum(w_bin)
        bar_angles0.append(0.5 * np.arctan2(B2, A2))
        bar_strength0.append(np.sqrt(A2**2 + B2**2))
    R_mid = (R_anulus[:-1] + R_anulus[1:]) / 2
    return R_mid, np.array(bar_angles0), np.array(bar_strength0)


# ═══════════════════════════════════════════════════════════════════
#  2D binned mean v_R
# ═══════════════════════════════════════════════════════════════════

def binned_mean_vR(x, y, vx, vy, weight, x_edges, y_edges):
    """
    Compute weighted mean radial velocity on a 2D (x, y) grid.
    v_R = (x * vx + y * vy) / R
    """
    R = np.sqrt(x**2 + y**2)
    mask = R > 0.01
    x, y, vx, vy, weight, R = x[mask], y[mask], vx[mask], vy[mask], weight[mask], R[mask]
    vR = (x * vx + y * vy) / R
    vphi = (x * vy - y * vx) / R

    sum_wvR, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=weight * vR)
    sum_w, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=weight)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_vR = np.where(sum_w > 0, sum_wvR / sum_w, np.nan)
    return mean_vR


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    figname = data_folder + '/plots/radial_velocity_field.png'
    figname_density = data_folder + '/plots/orbit_vs_data_density_field.png'
    # orbital_library_file = data_folder + '/best_fit_orbital_library.pkl'
    # orbital_library_file = data_folder + '/best_fit_orbital_library_0418_beta25_gamma140_D50_gal2_fixedbarlength.pkl'
    orbital_library_file = data_folder + '/best_fit_orbital_library_0415_beta25_gamma140_D50_gal2.pkl'

    # 2D grid for binning
    lim = 10.0
    n_bins = 100
    x_edges = np.linspace(-lim, lim, n_bins + 1)
    y_edges = np.linspace(-lim, lim, n_bins + 1)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])

    # ── Load N-body snapshot ──
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s)**2))
    w0_data, mass_data = agama.readSnapshot(
        data_folder + '/Bar_model_TG21/model/t_t0_7')
    mass_data = mass_data * mass_unit.value

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

    # Align bar with x-axis
    R_mid_ba, bar_angles0, _ = bar_angle_bar_strength(
        w0_data[:, 0], w0_data[:, 1], R_anulus=np.arange(1, 5, 0.1))
    bar_angle0 = np.mean(bar_angles0[R_mid_ba < 4])
    w0_data = rotate(w0_data, -bar_angle0)

    # ── N-body v_R field (disc particles, |z| < 1 kpc for thin disc) ──
    z_cut = np.abs(w0_data[mask_disc, 2]) < 5.0
    pos_disc = w0_data[mask_disc][z_cut]
    m_disc = mass_data[mask_disc][z_cut]
    vR_nbody = binned_mean_vR(
        pos_disc[:, 0], pos_disc[:, 1],
        pos_disc[:, 3], pos_disc[:, 4],
        m_disc, x_edges, y_edges)

    # ── Model v_R field ──
    print("Loading orbital library...")
    with open(orbital_library_file, 'rb') as f:
        lib = pickle.load(f)


    w_model = lib['weights']
    x_model = np.array(lib['x_orb'])  # list of (n_time, 3) arrays
    y_model = np.array(lib['y_orb'])
    z_model = np.array(lib['z_orb'])
    vx_model = np.array(lib['vx_orb'])
    vy_model = np.array(lib['vy_orb'])
    vz_model = np.array(lib['vz_orb'])
    vx_model = vx_model * KPCGYR_TO_KMS
    vy_model = vy_model * KPCGYR_TO_KMS
    mean_mass_per_orbit = lib['mean_mass_per_orbit']  # convert to mass/time

    w_model = np.repeat(w_model, x_model.shape[1])  # repeat weights for each time step
    # w_model = np.ones_like(vx_model)  # uniform weights for plotting

    x_model = x_model.flatten()
    y_model = y_model.flatten()
    vx_model = vx_model.flatten()
    vy_model = vy_model.flatten()
    mass_particle = np.ones_like(w_model) * mean_mass_per_orbit  # assign mass to each particle

    vR_model = binned_mean_vR(x_model, y_model, vx_model, vy_model, w_model,
                              x_edges, y_edges)

    # ── Plot ──
    vmax = np.nanpercentile(np.abs(vR_nbody), 95)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.pcolormesh(x_mid, y_mid, vR_nbody.T, cmap='coolwarm',
                       vmin=-vmax, vmax=vmax, shading='auto')
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_title('N-body')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$\langle v_R \rangle$ [km/s]')

    ax = axes[1]
    im = ax.pcolormesh(x_mid, y_mid, vR_model.T, cmap='coolwarm',
                       vmin=-vmax, vmax=vmax, shading='auto')
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_title('Model (best fit)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$\langle v_R \rangle$ [km/s]')

    fig.suptitle(r'Face-on mean $v_R$ field', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    print(f"Saved to {figname}")


    fig, ax2 = plt.subplots(1, 2, figsize=(12, 3.5))
    H, xedge, yedge, cb = ax2[0].hist2d(pos_disc[:,0], pos_disc[:,1], bins=100, range = [[-10,10], [-10,10]], 
                                     weights=m_disc, cmap='viridis', norm = 'log', vmin = 1e6, vmax = 1e9)
    xmid, ymid = 0.5*(xedge[1:]+xedge[:-1]), 0.5*(yedge[1:]+yedge[:-1])
    ax2[0].contour(xmid, ymid, np.log10(H).T, levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cb, ax=ax2[0])

    H, xedge, yedge, cb = ax2[1].hist2d(x_model, y_model, bins=100, range = [[-10,10], [-10,10]], 
                                     weights=w_model*mass_particle, cmap='viridis', norm = 'log', vmin = 1e6, vmax = 1e9)
    xmid, ymid = 0.5*(xedge[1:]+xedge[:-1]), 0.5*(yedge[1:]+yedge[:-1])
    ax2[1].contour(xmid, ymid, np.log10(H).T, levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cb, ax=ax2[1])

    fig.suptitle(r'Face-on density field', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figname_density, dpi=300, bbox_inches='tight')
    print(f"Saved to {figname_density}")

    fig, ax3 = plt.subplots(figsize=(6, 5))
    R_mid, bar_angles0, bar_strength0 = bar_angle_bar_strength_w(x_model, y_model, w_model, R_anulus=np.arange(1, 5, 0.1))
    ax3.plot(R_mid, bar_strength0, label='Bar strength')
    plt.show()
