"""
Plot the mean radial velocity (v_R) field in the face-on (x-y) plane,
comparing the fixed-potential model orbital library to the TG21 N-body snapshot
it was fit against.

Reads the orbital library produced by get_best_orbital_library_fixed_potential.py.

Usage:
    python plot_radial_velocity_field_fixed_potential.py
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


def binned_mean_vR(x, y, vx, vy, weight, x_edges, y_edges):
    R = np.sqrt(x**2 + y**2)
    mask = R > 0.01
    x, y, vx, vy, weight, R = x[mask], y[mask], vx[mask], vy[mask], weight[mask], R[mask]
    vR = (x * vx + y * vy) / R

    sum_wvR, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=weight * vR)
    sum_w, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=weight)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_vR = np.where(sum_w > 0, sum_wvR / sum_w, np.nan)
    return mean_vR


if __name__ == '__main__':

    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    figname = data_folder + '/plots/radial_velocity_field_fixed_potential_t_t0_7.png'
    orbital_library_file = data_folder + '/best_fit_orbital_library_fixed_potential_t_t0_7.pkl'
    nbody_snapshot = data_folder + '/Bar_model_TG21/model/t_t0_7'

    # 2D grid for binning
    lim = 10.0
    n_bins = 30
    x_edges = np.linspace(-lim, lim, n_bins + 1)
    y_edges = np.linspace(-lim, lim, n_bins + 1)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])

    # ── Load N-body snapshot ──
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s)**2))
    w0_data, mass_data = agama.readSnapshot(nbody_snapshot)
    mass_data = mass_data * mass_unit.value

    unique_masses = np.unique(mass_data)
    mask_halo = mass_data == unique_masses[-1]
    mask_disc = ~mask_halo

    for r_ap in [10.0, 5.0, 3.0, 2.0]:
        R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
        mask_center = mask_disc & (R < r_ap)
        m_c = mass_data[mask_center]
        for col in range(6):
            w0_data[:, col] -= np.sum(w0_data[mask_center, col] * m_c) / np.sum(m_c)

    w0_data[:, 0] = -w0_data[:, 0]
    w0_data[:, 3] = -w0_data[:, 3]

    R_mid_ba, bar_angles0, _ = bar_angle_bar_strength(
        w0_data[:, 0], w0_data[:, 1], R_anulus=np.arange(1, 5, 0.1))
    bar_angle0 = np.mean(bar_angles0[R_mid_ba < 4])
    w0_data = rotate(w0_data, -bar_angle0)

    # ── N-body v_R field (disc particles, |z| < 5 kpc) ──
    z_cut = np.abs(w0_data[mask_disc, 2]) < 5.0
    pos_disc = w0_data[mask_disc][z_cut]
    m_disc = mass_data[mask_disc][z_cut]
    vR_nbody = binned_mean_vR(
        pos_disc[:, 0], pos_disc[:, 1],
        pos_disc[:, 3], pos_disc[:, 4],
        m_disc, x_edges, y_edges)

    # ── Model v_R field (fixed-potential orbital library) ──
    print('Loading orbital library:', orbital_library_file)
    with open(orbital_library_file, 'rb') as f:
        lib = pickle.load(f)

    w_model = lib['weights']
    x_model = np.array(lib['x_orb'])
    y_model = np.array(lib['y_orb'])
    vx_model = np.array(lib['vx_orb'])
    vy_model = np.array(lib['vy_orb'])
    vx_model = vx_model * KPCGYR_TO_KMS
    vy_model = vy_model * KPCGYR_TO_KMS

    w_model = np.repeat(w_model, x_model.shape[1])

    x_model = x_model.flatten()
    y_model = y_model.flatten()
    vx_model = vx_model.flatten()
    vy_model = vy_model.flatten()

    vR_model = binned_mean_vR(x_model, y_model, vx_model, vy_model, w_model,
                              x_edges, y_edges)

    # ── Plot v_R field comparison ──
    vmax = np.nanpercentile(np.abs(vR_nbody), 95)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.pcolormesh(x_mid, y_mid, vR_nbody.T, cmap='coolwarm',
                       vmin=-vmax, vmax=vmax, shading='auto')
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_title('N-body (t_t0_7)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$\langle v_R \rangle$ [km/s]')

    ax = axes[1]
    im = ax.pcolormesh(x_mid, y_mid, vR_model.T, cmap='coolwarm',
                       vmin=-vmax, vmax=vmax, shading='auto')
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_title('Model (fixed-potential best fit)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$\langle v_R \rangle$ [km/s]')

    fig.suptitle(r'Face-on mean $v_R$ field — fixed-potential fit vs N-body', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    print(f'Saved v_R field comparison to {figname}')

    # ── Face-on surface density of the orbital library ──
    figname2 = data_folder + '/plots/radial_velocity_field_fixed_potential_t_t0_7_density.png'
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    H, xedge, yedge, _ = ax2.hist2d(x_model, y_model, bins=100, range=[[-6, 6], [-6, 6]],
                                    weights=w_model, cmap='viridis', norm='log')
    xmid2, ymid2 = 0.5 * (xedge[1:] + xedge[:-1]), 0.5 * (yedge[1:] + yedge[:-1])
    ax2.contour(xmid2, ymid2, np.log10(H).T, levels=10, colors='white', linewidths=0.5)
    ax2.set_xlabel('x [kpc]')
    ax2.set_ylabel('y [kpc]')
    ax2.set_title('Model: face-on surface density (weighted)')
    ax2.set_aspect('equal')
    fig2.tight_layout()
    fig2.savefig(figname2, dpi=300, bbox_inches='tight')
    print(f'Saved density map to {figname2}')

    # ── Bar strength comparison ──
    figname3 = data_folder + '/plots/radial_velocity_field_fixed_potential_t_t0_7_bar_strength.png'
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    R_mid_m, _, bar_strength_m = bar_angle_bar_strength_w(
        x_model, y_model, w_model, R_anulus=np.arange(1, 5, 0.1))
    R_mid_n, _, bar_strength_n = bar_angle_bar_strength_w(
        pos_disc[:, 0], pos_disc[:, 1], m_disc, R_anulus=np.arange(1, 5, 0.1))
    ax3.plot(R_mid_m, bar_strength_m, label='Model (fixed-potential)')
    ax3.plot(R_mid_n, bar_strength_n, label='N-body', linestyle='--')
    ax3.set_xlabel('R [kpc]')
    ax3.set_ylabel('Bar strength |A2|')
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(figname3, dpi=300, bbox_inches='tight')
    print(f'Saved bar strength to {figname3}')

    plt.show()
