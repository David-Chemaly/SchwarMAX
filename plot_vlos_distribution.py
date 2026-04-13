"""
Plot the mean radial velocity (v_R) field in the face-on (x-y) plane,
comparing model (from best-fit orbital library) to N-body data.

The v_R quadrupole pattern is a classic bar signature visible in face-on views.

Usage:
    python plot_radial_velocity_field.py
"""

import agama
from likelihoods_bar import get_dict_data_bootstrap
agama.setUnits(mass=1, length=1, velocity=1)
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
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


# ═══════════════════════════════════════════════════════════════════
#  2D binned mean v_R
# ═══════════════════════════════════════════════════════════════════

def vlos_distribution(X, Y, vlos, weight, vlos_edges, X_edges, Y_edges):
    """
    Compute weighted mean radial velocity on a 2D (x, y) grid.
    v_R = (x * vx + y * vy) / R
    """
    
    mask = (X>X_edges[0]) & (X<X_edges[-1]) & (Y>Y_edges[0]) & (Y<Y_edges[-1])
    vlos = vlos[mask]
    weight = weight[mask]

    sum_wvlos, _ = np.histogram(vlos, bins=vlos_edges, weights=weight)
    sum_w = np.sum(weight)

    vlos_norm = sum_wvlos / sum_w

    return vlos_norm


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    figname = data_folder + '/plots/radial_velocity_field.png'
    orbital_library_file = data_folder + '/best_fit_orbital_library.pkl'

    X_edges = [-5.0, -4.0]
    Y_edges = [0.5,  1.5]
    vlos_edges = np.linspace(-100, 300, 81)
    
    print("Loading orbital library...")
    with open(orbital_library_file, 'rb') as f:
        lib = pickle.load(f)
    rot_mat = lib['rotation_matrix'] 

    # data_filename = 'mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma170.pkl'\
    data_filename = 'mock_Nbody_bar_XY_withRot_gal2_Nbins1000.pkl'
    dict_data = get_dict_data_bootstrap(path, data_filename, n_samples = 5_000)

    # ── Load N-body snapshot ──
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s)**2))
    w0_data, mass_data = agama.readSnapshot(
        data_folder + '/Bar_model_TG21/model/t_t0_4')
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

    # Rotate the N-body data
    x_data, v_data = w0_data[:,:3], w0_data[:,3:]
    x_data = (rot_mat @ x_data.T).T
    v_data = (rot_mat @ v_data.T).T
    w0_data[:,:3] = x_data
    w0_data[:,3:] = v_data

    # ── N-body v_R field (disc particles, |z| < 1 kpc for thin disc) ──
    z_cut = np.abs(w0_data[mask_disc, 2]) < 5.0
    posvel_disc = w0_data[mask_disc][z_cut]
    m_disc = mass_data[mask_disc][z_cut]
    vlos_nbody = vlos_distribution(
        posvel_disc[:, 0], posvel_disc[:, 2],
        posvel_disc[:, 4],  m_disc, 
        vlos_edges, X_edges, Y_edges)
    
    # ── Model v_R field ──

    w_model = lib['weights']
    x_model = np.array(lib['x_orb'])  # list of (n_time, 3) arrays
    y_model = np.array(lib['y_orb'])
    z_model = np.array(lib['z_orb'])
    vx_model = np.array(lib['vx_orb'])
    vy_model = np.array(lib['vy_orb'])
    vz_model = np.array(lib['vz_orb'])
    vx_model = vx_model * KPCGYR_TO_KMS
    vy_model = vy_model * KPCGYR_TO_KMS
    vz_model = vz_model * KPCGYR_TO_KMS

    w_model = np.repeat(w_model, x_model.shape[1])  # repeat weights for each time step
    w_unity = np.ones_like(vx_model.flatten())  # uniform weights for plotting

    x_model = x_model.flatten()
    y_model = y_model.flatten()
    z_model = z_model.flatten()
    vx_model = vx_model.flatten()
    vy_model = vy_model.flatten()
    vz_model = vz_model.flatten()

    pos_model = np.array([x_model, y_model, z_model]).T
    vel_model = np.array([vx_model, vy_model, vz_model]).T
    pos_model = (rot_mat @ pos_model.T).T
    vel_model = (rot_mat @ vel_model.T).T
    x_model, y_model, z_model = pos_model.T
    vx_model, vy_model, vz_model = vel_model.T
    posvel_model = np.array([x_model, y_model, z_model, vx_model, vy_model, vz_model]).T
    vlos_model = vlos_distribution(
        posvel_model[:, 0], posvel_model[:, 2],
        posvel_model[:, 4],  w_model, 
        vlos_edges, X_edges, Y_edges)
    
    vlos_unity = vlos_distribution(
        posvel_model[:, 0], posvel_model[:, 2],
        posvel_model[:, 4],  w_unity, 
        vlos_edges, X_edges, Y_edges)


    # ── Plot ──

    fig, ax = plt.subplots(1, 3, figsize=(30, 3), gridspec_kw={'width_ratios': [1, 3, 1]})

    ax[0].plot(vlos_edges[:-1], vlos_nbody, label='N-body', drawstyle='steps-mid')
    ax[0].plot(vlos_edges[:-1], vlos_model, label='Model (best fit)', drawstyle='steps-mid')
    ax[0].plot(vlos_edges[:-1], vlos_unity, label='Model (uniform weights)', drawstyle='steps-mid')
    ax[0].set_xlabel(r'$v_{\rm los}$ [km/s]')
    ax[0].set_ylabel(r'Weighted counts')
    ax[0].set_title('Line-of-sight velocity distribution')
    # ax[0].legend()

    surface_density = dict_data['XY_density_data']
    X_regular_grid = dict_data['X_regular_grid']
    Y_regular_grid = dict_data['Y_regular_grid']
    bin_mapping = dict_data['bin_mapping']
    index_remap = bin_mapping[:-1]
    density_2DXY_data = surface_density[index_remap]

    cb = ax[1].scatter(X_regular_grid, Y_regular_grid, c=density_2DXY_data,
                    s = 22, cmap=cmr.sepia, marker = 's', norm = 'log',
                    vmin = 1e1, vmax = 1e4, rasterized = True)
    ax[1].set_title(f'Data', fontsize=15)
    ax[1].set_xlabel('X [kpc]', fontsize=12)
    ax[1].set_ylabel('Y [kpc]', fontsize=12)
    ax[1].set_xlim(-10, 10)
    ax[1].set_ylim(-3, 3)
    
    # Draw a box to indicate the region where vlos distribution is computed
    rect = plt.Rectangle((X_edges[0], Y_edges[0]), X_edges[1]-X_edges[0], Y_edges[1]-Y_edges[0],
                         edgecolor='red', facecolor='none', linestyle='--')
    ax[1].add_patch(rect)


    # cbar = fig1.colorbar(cb, ax=ax1[i][1])
    # cbar.set_label(fig_names[i], fontsize=18)
    # cbar.ax.tick_params(labelsize=14)

    plt.show()
