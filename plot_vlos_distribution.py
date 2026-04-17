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
import scipy as sp
import matplotlib.pyplot as plt
import cmasher as cmr
import pickle
from astropy import units as u
from astropy.constants import G
from constants import KPCGYR_TO_KMS


def plot_prettier(dpi=200, fontsize=18, usetex=False):
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

def hist_mean_std(vlos_edges, hist):
    """Compute mean and std from a normalized histogram."""
    v_mid = 0.5 * (vlos_edges[:-1] + vlos_edges[1:])
    dv = vlos_edges[1] - vlos_edges[0]
    norm = np.sum(hist) * dv
    if norm == 0:
        return np.nan, np.nan
    mean = np.sum(v_mid * hist * dv) / norm
    std = np.sqrt(np.sum((v_mid - mean)**2 * hist * dv) / norm)
    return mean, std

def weighted_kde(vlos, weight, v_eval, bw_method='scott'):
    """Compute weighted KDE from raw samples and evaluate on v_eval grid."""
    w = weight / weight.sum()
    kde = sp.stats.gaussian_kde(vlos, bw_method=bw_method, weights=w)
    return kde(v_eval)

def select_in_box(X, Y, vlos, weight, X_edges, Y_edges):
    """Select particles within a spatial box, return vlos and weights."""
    mask = (X > X_edges[0]) & (X < X_edges[-1]) & (Y > Y_edges[0]) & (Y < Y_edges[-1])
    return vlos[mask], weight[mask]

def vlos_distribution(X, Y, vlos, weight, vlos_edges, X_edges, Y_edges):
    """Compute normalized histogram of vlos in a spatial box."""
    v, w = select_in_box(X, Y, vlos, weight, X_edges, Y_edges)
    sum_wvlos, _ = np.histogram(v, bins=vlos_edges, weights=w)
    sum_w = np.sum(w)
    return sum_wvlos / sum_w

def get_orb_samples(lib, X_edges, Y_edges):
    """Get vlos samples and weights from orbital library within a spatial box."""
    w_model = lib['weights']
    x_model = np.array(lib['x_orb'])
    y_model = np.array(lib['y_orb'])
    z_model = np.array(lib['z_orb'])
    vx_model = np.array(lib['vx_orb'])
    vy_model = np.array(lib['vy_orb'])
    vz_model = np.array(lib['vz_orb'])
    vx_model = vx_model * KPCGYR_TO_KMS
    vy_model = vy_model * KPCGYR_TO_KMS
    vz_model = vz_model * KPCGYR_TO_KMS

    w_model = np.repeat(w_model, x_model.shape[1])

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
    x_rot, _, z_rot = pos_model.T
    _, vy_rot, _ = vel_model.T

    # Select particles in spatial box (X=x_rot, Y=z_rot, vlos=vy_rot)
    vlos_sel, w_sel = select_in_box(x_rot, z_rot, vy_rot, w_model, X_edges, Y_edges)
    return vlos_sel, w_sel


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    fig, ax = plt.subplots(1, 5, figsize=(21, 4),
                           gridspec_kw={'width_ratios': [1, 1, 2, 1, 1], 'wspace': 0.05})

    STAT_FONTSIZE = 13

    path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    figname = data_folder + '/plots/vlos_distribution.png'
    orbital_library_file = data_folder + '/best_fit_orbital_library.pkl'
    print("Loading orbital library...")
    with open(orbital_library_file, 'rb') as f:
        lib = pickle.load(f)
    rot_mat = lib['rotation_matrix'] 

    orbital_library_file = data_folder + '/best_fit_orbital_library_Omega10.pkl'
    with open(orbital_library_file, 'rb') as f:
        lib_10 = pickle.load(f)

    orbital_library_file = data_folder + '/best_fit_orbital_library_Omega40.pkl'
    with open(orbital_library_file, 'rb') as f:
        lib_40 = pickle.load(f)

    # data_filename = 'mock_Nbody_bar_XY_withRot_gal2_Nbins1000.pkl'
    data_filename = 'mock_data/mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma140_D50_gal2.pkl'
    dict_data = get_dict_data_bootstrap(path, data_filename, n_samples = 5_000)
    X_minmax = dict_data['X_minmax']
    Y_minmax = dict_data['Y_minmax']
    surface_density = dict_data['XY_density_data']
    X_regular_grid = dict_data['X_regular_grid']
    Y_regular_grid = dict_data['Y_regular_grid']
    bin_mapping = dict_data['bin_mapping']
    index_remap = bin_mapping[:-1]
    density_2DXY_data = surface_density[index_remap]
    cb = ax[2].scatter(X_regular_grid, Y_regular_grid, c=density_2DXY_data,
                    s = 40, cmap=cmr.sepia, marker = 's', norm = 'log',
                    vmin = 1e1, vmax = 1e4, rasterized = True)
    ax[2].set_xlim(X_minmax)
    ax[2].set_ylim(Y_minmax)
    # Draw a box to indicate the region where vlos distribution is computed

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

    # Rotate the N-body data
    x_data, v_data = w0_data[:,:3], w0_data[:,3:]
    x_data = (rot_mat @ x_data.T).T
    v_data = (rot_mat @ v_data.T).T
    w0_data[:,:3] = x_data
    w0_data[:,3:] = v_data

    x, y = w0_data[:, 0], w0_data[:, 1]
    R, phi = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2), np.arctan2(w0_data[:, 1], w0_data[:, 0])
    z, vy = w0_data[:, 2], w0_data[:, 4]
    XY_stars = np.stack([x, z], axis=-1) # prefect edge-on view

    # Plot contours of the 2D histogram of (x, z) for disc particles
    X_min, X_max = -12., 12.
    Y_min, Y_max = -4., 4.
    nX, nY = 60,40
    area_XY = ((X_max - X_min)/nX) * ((Y_max - Y_min)/nY)
    X_edge = np.linspace(X_min, X_max, nX+1)
    Y_edge = np.linspace(Y_min, Y_max, nY+1)
    X_mids, Y_mids = 0.5 * (X_edge[:-1] + X_edge[1:]), 0.5 * (Y_edge[:-1] + Y_edge[1:])
    H, xedge, yedge = np.histogram2d(XY_stars[:, 0], XY_stars[:, 1], bins=[X_edge, Y_edge])
    signal = H.flatten()
    noise = np.sqrt(signal + 1)
    xmid, ymid = 0.5 * (xedge[1:] + xedge[:-1]), 0.5 * (yedge[1:] + yedge[:-1])
    H = sp.ndimage.gaussian_filter(H, sigma=0.5)
    for j in range (0,3):
        ax[2].contour(xmid, ymid, np.log10(H).T, levels=[2.2, 2.8, 3.1, 3.5, 4.], colors='grey', linewidths=1)


    X_edges = [-3.5, -2.5]
    Y_edges = [ 0.5,  1.5]
    vlos_edges = np.linspace(-100, 300, 51)
    # ── N-body v_R field (disc particles, |z| < 1 kpc for thin disc) ──
    z_cut = np.abs(w0_data[mask_disc, 2]) < 5.0
    posvel_disc = w0_data[mask_disc][z_cut]
    m_disc = mass_data[mask_disc][z_cut]
    vlos_nbody = vlos_distribution(
        posvel_disc[:, 0], posvel_disc[:, 2],
        posvel_disc[:, 4],  m_disc, 
        vlos_edges, X_edges, Y_edges)

    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    colors = ['C0', 'black', 'orange', 'purple']
    def _colored_stat_line(target_ax, y_anchor, label, vals, colors):
        fs = STAT_FONTSIZE
        parts = [TextArea(f'{label} = (', textprops=dict(fontsize=fs))]
        for i, v in enumerate(vals):
            parts.append(TextArea(f'{v:.0f}', textprops=dict(fontsize=fs, color=colors[i])))#, fontweight='bold'
            if i < len(vals) - 1:
                parts.append(TextArea(', ', textprops=dict(fontsize=fs)))
        parts.append(TextArea(') km/s', textprops=dict(fontsize=fs)))
        box = HPacker(children=parts, pad=0, sep=0, align='baseline')
        ab = AnchoredOffsetbox(loc='upper left', child=box, frameon=False,
                               bbox_to_anchor=(0.02, y_anchor), bbox_transform=target_ax.transAxes)
        target_ax.add_artist(ab)

    def plot_vlos_panel(panel_ax, X_edges, Y_edges, vlos_range, rect_color):
        """Plot weighted KDE of vlos for one spatial box on a given axis."""
        v_eval = np.linspace(vlos_range[0], vlos_range[1], 200)

        # N-body KDE
        z_cut = np.abs(w0_data[mask_disc, 2]) < 5.0
        posvel_disc = w0_data[mask_disc][z_cut]
        m_disc = mass_data[mask_disc][z_cut]
        vlos_nb, w_nb = select_in_box(posvel_disc[:, 0], posvel_disc[:, 2],
                                       posvel_disc[:, 4], m_disc, X_edges, Y_edges)
        kde_nb = weighted_kde(vlos_nb, w_nb, v_eval)
        panel_ax.fill_between(v_eval, 0, kde_nb, label='N-body', alpha=0.3, color='C0')
        panel_ax.plot(v_eval, kde_nb, color='C0', lw=1)
        mu_nb = np.average(vlos_nb, weights=w_nb)
        sig_nb = np.sqrt(np.average((vlos_nb - mu_nb)**2, weights=w_nb))

        # Best-fit model KDE
        vlos_bf, w_bf = get_orb_samples(lib, X_edges, Y_edges)
        kde_bf = weighted_kde(vlos_bf, w_bf, v_eval)
        panel_ax.plot(v_eval, kde_bf, label='Model (best fit)', color='black', lw=3)
        mu_bf = np.average(vlos_bf, weights=w_bf)
        sig_bf = np.sqrt(np.average((vlos_bf - mu_bf)**2, weights=w_bf))

        # Omega=10 model KDE
        vlos_o10, w_o10 = get_orb_samples(lib_10, X_edges, Y_edges)
        kde_10 = weighted_kde(vlos_o10, w_o10, v_eval)
        panel_ax.plot(v_eval, kde_10, label='Model (Omega=10)', color='orange', lw=1.5)
        mu_10 = np.average(vlos_o10, weights=w_o10)
        sig_10 = np.sqrt(np.average((vlos_o10 - mu_10)**2, weights=w_o10))

        # Omega=40 model KDE
        vlos_o40, w_o40 = get_orb_samples(lib_40, X_edges, Y_edges)
        kde_40 = weighted_kde(vlos_o40, w_o40, v_eval)
        panel_ax.plot(v_eval, kde_40, label='Model (Omega=40)', color='purple', lw=1.5)
        mu_40 = np.average(vlos_o40, weights=w_o40)
        sig_40 = np.sqrt(np.average((vlos_o40 - mu_40)**2, weights=w_o40))

        ymax = max(kde_nb.max(), kde_bf.max(), kde_10.max(), kde_40.max()) * 1.3

        panel_ax.set_xlabel(r'$v_{\rm los}$ [km/s]')
        panel_ax.set_ylabel(r'Weighted counts')
        panel_ax.set_xlim(vlos_range)
        panel_ax.set_ylim(0, ymax)

        # Annotate mean and std
        _colored_stat_line(panel_ax, 1.03, r'$\langle v \rangle$', [mu_nb, mu_bf, mu_10, mu_40], colors)
        _colored_stat_line(panel_ax, 0.95, r'$\sigma_{v}$', [sig_nb, sig_bf, sig_10, sig_40], colors)

        # Rectangle on map + colored frame
        rect = plt.Rectangle((X_edges[0], Y_edges[0]), X_edges[1]-X_edges[0], Y_edges[1]-Y_edges[0],
                             edgecolor=rect_color, facecolor='none', linestyle='--', linewidth=2)
        ax[2].add_patch(rect)
        for spine in panel_ax.spines.values():
            spine.set_edgecolor(rect_color)
            spine.set_linewidth(2)

    # Red and blue box centers: (-3.0, 1.0) and (-1.8, 0.6)
    # Direction vector: (1.2, -0.4) per step
    # Box 0 (green):  center (-4.2, 1.4) — one step left of red
    plot_vlos_panel(ax[0], [-4.7, -3.7], [0.9, 1.9], [-100, 300], 'darkorange')
    # Box 1 (red):    center (-3.0, 1.0)
    plot_vlos_panel(ax[1], [-3.5, -2.5], [0.5, 1.5], [-100, 300], 'red')
    # Box 3 (blue):   center (-1.8, 0.6)
    plot_vlos_panel(ax[3], [-2.3, -1.3], [0.1, 1.1], [-200, 350], 'blue')
    # Box 4 (green): center (-0.6, 0.2) — one step right of blue
    plot_vlos_panel(ax[4], [-1.1, -0.1], [-0.3, 0.7], [-300, 350], 'green')

    # Hide y-axis on all vlos panels and center map
    for i in [0, 1, 3, 4]:
        ax[i].set_yticks([])
        ax[i].set_ylabel('')
    ax[2].set_yticks([])
    ax[2].set_ylabel('')
    ax[2].set_xticks([])
    ax[2].set_xlabel('')

    # Add a shared legend at the bottom
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='C0', alpha=0.3, edgecolor='C0', label='Mock obs (N-body)'),
        Line2D([0], [0], color='black', lw=3, label='Model (best-fit)'),
        Line2D([0], [0], color='orange', lw=2, label=r'Model ($\Omega$ = 10 km/s/kpc)'),
        Line2D([0], [0], color='purple', lw=2, label=r'Model ($\Omega$ = 40 km/s/kpc)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.88))

    # plt.show()
    fig.savefig(figname, bbox_inches='tight')
