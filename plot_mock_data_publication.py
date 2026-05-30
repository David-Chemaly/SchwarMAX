"""
Publication figure for the mock IFU data.

Layout (3 columns):
  Left column (2 panels stacked):
    - Edge-on (N-body, intrinsic frame, bar along X)
    - Face-on (N-body, intrinsic frame)
  Middle column (3 panels stacked):
    - Mock IFU surface luminosity density Sigma_*
    - V_los
    - sigma_v
  Right column (2 panels stacked, aligned with V and sigma):
    - h_3
    - h_4

A stellar surface-luminosity contour (from the N-body data projected
through the same viewing rotation as the mock IFU) is overlaid on each
IFU panel.

The N-body snapshot is processed exactly as in generate_mock_formal.py
(COM iteration, X/Vx sign flip, bar-angle alignment). For the
face-on / edge-on panels the viewing rotation is NOT applied (we want
the intrinsic frame). For the IFU contours the viewing rotation is
applied. The IFU panels themselves are loaded from the mock pickle.
"""

import os
import pickle
import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr

import agama
from astropy.constants import G
import astropy.units as u


# =====================================================================
# CONFIG
# =====================================================================
DATA_FOLDER    = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
SCHWARMAX_PATH = '/Users/hanyuan/Dropbox/python_script/SchwarMAX'
BETA           = 25
GAMMA          = 140
D_KPC          = 50_000

NBODY_SNAPSHOT = f'{DATA_FOLDER}/Bar_model_TG21/model/t_t0_7'
MOCK_FILE      = (f'{SCHWARMAX_PATH}/mock_data/'
                  f'mock_Nbody_bar_XY_withRot_Nbins600'
                  f'_beta{BETA}_gamma{GAMMA}_D{D_KPC//1000}_gal2.pkl')
FIG_OUT        = (f'{DATA_FOLDER}/plots/'
                  f'mock_data_summary_beta{BETA}_gamma{GAMMA}_D{D_KPC//1000}.png')
FIG_PAPER        = (f'{DATA_FOLDER}/figs_paper/'
                  f'mock_data_summary.pdf')

# Face-on / edge-on extents (kpc)
FO_RANGE = ((-10, 10), (-10, 10))   # X, Y
EO_RANGE = ((-10, 10), (-3, 3))     # X, Z
NBIN_2D  = 200

# Colorbar geometry (figure-fraction units, same for every panel)
CBAR_W   = 0.004
CBAR_PAD = 0.003

# Publication style
plt.rcParams['figure.dpi']  = 150
plt.rc('font', size=15)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman'] + plt.rcParams['font.serif']


# =====================================================================
# Helpers ported from generate_mock_formal.py
# =====================================================================
def bar_angle_bar_strength(x, y, R_annulus=np.arange(1, 5, 0.25)):
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    angles, strengths = [], []
    for i in range(len(R_annulus) - 1):
        sel = (R > R_annulus[i]) & (R < R_annulus[i + 1])
        if sel.sum() == 0:
            angles.append(0.0); strengths.append(0.0); continue
        ph = phi[sel]
        A2 = np.sum(np.cos(2 * ph)) / len(ph)
        B2 = np.sum(np.sin(2 * ph)) / len(ph)
        angles.append(0.5 * np.arctan2(B2, A2))
        strengths.append(np.sqrt(A2**2 + B2**2))
    R_mid = (R_annulus[:-1] + R_annulus[1:]) / 2
    return R_mid, np.array(angles), np.array(strengths)


def rotate(posvel, angle):
    x, y, z, vx, vy, vz = posvel.T
    s, c = np.sin(angle), np.cos(angle)
    return np.array([x*c - y*s, x*s + y*c, z,
                     vx*c - vy*s, vx*s + vy*c, vz]).T


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])

def make_rotation_matrix_deg(alpha, beta, gamma):
    """Same as generate_mock_formal.py makeRotationMatrix: inputs in DEGREES."""
    a, b, g = np.radians(alpha), np.radians(beta), np.radians(gamma)
    return (Rz(g) @ Rx(b) @ Rz(a)).T


def prepare_nbody():
    """Read the N-body snapshot and align to the bar frame (no viewing rot)."""
    agama.setUnits(length=1, velocity=1, mass=1)
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s) ** 2))
    print(f'loading N-body <- {NBODY_SNAPSHOT}')
    w0, mass = agama.readSnapshot(NBODY_SNAPSHOT)
    mass = mass * mass_unit.value

    # Drop the heaviest particle species (halo) -- same as the mock script
    mask = (mass != np.unique(mass)[-1])

    # Iterative COM about the (X, Y)-centre, tightening radii
    for r_ap in [10.0, 5.0, 3.0, 2.0]:
        R = np.sqrt(w0[:, 0]**2 + w0[:, 1]**2)
        mc = mask & (R < r_ap)
        m_c = mass[mc]
        for col in range(6):
            w0[:, col] -= np.sum(w0[mc, col] * m_c) / np.sum(m_c)

    w0   = w0[mask]
    mass = mass[mask]

    # Sign flip on X / Vx (matches generate_mock_formal.py)
    w0[:, 0] = -w0[:, 0]
    w0[:, 3] = -w0[:, 3]

    # Rotate so the bar lies along the X-axis
    R_mid, bar_angles, _ = bar_angle_bar_strength(w0[:, 0], w0[:, 1])
    bar_angle = np.mean(bar_angles[R_mid < 4])
    w0 = rotate(w0, -bar_angle)
    print(f'  bar-angle alignment: rotated by {-bar_angle * 180 / np.pi:.2f} deg')

    return w0, mass


def density_imshow(ax, x, y, weights, xy_range, cmap, title, ylabel, cax=None):
    H, xe, ye = np.histogram2d(x, y, bins=NBIN_2D, range=xy_range, weights=weights)
    H = np.ma.masked_where(H <= 0, H)
    dx, dy = xe[1] - xe[0], ye[1] - ye[0]
    dens = H / (dx * dy) / 1e6
    vmin = np.percentile(dens.compressed(), 5)
    vmax = np.percentile(dens.compressed(), 99.5)
    # aspect='auto' so the image fills the axes box (alignment > data aspect).
    im = ax.imshow(dens.T,
                   extent=[xy_range[0][0], xy_range[0][1],
                           xy_range[1][0], xy_range[1][1]],
                   origin='lower', cmap=cmap,
                   norm=LogNorm(vmin = vmin, vmax = vmax),
                   interpolation='nearest',
                   aspect='auto', rasterized=True)
    # ax.set_title(title, fontsize=13)
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel(ylabel)
    if cax is None:
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    else:
        cb = plt.colorbar(im, cax=cax)
    cb.set_label(r'$\Sigma$ [M$_\odot$ / pc$^2$]')


def voronoi_panel(ax, X_grid, Y_grid, per_bin_vals, index_remap,
                  cmap, vmin, vmax, label, xy_lim, log_norm=False,
                  contour=None, cax=None, sigma_err=0.):
    per_bin_vals = np.random.normal(per_bin_vals, sigma_err)
    v_pix = np.asarray(per_bin_vals)[index_remap]
    kw = dict(cmap=cmap, marker='s', s=40, linewidths=0, rasterized=True)
    if log_norm:
        kw['norm'] = LogNorm(vmin=vmin, vmax=vmax)
    else:
        kw['vmin'] = vmin; kw['vmax'] = vmax

    im = ax.scatter(X_grid, Y_grid, c=v_pix, **kw)
    if contour is not None:
        xmid, ymid, logH, levels = contour
        ax.contour(xmid, ymid, logH.T, levels=levels,
                   colors='darkgrey', linewidths=1, alpha=1)
    ax.set_xlim(xy_lim[0]); ax.set_ylim(xy_lim[1])
    ax.set_xlabel('X [arcsec]'); ax.set_ylabel('Y [arcsec]')
    if cax is None:
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    else:
        cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=18)


def build_stellar_contour(w0_intrinsic, mass, alpha, beta, gamma,
                          X_minmax, Y_minmax, nX_nY):
    """Project N-body to the observer frame and build a smoothed log-count
    map on the IFU pixel grid, return (xmid, ymid, log10H, levels) for
    overlaying as contours on the IFU panels."""
    R = make_rotation_matrix_deg(alpha, beta, gamma)
    xyz_obs = (R @ w0_intrinsic[:, :3].T).T
    x_obs, z_obs = xyz_obs[:, 0], xyz_obs[:, 2]

    nX, nY = int(nX_nY[0]), int(nX_nY[1])
    X_edge = np.linspace(X_minmax[0], X_minmax[1], nX + 1)
    Y_edge = np.linspace(Y_minmax[0], Y_minmax[1], nY + 1)
    H, _, _ = np.histogram2d(x_obs, z_obs, bins=[X_edge, Y_edge])
    H = sp.ndimage.gaussian_filter(H, sigma=0.8)
    H = np.maximum(H, 1e-10)
    xmid = 0.5 * (X_edge[1:] + X_edge[:-1])
    ymid = 0.5 * (Y_edge[1:] + Y_edge[:-1])
    log10H = np.log10(H)

    # Levels spread across the populated dynamic range
    pop = log10H[H > 1]
    if pop.size == 0:
        levels = [0.0]
    else:
        levels = np.percentile(pop, [40, 65, 80, 92, 98]).tolist()
    return xmid, ymid, log10H, levels

# def turn_off_axis_ticks(ax):
#     """Turn off the ticks on the right and top axes spines."""
#     ax.tick_params(top=False, right=False)
def turn_off_xaxis_ticks(ax):
    """Turn off the ticks on the right and top axes spines."""
    ax.set_xticks([])
    ax.set_xlabel('')

def turn_off_yaxis_ticks(ax):
    """Turn off the ticks on the right and top axes spines."""
    ax.set_yticks([])
    ax.set_ylabel('')

# =====================================================================
def main():
    w0_nb, mass_nb = prepare_nbody()
    x_nb, y_nb, z_nb = w0_nb[:, 0], w0_nb[:, 1], w0_nb[:, 2]

    print(f'loading mock IFU <- {MOCK_FILE}')
    with open(MOCK_FILE, 'rb') as f:
        bd = pickle.load(f)
    bin_mapping = np.asarray(bd['bin_mapping'])
    index_remap = bin_mapping[:-1]
    X_grid_kpc = np.asarray(bd['X_regular_grid'])
    Y_grid_kpc = np.asarray(bd['Y_regular_grid'])
    X_min, X_max = bd['X_minmax']
    Y_min, Y_max = bd['Y_minmax']
    nX_nY  = bd['nX_nY']
    alpha, beta, gamma = bd['orientation']

    # Stellar contour from N-body, projected through the same viewing rotation
    # used for the mock IFU. Same grid as the IFU pixel grid (still in kpc here).
    contour_kpc = build_stellar_contour(w0_nb, mass_nb,
                                        alpha, beta, gamma,
                                        [X_min, X_max], [Y_min, Y_max], nX_nY)

    # ---- Convert IFU coordinates kpc -> arcsec (D = D_KPC) ----
    KPC_TO_ARCSEC = 3600.0 * 180.0 / (np.pi * D_KPC)
    X_grid = X_grid_kpc * KPC_TO_ARCSEC
    Y_grid = Y_grid_kpc * KPC_TO_ARCSEC
    xy_lim = ((X_min * KPC_TO_ARCSEC, X_max * KPC_TO_ARCSEC),
              (Y_min * KPC_TO_ARCSEC, Y_max * KPC_TO_ARCSEC))
    contour = (contour_kpc[0] * KPC_TO_ARCSEC,
               contour_kpc[1] * KPC_TO_ARCSEC,
               contour_kpc[2], contour_kpc[3])
    print(f"  kpc -> arcsec scale: {KPC_TO_ARCSEC:.4f}  (D = {D_KPC/1000:.0f} Mpc)")

    # ---- Layout: 6-row * 3-col gridspec ----
    # Left col panels each span 3 sub-rows (full height).
    # Middle col has 3 panels, each spanning 2 sub-rows.
    # Right col has 2 panels (rows 2-3 and 4-5) aligned with V and sigma.
    # fig = plt.figure(figsize=(16, 11))
    # gs  = fig.add_gridspec(6, 3,
    #                        hspace=0.55, wspace=0.30,
    #                        left=0.06, right=0.985, top=0.96, bottom=0.05)

    # ax_eo = fig.add_subplot(gs[0:3, 0])    # left top: edge-on
    # ax_fo = fig.add_subplot(gs[3:6, 0])    # left bottom: face-on
    # ax_sd = fig.add_subplot(gs[0:2, 1])    # middle top: Sigma
    # ax_V  = fig.add_subplot(gs[2:4, 1])    # middle mid: V
    # ax_s  = fig.add_subplot(gs[4:6, 1])    # middle bot: sigma
    # ax_h3 = fig.add_subplot(gs[2:4, 2])    # right mid: h3
    # ax_h4 = fig.add_subplot(gs[4:6, 2])    # right bot: h4

    fig = plt.figure(figsize=(20, 8))

    # Row y-extents (top->bottom): [0.70-0.95], [0.40-0.65], [0.10-0.35]
    # Edge-on aligned with the top row; face-on spans the V + sigma rows.
    panel_rects = {
        'eo': [0.02, 0.70, 0.30, 0.25],
        'fo': [0.02, 0.10, 0.30, 0.55],
        'sd': [0.40, 0.70, 0.23, 0.25],
        'V' : [0.40, 0.40, 0.23, 0.25],
        's' : [0.40, 0.10, 0.23, 0.25],
        'h3': [0.72, 0.40, 0.23, 0.25],
        'h4': [0.72, 0.10, 0.23, 0.25],
    }
    axes, caxes = {}, {}
    for key, (x0, y0, w, h) in panel_rects.items():
        axes[key]  = fig.add_axes([x0, y0, w, h])
        caxes[key] = fig.add_axes([x0 + w + CBAR_PAD, y0, CBAR_W, h])

    ax_eo, cax_eo = axes['eo'], caxes['eo']
    ax_fo, cax_fo = axes['fo'], caxes['fo']
    ax_sd, cax_sd = axes['sd'], caxes['sd']
    ax_V,  cax_V  = axes['V'],  caxes['V']
    ax_s,  cax_s  = axes['s'],  caxes['s']
    ax_h3, cax_h3 = axes['h3'], caxes['h3']
    ax_h4, cax_h4 = axes['h4'], caxes['h4']

    # ---- Left column: N-body intrinsic-frame views ----
    density_imshow(ax_eo, x_nb, z_nb, mass_nb, EO_RANGE,
                   cmr.sepia, 'Edge-on (N-body, intrinsic frame)', 'Z [kpc]',
                   cax=cax_eo)
    density_imshow(ax_fo, x_nb, y_nb, mass_nb, FO_RANGE,
                   cmr.sepia, 'Face-on (N-body, intrinsic frame)', 'Y [kpc]',
                   cax=cax_fo)

    # ---- Middle column: Sigma, V, sigma  (with stellar contour overlay) ----
    voronoi_panel(ax_sd, X_grid, Y_grid, bd['surface_density'], index_remap,
                  cmr.sepia, 10, 1e4, r'$\Sigma_*$  [L$_\odot$/pc$^2$]',
                  xy_lim, log_norm=True, contour=contour, cax=cax_sd)

    voronoi_panel(ax_V,  X_grid, Y_grid, bd['V_mean'],  index_remap,
                  cmr.iceburn, -200, 200, r'$V_{\rm los}$ [km/s]', xy_lim,
                  contour=contour, cax=cax_V, sigma_err = 0)
    voronoi_panel(ax_s,  X_grid, Y_grid, bd['V_sigma'], index_remap,
                  cmr.amber,    20, 150, r'$\sigma_v$ [km/s]', xy_lim,
                  contour=contour, cax=cax_s, sigma_err = 0)

    # ---- Right column: h3, h4  (aligned with V and sigma) ----
    voronoi_panel(ax_h3, X_grid, Y_grid, bd['h3'], index_remap,
                  cmr.iceburn, -0.2, 0.2, r'$h_3$', xy_lim,
                  contour=contour, cax=cax_h3, sigma_err = 0.)
    voronoi_panel(ax_h4, X_grid, Y_grid, bd['h4'], index_remap,
                  cmr.amber,  -0.2, 0.1, r'$h_4$', xy_lim,
                  contour=contour, cax=cax_h4, sigma_err = 0.)
    
    turn_off_xaxis_ticks(ax_eo)
    turn_off_xaxis_ticks(ax_sd)
    turn_off_xaxis_ticks(ax_V)
    turn_off_xaxis_ticks(ax_h3)


    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches='tight', dpi=200)
    fig.savefig(FIG_PAPER, bbox_inches='tight', dpi = 300)
    print(f'saved -> {FIG_OUT}')


if __name__ == '__main__':
    main()
