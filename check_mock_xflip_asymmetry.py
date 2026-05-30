"""
Diagnose whether the N-body mock galaxy is intrinsically asymmetric at
high |Z| (the mismatch you see in the top row of plot_data_vs_model.py).

The test: run the *same* mock-IFU pipeline as generate_mock_formal.py
twice on the same N-body snapshot. Run A is the original. Run B
flips the *intrinsic* coordinates  x -> -x  and  vx -> -vx  AFTER the
bar-alignment step, then applies the same viewing rotation. If the bar
is a perfect mirror-symmetric structure, the two mock observations
should look similar (modulo the projection's geometric mixing of axes).
Any residual difference -- especially as a function of observed Y --
shows real intrinsic asymmetry that the symmetric T3+V4 model cannot fit.

Output: 5 rows (Sigma, V, sigma, h3, h4) x 3 cols (original, x-flipped,
difference), each panel a Voronoi-bin scatter using the same bins as
the stored mock pickle.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr

import agama
from astropy.constants import G
import astropy.units as u


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
                  f'mock_xflip_asymmetry_beta{BETA}_gamma{GAMMA}_D{D_KPC//1000}.png')

EPS = 1e-30

plt.rc('font', size=11)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman'] + plt.rcParams['font.serif']


# =====================================================================
# Helpers (ported from generate_mock_formal.py)
# =====================================================================
def bar_angle_bar_strength(x, y, R_anulus=np.arange(1, 5, 0.25)):
    R0 = np.sqrt(x**2 + y**2); phi0 = np.arctan2(y, x)
    angles, strengths = [], []
    for i in range(len(R_anulus) - 1):
        sel = (R0 > R_anulus[i]) & (R0 < R_anulus[i+1])
        if sel.sum() == 0:
            angles.append(0.0); strengths.append(0.0); continue
        ph = phi0[sel]
        A2 = np.sum(np.cos(2 * ph)) / len(ph)
        B2 = np.sum(np.sin(2 * ph)) / len(ph)
        angles.append(0.5 * np.arctan2(B2, A2))
        strengths.append(np.sqrt(A2**2 + B2**2))
    return (R_anulus[:-1] + R_anulus[1:]) / 2, np.array(angles), np.array(strengths)


def rotate2D(posvel, angle):
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
    a, b, g = np.radians(alpha), np.radians(beta), np.radians(gamma)
    return (Rz(g) @ Rx(b) @ Rz(a)).T


def prepare_nbody_intrinsic(snapshot):
    """Return bar-aligned intrinsic-frame (w0, mass), without viewing rotation."""
    agama.setUnits(mass=1, length=1, velocity=1)
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s) ** 2))
    print(f'loading N-body <- {snapshot}')
    w0, mass = agama.readSnapshot(snapshot)
    mass = mass * mass_unit.value

    mask = (mass != np.unique(mass)[-1])
    for r_ap in [10.0, 5.0, 3.0, 2.0]:
        R = np.sqrt(w0[:, 0]**2 + w0[:, 1]**2)
        mc = mask & (R < r_ap)
        m_c = mass[mc]
        for col in range(6):
            w0[:, col] -= np.sum(w0[mc, col] * m_c) / np.sum(m_c)
    w0   = w0[mask]
    mass = mass[mask]

    # Sign flip on intrinsic X / Vx (matches generate_mock_formal.py)
    w0[:, 0] = -w0[:, 0]
    w0[:, 3] = -w0[:, 3]

    R_mid, ba, _ = bar_angle_bar_strength(w0[:, 0], w0[:, 1])
    bar_angle = np.mean(ba[R_mid < 4])
    w0 = rotate2D(w0, -bar_angle)
    print(f'  bar-angle alignment: rotated by {-bar_angle * 180 / np.pi:.2f} deg')
    return w0, mass


# =====================================================================
# Projection + binning (mirror of the maps that generate_mock_formal.py
# computes per-Voronoi-bin: Sigma, V_mean, V_sigma, h1..h4)
# =====================================================================
def project_and_bin(w0, mass, bin_dict, alpha, beta, gamma):
    R = make_rotation_matrix_deg(alpha, beta, gamma)
    pos_obs = (R @ w0[:, :3].T).T
    vel_obs = (R @ w0[:, 3:].T).T

    # In the mock pipeline observed X = rotated x; observed Y (on sky) = rotated z;
    # LOS direction = rotated y.
    Xobs   = pos_obs[:, 0]
    Yobs   = pos_obs[:, 2]
    v_los  = vel_obs[:, 1]

    nX, nY = int(bin_dict['nX_nY'][0]), int(bin_dict['nX_nY'][1])
    Xmin, Xmax = bin_dict['X_minmax']
    Ymin, Ymax = bin_dict['Y_minmax']
    bin_mapping = np.asarray(bin_dict['bin_mapping'])[:-1]   # length nX*nY
    total_bins  = int(bin_dict['total_bins'])
    num_per_bin = np.asarray(bin_dict['num_per_bin'], dtype=np.float64)
    area_pixel  = ((Xmax - Xmin) / nX) * ((Ymax - Ymin) / nY)   # kpc^2

    # Assign each particle to a pixel; pixel layout matches the mock
    # script's strides = [1, nX] -> idx = ix + iy*nX
    ix = np.floor((Xobs - Xmin) / (Xmax - Xmin) * nX).astype(np.int64)
    iy = np.floor((Yobs - Ymin) / (Ymax - Ymin) * nY).astype(np.int64)
    in_fov = (ix >= 0) & (ix < nX) & (iy >= 0) & (iy < nY)
    pixel_idx = np.where(in_fov, ix + iy * nX, 0)
    bin_idx = np.where(in_fov, bin_mapping[pixel_idx], -1)

    valid = bin_idx >= 0
    bidx  = bin_idx[valid]
    m     = mass[valid]
    vl    = v_los[valid]

    counts = np.bincount(bidx, weights=m,         minlength=total_bins)
    sum_v  = np.bincount(bidx, weights=m * vl,    minlength=total_bins)
    sum_v2 = np.bincount(bidx, weights=m * vl**2, minlength=total_bins)
    V_mean  = sum_v  / (counts + EPS)
    V_sigma = np.sqrt(np.maximum(sum_v2 / (counts + EPS) - V_mean**2, 0.0))

    # Use the same v0, s as the stored mock so h_k are normalised the same way
    v0 = np.asarray(bin_dict['v0'])
    s  = np.asarray(bin_dict['s'])
    w_norm = (vl - v0[bidx]) / (s[bidx] + EPS)
    s_w  = np.bincount(bidx, weights=m * w_norm,    minlength=total_bins)
    s_w2 = np.bincount(bidx, weights=m * w_norm**2, minlength=total_bins)
    s_w3 = np.bincount(bidx, weights=m * w_norm**3, minlength=total_bins)
    s_w4 = np.bincount(bidx, weights=m * w_norm**4, minlength=total_bins)
    norm = counts + EPS
    w1, w2, w3, w4 = s_w/norm, s_w2/norm, s_w3/norm, s_w4/norm

    h1 = w1
    h2 = (w2 - 1.) / np.sqrt(2.)
    h3 = (w3 - 3.*w1) / np.sqrt(6.)
    h4 = (w4 - 6.*w2 + 3.) / np.sqrt(24.)

    # Surface density per Voronoi bin in M_sun/pc^2
    rho = counts / (num_per_bin * area_pixel * 1e6 + EPS)

    return dict(Sigma=rho, V_mean=V_mean, V_sigma=V_sigma,
                h1=h1, h2=h2, h3=h3, h4=h4)


# =====================================================================
def main():
    print(f'loading mock IFU bins <- {MOCK_FILE}')
    with open(MOCK_FILE, 'rb') as f:
        bin_dict = pickle.load(f)
    alpha, beta, gamma = bin_dict['orientation']
    print(f'  viewing (alpha, beta, gamma) = '
          f'({alpha:.2f}, {beta:.2f}, {gamma:.2f}) [deg]')

    w0, mass = prepare_nbody_intrinsic(NBODY_SNAPSHOT)

    print('projecting + binning ORIGINAL intrinsic frame ...')
    maps_A = project_and_bin(w0, mass, bin_dict, alpha, beta, gamma)

    print('projecting + binning  x -> -x  flipped frame ...')
    w0_flip = w0.copy()
    w0_flip[:, 0] *= -1.0      # intrinsic x
    w0_flip[:, 3] *= -1.0      # intrinsic v_x
    maps_B = project_and_bin(w0_flip, mass, bin_dict, alpha, beta, gamma)

    # ---- Plot ----
    bin_mapping = np.asarray(bin_dict['bin_mapping'])
    index_remap = bin_mapping[:-1]
    X_grid = np.asarray(bin_dict['X_regular_grid'])
    Y_grid = np.asarray(bin_dict['Y_regular_grid'])
    Xmin, Xmax = bin_dict['X_minmax']
    Ymin, Ymax = bin_dict['Y_minmax']

    # Rows: Sigma, V, sigma, h3, h4
    rows = [
        ('Sigma',  r'$\Sigma_*$ [M$_\odot$/pc$^2$]',
         cmr.sepia,    1e1, 1e4,  True,  cmr.iceburn, 0.5),
        ('V_mean', r'$V_{\rm los}$ [km/s]',
         cmr.iceburn, -200, 200, False,  cmr.iceburn, 30.0),
        ('V_sigma', r'$\sigma_v$ [km/s]',
         cmr.amber,    20, 150, False,   cmr.iceburn, 30.0),
        ('h3',     r'$h_3$',
         cmr.iceburn, -0.2, 0.2, False,  cmr.iceburn, 0.15),
        ('h4',     r'$h_4$',
         cmr.amber,  -0.2,  0.1, False,  cmr.iceburn, 0.15),
    ]

    fig = plt.figure(figsize=(15, 14))
    gs  = fig.add_gridspec(len(rows), 3,
                           hspace=0.45, wspace=0.45,
                           left=0.06, right=0.97, top=0.96, bottom=0.05)

    for i, (key, label, cmap_main, vmin, vmax, log_norm,
            cmap_diff, vdiff) in enumerate(rows):
        vec_A = np.asarray(maps_A[key])[index_remap]
        vec_B = np.asarray(maps_B[key])[index_remap]
        if log_norm:
            # log-residual for density to keep symmetric meaning
            with np.errstate(divide='ignore', invalid='ignore'):
                diff = np.log10(np.maximum(vec_A, 1e-30)) - np.log10(np.maximum(vec_B, 1e-30))
        else:
            diff = vec_A - vec_B

        def panel(ax, c, vlims, cm, log_norm=False, title=''):
            kw = dict(cmap=cm, marker='s', linewidths=0, rasterized=True, s=14)
            if log_norm:
                kw['norm'] = LogNorm(vmin=vlims[0], vmax=vlims[1])
            else:
                kw['vmin'] = vlims[0]; kw['vmax'] = vlims[1]
            im = ax.scatter(X_grid, Y_grid, c=c, **kw)
            ax.set_xlim(Xmin, Xmax); ax.set_ylim(Ymin, Ymax)
            ax.set_xlabel('X [kpc]')
            if title:
                ax.set_title(title, fontsize=11)
            cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            cb.set_label(label, fontsize=10)
            return im

        ax_a = fig.add_subplot(gs[i, 0])
        ax_b = fig.add_subplot(gs[i, 1])
        ax_d = fig.add_subplot(gs[i, 2])
        panel(ax_a, vec_A, (vmin, vmax), cmap_main, log_norm,
              title='Original' if i == 0 else '')
        panel(ax_b, vec_B, (vmin, vmax), cmap_main, log_norm,
              title=r'$x \to -x$ flipped' if i == 0 else '')
        # Diff panel
        diff_lim = (-vdiff, vdiff)
        if key == 'Sigma':
            diff_label = r'$\Delta\log_{10}\,\Sigma$'
        else:
            diff_label = r'$\Delta$ ' + label.split('[')[0].strip()
        kw = dict(cmap=cmap_diff, marker='s', linewidths=0, rasterized=True, s=14,
                  vmin=diff_lim[0], vmax=diff_lim[1])
        im = ax_d.scatter(X_grid, Y_grid, c=diff, **kw)
        ax_d.set_xlim(Xmin, Xmax); ax_d.set_ylim(Ymin, Ymax)
        ax_d.set_xlabel('X [kpc]')
        if i == 0:
            ax_d.set_title('Original − Flipped', fontsize=11)
        cb = plt.colorbar(im, ax=ax_d, fraction=0.045, pad=0.02)
        cb.set_label(diff_label, fontsize=10)

        ax_a.set_ylabel('Y [kpc]')

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches='tight', dpi=200)
    print(f'saved -> {FIG_OUT}')


if __name__ == '__main__':
    main()
