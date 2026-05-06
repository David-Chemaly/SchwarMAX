"""
Plot the kinematic Fourier decomposition of the line-of-sight velocity field,
comparing model (from orbit trajectories + weights) to N-body data.

Computes k_m(R) = sqrt(a_m^2 + b_m^2) for m=1 (rotation), m=2 (quadrupole),
and optionally higher-order terms, in circular annuli on the sky plane.

Usage:
    python plot_velocity_quadrupole.py
"""
from likelihoods_bar import get_dict_data_bootstrap
from model_bar import * 

import agama
agama.setUnits(mass=1, length=1, velocity=1)
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pickle
from tqdm import tqdm
from constants import KPCGYR_TO_KMS

from scipy.interpolate import CubicSpline
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


def get_dict_data_bootstrap(path, filename, N_BOOTSTRAP = 100, n_samples = 5_000):

    with open(path + filename, 'rb') as f:
        bin_dict = pickle.load(f)

    X_minmax = jnp.array(bin_dict['X_minmax'])
    Y_minmax = jnp.array(bin_dict['Y_minmax'])
    nX_nY = jnp.array(bin_dict['nX_nY'])

    # voronoi binning mapping and data
    num_per_bin = jnp.array(bin_dict['num_per_bin'])
    total_bins = jnp.array(bin_dict['total_bins'])
    bin_mapping = jnp.array(bin_dict['bin_mapping'])
    surface_density = jnp.array(bin_dict['surface_density'])
    V_data = jnp.array(bin_dict['V_mean'])
    sigma_data = jnp.array(bin_dict['V_sigma'])
    h1_data = jnp.array(bin_dict['h1'])
    h2_data = jnp.array(bin_dict['h2'])
    h3_data = jnp.array(bin_dict['h3'])
    h4_data = jnp.array(bin_dict['h4'])
    v0 = jnp.array(bin_dict['v0'])
    s = jnp.array(bin_dict['s'])
    alpha, beta, gamma = bin_dict['orientation']


    XY_density_data_err = 0.01 * surface_density + EPSILON
    V_data_err = jnp.array(bin_dict['V_mean_err'])
    sigma_data_err = jnp.array(bin_dict['V_sigma_err'])
    h1_data_err = jnp.array(bin_dict['h1_err'])
    h2_data_err = jnp.array(bin_dict['h2_err'])
    h3_data_err = jnp.array(bin_dict['h3_err'])
    h4_data_err = jnp.array(bin_dict['h4_err'])

    '''
    Bootstrap the observation
    '''
    # rng = np.random.default_rng(42)
    # N_BOOTSTRAP = 100
    # y_xy_boot = np.array(surface_density[None, :] + rng.normal(size=(N_BOOTSTRAP, len(surface_density))) * XY_density_data_err[None, :])
    # y_h1_boot = np.array(h1_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h1_data))) * h1_data_err[None, :])
    # y_h2_boot = np.array(h2_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h2_data))) * h2_data_err[None, :])
    # y_h3_boot = np.array(h3_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h3_data))) * h3_data_err[None, :])
    # y_h4_boot = np.array(h4_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h4_data))) * h4_data_err[None, :])
    # V_boot = np.array(V_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(V_data))) * V_data_err[None, :])
    # sigma_boot = np.array(sigma_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(sigma_data))) * sigma_data_err[None, :])
    # # Fix the first one to always be the unperturbed system
    # y_xy_boot[0, :] = surface_density
    # y_h1_boot[0, :] = h1_data
    # y_h2_boot[0, :] = h2_data
    # y_h3_boot[0, :] = h3_data
    # y_h4_boot[0, :] = h4_data
    # V_boot[0, :] = V_data
    # sigma_boot[0, :] = sigma_data
    # y_xy_boot = jnp.array(y_xy_boot)
    # y_h1_boot = jnp.array(y_h1_boot)
    # y_h2_boot = jnp.array(y_h2_boot)
    # y_h3_boot = jnp.array(y_h3_boot)
    # y_h4_boot = jnp.array(y_h4_boot)
    # V_boot = jnp.array(V_boot)
    # sigma_boot = jnp.array(sigma_boot)

    rng = np.random.default_rng(42)
    XY_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(surface_density)))
    h1_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h1_data)))
    h2_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h2_data)))
    h3_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h3_data)))
    h4_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h4_data)))
    V_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(V_data)))
    sigma_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(sigma_data)))
    XY_standard_normal[0, :] = 0.0
    h1_standard_normal[0, :] = 0.0
    h2_standard_normal[0, :] = 0.0
    h3_standard_normal[0, :] = 0.0
    h4_standard_normal[0, :] = 0.0
    V_standard_normal[0, :] = 0.0
    sigma_standard_normal[0, :] = 0.0
    XY_standard_normal = jnp.array(XY_standard_normal)
    h1_standard_normal = jnp.array(h1_standard_normal)
    h2_standard_normal = jnp.array(h2_standard_normal)
    h3_standard_normal = jnp.array(h3_standard_normal)
    h4_standard_normal = jnp.array(h4_standard_normal)
    V_standard_normal = jnp.array(V_standard_normal)
    sigma_standard_normal = jnp.array(sigma_standard_normal)

    from scipy.stats import qmc

    # with open(path + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
    #     Rzphi_density_data = pickle.load(f)
    # R_grid, z_grid, phi_grid = Rzphi_density_data['R_grid'], Rzphi_density_data['z_grid'], Rzphi_density_data['phi_grid']
    # dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    # dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    # dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    # sample_for_integration = Rzphi_density_data['sample_for_integration']
    R_min, R_max =0., 10.
    z_min, z_max = -3., 3.
    phi_min, phi_max = -jnp.pi, jnp.pi
    n_R, n_z, n_phi =10, 6, 10
    n_tot = int(n_R * n_z * n_phi)
    R_edge = jnp.linspace(R_min, R_max, n_R+1)
    z_edge = jnp.linspace(z_min, z_max, n_z+1)
    phi_edge = jnp.linspace(phi_min, phi_max, n_phi+1)
    R_mids, z_mids, phi_mids = 0.5 * (R_edge[:-1] + R_edge[1:]), 0.5 * (z_edge[:-1] + z_edge[1:]), 0.5 * (phi_edge[:-1] + phi_edge[1:])
    dR, dz, dphi = R_edge[1]-R_edge[0], z_edge[1]-z_edge[0], phi_edge[1]-phi_edge[0]
    R_mids_mesh, z_mids_mesh, phi_mids_mesh = jnp.meshgrid(R_mids, z_mids, phi_mids, indexing='ij')
    Rzphi_mid_grid = jnp.stack([R_mids_mesh.ravel(), z_mids_mesh.ravel(), phi_mids_mesh.ravel()], axis=-1)  # (n_R*n_z*n_phi, 3)
    R_grid = Rzphi_mid_grid[:,0]
    z_grid = Rzphi_mid_grid[:,1]
    phi_grid = Rzphi_mid_grid[:,2]
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    Rzphi_minmax=jnp.array([[R_min, R_max],[z_min, z_max],[phi_min, phi_max]])
    nRzphi=jnp.array([n_R,n_z,n_phi])
    num_segments_Rzphi=nRzphi.prod()
    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_grid_indices = assign_regular_grid(Rzphi_mid_grid,
                                        grid_min=Rzphi_minmax[:,0],
                                        grid_max=Rzphi_minmax[:,1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)
    _, COUNTS = jnp.unique(Rzphi_grid_indices, return_counts=True)
    argsort = jnp.argsort(Rzphi_grid_indices)
    R_grid = R_grid[argsort]
    z_grid = z_grid[argsort]
    phi_grid = phi_grid[argsort]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample_for_integration = sampler.random_base2(m=10)

    
    X_regular_grid, Y_regular_grid = bin_dict['X_regular_grid'], bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)

    n_samples = n_samples  # Same number as original data
    x_grid = np.linspace(0., 12., 1000)
    logP_xexp = XexpX_pdf_log(x_grid, 4.0)
    key = jax.random.PRNGKey(10086)
    R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.default_rng(42).uniform(0, 2*np.pi, size=n_samples)

    x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)

    x_grid = np.linspace(0, 4, 1000)
    logP_exp = expX_pdf_log(x_grid, 1.5)
    key = jax.random.PRNGKey(10010)
    z_samples = sample_from_logP(x_grid, logP_exp, n_samples, key)
    w0 = np.array([
        x_samples,
        y_samples,
        z_samples,
    ]).T


    dict_data = {
        # 'w0': w0,
        'v0': v0,
        's': s,

        # 'Rzphi_density_data': Rzphi_density_data,
        'XY_density_data': surface_density,
        'XY_density_data_err': XY_density_data_err,
        'V_data': V_data,
        'V_data_err': V_data_err,
        'sigma_data': sigma_data,
        'sigma_data_err': sigma_data_err,
        'h1_data': h1_data,
        'h1_data_err': h1_data_err,
        'h2_data': h2_data,
        'h2_data_err': h2_data_err,
        'h3_data': h3_data,
        'h3_data_err': h3_data_err,
        'h4_data': h4_data,
        'h4_data_err': h4_data_err,
        'num_per_bin': num_per_bin,
        'bin_mapping': bin_mapping,
        'total_bins': total_bins.item(),

        # 'y_xy_boot': y_xy_boot,
        # 'y_h1_boot': y_h1_boot,
        # 'y_h2_boot': y_h2_boot,
        # 'y_h3_boot': y_h3_boot,
        # 'y_h4_boot': y_h4_boot,
        # 'V_boot': V_boot,
        # 'sigma_boot': sigma_boot,
        'XY_standard_normal': XY_standard_normal,
        'h1_standard_normal': h1_standard_normal,
        'h2_standard_normal': h2_standard_normal,
        'h3_standard_normal': h3_standard_normal,
        'h4_standard_normal': h4_standard_normal,
        'V_standard_normal': V_standard_normal,
        'sigma_standard_normal': sigma_standard_normal,

        'R_grid': R_grid,
        'z_grid': z_grid,
        'phi_grid': phi_grid,
        'R_minmax': [R_min, R_max],
        'z_minmax': [z_min, z_max],
        'phi_minmax': [-jnp.pi, jnp.pi],
        'Rzphi_n_tot': n_tot,
        'Rzphi_n_grid': jnp.array([n_R, n_z, n_phi]),
        'dR': dR,
        'dz': dz,
        'dphi': dphi,
        'sample_for_integration': sample_for_integration,

        'X_regular_grid': X_regular_grid,
        'Y_regular_grid': Y_regular_grid,
        'dX': dX,
        'dY': dY,
        'sample_for_integration_XY': sample,

        'X_minmax': X_minmax,
        'Y_minmax': Y_minmax,
        'nX_nY': nX_nY,

        'w0': w0
    }

    return dict_data

# ═══════════════════════════════════════════════════════════════════
#  Velocity field Fourier decomposition
# ═══════════════════════════════════════════════════════════════════

def fourier_decompose_velocity(X, Y, V_LOS, weight, R_edges, m_max=5):
    """
    Decompose V_LOS(X, Y) into azimuthal Fourier modes in circular annuli.

    Parameters
    ----------
    X, Y : (N,) sky-plane coordinates
    V_LOS : (N,) line-of-sight velocity
    weight : (N,) weight per data point (dt * orbit_weight for model, mass for N-body)
    R_edges : (n_annuli+1,) radial bin edges
    m_max : int, maximum Fourier order

    Returns
    -------
    R_mid : (n_annuli,) bin centres
    k_m : (m_max+1, n_annuli) amplitude of each mode, k_0 is mean V
    phi_m : (m_max+1, n_annuli) phase of each mode
    """
    R = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    n_annuli = len(R_edges) - 1
    R_mid = 0.5 * (R_edges[:-1] + R_edges[1:])

    k_m = np.zeros((m_max + 1, n_annuli))
    phi_m = np.zeros((m_max + 1, n_annuli))

    for j in range(n_annuli):
        mask = (R >= R_edges[j]) & (R < R_edges[j + 1]) & np.isfinite(V_LOS)
        if mask.sum() < 10:
            continue
        w_j = weight[mask]
        v_j = V_LOS[mask]
        phi_j = phi[mask]
        W = np.sum(w_j)
        if W == 0:
            continue

        # m=0: weighted mean velocity
        k_m[0, j] = np.sum(w_j * v_j) / W

        for m in range(1, m_max + 1):
            a_m = 2.0 * np.sum(w_j * v_j * np.cos(m * phi_j)) / W
            b_m = 2.0 * np.sum(w_j * v_j * np.sin(m * phi_j)) / W
            k_m[m, j] = np.sqrt(a_m**2 + b_m**2)
            phi_m[m, j] = 0.5 * np.arctan2(b_m, a_m)

    return R_mid, k_m, phi_m


# ═══════════════════════════════════════════════════════════════════
#  Build model velocity field from orbit trajectories + weights
# ═══════════════════════════════════════════════════════════════════

def build_model_velocity_field(diag, R_edges, m_max=5):
    """
    From model_diagnostic output, project orbits onto the sky plane
    and compute the Fourier decomposition of V_LOS.

    Parameters
    ----------
    diag : dict from model_diagnostic
    R_edges : radial bin edges on the sky plane [kpc]
    m_max : maximum Fourier order

    Returns
    -------
    R_mid, k_m, phi_m  (same as fourier_decompose_velocity)
    """
    y_traj = diag['y_traj']        # (n_orbits, N_max, 6)
    dt_traj = diag['dt_traj']      # (n_orbits, N_max)
    valid = diag['valid']          # (n_orbits,)
    weights = diag['weights']      # (n_orbits,)
    rot = diag['rotation_matrix']  # (3, 3)

    n_orbits, N_max, _ = y_traj.shape

    # Apply 4-fold bar symmetry
    sign_sym = np.array([
        [ 1,  1,  1,  1,  1,  1],   # identity
        [ 1,  1, -1,  1,  1, -1],   # z-reflection
        [-1, -1,  1, -1, -1,  1],   # xy point symmetry
        [-1, -1, -1, -1, -1, -1],   # combined
    ])

    X_all, Y_all, VLOS_all, W_all = [], [], [], []

    for i in tqdm(range(n_orbits), desc="Projecting orbits"):
        if valid[i] < 0.5 or weights[i] < 1e-30:
            continue

        w_i = weights[i]
        dt_i = dt_traj[i]      # (N_max,)
        pos_i = y_traj[i]      # (N_max, 6)

        # Apply symmetry: 4 copies
        for s in range(4):
            sym_pos = pos_i * sign_sym[s]  # (N_max, 6)

            # Rotate to observer frame
            x_rot = (rot @ sym_pos[:, :3].T).T  # (N_max, 3)
            v_rot = (rot @ sym_pos[:, 3:].T).T  # (N_max, 3)

            # Sky plane: X = x_rot[:,0], Y = x_rot[:,2]; LOS = y_rot[:,1]
            X_proj = x_rot[:, 0]
            Y_proj = x_rot[:, 2]
            V_LOS = v_rot[:, 1] * KPCGYR_TO_KMS  # convert to km/s

            wt = dt_i * w_i  # weight per timestep

            # Only accepted steps (dt > 0)
            mask = dt_i > 0
            X_all.append(X_proj[mask])
            Y_all.append(Y_proj[mask])
            VLOS_all.append(V_LOS[mask])
            W_all.append(wt[mask])

    X_all = np.concatenate(X_all)
    Y_all = np.concatenate(Y_all)
    VLOS_all = np.concatenate(VLOS_all)
    W_all = np.concatenate(W_all)

    R_mid, k_m, phi_m = fourier_decompose_velocity(
        X_all, Y_all, VLOS_all, W_all, R_edges, m_max=m_max)

    return R_mid, k_m, phi_m


# ═══════════════════════════════════════════════════════════════════
#  N-body data loading and processing
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


def nbody_velocity_fourier(w0_data, mass_data, mask_disc, rotation_matrix,
                           R_edges, m_max=5):
    """
    Compute Fourier decomposition of V_LOS from N-body stellar particles.

    Parameters
    ----------
    w0_data : (N, 6) positions+velocities in bar frame
    mass_data : (N,) particle masses
    mask_disc : (N,) bool mask for disc/stellar particles
    rotation_matrix : (3, 3) observer projection
    R_edges : radial bin edges
    m_max : max Fourier order

    Returns
    -------
    R_mid, k_m, phi_m
    """
    pos = w0_data[mask_disc, :3]
    vel = w0_data[mask_disc, 3:]
    m = mass_data[mask_disc]

    # Project to observer frame
    x_rot = (rotation_matrix @ pos.T).T
    v_rot = (rotation_matrix @ vel.T).T

    X = x_rot[:, 0]
    Y = x_rot[:, 2]
    V_LOS = v_rot[:, 1]  # already in km/s for N-body (check units)

    R_mid, k_m, phi_m = fourier_decompose_velocity(X, Y, V_LOS, m, R_edges, m_max)
    return R_mid, k_m, phi_m


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    from utils import logMenc_logc_to_logM_logRs, makeRotationMatrix
    from model_bar import model_diagnostic

    path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    # CHECKPOINT_FILE = data_folder + '/ensemble_checkpoint_gal2_0406.pkl'
    # data_filename = 'mock_Nbody_bar_XY_withRot_gal2_Nbins1000.pkl'
    # data_filename = 'mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma170.pkl'

    # CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_0415_beta25_gamma140_D50_gal2.pkl'
    # CHECKPOINT_FILE = data_folder+'/mcmc_checkpoint_0415_beta25_gamma140_D50_gal2.pkl'
    CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_0418_beta25_gamma140_D50_gal2_fixedbarlength.pkl'
    data_filename = 'mock_data/mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma140_D50_gal2.pkl'

    # output_filename = data_folder + '/best_fit_orbital_library_0415_beta25_gamma140_D50_gal2_fixedbarlength.pkl'
    output_filename = data_folder + '/best_fit_orbital_library_0418_beta25_gamma140_D50_gal2_fixedbarlength.pkl'
    DISCARD = 600
    THIN = 1

    # ── Model: run diagnostic for best-fit params ──

    dict_data = get_dict_data_bootstrap(path, data_filename, n_samples = 10_000, N_BOOTSTRAP=10)

    param_names = ['logM_halo', 'logM_disk', 'logM_bar', 'logRs_halo', 'logRs_disk', 'logHs_disk', 'logRs_bar', 
                r'$\alpha$', r'$\beta$', r'$\gamma$', 
                r'$\log_{10}(L/M)$', r'$\log_{10}(\Omega)$', r'$\log_{10}(\sigma amplifier)$']

    posterior, logprob, step = load_checkpoint(CHECKPOINT_FILE)
    posterior = posterior[:, logprob[-1, :]>np.amax(logprob[-1, :])-100, :]
    posterior = posterior[DISCARD::THIN, :, :]
    posterior = posterior.reshape(-1, posterior.shape[-1])

    best_fit_param = np.percentile(posterior, 50, axis=0)
    print("Best-fit parameters:")
    for i, name in enumerate(param_names):
        print(f"{name}: {best_fit_param[i]:.4f}")

    dict_data['logl_density_max'] = -0.24
    logMhalo_10_best_fit, logMdisk_best_fit, logMbar_best_fit, logC_halo_best_fit, logRs_disk_best_fit, logHs_disk_best_fit, logRs_bar_best_fit,\
        alpha_best_fit, beta_best_fit, gamma_best_fit, logLM_best_fit, logOmega_best_fit, logSigma_amplifier_best_fit = best_fit_param
    logMhalo_best_fit, logRh_halo_best_fit = logMenc_logc_to_logM_logRs(logMhalo_10_best_fit, logC_halo_best_fit, r_enc=10.0, Delta=200., rho_crit=277.54)

    alpha = alpha_best_fit * 180/np.pi
    beta = beta_best_fit * 180/np.pi
    gamma = gamma_best_fit * 180/np.pi# gamma_best_fit
    ground_truth = [logMhalo_best_fit,
                    logMdisk_best_fit,
                    logMbar_best_fit,
                    logRh_halo_best_fit,
                    logRs_disk_best_fit,
                    logHs_disk_best_fit,
                    logRs_bar_best_fit,   
                    # 0.68,
                    alpha,
                    beta,
                    gamma,
                    logLM_best_fit,
                    logOmega_best_fit,
                    # jnp.log10(40.0).item(),
                    logSigma_amplifier_best_fit
    ]

    params_halo_pot = {
        'logM': ground_truth[0],
        'Rs':10 ** ground_truth[3],
        'a':1.0,
        'b':1.0,
        'c':1.0,
        'x_origin':0.0,
        'y_origin':0.0,
        'z_origin':0.0,
        'dirx':0.0,
        'diry':0.0,
        'dirz':1.0
    }

    params_disk_rho = {
        'logM_disc': ground_truth[1],
        'Rs_disc': 10 ** ground_truth[4],
        'Hs_disc': 10 ** ground_truth[5],
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
        'alpha': ground_truth[7],
        'beta': ground_truth[8],
        'gamma': ground_truth[9],
        'logM_bar': ground_truth[2],
        'L_bar': 10 ** ground_truth[6],
        'a_bar': 10 ** ground_truth[6] / 5,
        'b_bar': 10 ** ground_truth[5],
        'light_to_mass_ratio': 10 ** ground_truth[10],
        'Omega_bar': 10 ** ground_truth[11],
        'sigma_amplifier': 10 ** ground_truth[12],
    }
    print('Model pattern speed (Omega_bar):', params_disk_rho['Omega_bar'])

    surface_density_model = projection(params_disk_rho, dict_data, dict_data['total_bins'])
    surface_density_gt = dict_data['XY_density_data'] / params_disk_rho['light_to_mass_ratio']

    chi2 = jnp.sum((surface_density_gt - surface_density_model)**2 / (0.1 * surface_density_gt)**2)
    _logl_density = -0.5 * chi2 / dict_data['total_bins']
    print(_logl_density)

    X_minmax = dict_data['X_minmax']
    Y_minmax = dict_data['Y_minmax']
    nX, nY = dict_data['nX_nY']
    xy_lim_grid = jnp.array([X_minmax, Y_minmax])
    xy_n_grid = jnp.array([nX, nY])

    Rmin, Rmax = dict_data['R_minmax']
    zmin, zmax = dict_data['z_minmax']
    phimin, phimax = dict_data['phi_minmax']
    Rzphi_n_tot = dict_data['Rzphi_n_tot']
    Rzphi_n_grid = dict_data['Rzphi_n_grid']

    output_dict = model_diagnostic(params_halo_pot, params_disk_rho, dict_data, dict_data['total_bins'],
                                   Rzphi_n_tot, Rzphi_n_grid, Rzphi_lim_grid=jnp.array([[Rmin, Rmax],[zmin, zmax],[phimin, phimax]]),
                                    xy_lim_grid=xy_lim_grid,
                                    xy_n_grid=xy_n_grid)
    logL = output_dict['logl_all'][0]
    print('logL:', logL)
    print('Model Done')

    weights = output_dict['weights']        # (n_orbits,)
    o_traj = output_dict['y_traj']            # (n_orbits, 4, N_max, 6)
    t_traj = output_dict['t_traj']            # (n_orbits, 4, N_max)
    mean_mass_per_orbit = output_dict['mass_per_orbit']  # (n_orbits,)

    # weights = np.repeat(weights, 4)
    # y_traj = np.stack(y_traj, axis = 1)
    # t_traj = np.stack(t_traj, axis = 1)
    weights = np.repeat(weights, o_traj.shape[1])
    # weights = np.concatenate([weights] * o_traj.shape[1])
    o_traj = o_traj.reshape(-1, o_traj.shape[2], 6)
    t_traj = t_traj.reshape(-1, t_traj.shape[2])

    # x_traj = o_traj[:, :, 0]
    # y_traj = o_traj[:, :, 1]
    # z_traj = o_traj[:, :, 2]
    # weights_traj = np.repeat(weights, o_traj.shape[1])

    # plt.hist2d(x_traj.flatten(), y_traj.flatten(), bins=200, cmap='viridis', norm='log', weights=weights_traj)
    # plt.xlabel('x [kpc]')
    # plt.ylabel('y [kpc]')
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)
    # plt.show()


    print('weights shape', weights.shape)
    print('orbits shape', o_traj.shape)
    n_time = 1000

    orbits_uniform = np.empty((o_traj.shape[0], n_time, 6), dtype=np.float32)
    times_uniform = np.empty((t_traj.shape[0], n_time), dtype=np.float32)

    t_orb = []
    x_orb = []
    y_orb = []
    z_orb = []
    vx_orb = []
    vy_orb = []
    vz_orb = []
    for i in tqdm(range(o_traj.shape[0])):
        t_old = t_traj[i]
        o_old = o_traj[i]

        x_old = o_old[:, 0]
        y_old = o_old[:, 1]
        z_old = o_old[:, 2]
        vx_old = o_old[:, 3]
        vy_old = o_old[:, 4]
        vz_old = o_old[:, 5]
        # print(t_old.shape, x_old.shape, y_old.shape)

        # remove duplicate timestamps if needed
        keep = np.concatenate(([True], np.diff(t_old) > 0))
        t_old = t_old[keep]
        x_old = x_old[keep]
        y_old = y_old[keep]
        z_old = z_old[keep]
        vx_old = vx_old[keep]
        vy_old = vy_old[keep]
        vz_old = vz_old[keep]

        t_new = np.linspace(t_old[0], t_old[-1], n_time)

        try: 
            x_interp = CubicSpline(t_old, x_old)   # (N_max, 6) -> (n_time, 6)
            y_interp = CubicSpline(t_old, y_old)
            z_interp = CubicSpline(t_old, z_old)
            vx_interp = CubicSpline(t_old, vx_old)
            vy_interp = CubicSpline(t_old, vy_old)
            vz_interp = CubicSpline(t_old, vz_old)

            t_orb.append(t_new)
            x_orb.append(x_interp(t_new))
            y_orb.append(y_interp(t_new))
            z_orb.append(z_interp(t_new))
            vx_orb.append(vx_interp(t_new))
            vy_orb.append(vy_interp(t_new))
            vz_orb.append(vz_interp(t_new))
        except Exception as e:
            print(e)
            print('t_old:', t_old)
            zeros = np.zeros(n_time)
            t_orb.append(t_new)
            x_orb.append(zeros)
            y_orb.append(zeros)
            z_orb.append(zeros)
            vx_orb.append(zeros)
            vy_orb.append(zeros)
            vz_orb.append(zeros)


    with open(output_filename, 'wb') as f:
        pickle.dump({
            't_orb': t_orb,                 # list of (n_time,) arrays
            'x_orb': x_orb,                 # list of (n_time, 3) arrays
            'y_orb': y_orb,                 # list of (n_time, 3) arrays
            'z_orb': z_orb,                 # list of (n_time, 3) arrays
            'vx_orb': vx_orb,               # list of (n_time, 3) arrays
            'vy_orb': vy_orb,               # list of (n_time, 3) arrays
            'vz_orb': vz_orb,               # list of (n_time, 3) arrays
            'weights': weights,                 # (n_orbits_total,)
            'mean_mass_per_orbit': mean_mass_per_orbit / n_time,   # (n_orbits_total,)
            'rotation_matrix': output_dict['rotation_matrix'],
            'Omega_bar': output_dict['Omega_bar'],
            'logl_all': output_dict['logl_all'],
        }, f)
