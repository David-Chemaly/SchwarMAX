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
    CHECKPOINT_FILE = data_folder + '/ensemble_checkpoint_gal2_0406.pkl'
    data_filename = 'mock_Nbody_bar_XY_withRot_gal2_Nbins1000.pkl'
    # data_filename = 'mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma170.pkl'

    figname = data_folder + '/plots/velocity_quadrupole.png'
    output_filename = data_folder + '/best_fit_orbital_library.pkl'
    DISCARD = 500
    THIN = 100
    m_max = 5

    # Radial bins for Fourier decomposition (sky plane)
    R_edges = np.linspace(0.5, 6.0, 12)

    # ── Load N-body snapshot ──
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s)**2))
    w0_data, mass_data = agama.readSnapshot(
        data_folder + '/Bar_model_TG21/model/t_t0_4')
    mass_data = mass_data * mass_unit.value

    unique_masses = np.unique(mass_data)
    mask_halo = mass_data == unique_masses[-1]
    mask_disc = ~mask_halo

    # Centre on stellar component
    R = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2)
    mask_center = R < 5
    for col in range(6):
        w0_data[:, col] -= np.sum(w0_data[mask_center, col] * mass_data[mask_center]) / np.sum(mass_data[mask_center])

    w0_data[:, 0] = -w0_data[:, 0]
    w0_data[:, 3] = -w0_data[:, 3]

    # Align bar with x-axis
    R_mid_ba, bar_angles0, _ = bar_angle_bar_strength(
        w0_data[:, 0], w0_data[:, 1], R_anulus=np.arange(1, 5, 0.1))
    bar_angle0 = np.mean(bar_angles0[R_mid_ba < 4])
    w0_data = rotate(w0_data, -bar_angle0)

    # ── N-body velocity Fourier decomposition ──
    # Use identity rotation for now (bar aligned with x); override with model rotation if known
    rot_identity = np.eye(3)
    R_mid_data, k_m_data, phi_m_data = nbody_velocity_fourier(
        w0_data, mass_data, mask_disc, rot_identity, R_edges, m_max)

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
                    alpha,
                    beta,
                    gamma,
                    logLM_best_fit,
                    # 0.,
                    logOmega_best_fit,
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

    surface_density_model = projection(params_disk_rho, dict_data, dict_data['total_bins'])
    surface_density_gt = dict_data['XY_density_data'] / params_disk_rho['light_to_mass_ratio']

    chi2 = jnp.sum((surface_density_gt - surface_density_model)**2 / (0.1 * surface_density_gt)**2)
    _logl_density = -0.5 * chi2 / dict_data['total_bins']
    print(_logl_density)

    output_dict = model_diagnostic(params_halo_pot, params_disk_rho, dict_data, dict_data['total_bins'])
    logL = output_dict['logl_all'][0]
    print('logL:', logL)
    print('Model Done')

    weights = output_dict['weights']        # (n_orbits,)
    o_traj = output_dict['y_traj']            # (n_orbits, 4, N_max, 6)
    t_traj = output_dict['t_traj']            # (n_orbits, 4, N_max)

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
            'rotation_matrix': output_dict['rotation_matrix'],
            'Omega_bar': output_dict['Omega_bar'],
            'logl_all': output_dict['logl_all'],
        }, f)
