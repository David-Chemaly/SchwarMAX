import agama
from likelihoods_bar import get_dict_data_bootstrap
agama.setUnits(mass=1, length=1, velocity=1)
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
import pickle
from astropy import units as u
from astropy.constants import G
from constants import KPCGYR_TO_KMS
from utils import get_rotation_curve, build_Lcirc_of_E, logMenc_logc_to_logM_logRs
from potentials import NFW_potential, MiyamotoNagai_potential, T3_potential
from dehnen_bar import V4_potential

from tqdm import tqdm


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

@jax.jit
def potential_func(x, y, z, params_baryon, params_halo):
    """Returns Phi(x,y,z) = NFW + MiyamotoNagai + T3 + V4"""
    phi_halo = NFW_potential(x, y, z, params_halo)

    mn_params = {
        'logM_disc': params_baryon['logM_disc'],
        'Rs_disc': params_baryon['Rs_disc'],
        'Hs_disc': params_baryon['Hs_disc'],
        'x_origin': params_baryon['x_origin'],
        'y_origin': params_baryon['y_origin'],
        'z_origin': params_baryon['z_origin'],
        'dirx': params_baryon['dirx'],
        'diry': params_baryon['diry'],
        'dirz': params_baryon['dirz'],
    }
    phi_mn = MiyamotoNagai_potential(x, y, z, mn_params)

    M_bar = 10.0 ** params_baryon['logM_bar']
    L_bar = params_baryon['L_bar']
    a_bar = params_baryon['a_bar']
    b_bar = params_baryon['b_bar']
    phi_t3 = T3_potential(x, y, z, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)
    phi_v4 = V4_potential(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)

    return phi_halo + phi_mn + phi_t3 + phi_v4


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

def get_orb_samples(lib, X_edges, Y_edges, N_sample=None, seed=42):
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

    if N_sample is not None and N_sample < len(x_model):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_model), size=N_sample, replace=False)
        w_model = w_model[idx]
        x_model = x_model[idx]
        y_model = y_model[idx]
        z_model = z_model[idx]
        vx_model = vx_model[idx]
        vy_model = vy_model[idx]
        vz_model = vz_model[idx]

    return w_model, x_model, y_model, z_model, vx_model, vy_model, vz_model


def load_potential_dict(params):
    """Parse parameter vector into potential dicts."""
    logM_enc = params[0]
    log_c = params[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54)

    logM_disc = params[1]
    logM_bar = params[2]
    logRs_disk = params[4]
    logHs_disk = params[5]
    logL_bar = params[6]
    log_light_to_mass_ratio = params[10]
    log_Omega_bar = params[11]

    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc

    params_halo_pot = {
        'logM': logM_halo,
        'Rs': 10 ** logRs_halo,
        'a': 1.0, 'b': 1.0, 'c': 1.0,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    }

    params_baryon_rho = {
        'logM_disc': logM_disc,
        'Rs_disc': 10 ** logRs_disk,
        'Hs_disc': Hs_disc,
        'logM_bar': logM_bar,
        'L_bar': L_bar,
        'a_bar': a_bar,
        'b_bar': b_bar,
        'light_to_mass_ratio': 10 ** log_light_to_mass_ratio,
        'Omega_bar': 10 ** log_Omega_bar,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    }
    return params_baryon_rho, params_halo_pot


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
    GAMMA_BAR = 1.0

    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
    figname = data_folder + '/plots/orbital_structure_R_vs_circularity.png'
    orbital_library_file = data_folder + '/best_fit_orbital_library.pkl'
    CHECKPOINT_FILE = data_folder + '/ensemble_checkpoint_0415_beta25_gamma140_D50_gal2.pkl'

    # ── Load best-fit parameters ──
    with open(CHECKPOINT_FILE, 'rb') as f:
        checkpoint = pickle.load(f)
    posterior = np.stack(checkpoint['all_samples'], axis=0)  # (steps, chains, params)
    logprob = np.stack(checkpoint['all_logprob'], axis=0)
    DISCARD = 400
    posterior = posterior[:, logprob[-1, :] > np.amax(logprob[-1, :]) - 100, :]
    posterior = posterior[DISCARD::1, :, :]
    posterior = posterior.reshape(-1, posterior.shape[-1])
    best_params = np.percentile(posterior, 50, axis=0)
    params_baryon, params_halo = load_potential_dict(best_params)

    # ── Load orbital library ──
    print("Loading orbital library...")
    with open(orbital_library_file, 'rb') as f:
        lib = pickle.load(f)
    rot_mat = lib['rotation_matrix']

    # ══════════════════════════════════════════════════════════════
    #  N-body: circularity from agama CylSpline potential
    # ══════════════════════════════════════════════════════════════

    print("Loading N-body snapshot...")
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s)**2))
    w0_data, mass_data = agama.readSnapshot(data_folder + '/Bar_model_TG21/model/t_t0_7')
    mass_data = mass_data * mass_unit.value

    unique_masses = np.unique(mass_data)
    mask_halo = mass_data == unique_masses[-1]
    mask_disc = ~mask_halo

    # Iterative centering on disc particles
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

    try:
        pot_nbody = agama.Potential(data_folder + '/Bar_model_TG21/model/t_t0_7.ini')
    except Exception as e:
        print(e)
        print("Agama potential file not found, rebuilding from particles...")
        # Build agama potential from particles (CylSpline expansion)
        print("Building agama CylSpline potential from N-body particles...")
        pos_nbody = w0_data[:, :3]
        pot_nbody = agama.Potential(
            type='CylSpline',
            particles=(pos_nbody, mass_data),
            symmetry='None',
            mmax=6,
            gridSizeR=30,
            gridSizeZ=30,
            Rmin=0.1,
            Rmax=50.0,
            zmin=0.1,
            zmax=50.0,
        )
        pot_nbody.export(data_folder + '/Bar_model_TG21/model/t_t0_7.ini')
        print("  Done.")

    # Compute circularity for disc particles
    pos_disc = w0_data[mask_disc, :3]
    vel_disc = w0_data[mask_disc, 3:]
    R_disc = np.sqrt(pos_disc[:, 0]**2 + pos_disc[:, 1]**2)
    Lz_disc = -(pos_disc[:, 0] * vel_disc[:, 1] - pos_disc[:, 1] * vel_disc[:, 0])

    # Energy = 0.5 v^2 + Phi
    v2_disc = np.sum(vel_disc**2, axis=1)
    Phi_disc = pot_nbody.potential(pos_disc)
    E_disc = 0.5 * v2_disc + Phi_disc

    # Lcirc(E) via agama: Rcirc gives the circular orbit radius for a given energy
    R_circ_disc = pot_nbody.Rcirc(E=E_disc)
    # Vc at Rcirc: v_c^2 = R * |dPhi/dR|, get from agama force
    pos_circ = np.column_stack([R_circ_disc, np.zeros_like(R_circ_disc), np.zeros_like(R_circ_disc)])
    force_circ = pot_nbody.force(pos_circ)  # (N, 3)
    # radial force = -(dPhi/dR) in cylindrical, for points on x-axis: f_R = force_x
    Vc_circ = np.sqrt(np.maximum(R_circ_disc * (-force_circ[:, 0]), 0.0))
    Lcirc_disc = R_circ_disc * Vc_circ
    circularity_disc = Lz_disc / (Lcirc_disc + 1e-10)

    # ══════════════════════════════════════════════════════════════
    #  Model: circularity from parametrised potential
    # ══════════════════════════════════════════════════════════════

    print("Computing model circularity...")
    N_SAMPLE = 10_000_000
    w_model, x_m, y_m, z_m, vx_m, vy_m, vz_m = get_orb_samples(lib, [-100, 100], [-100, 100], N_sample=N_SAMPLE)

    R_model = np.sqrt(x_m**2 + y_m**2)

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        R_model,
        potential_func,
        (params_baryon, params_halo),
        z_m
    )
    key = jax.random.PRNGKey(2024)
    keys = jax.random.split(key, 6)
    n_particles = len(x_m)
    d_scale = 0.1 * jnp.ones(n_particles)
    v_scale = 0.1 * _Vc
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
    x_m = jnp.array((jax.random.normal(keys[0], (n_particles,))) * d_scale + x_m)
    y_m = jnp.array((jax.random.normal(keys[1], (n_particles,))) * d_scale + y_m)
    z_m = jnp.array((jax.random.normal(keys[2], (n_particles,))) * d_scale + z_m)
    vx_m = jnp.array((jax.random.normal(keys[3], (n_particles,))) * v_scale + vx_m)
    vy_m = jnp.array((jax.random.normal(keys[4], (n_particles,))) * v_scale + vy_m)
    vz_m = jnp.array((jax.random.normal(keys[5], (n_particles,))) * v_scale + vz_m)

    Lz_model = -(x_m * vy_m - y_m * vx_m)  # km/s * kpc

    # Convert velocities to kpc/Gyr for energy calculation with our potential
    vx_int = vx_m / KPCGYR_TO_KMS
    vy_int = vy_m / KPCGYR_TO_KMS
    vz_int = vz_m / KPCGYR_TO_KMS
    v2_model = vx_int**2 + vy_int**2 + vz_int**2

    Phi_model = jax.vmap(potential_func, in_axes=(0,0,0, None, None))(
        x_m, y_m, z_m,
        params_baryon, params_halo,
    )
    # # Evaluate potential at orbit positions (in kpc^2/Gyr^2)
    # Phi_model = np.array([float(potential_func(x_m[i], y_m[i], z_m[i], params_baryon, params_halo))
    #                       for i in tqdm(range(len(x_m)))])
    E_model = 0.5 * v2_model + Phi_model

    # Build Lcirc(E) lookup from parametrised potential
    Lcirc_interp, _, _ = build_Lcirc_of_E(potential_func, (params_baryon, params_halo))
    Lcirc_model = Lcirc_interp(E_model)
    # Convert Lz to kpc^2/Gyr for consistency
    Lz_model_internal = Lz_model / KPCGYR_TO_KMS  # kpc^2/Gyr
    circularity_model = Lz_model_internal / (Lcirc_model + 1e-10)

    # ══════════════════════════════════════════════════════════════
    #  Plot: R vs circularity
    # ══════════════════════════════════════════════════════════════

    print("Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)


    ax = axes[0]
    R_cut = R_disc < 10

    # H, xedges, yedges = np.histogram2d(R_disc[R_cut], circularity_disc[R_cut],
    #                                   bins=[100, 100], range=[[0, 10], [-1., 1.]],
    #                                   weights=mass_data[mask_disc][R_cut])
    # H_col = H / np.sum(H, axis = 1)[:, np.newaxis]
    # H_col = H_col/np.amax(H_col, axis = 1)[:, np.newaxis]
    # ax.pcolormesh(xedges, yedges, H_col.T, cmap='inferno',)
    # ax.set_xlabel('R [kpc]')
    # ax.set_ylabel(r'Circularity $\lambda = L_z / L_{\rm circ}(E)$')
    # ax.set_title('N-body')
    # ax.axhline(0, color='white', ls='--', lw=0.5)
    # ax.axhline(1, color='white', ls=':', lw=0.5)
    # ax.axhline(-1, color='white', ls=':', lw=0.5)

    # # Model
    # ax = axes[1]
    # R_cut_m = R_model < 10
    # H, xedges, yedges = np.histogram2d(R_model[R_cut_m], circularity_model[R_cut_m],
    #                                   bins=[100, 100], range=[[0, 10], [-1., 1.]],
    #                                   weights=w_model[R_cut_m])
    # H_col = H / np.sum(H, axis = 1)[:, np.newaxis]
    # H_col = H_col/np.amax(H_col, axis = 1)[:, np.newaxis]
    # ax.pcolormesh(xedges, yedges, H_col.T, cmap='inferno',)
    # ax.set_xlabel('R [kpc]')
    # ax.set_title('Model (best fit)')
    # ax.axhline(0, color='white', ls='--', lw=0.5)
    # ax.axhline(1, color='white', ls=':', lw=0.5)
    # ax.axhline(-1, color='white', ls=':', lw=0.5)

    ax = axes[0]
    R_cut = R_disc < 10
    H, xedges, yedges = np.histogram2d(np.log10(R_disc[R_cut]), circularity_disc[R_cut],
                                      bins=[50, 70], range=[[-1., 1.], [-1., 1.]],
                                      weights=mass_data[mask_disc][R_cut])
    H_col = H / np.sum(H, axis = 1)[:, np.newaxis]
    H_col1 = H_col/np.amax(H_col, axis = 1)[:, np.newaxis]
    ax.pcolormesh(xedges, yedges, H_col1.T, cmap='inferno',)
    ax.set_xlabel(r'$\log_{10}(R \, [{\rm kpc}])$')
    ax.set_ylabel(r'Circularity $\lambda = L_z / L_{\rm circ}(E)$')
    ax.set_title('N-body')
    ax.axhline(0, color='white', ls='--', lw=0.5)
    ax.axhline(1, color='white', ls=':', lw=0.5)
    ax.axhline(-1, color='white', ls=':', lw=0.5)

    # Model
    ax = axes[1]
    R_cut_m = R_model < 10
    H, xedges, yedges = np.histogram2d(np.log10(R_model[R_cut_m]), circularity_model[R_cut_m],
                                      bins=[50, 70], range=[[-1., 1.], [-1., 1.]],
                                      weights=w_model[R_cut_m])
    H_col = H / np.sum(H, axis = 1)[:, np.newaxis]
    H_col2 = H_col/np.amax(H_col, axis = 1)[:, np.newaxis]
    ax.pcolormesh(xedges, yedges, H_col2.T, cmap='inferno',)
    ax.set_xlabel(r'$\log_{10}(R \, [{\rm kpc}])$')
    ax.set_title('Model (best fit)')
    ax.axhline(0, color='white', ls='--', lw=0.5)
    ax.axhline(1, color='white', ls=':', lw=0.5)
    ax.axhline(-1, color='white', ls=':', lw=0.5)

    ax = axes[2]
    ax.pcolormesh(xedges, yedges, (H_col1 - H_col2).T, cmap='coolwarm',vmin=-0.5, vmax=0.5)
    ax.set_xlabel(r'$\log_{10}(R \, [{\rm kpc}])$')
    ax.set_title('Difference (N-body - Model)')
    ax.axhline(0, color='white', ls='--', lw=0.5)
    ax.axhline(1, color='white', ls=':', lw=0.5)
    ax.axhline(-1, color='white', ls=':', lw=0.5)


    fig.tight_layout()
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    print(f"Saved to {figname}")