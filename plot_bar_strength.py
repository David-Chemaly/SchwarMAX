import agama
agama.setUnits(mass=1, length=1, velocity=1)
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from utils import logMenc_logc_to_logM_logRs
from densities import MiyamotoNagai_density
from dehnen_bar import T3_density, V4_density
from tqdm import tqdm

import pickle

from astropy import units as u
from astropy.constants import G

def plot_prettier(dpi=200, fontsize=12, usetex=False):
    '''
    Make plots look nicer compared to Matplotlib defaults
    Parameters:
        dpi - int, "dots per inch" - controls resolution of PNG images that are produced
                by Matplotlib
        fontsize - int, font size to use overall
        usetex - bool, whether to use LaTeX to render fonds of axes labels
                use False if you don't have LaTeX installed on your system
    '''
    plt.rcParams['figure.dpi']= dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    # if you don't have LaTeX installed on your laptop and this statement
    # generates error, comment it out
    plt.rc('text', usetex=usetex)
    #fm.fontManager.addfont('Serif.ttf')
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plot_prettier(usetex=False)


def bar_angle_bar_strength(x, y, R_anulus = np.arange(1,5,0.25)):

    ''' 
    Calculate the angle and strength of the dipole strength in each radial bin.
    Parameters:
        x, y - ndarray, Cartesian coordinates of the stars
        R_anulus - ndarray, edges of the radial bins to use for calculating the dipole angle and strength
    Returns:
        R_mid - ndarray, midpoints of the radial bins
        bar_angles0 - ndarray, angles of the dipole strength in each radial bin
        bar_strength0 - ndarray, strengths of the dipole strength in each radial bin

    '''

    R0,phi0 = np.sqrt(x**2 + y**2), np.arctan2(y,x)

    bar_angles0 = []
    bar_strength0 = []
    R_anulus = R_anulus
    for i in range(len(R_anulus)-1):
        sel0 = (R0>R_anulus[i]) & (R0<R_anulus[i+1])

        r_min_i = (R_anulus[i]+R_anulus[i+1])/2
        
        R0_bin, phi0_bin = R0[sel0], phi0[sel0]

        A2_0 = np.sum(np.cos(2*phi0_bin))/len(phi0_bin)
        B2_0 = np.sum(np.sin(2*phi0_bin))/len(phi0_bin)

        angle_bar0 = 0.5*np.arctan2(B2_0,A2_0)
        bar_strength = np.sqrt(A2_0**2 + B2_0**2)

        bar_angles0.append(angle_bar0)
        bar_strength0.append(bar_strength
                             )
    R_mid = (R_anulus[:-1]+R_anulus[1:])/2
    bar_angles0 = np.array(bar_angles0)
    bar_strength0 = np.array(bar_strength0)

    return R_mid, bar_angles0, bar_strength0

def get_model_A2(params_baryon_rho, R_edges, zmax=5.0, n_phi=128, n_z=50):
    """
    Compute bar strength A2/A0 as a function of radius from the analytic density model.

    For each radial bin, evaluates the density on a (phi, z) grid, integrates
    over z to get Sigma(R, phi), then computes the m=2 Fourier amplitude.

    Parameters
    ----------
    params_baryon_rho : dict
        Baryon density parameters (same format as load_density_dict output).
    R_edges : array
        Radial bin edges [kpc].
    zmax : float
        Vertical integration range [-zmax, zmax] in kpc.
    n_phi : int
        Number of azimuthal grid points.
    n_z : int
        Number of vertical grid points.

    Returns
    -------
    R_mid : array
        Midpoints of radial bins.
    A2 : array
        Bar strength A2/A0 in each radial bin.
    """
    R_mid = 0.5 * (R_edges[:-1] + R_edges[1:])
    phi = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)
    z = jnp.linspace(-zmax, zmax, n_z)
    dz = z[1] - z[0]

    # Build 3D grid: (n_R, n_phi, n_z)
    R_3d, PHI_3d, Z_3d = jnp.meshgrid(jnp.array(R_mid), phi, z, indexing='ij')
    x_flat = (R_3d * jnp.cos(PHI_3d)).ravel()
    y_flat = (R_3d * jnp.sin(PHI_3d)).ravel()
    z_flat = Z_3d.ravel()

    n_R = len(R_mid)
    rho = density_func(x_flat, y_flat, z_flat, params_baryon_rho)
    rho = rho.reshape(n_R, n_phi, n_z)

    # Integrate over z -> Sigma(R, phi)
    Sigma = jnp.sum(rho, axis=2) * dz  # (n_R, n_phi)

    # Fourier decomposition per radial bin
    a0 = jnp.mean(Sigma, axis=1)  # (n_R,)
    a2_cos = jnp.mean(Sigma * jnp.cos(2 * phi[None, :]), axis=1)
    a2_sin = jnp.mean(Sigma * jnp.sin(2 * phi[None, :]), axis=1)
    A2 = jnp.sqrt(a2_cos**2 + a2_sin**2) / (a0 + 1e-30)

    return np.array(R_mid), np.array(A2)


def rotate(posvel, angle):
    # Rotate contourclockwise with positive angle
    x, y, z, vx, vy, vz = posvel.T
    sina, cosa = np.sin(angle), np.cos(angle)
    return np.array([x*cosa-y*sina, x*sina+y*cosa, z, vx*cosa-vy*sina, vx*sina+vy*cosa, vz]).T


def scalarize(x):
    return x if len(x)>1 else x[0]

def v_circ(pot, r):
    return scalarize((-r * pot.force(np.column_stack((r, r*0, r*0)))[:,0]) ** 0.5)

def compute_potential(pos, mass):
    pot_axi = agama.Potential(type = "CylSpline", particles=(pos,mass), symmetry = "axisymmetric",gridSizeR = 30, gridSizez = 25, Rmin = 0.15, Rmax = 50, zmin = 0.1, zmax = 20)
    return pot_axi

@jax.jit
def density_func(x, y, z, params):
    """
    Returns Stellar Density nu(R, z) using:
      MiyamotoNagai disc + T3 bar + V4 bulge
    """
    # MN disc density
    mn_params = {
        'logM_disc': params['logM_disc'],
        'Rs_disc': params['Rs_disc'],
        'Hs_disc': params['Hs_disc'],
        'x_origin': params['x_origin'],
        'y_origin': params['y_origin'],
        'z_origin': params['z_origin'],
        'dirx': params['dirx'],
        'diry': params['diry'],
        'dirz': params['dirz'],
    }
    rho_mn = MiyamotoNagai_density(x, y, z, mn_params)

    # T3 bar density
    M_bar = 10.0 ** params['logM_bar']
    L_bar = params['L_bar']
    a_bar = params['a_bar']
    b_bar = params['b_bar']
    rho_t3 = T3_density(x, y, z, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)

    # V4 bulge density (M = M_bar, fixed shape)
    rho_v4 = V4_density(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)

    return rho_mn + rho_t3 + rho_v4


def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        ckpt = pickle.load(f)

    all_samples = ckpt['all_samples']   # list of (N_CHAINS, NDIM) arrays
    all_logprob = ckpt['all_logprob']   # list of (N_CHAINS,) arrays
    step = ckpt['step']


    # Stack into (N_STEPS, N_CHAINS, NDIM) and (N_STEPS, N_CHAINS)
    posterior = np.stack(all_samples, axis=0)
    logprob = np.stack(all_logprob, axis=0)
    
    print(f"Loaded checkpoint: {step} steps, "
          f"{posterior.shape[1]} chains, {posterior.shape[2]} params")
    print(f"Chain shape: {posterior.shape}")
    return posterior, logprob, step


def load_density_dict(params):


    logM_enc = params[0]
    log_c = params[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54)

    logM_disc = params[1]
    logM_bar = params[2]
    logRs_disk = params[4]
    logHs_disk = params[5]
    logL_bar = params[6]
    alpha = params[7]
    beta = params[8]
    gamma = params[9]
    log_light_to_mass_ratio = params[10]
    log_Omega_bar = params[11]

    sigma_density_model = 0#10**params[12]
    sigma_kine_model = 0. 
    sigma_amplifier = 10**params[12]

    alpha = alpha * 180 / jnp.pi
    beta = beta * 180 / jnp.pi
    gamma = gamma * 180 / jnp.pi

    # Derived bar parameters
    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc

    params_halo_pot = {
        'logM': logM_halo,
        'Rs':10 ** logRs_halo,
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
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
    }

    return params_baryon_rho, params_halo_pot

if __name__ == '__main__':
    V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
    GAMMA_BAR = 1.0


    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

    figname = data_folder+'/plots/bar_strength_data_vs_model.png'
    # CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_gal2_0406.pkl'
    CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_0415_beta25_gamma140_D50_gal2.pkl'
    
    DISCARD=300
    THIN=100
    posterior, logprob, step = load_checkpoint(CHECKPOINT_FILE)
    posterior = posterior[:, logprob[-1, :]>np.amax(logprob[-1, :])-100, :]
    posterior = posterior[DISCARD::THIN, :, :]
    posterior = posterior.reshape(-1, posterior.shape[-1])

    best_params = np.percentile(posterior, 50, axis=0)

    print('shape of posterior', posterior.shape)
    param_names = ['logM_halo', 'logM_disk', 'logM_bar', 'logRs_halo', 'logRs_disk', 'logHs_disk', 'logRs_bar', 
               r'$\alpha$', r'$\beta$', r'$\gamma$', 
               r'$\log_{10}(L/M)$', r'$\log_{10}(\Omega)$', r'$\log_{10}(\sigma amplifier)$']
    print("Best-fit parameters:")
    for i, name in enumerate(param_names):
        print(f"{name}: {best_params[i]:.4f}")

    mass_unit = 1/((G*u.Msun).to(u.kpc*(u.km/u.s)**2))
    w0_data, mass_data = agama.readSnapshot(data_folder+'/Bar_model_TG21/model/t_t0_7')#snap_t0_3
    mass_data = mass_data * mass_unit.value


    print(np.unique(mass_data, return_counts=True))

    # R = np.sqrt(w0_data[:,0]**2 + w0_data[:,1]**2)
    # mask = R < 5
    # w0_data[:,0] = w0_data[:,0] - np.sum(w0_data[mask,0] * mass_data[mask]) / np.sum(mass_data[mask])
    # w0_data[:,1] = w0_data[:,1] - np.sum(w0_data[mask,1] * mass_data[mask]) / np.sum(mass_data[mask])
    # w0_data[:,2] = w0_data[:,2] - np.sum(w0_data[mask,2] * mass_data[mask]) / np.sum(mass_data[mask])
    # w0_data[:,3] = w0_data[:,3] - np.sum(w0_data[mask,3] * mass_data[mask]) / np.sum(mass_data[mask])
    # w0_data[:,4] = w0_data[:,4] - np.sum(w0_data[mask,4] * mass_data[mask]) / np.sum(mass_data[mask])
    # w0_data[:,5] = w0_data[:,5] - np.sum(w0_data[mask,5] * mass_data[mask]) / np.sum(mass_data[mask])
    
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


    w0_data[:,0] = -w0_data[:,0]
    w0_data[:,3] = -w0_data[:,3]

    R_mid, bar_angles0, bar_strength0 = bar_angle_bar_strength(w0_data[:,0], w0_data[:,1], R_anulus = np.arange(1,5,0.1))
    bar_angle0 = np.mean(bar_angles0[R_mid<4])
    rot_angle = -bar_angle0
    w0_data = rotate(w0_data, rot_angle)  # rotate to make it anticlockwise

    # ── Data A2 from N-body particles ──
    R_anulus_A2 = np.arange(0.25, 10.0, 0.25)
    R_mid_data, bar_angles0, bar_strength_data = bar_angle_bar_strength(
        w0_data[:,0], w0_data[:,1], R_anulus=R_anulus_A2)

    # ── Model A2 from posterior samples ──
    A2_R_edges = R_anulus_A2
    A2_all = []
    for i in tqdm(range(posterior.shape[0]), desc="Computing A2"):
        params_baryon_rho, params_halo_pot = load_density_dict(posterior[i])
        _, A2_i = get_model_A2(params_baryon_rho, A2_R_edges)
        A2_all.append(A2_i)

    A2_all = np.array(A2_all)
    A2_16, A2_50, A2_84 = np.percentile(A2_all, [5, 50, 95], axis=0)
    R_mid_model = 0.5 * (A2_R_edges[:-1] + A2_R_edges[1:])

    # ── Plot ──
    plt.figure(figsize=(6, 4))
    plt.plot(R_mid_data, bar_strength_data, lw=3, alpha=1, label='N-body data', color='royalblue')
    plt.fill_between(R_mid_model, A2_16, A2_84, color='tomato', alpha=0.2, label=r'Model $1\sigma$')
    plt.plot(R_mid_model, A2_50, lw=3, ls='--', alpha=0.7, label='Model median', color='tomato')

    plt.xlabel('Galactocentric Radius, R [kpc]')
    plt.ylabel(r'Bar strength, $A_2 / A_0$')
    plt.xlim(0, 10)
    plt.ylim(0, None)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.show()
