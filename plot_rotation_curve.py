import agama
agama.setUnits(mass=1, length=1, velocity=1)
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from utils import get_rotation_curve, logMenc_logc_to_logM_logRs
from potentials import *
from dehnen_bar import T3_potential, V4_potential
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


def load_potential_dict(params):


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

    figname = data_folder+'/plots/rotation_curve_data_vs_model.png'
    # CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_gal2_0406.pkl'
    CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_0415_beta25_gamma140_D50_gal2.pkl'
    
    DISCARD=400
    THIN=1
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

    R = np.sqrt(w0_data[:,0]**2 + w0_data[:,1]**2)
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

    R_mid, bar_angles0, bar_strength0 = bar_angle_bar_strength(w0_data[:,0], w0_data[:,1], R_anulus = np.arange(1,5,0.25))
    bar_angle0 = np.mean(bar_angles0[R_mid<4])
    rot_angle = -bar_angle0
    w0_data = rotate(w0_data, rot_angle)  # rotate to make it anticlockwise

    r = np.sqrt(w0_data[:,0]**2 + w0_data[:,1]**2 + w0_data[:,2]**2)
    pos = w0_data[r<50,:3]
    mass = mass_data[r<50]

    pot_axi = compute_potential(pos, mass)
    r_plot = np.linspace(0.01, 20, 100)
    v_circular = v_circ(pot_axi, r_plot)
    plt.figure(figsize=(4,3))
    plt.plot(r_plot, v_circular, lw = 3, alpha = 1, label='Ground truth', color = 'royalblue')

    # params_baryon_rho, params_halo_pot = load_potential_dict(best_params)
    # Vc_model = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
    #     jnp.array(r_plot),
    #     potential_func,
    #     (params_baryon_rho, params_halo_pot),
    #     jnp.array(r_plot*0)
    # )
    # plt.plot(r_plot, Vc_model, lw = 3, alpha = 0.8, label='Best-fit model', color = 'tomato')

    Vc_model = []
    for i in tqdm(range(0, posterior.shape[0])):
        params_baryon_rho, params_halo_pot = load_potential_dict(posterior[i])
        _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
            jnp.array(r_plot),
            potential_func,
            (params_baryon_rho, params_halo_pot),
            jnp.array(r_plot*0)
        )
        Vc_model.append(_Vc)
    
    Vc_model = np.array(Vc_model)
    Vc_model_16, Vc_model_50, Vc_model_84 = np.percentile(Vc_model, [5, 50, 95], axis=0)
    plt.fill_between(r_plot, Vc_model_16, Vc_model_84, color='tomato', alpha=0.2, label=r'$1\sigma$ interval')
    plt.plot(r_plot, Vc_model_50, lw = 2, ls = '--', alpha = 0.7, label='Best-fit model', color = 'tomato')
    

    plt.xlim(0., 20.)
    plt.axvline(10., color='grey', ls='--', lw=1, alpha=0.7, label = 'Data extent')
    plt.ylim(50, None)
    plt.xlabel('Galactocentric Radius, R [kpc]')
    plt.ylabel('Circular Velocity, $V_c$ [km/s]')
    plt.legend(frameon=False, fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.show()

