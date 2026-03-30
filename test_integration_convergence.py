"""
Realistic comparison: adaptive BS23
Uses the full 5k particle setup from test_integration.py (no realisations).
"""
import agama
agama.setUnits(mass=1, length=1, velocity=1)

from constants import *
from integrants_with_binning import (
    _integrate_adaptive_batch_vmap, _integrate_adaptive_vmap, _integrate_adaptive_chunked_vmap
)
from sample_from_density import sample_from_density_grid
from densities import *
from potentials import *
from utils import *
from model import *
from CylindricalSpline import get_phi_m, get_acc, evaluate_phi_axisymmetric

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pickle
import time

from densities import MiyamotoNagai_density
from dehnen_bar import T3_density, T3_acceleration, T3_potential, V4_density, V4_acceleration, V4_potential

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

# ══════════════════════════════════════════════════════════════
# Setup (copied from test_integration.py)
# ══════════════════════════════════════════════════════════════


def get_dict_data(path):

    with open(path + 'mock_Nbody_bar_XY_withRot.pkl', 'rb') as f:
        bin_dict = pickle.load(f)

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

    # V_data_err = jnp.where(0.1 * jnp.fabs(V_data) < 10, 10, 0.1 * V_data)
    # sigma_data_err = jnp.where(0.1 * jnp.fabs(sigma_data) < 5, 5, 0.1 * sigma_data)
    # h1_data_err = jnp.where(0.1 * jnp.fabs(h1_data) < 0.03, 0.03, 0.1 * jnp.fabs(h1_data))
    # h2_data_err = jnp.where(0.1 * jnp.fabs(h2_data) < 0.03, 0.03, 0.1 * jnp.fabs(h2_data))
    # h3_data_err = jnp.where(0.1 * jnp.fabs(h3_data) < 0.03, 0.03, 0.1 * jnp.fabs(h3_data))
    # h4_data_err = jnp.where(0.1 * jnp.fabs(h4_data) < 0.03, 0.03, 0.1 * jnp.fabs(h4_data))
    V_data_err = jnp.array(bin_dict['V_mean_err'])
    sigma_data_err = jnp.array(bin_dict['V_sigma_err'])
    h1_data_err = jnp.array(bin_dict['h1_err'])
    h2_data_err = jnp.array(bin_dict['h2_err'])
    h3_data_err = jnp.array(bin_dict['h3_err'])
    h4_data_err = jnp.array(bin_dict['h4_err'])

    # df_Rzphi_data = pd.read_csv(path + 'mock_axisymmetric_disc_Rzphi.csv')
    # Rzphi_density_data = jnp.array(df_Rzphi_data['mass'].to_numpy()).astype(jnp.float32)
    with open(path + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
        Rzphi_density_data = pickle.load(f)

    R_grid, z_grid, phi_grid = Rzphi_density_data['R_grid'], Rzphi_density_data['z_grid'], Rzphi_density_data['phi_grid']
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    sample_for_integration = Rzphi_density_data['sample_for_integration']

    from scipy.stats import qmc
    X_regular_grid, Y_regular_grid = bin_dict['X_regular_grid'], bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)

    n_samples = 5_000  # Same number as original data
    x_grid = np.linspace(0, 12, 1000)
    logP_xexp = XexpX_pdf_log(x_grid, 4.0)
    key = jax.random.PRNGKey(10086)
    R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.uniform(0, 2*np.pi, size=n_samples)

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

        'R_grid': R_grid,
        'z_grid': z_grid,
        'phi_grid': phi_grid,
        'dR': dR,
        'dz': dz,
        'dphi': dphi,
        'sample_for_integration': sample_for_integration,

        'X_regular_grid': X_regular_grid,
        'Y_regular_grid': Y_regular_grid,
        'dX': dX,
        'dY': dY,
        'sample_for_integration_XY': sample,

        'w0': w0
    }

    return dict_data

# ---- Fixed V4 bulge parameters ----
V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
GAMMA_BAR = 1.0


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

@jax.jit
def dPhi_dz(x, y, z, params_baryon, params_halo):
    # Numerical derivative of Potential w.r.t z
    d = 5e-3
    return (potential_func(x, y, z+d, params_baryon, params_halo) - potential_func(x, y, z-d, params_baryon, params_halo)) / (2*d)

@jax.jit
def dPhi_dR(x, y, z, params_baryon, params_halo):
    # Numerical derivative of Potential w.r.t R
    d = 5e-3
    R = jnp.sqrt(x**2 + y**2)
    return (potential_func(R+d, 0, z, params_baryon, params_halo) - potential_func(R-d, 0, z, params_baryon, params_halo)) / (2*d)

@jax.jit
def get_jeans_moments(x_star, y_star, z_star, params_baryon, params_disc, params_halo, anisotropy_b=1.0):
    """
    Computes (v_mean, sigma_R, sigma_z, sigma_phi) for a star at (R, z).
    """
    R_star = jnp.sqrt(x_star**2 + y_star**2)

    # --- Step 1: Compute Sigma_z (Vertical Integration) ---
    def integrand(z_prime):
        return density_func(x_star, y_star, z_prime, params_disc) * dPhi_dz(x_star, y_star, z_prime, params_baryon, params_halo)

    pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
    dx = pts[1] - pts[0]
    integrand_val = jax.vmap(integrand, in_axes = (0))(pts)
    integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)

    nu_val = density_func(x_star, y_star, z_star, params_disc)

    sigma_z2 = (1.0 / nu_val) * integral_val
    sigma_z2 = jnp.maximum(sigma_z2, 0.0)
    sigma_z = jnp.sqrt(sigma_z2)

    # --- Step 2: Compute Sigma_R (Anisotropy assumption) ---
    sigma_R2 = anisotropy_b * sigma_z2
    sigma_R = jnp.sqrt(sigma_R2)

    # --- Step 3: Compute v_phi_total^2 (Radial Equation) ---
    def vertical_pressure(r_in):
        def integrand_r(z_prime):
            return density_func(r_in, 0, z_prime, params_disc) * dPhi_dz(r_in, 0, z_prime, params_baryon, params_halo)

        pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
        dx = pts[1] - pts[0]
        integrand_val = jax.vmap(integrand_r, in_axes = (0))(pts)
        integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)
        return integral_val

    dR = 5e-3
    Pzz_plus = vertical_pressure(R_star + dR)
    Pzz_minus = vertical_pressure(R_star - dR)
    d_nu_sigR2_dR = anisotropy_b * (Pzz_plus - Pzz_minus) / (2*dR)

    term1 = sigma_R2
    term2 = (R_star / nu_val) * d_nu_sigR2_dR
    term3 = R_star * dPhi_dR(x_star, y_star, z_star, params_baryon, params_halo)

    v_phi_total_sq = term1 + term2 + term3

    # --- Step 4: Separate Rotation vs Dispersion ---
    sigma_phi = sigma_R
    v_streaming_sq = v_phi_total_sq - sigma_phi**2
    v_streaming_sq = jnp.maximum(v_streaming_sq, 0.0)
    v_mean_phi = jnp.sqrt(v_streaming_sq)

    output = jax.lax.cond(nu_val<=0, lambda: (0.0, 0.0, 0.0, 0.0), lambda: (v_mean_phi, sigma_R, sigma_z, sigma_phi))

    return output


dict_data = get_dict_data(path)

# Parameters
logM_halo, logM_disc, logM_bar, logRs_halo = 11.8, 10.7, 10.1, 1.2
logRs_disc, logHs_disc, logL_bar = 0.8, -0.24, 0.5
alpha_best_fit, beta_best_fit, gamma_best_fit = 30, 20, 140
logLM, logOmega_bar = 0, 1.6

L_bar = 10.0 ** logL_bar
a_bar = L_bar / 5.0
Hs_disc = 10.0 ** logHs_disc
b_bar = Hs_disc
light_to_mass_ratio = 10.0 ** logLM

params_disk_rho = {
    'logM_disc': logM_disc,
    'Rs_disc': 10.0 ** logRs_disc,
    'Hs_disc': 10.0 ** logHs_disc,
    'logM_bar': logM_bar,
    'L_bar': L_bar,
    'a_bar': a_bar,
    'b_bar': b_bar,
    'x_origin': 0.0,
    'y_origin': 0.0,
    'z_origin': 0.0,
    'dirx': 0.0,
    'diry': 0.0,
    'dirz': 1.0,
    'alpha': alpha_best_fit,
    'beta': beta_best_fit,
    'gamma': gamma_best_fit,
    'light_to_mass_ratio': light_to_mass_ratio,
    'Omega_bar': 10 ** logOmega_bar,
}

params_baryon = {
    'logM_disc': params_disk_rho['logM_disc'],
    'Rs_disc': params_disk_rho['Rs_disc'],
    'Hs_disc': params_disk_rho['Hs_disc'],
    'x_origin': params_disk_rho['x_origin'],
    'y_origin': params_disk_rho['y_origin'],
    'z_origin': params_disk_rho['z_origin'],
    'dirx': params_disk_rho['dirx'],
    'diry': params_disk_rho['diry'],
    'dirz': params_disk_rho['dirz'],
    'logM_bar': params_disk_rho['logM_bar'],
    'L_bar': L_bar,
    'a_bar': a_bar,
    'b_bar': b_bar,
}

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

w0 = dict_data['w0']
n_particles = w0.shape[0]
v0 = dict_data['v0']
s = dict_data['s']
num_per_bin = dict_data['num_per_bin']
bin_mapping = dict_data['bin_mapping']
Omega_bar = params_disk_rho['Omega_bar']
alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
rotation_matrix = makeRotationMatrix(alpha, beta, gamma)
num_Vbin = dict_data['total_bins']

#=========================================== GET INITIAL VELOCITY ===================================================

get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
def get_w0_new(w0, key1, key2, key3, n_particles):
    jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], params_baryon, params_disk_rho, params_halo_pot, 1.)
    v_rot, sig_R, sig_z, sig_phi = jeans_moments
    g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
    vR = g1 * sig_R
    vz = g2 * sig_z
    vphi = v_rot + g3 * sig_phi
    x, y, vx, vy = getCartesianFromCylindrical_clockwise(jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
    return jnp.array([x, y, w0[:,2], vx, vy, vz]).T
key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
w0_new = get_w0_new(w0, key1, key2, key3, n_particles)

#======================================== Calculate orbital timescale =====================================================
_R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
_z = w0_new[:,2]

_Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
    _R,
    potential_func,
    (params_baryon, params_halo_pot),
    _z
)

n_realizations = 4
key = jax.random.PRNGKey(911)
keys = jax.random.split(key, 6)
d_scale = 0.1 * jnp.ones(_R.shape)
v_scale = 0.1 * _Vc
v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
noise_x = (jax.random.uniform(keys[0], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
noise_y = (jax.random.uniform(keys[1], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
noise_z = (jax.random.uniform(keys[2], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]

w0_new_batch = w0_new[:, jnp.newaxis, :]
w0_new_batch = w0_new_batch + jnp.stack([noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)
T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
    _R,
    potential_func,
    (params_baryon, params_halo_pot),
    _z
)
T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

#=========================================== Integrate orbits =======================================================
# Single combined potential + grad for acceleration — one forward + one backward pass
@jax.jit
def acc_fn(x, y, z):
    def _pot(pos):
        return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
    grad_phi = jax.grad(_pot)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def pot_fn(x, y, z):
    return potential_func(x, y, z, params_baryon, params_halo_pot)

Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]])
Rzphi_n_grid = jnp.array([10,6,6])
xy_n_grid = jnp.array([60,40])
Rzphi_n_tot = 360

N_step_per_orb = 100
N_dynamical_time = 50
N_max = N_step_per_orb * N_dynamical_time
T_total = T_orb * N_dynamical_time
dt_init = T_orb / N_step_per_orb
atol, rtol = 1e-7, 1e-4
dt_min, dt_max = 1e-5, 0.3

time_start = time.time()
# Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, _, t_total = _integrate_adaptive_vmap(
#                     w0_new, acc_fn, pot_fn, N_max, T_total,
#                     dt_init, -Omega_bar,
#                     atol, rtol,
#                     dt_min, dt_max,
#                     num_Vbin, bin_mapping, num_per_bin,
#                     Rzphi_lim_grid, xy_lim_grid,
#                     Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
#                     v0, s, rotation_matrix)
_integrate_adaptive_chunked_vmap
Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, _, t_total = _integrate_adaptive_chunked_vmap(
                    w0_new, acc_fn, pot_fn, N_max, T_total,
                    dt_init, -Omega_bar,
                    atol, rtol,
                    dt_min, dt_max,
                    num_Vbin, bin_mapping, num_per_bin,
                    Rzphi_lim_grid, xy_lim_grid,
                    Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                    v0, s, rotation_matrix,
                    5000)
A_Rzphi = Rzphi_bin_counts.T
A_xy = surface_density.T
A_h1 = h1.T
A_h2 = h2.T
A_h3 = h3.T
A_h4 = h4.T
Rzphi_bin_counts.block_until_ready()
print("Number of valid orbits:", jnp.sum(valid))
print("Integration time:", time.time() - time_start, "seconds")

@jax.jit
def density_func_Rz(R, z, phi, params):
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    return density_func(x, y, z, params)

@partial(jax.jit, static_argnames=['rho_fct'])
def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
    R_samples = R_grid + (sample[:,0] - 0.5) * dR
    z_samples = z_grid + (sample[:,1] - 0.5) * dz
    phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
    density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
    mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
    return mass_tot

R_grid, dR = dict_data['R_grid'], dict_data['dR']
z_grid, dz = dict_data['z_grid'], dict_data['dz']
phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
            R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
)

y_xy = dict_data['XY_density_data'].astype(jnp.float32)
y_h1 = dict_data['h1_data'].astype(jnp.float32)
y_h2 = dict_data['h2_data'].astype(jnp.float32)
y_h3 = dict_data['h3_data'].astype(jnp.float32)
y_h4 = dict_data['h4_data'].astype(jnp.float32)

y_xy = y_xy / params_disk_rho['light_to_mass_ratio']

sig_Rzphi = 0.02 * y_Rzphi + 1e-10
sig_xy = 0.01 * y_xy + 1e-10
h_err_min = 0.03
sig_A1 = jnp.where(h_err_min > dict_data['h1_data_err'], h_err_min, dict_data['h1_data_err']) + EPSILON
sig_A2 = jnp.where(h_err_min > dict_data['h2_data_err'], h_err_min, dict_data['h2_data_err']) + EPSILON
sig_A3 = jnp.where(h_err_min > dict_data['h3_data_err'], h_err_min, dict_data['h3_data_err']) + EPSILON
sig_A4 = jnp.where(h_err_min > dict_data['h4_data_err'], h_err_min, dict_data['h4_data_err']) + EPSILON

mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]

y_xy = y_xy / mean_mass_per_orb
sig_xy = sig_xy / mean_mass_per_orb
y_Rzphi = y_Rzphi / mean_mass_per_orb
sig_Rzphi = sig_Rzphi / mean_mass_per_orb

path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
with open(path_data + 'orbital_library_adaptive_Nmax5e3_Chunk5000.pkl', 'wb') as f:
    orb_lib = (A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, \
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4, \
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, t_total/T_orb)
    pickle.dump(orb_lib, f)


compare=False
if compare:

    path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
    with open(path_data + 'orbital_library_adaptive_Nmax1e4.pkl', 'rb') as f:
        orb_lib = pickle.load(f)
    (A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, \
            y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4, \
            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, N_dyn) = orb_lib

    fig, ax = plt.subplots(1,1,figsize = (10,4))
    ax.hist(np.log10(N_dyn), bins=20, alpha=0.5, range = [1,3])
    ax.set(xlabel='log10(Number of Dynamical Times Integrated)', ylabel='Number of Orbits', yscale = 'log')

    weights = jnp.ones(A_Rzphi.shape[1])
    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1 @ weights) / density_2DXY # density_2DXY
    h2_model = (A_h2 @ weights) / density_2DXY # density_2DXY
    h3_model = (A_h3 @ weights) / density_2DXY # density_2DXY
    h4_model = (A_h4 @ weights) / density_2DXY # density_2DXY

    X_regular_grid = dict_data['X_regular_grid']
    Y_regular_grid = dict_data['Y_regular_grid']

    bin_mapping = dict_data['bin_mapping']
    index_remap = bin_mapping[:-1]
    density_2DXY_weighted1 = density_2DXY[index_remap]
    h1_model_weighted1 = h1_model[index_remap]
    h2_model_weighted1 = h2_model[index_remap]
    h3_model_weighted1 = h3_model[index_remap]
    h4_model_weighted1 = h4_model[index_remap]


    path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
    with open(path_data + 'orbital_library_adaptive_Nmax1e4.pkl', 'rb') as f:
        orb_lib = pickle.load(f)
    (A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, \
            y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4, \
            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, N_dyn) = orb_lib

    ax.hist(np.log10(N_dyn), bins=20, alpha=0.5, range = [1,3])


    weights = jnp.ones(A_Rzphi.shape[1])
    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1 @ weights) / density_2DXY # density_2DXY
    h2_model = (A_h2 @ weights) / density_2DXY # density_2DXY
    h3_model = (A_h3 @ weights) / density_2DXY # density_2DXY
    h4_model = (A_h4 @ weights) / density_2DXY # density_2DXY

    X_regular_grid = dict_data['X_regular_grid']
    Y_regular_grid = dict_data['Y_regular_grid']

    bin_mapping = dict_data['bin_mapping']
    index_remap = bin_mapping[:-1]
    density_2DXY_weighted2 = density_2DXY[index_remap]
    h1_model_weighted2 = h1_model[index_remap]
    h2_model_weighted2 = h2_model[index_remap]
    h3_model_weighted2 = h3_model[index_remap]
    h4_model_weighted2 = h4_model[index_remap]

    fig, ax = plt.subplots(1,5, figsize=(25,4))
    res0 = ((density_2DXY_weighted2 - density_2DXY_weighted1)/density_2DXY_weighted1)
    im4 = ax[0].scatter(X_regular_grid, Y_regular_grid, c=res0.T, cmap='coolwarm', s = 20, marker='s', vmin=-0.5, vmax=0.5)
    ax[0].set_title('Density (Model)', fontsize=16)
    ax[0].set_xlabel('X (kpc)', fontsize=14)
    ax[0].set_ylabel('Y (kpc)', fontsize=14)
    fig.colorbar(im4, ax=ax[0])

    res1 = ((h1_model_weighted2 - h1_model_weighted1))
    im4 = ax[1].scatter(X_regular_grid, Y_regular_grid, c=res1.T, cmap='coolwarm', s = 20, marker='s', vmin = -0.3, vmax = 0.3)
    ax[1].set_title('h1 (Model)', fontsize=16)
    ax[1].set_xlabel('X (kpc)', fontsize=14)
    ax[1].set_ylabel('Y (kpc)', fontsize=14)
    fig.colorbar(im4, ax=ax[1])

    res2 = ((h2_model_weighted2 - h2_model_weighted1))
    im4 = ax[2].scatter(X_regular_grid, Y_regular_grid, c=res2.T, cmap='coolwarm', s = 20, marker='s', vmin = -0.3, vmax = 0.3)
    ax[2].set_title('h2 (Model)', fontsize=16)
    ax[2].set_xlabel('X (kpc)', fontsize=14)
    ax[2].set_ylabel('Y (kpc)', fontsize=14)
    fig.colorbar(im4, ax=ax[2])

    res3 = ((h3_model_weighted2 - h3_model_weighted1))
    im4 = ax[3].scatter(X_regular_grid, Y_regular_grid, c=res3.T, cmap='coolwarm', s = 20, marker='s', vmin = -0.3, vmax = 0.3)
    ax[3].set_title('h3 (Model)', fontsize=16)
    ax[3].set_xlabel('X (kpc)', fontsize=14)
    ax[3].set_ylabel('Y (kpc)', fontsize=14)
    fig.colorbar(im4, ax=ax[3])

    res4 = ((h4_model_weighted2 - h4_model_weighted1))
    im4 = ax[4].scatter(X_regular_grid, Y_regular_grid, c=res4.T, cmap='coolwarm', s = 20, marker='s', vmin = -0.3, vmax = 0.3)
    ax[4].set_title('h4 (Model)', fontsize=16)
    ax[4].set_xlabel('X (kpc)', fontsize=14)
    ax[4].set_ylabel('Y (kpc)', fontsize=14)
    fig.colorbar(im4, ax=ax[4])

    print('Residual sum:', jnp.sum(np.fabs(res0)) + jnp.sum(np.fabs(res1)) + jnp.sum(np.fabs(res2)) + jnp.sum(np.fabs(res3)) + jnp.sum(np.fabs(res4)))