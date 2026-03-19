import agama
agama.setUnits(mass=1, length=1, velocity=1)

from constants import *
from integrants_with_binning import integrate_leapfrog_barred, _integrate_batch_vmap, _integrate_barred_vmap
from sample_from_density import sample_from_density_grid
from densities import *
from potentials import *
from utils import *
from model import *
from CylindricalSpline import get_phi_m, get_acc, evaluate_phi_axisymmetric

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd
import pickle

import time

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
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
    }

    return dict_data


@jax.jit
def Ferrers_density(x, y, z, params):

    '''
    params include: 'logM_bar', 'Rs_bar', 'q_bar', 'p_bar'
    '''
    p, q, Rs = params['p_bar'], params['q_bar'], params['Rs_bar']
    M = 10.0 ** params['logM_bar']
    r = jnp.sqrt(x**2 + (y / p)**2 + (z / q)**2)
    rho = 105 * M / (32 * jnp.pi * p * q * Rs**3) * (1 - (r / Rs)**2)**2
    rho = jnp.where(r <= Rs, rho, 0.0)
    return rho

@jax.jit
def Dehnen_density(x, y, z, params):

    '''
    Dehnen profile: rho = (M / (4π p q Rs^3)) * (r/Rs)^(-n) * (1 + r/Rs)^(n-4), where n = 2
    '''

    p, q, Rs = params['p_bar'], params['q_bar'], params['Rs_bar']
    M = 10.0 ** params['logM_bar']
    r = jnp.sqrt(x**2 + (y / p)**2 + (z / q)**2) + EPSILON

    val = M / (4 * jnp.pi * p * q * Rs**3) * (r / Rs)**(-2) * (1 + r / Rs)**(-2) * jnp.exp(-(z / 3)**4) * jnp.exp(-(r / 10)**4)

    return val

def density_func(x, y, z, params):
    return Dehnen_density(x, y, z, params) + DoubleExponentialDisk_density(x, y, z, params)

@jax.jit
def potential_func(x, y, z, dict_phi, params_halo):
    """ Returns Phi(R, z) """
    phi_halo = NFW_potential(x, y, z, params_halo)
    phi_disk = evaluate_phi_axisymmetric(x, y, z, dict_phi)
    return phi_halo + phi_disk


def density_func_agama(x):
    return np.array(density_func(x[:, 0], x[:, 1], x[:, 2], params_dict))

if __name__ == "__main__":

    dict_data = get_dict_data(path)

    logMhalo_best_fit, logrho0_best_fit, logM_bar_best_fit, logRh_disk_best_fit, logRs_disk_best_fit, logHs_disk_best_fit, logRs_bar_best_fit,\
        alpha_best_fit, beta_best_fit, gamma_best_fit, logLM_best_fit, logOmega_bar = (11.8, 8.8, 10.4, 1.2, 0.45, -0.24, 0.3, 
                                                                                        30*np.pi/180, 20*np.pi/180, 130*np.pi/180, 0, 1.6)

    # logMhalo_best_fit, logrho0_best_fit, logM_bar_best_fit, logRh_disk_best_fit, logRs_disk_best_fit, logHs_disk_best_fit, logRs_bar_best_fit,\
    #     alpha_best_fit, beta_best_fit, gamma_best_fit, logLM_best_fit, logOmega_bar = (10.07, 9.22, 9.2, 1.95, 0.54, -0.94, 0.35, 0.73, 0.71, 2.45, -0.25, 1.48)  


    alpha = alpha_best_fit * 180/np.pi
    beta = beta_best_fit * 180/np.pi
    gamma = gamma_best_fit * 180/np.pi
    ground_truth = [logMhalo_best_fit,
                    logrho0_best_fit,
                    logM_bar_best_fit,
                    logRh_disk_best_fit,
                    logRs_disk_best_fit,
                    logHs_disk_best_fit,
                    logRs_bar_best_fit,
                    alpha,
                    beta,
                    gamma,
                    logLM_best_fit,
                    logOmega_bar,
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

    params_dict = {
        'rho0_disc': 10 ** ground_truth[1],
        'Rd_disc': 10 ** ground_truth[4],
        'hz_disc': 10 ** ground_truth[5],
        'light_to_mass_ratio': 10 ** ground_truth[10],
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
        'Rs_bar': 10 ** ground_truth[6],
        'Omega_bar': 10 ** ground_truth[11],
        'p_bar': 0.3,
        'q_bar': 0.3,
    }

    n_samples = 5_000  # Same number as original data
    x_grid = np.linspace(0, 15, 1000)
    logP_xexp = XexpX_pdf_log(x_grid, 4.0)
    key = jax.random.PRNGKey(10086)
    R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.uniform(0, 2*np.pi, size=n_samples)

    x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)

    x_grid = np.linspace(0, 4, 1000)
    logP_exp = expX_pdf_log(x_grid, 1.5)
    key = jax.random.PRNGKey(10010)
    z_samples = sample_from_logP(x_grid, logP_exp, n_samples, key)
    samples = np.array([
        x_samples,
        y_samples,
        z_samples,
    ]).T

    # samples = np.array([
    #     jnp.array(np.random.normal(0, 4., 5000)),
    #     jnp.array(np.random.normal(0, 4., 5000)),
    #     jnp.array(np.random.normal(0, 2., 5000)),
    # ]).T

    w0 = jnp.array(samples)
    n_particles = w0.shape[0]
    params_disk_rho = params_dict
    NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-3, 30.0, 1e-3, 15.0, 8.
    Nphi = 300
    N_int = 10_000
    dict_phi = get_phi_m(density_func, params_disk_rho, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)

    #=========================================== GET INITIAL VELOCITY ===================================================
    @jax.jit
    def dPhi_dz(x, y, z, dict_phi, params_halo):
        # Numerical derivative of Potential w.r.t z
        d = 5e-3
        return (potential_func(x, y, z+d, dict_phi, params_halo) - potential_func(x, y, z-d, dict_phi, params_halo)) / (2*d)

    @jax.jit
    def dPhi_dR(x, y, z, dict_phi, params_halo):
        # Numerical derivative of Potential w.r.t R
        d = 5e-3
        R = jnp.sqrt(x**2 + y**2)
        return (potential_func(R+d, 0, z, dict_phi, params_halo) - potential_func(R-d, 0, z, dict_phi, params_halo)) / (2*d)

    @jax.jit
    def get_jeans_moments(x_star, y_star, z_star, dict_phi, params_disc, params_halo, anisotropy_b=1.0):
        """
        Computes (v_mean, sigma_R, sigma_z, sigma_phi) for a star at (R, z).

        Parameters:
        -----------
        x_star, y_star, z_star : float
            Coordinates of the particle.
        density_fn : function
            Function returning stellar density nu(R, z).
        anisotropy_b : float
            sigma_R^2 / sigma_z^2. Default 1.0 (isotropic).
        """
        R_star = jnp.sqrt(x_star**2 + y_star**2)

        # --- Step 1: Compute Sigma_z (Vertical Integration) ---

        # Integrand: nu(R, z) * dPhi/dz
        def integrand(z_prime):
            return density_func(x_star, y_star, z_prime, params_disc) * dPhi_dz(x_star, y_star, z_prime, dict_phi, params_halo)

        # Integrate from z_star to infinity (e.g., 20 kpc)
        # Note: We assume symmetry, so we integrate |z| to infinity
        pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
        dx = pts[1] - pts[0]
        # integrand_val = integrand(pts)
        integrand_val = jax.vmap(integrand, in_axes = (0))(pts)
        integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)

        nu_val = density_func(x_star, y_star, z_star, params_disc)
        # if nu_val <= 0: return 0, 0, 0, 0

        sigma_z2 = (1.0 / nu_val) * integral_val

        sigma_z2 = jnp.maximum(sigma_z2, 0.0)

        sigma_z = jnp.sqrt(sigma_z2)

        # --- Step 2: Compute Sigma_R (Anisotropy assumption) ---
        sigma_R2 = anisotropy_b * sigma_z2
        sigma_R = jnp.sqrt(sigma_R2)

        # --- Step 3: Compute v_phi_total^2 (Radial Equation) ---
        # We need d(nu * sigma_R^2) / dR
        # Since sigma_R^2 = b * sigma_z^2, we need d(b * Integral) / dR

        # Define a helper to calculate the "Vertical Pressure" P_zz = nu * sigma_z^2 at any R
        def vertical_pressure(r_in):
            # We need to re-integrate at radius r_in to get the pressure there
            def integrand_r(z_prime):
                return density_func(r_in, 0, z_prime, params_disc) * dPhi_dz(r_in, 0, z_prime, dict_phi, params_halo)

            pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
            dx = pts[1] - pts[0]
            # integrand_val = integrand(pts)
            integrand_val = jax.vmap(integrand_r, in_axes = (0))(pts)
            integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)
            return integral_val # This is (nu * sigma_z^2)

        # Calculate derivative w.r.t R using central difference
        dR = 5e-3
        Pzz_plus = vertical_pressure(R_star + dR)
        Pzz_minus = vertical_pressure(R_star - dR)
        d_nu_sigR2_dR = anisotropy_b * (Pzz_plus - Pzz_minus) / (2*dR)

        # Radial Jeans Equation:
        # v_phi_sq = sigma_R^2 + (R/nu) * d(nu*sigR^2)/dR + R * dPhi/dR
        term1 = sigma_R2
        term2 = (R_star / nu_val) * d_nu_sigR2_dR
        term3 = R_star * dPhi_dR(x_star, y_star, z_star, dict_phi, params_halo)

        v_phi_total_sq = term1 + term2 + term3

        # --- Step 4: Separate Rotation vs Dispersion ---
        # Assumption: sigma_phi approx sigma_R (Round velocity ellipsoid in plane)
        sigma_phi = sigma_R

        v_streaming_sq = v_phi_total_sq - sigma_phi**2

        v_streaming_sq = jnp.maximum(v_streaming_sq, 0.0)
        v_mean_phi = jnp.sqrt(v_streaming_sq)

        output = jax.lax.cond(nu_val<=0, lambda: (0.0, 0.0, 0.0, 0.0), lambda: (v_mean_phi, sigma_R, sigma_z, sigma_phi))

        return output


    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    # jeans_moments = get_jeans_moments(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot, anisotropy_b=1.0)
    def get_w0_new(w0, key1, key2, key3, n_particles):
        jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], dict_phi, params_disk_rho, params_halo_pot, 1.)
        v_rot, sig_R, sig_z, sig_phi = jeans_moments
        g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
        vR = g1 * sig_R # 2 sigma dispersion
        vz = g2 * sig_z
        vphi = v_rot + g3 * sig_phi
        x, y, vx, vy = getCartesianFromCylindrical_clockwise(jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
        return jnp.array([x, y, w0[:,2], vx, vy, vz]).T
    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    w0_new = get_w0_new(w0, key1, key2, key3, n_particles)

    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (dict_phi, params_halo_pot),
        _z
    )

    n_realizations = 10
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape) # 0.1 kpc positional ditching
    v_scale = 0.1 * _Vc # 10% velocity ditching
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15) # limiting the noise with 1 - 15 kpc/Gyr
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
        (dict_phi, params_halo_pot),
        _z
    )
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)


    def _split(w):
        return w[:3], w[3:]

    def _merge(r, v):
        return jnp.concatenate([r, v], axis=0)

    @partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll',))
    def integrate_leapfrog_barred_traj(w0, acc_fn, n_steps, dt = 0.010, t0 = 0.0, Omega = 0.0, unroll=True):
        """Leapfrog (KDK) — returns final time and final state only.

        num_segments_Rzphi: int
            Number of segments for Rzphi bin counting. MUST equal to nRzphi.prod()
        num_segments_XY: int
            Number of segments for XY bin counting. MUST equal to nXY.prod()
        
        v0 and s are arrays of reference velocity and dispersion for each XY cell for the GH coefficent calculation. Length should equal to nXY.prod()
        """

        def step(carry, _):
            t, y = carry
            r, v = _split(y)

            # Gravity half-kick
            a0 = acc_fn(*r)
            v_half = v + 0.5 * dt * a0

            # Exact Omega-subflow for:
            #   xdot = vx + Omega*y, ydot = vy - Omega*x
            #   vxdot = Omega*vy,     vydot = -Omega*vx
            theta = Omega * dt
            c, s_theta = jnp.cos(theta), jnp.sin(theta)

            x_bar = r[0] + dt * v_half[0]
            y_bar = r[1] + dt * v_half[1]
            x_new = c * x_bar + s_theta * y_bar
            y_new = -s_theta * x_bar + c * y_bar
            z_new = r[2] + dt * v_half[2]

            vx_rot = c * v_half[0] + s_theta * v_half[1]
            vy_rot = -s_theta * v_half[0] + c * v_half[1]
            vz_rot = v_half[2]

            r_new = jnp.array([x_new, y_new, z_new])
            v_rot = jnp.array([vx_rot, vy_rot, vz_rot])
            t_new = t + dt

            # Gravity half-kick at updated position
            a1 = acc_fn(*r_new)
            v_new = v_rot + 0.5 * dt * a1
            y_new = _merge(r_new, v_new)
            return (t_new, y_new), (t_new, y_new)

        (_, _), (tN, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

        return tN, wN 

    N_step_per_orb = 100
    N_dynamical_time = 20
    dt = T_orb / N_step_per_orb
    time_integrate = T_orb * N_dynamical_time
    N_steps = N_step_per_orb * N_dynamical_time
    #=========================================== Integrate orbits =======================================================
    @jax.jit
    def acc_fn(x, y, z):
        a_halo = NFW_acceleration(x, y, z,  params_halo_pot)
        a_disk = get_acc(x, y, z, dict_phi)
        return a_halo + a_disk
    
    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, dict_phi, params_halo_pot)
    pot_fn = jax.vmap(pot_fn, in_axes=(0, 0, 0))

    time_start = time.time()
    _integrate_vmap = jax.vmap(integrate_leapfrog_barred_traj, 
                            in_axes=(0, None, None, 0, None, None, None))
    tN, wN = _integrate_vmap(w0_new, acc_fn, N_steps, dt, 0.0, -34.0, False)
    tN.block_until_ready()
    time_end = time.time()
    print(f"Time taken to integrate orbits with reconstructed potential: {time_end - time_start:.2f} seconds")
    x_lf, y_lf, z_lf = wN[:,:,0].flatten(), wN[:,:,1].flatten(), wN[:,:,2].flatten()
    vx_lf, vy_lf, vz_lf = wN[:,:,3].flatten(), wN[:,:,4].flatten(), wN[:,:,5].flatten()
    print("Leapfrog integration done.")

    alpha, beta, gamma = params_dict['alpha'], params_dict['beta'], params_dict['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)
    num_Vbin = dict_data['total_bins']
    bin_mapping, num_per_bin = dict_data['bin_mapping'], dict_data['num_per_bin']
    v0, s = dict_data['v0'], dict_data['s']

    Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
    xy_lim_grid = jnp.array([[-12.,12.],[-4.,4.]])
    Rzphi_n_grid = jnp.array([10,6,6])
    xy_n_grid = jnp.array([60,40])
    Rzphi_n_tot = 360

    time_start = time.time()
    Omega_bar = params_dict['Omega_bar']
    # time = time_integrate #Gyr
    n_steps = N_steps
    dt_batch = T_orb_batch / N_step_per_orb
    unroll = False
    initial_time = 0.0
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid = _integrate_batch_vmap(
                        w0_new_batch, acc_fn, pot_fn, n_steps, dt_batch, initial_time, -Omega_bar, unroll,
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix)
    Rzphi_bin_counts.block_until_ready()
    A_Rzphi_batch = Rzphi_bin_counts.T# / n_steps
    A_xy_batch = surface_density.T# / n_steps
    A_h1_batch = h1.T
    A_h2_batch = h2.T
    A_h3_batch = h3.T
    A_h4_batch = h4.T
    valid_batch = valid
    time_end = time.time()
    print(f"Time taken to integrate orbits and compute binned moments with vmap in batch: {time_end - time_start:.2f} seconds")

    time_start = time.time()
    w0_new_batch_flattened = w0_new_batch.reshape(-1, 6)
    dt_batch_flattened = dt_batch.reshape(-1)
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid_unbatch = _integrate_barred_vmap(
                        w0_new_batch_flattened, acc_fn, pot_fn, n_steps, dt_batch_flattened, initial_time, -Omega_bar, unroll,
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix)
    
    Rzphi_bin_counts.block_until_ready()
    A_Rzphi_unbatch = Rzphi_bin_counts.T / n_steps
    A_xy_unbatch = surface_density.T / n_steps
    A_h1_unbatch = h1.T
    A_h2_unbatch = h2.T
    A_h3_unbatch = h3.T
    A_h4_unbatch = h4.T

    #### Codex, below write a jax-jitted function that takes A_XXX_unbatched as input and rebatches it so that the results is the same as A_XXX_batch above
    def rebatch_orbit_matrices(A_Rzphi_unbatch, A_xy_unbatch, A_h1_unbatch, A_h2_unbatch, A_h3_unbatch, A_h4_unbatch, valid_unbatch, n_orb_per_batch):
        n_orb_tot = A_Rzphi_unbatch.shape[1]
        n_batch = n_orb_tot // n_orb_per_batch
        A_Rzphi_grouped = A_Rzphi_unbatch.reshape(A_Rzphi_unbatch.shape[0], n_batch, n_orb_per_batch)
        A_xy_grouped = A_xy_unbatch.reshape(A_xy_unbatch.shape[0], n_batch, n_orb_per_batch)
        A_h1_grouped = A_h1_unbatch.reshape(A_h1_unbatch.shape[0], n_batch, n_orb_per_batch)
        A_h2_grouped = A_h2_unbatch.reshape(A_h2_unbatch.shape[0], n_batch, n_orb_per_batch)
        A_h3_grouped = A_h3_unbatch.reshape(A_h3_unbatch.shape[0], n_batch, n_orb_per_batch)
        A_h4_grouped = A_h4_unbatch.reshape(A_h4_unbatch.shape[0], n_batch, n_orb_per_batch)
        valid_grouped = valid_unbatch.reshape(n_batch, n_orb_per_batch)

        def _reduce_one_batch(A_Rzphi_one, A_xy_one, A_h1_one, A_h2_one, A_h3_one, A_h4_one, valid_one):
            valid_sum = valid_one.sum()
            weights = jnp.ones((A_Rzphi_one.shape[1],), dtype=A_Rzphi_one.dtype) / (valid_sum + 0.1)

            A_h1w_one = A_h1_one * A_xy_one
            A_h2w_one = A_h2_one * A_xy_one
            A_h3w_one = A_h3_one * A_xy_one
            A_h4w_one = A_h4_one * A_xy_one

            A_Rzphi_batch = A_Rzphi_one @ weights
            A_xy_batch = A_xy_one @ weights
            A_h1_batch = (A_h1w_one @ weights) / (A_xy_batch + EPSILON)
            A_h2_batch = (A_h2w_one @ weights) / (A_xy_batch + EPSILON)
            A_h3_batch = (A_h3w_one @ weights) / (A_xy_batch + EPSILON)
            A_h4_batch = (A_h4w_one @ weights) / (A_xy_batch + EPSILON)
            return A_Rzphi_batch, A_xy_batch, A_h1_batch, A_h2_batch, A_h3_batch, A_h4_batch, valid_sum

        A_Rzphi_batch_from_unbatch, A_xy_batch_from_unbatch, A_h1_batch_from_unbatch, A_h2_batch_from_unbatch, A_h3_batch_from_unbatch, A_h4_batch_from_unbatch, valid_batch_from_unbatch = jax.vmap(
            _reduce_one_batch,
            in_axes=(1, 1, 1, 1, 1, 1, 0),
            out_axes=(1, 1, 1, 1, 1, 1, 0),
        )(
            A_Rzphi_grouped,
            A_xy_grouped,
            A_h1_grouped,
            A_h2_grouped,
            A_h3_grouped,
            A_h4_grouped,
            valid_grouped,
        )

        return (
            A_Rzphi_batch_from_unbatch,
            A_xy_batch_from_unbatch,
            A_h1_batch_from_unbatch,
            A_h2_batch_from_unbatch,
            A_h3_batch_from_unbatch,
            A_h4_batch_from_unbatch,
            valid_batch_from_unbatch,
        )

    rebatch_orbit_matrices = jax.jit(rebatch_orbit_matrices, static_argnames=("n_orb_per_batch",))
    A_Rzphi_rebatch, A_xy_rebatch, A_h1_rebatch, A_h2_rebatch, A_h3_rebatch, A_h4_rebatch, valid_rebatch = rebatch_orbit_matrices(
            A_Rzphi_unbatch,
            A_xy_unbatch,
            A_h1_unbatch,
            A_h2_unbatch,
            A_h3_unbatch,
            A_h4_unbatch,
            valid_unbatch,
            n_realizations,
    )

    A_Rzphi_rebatch.block_until_ready()
    time_end = time.time()
    print(f"Time taken to integrate orbits in vmap and rebatch afterwards: {time_end - time_start:.2f} seconds")

    print("The total difference between two methods")
    def print_diff_stats(name, x0, x1):
        diff = jnp.abs(x0 - x1)
        print(f"{name}: sum={float(jnp.sum(diff)):.6e}, mean={float(jnp.mean(diff)):.6e}, max={float(jnp.max(diff)):.6e}")

    print_diff_stats("A_Rzphi", A_Rzphi_batch, A_Rzphi_rebatch)
    print_diff_stats("A_xy", A_xy_batch, A_xy_rebatch)
    print_diff_stats("A_h1", A_h1_batch, A_h1_rebatch)
    print_diff_stats("A_h2", A_h2_batch, A_h2_rebatch)
    print_diff_stats("A_h3", A_h3_batch, A_h3_rebatch)
    print_diff_stats("A_h4", A_h4_batch, A_h4_rebatch)
    print_diff_stats("valid", valid_batch, valid_rebatch)

    # extra diagnostic: test alternative flat->batch order assumptions for valid
    valid_rebatch_contig = valid_unbatch.reshape(-1, n_realizations).sum(axis=1)
    valid_rebatch_strided = valid_unbatch.reshape(n_realizations, -1).sum(axis=0)
    print_diff_stats("valid_contig_check", valid_batch, valid_rebatch_contig)
    print_diff_stats("valid_strided_check", valid_batch, valid_rebatch_strided)

    mismatch_idx = jnp.where(valid_batch != valid_rebatch)[0]
    if mismatch_idx.size > 0:
        b = int(mismatch_idx[0])
        print(f"first_valid_mismatch_batch={b}, valid_batch={float(valid_batch[b]):.1f}, valid_rebatch={float(valid_rebatch[b]):.1f}")

        flat_slice = valid_unbatch[b * n_realizations : (b + 1) * n_realizations]
        print(f"flat_slice_valid_sum={float(flat_slice.sum()):.1f}")

        _, _, _, _, _, _, valid_single = _integrate_barred_vmap(
            w0_new_batch[b], acc_fn, pot_fn, n_steps, dt_batch[b], initial_time, -Omega_bar, unroll,
            num_Vbin, bin_mapping, num_per_bin,
            Rzphi_lim_grid, xy_lim_grid,
            Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
            v0, s, rotation_matrix
        )
        print(f"single_batch_valid_sum={float(valid_single.sum()):.1f}")
        print(f"single_vs_flat_orbit_valid_maxdiff={float(jnp.max(jnp.abs(valid_single - flat_slice))):.1f}")

        local_mismatch = jnp.where(valid_single != flat_slice)[0]
        if local_mismatch.size > 0:
            o = int(local_mismatch[0])
            g = b * n_realizations + o
            print(f"first_orbit_valid_mismatch: local={o}, global={g}, flat_valid={float(flat_slice[o]):.1f}, single_valid={float(valid_single[o]):.1f}")

            _, _, _, _, _, _, valid_orbit_alone = integrate_leapfrog_barred(
                w0_new_batch_flattened[g], acc_fn, pot_fn, n_steps, dt_batch_flattened[g], initial_time, -Omega_bar, unroll,
                num_Vbin, bin_mapping, num_per_bin,
                Rzphi_lim_grid, xy_lim_grid,
                Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                v0, s, rotation_matrix
            )
            print(f"single_orbit_call_valid={float(valid_orbit_alone):.1f}")

            lo = max(0, g - 64)
            hi = min(w0_new_batch_flattened.shape[0], g + 64)
            _, _, _, _, _, _, valid_local_chunk = _integrate_barred_vmap(
                w0_new_batch_flattened[lo:hi], acc_fn, pot_fn, n_steps, dt_batch_flattened[lo:hi], initial_time, -Omega_bar, unroll,
                num_Vbin, bin_mapping, num_per_bin,
                Rzphi_lim_grid, xy_lim_grid,
                Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                v0, s, rotation_matrix
            )
            print(f"local_chunk_orbit_valid={float(valid_local_chunk[g-lo]):.1f}, chunk_size={hi-lo}")

        _, _, _, _, _, _, valid_batch_singleton = _integrate_batch_vmap(
            w0_new_batch[b:b+1], acc_fn, pot_fn, n_steps, dt_batch[b:b+1], initial_time, -Omega_bar, unroll,
            num_Vbin, bin_mapping, num_per_bin,
            Rzphi_lim_grid, xy_lim_grid,
            Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
            v0, s, rotation_matrix
        )
        print(f"batch_vmap_singleton_valid={float(valid_batch_singleton[0]):.1f}")
