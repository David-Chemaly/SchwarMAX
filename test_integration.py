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

    n_samples = 10_000  # Same number as original data
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
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid = _integrate_barred_vmap(
                        w0_new, acc_fn, pot_fn, n_steps, dt, initial_time, -Omega_bar, unroll,
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix)
    # Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid = _integrate_batch_vmap(
    #                     w0_new_batch, acc_fn, pot_fn, n_steps, dt_batch, initial_time, -Omega_bar, unroll,
    #                     num_Vbin, bin_mapping, num_per_bin,
    #                     Rzphi_lim_grid, xy_lim_grid,
    #                     Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
    #                     v0, s, rotation_matrix)
    Rzphi_bin_counts.block_until_ready()
    A_Rzphi = Rzphi_bin_counts.T / n_steps
    A_xy = surface_density.T / n_steps
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T
    time_end = time.time()
    print(f"Time taken to integrate orbits and compute observables: {time_end - time_start:.2f} seconds")
    print(A_Rzphi.shape)
    print(jnp.unique(valid, return_counts=True))
    print(np.isnan(A_h1).sum(), np.isnan(A_h2).sum(), np.isnan(A_h3).sum(), np.isnan(A_h4).sum())
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
                R_grid, z_grid, phi_grid, density_func_Rz, params_dict, dR, dz, dphi, dict_data['sample_for_integration']
    )
    # y_Rzphi = dict_data['Rzphi_density_data'].astype(jnp.float32)
    y_xy = dict_data['XY_density_data'].astype(jnp.float32)
    y_h1 = dict_data['h1_data'].astype(jnp.float32)
    y_h2 = dict_data['h2_data'].astype(jnp.float32)
    y_h3 = dict_data['h3_data'].astype(jnp.float32)
    y_h4 = dict_data['h4_data'].astype(jnp.float32)
    y_xy = y_xy / params_dict['light_to_mass_ratio'] # convert from light to mass
    sig_Rzphi = 0.02 * y_Rzphi + 1e-10
    sig_xy = 0.01 * y_xy + 1e-10
    # frac_err_min = 0.1
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
    with open(path_data + 'orbital_library_bar_1.pkl', 'wb') as f:
        orb_lib = (A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, \
           y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4, \
           sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)
        pickle.dump(orb_lib, f)


    weights = jnp.ones(A_Rzphi.shape[1])
    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1 @ weights) / density_2DXY # density_2DXY
    h2_model = (A_h2 @ weights) / density_2DXY # density_2DXY
    h3_model = (A_h3 @ weights) / density_2DXY # density_2DXY
    h4_model = (A_h4 @ weights) / density_2DXY # density_2DXY

    plot = True
    if plot:

        X_regular_grid = dict_data['X_regular_grid']
        Y_regular_grid = dict_data['Y_regular_grid']

        bin_mapping = dict_data['bin_mapping']
        index_remap = bin_mapping[:-1]
        density_2DXY_weighted = density_2DXY[index_remap]
        h1_model_weighted = h1_model[index_remap]
        h2_model_weighted = h2_model[index_remap]
        h3_model_weighted = h3_model[index_remap]
        h4_model_weighted = h4_model[index_remap]

        density_mock = y_xy[index_remap]
        h1_mock = y_h1[index_remap]
        h2_mock = y_h2[index_remap]
        h3_mock = y_h3[index_remap]
        h4_mock = y_h4[index_remap]

        fig, ax = plt.subplots(2, 3, figsize=(24, 10), gridspec_kw={'wspace': 0.5, 'hspace': 0.4,})
        cb = ax[0,0].scatter(X_regular_grid, Y_regular_grid, c=density_2DXY_weighted.T, cmap='viridis', s = 20, marker='s', norm='log')
        ax[0,0].set_title('Surface Density (Model)', fontsize=16)
        ax[0,0].set_xlabel('X (kpc)', fontsize=14)
        ax[0,0].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(cb, ax=ax[0,0])

        im1 = ax[0,1].scatter(X_regular_grid, Y_regular_grid, c=h1_model_weighted.T, cmap='coolwarm', s = 20, marker='s')
        ax[0,1].set_title('h1 (Model)', fontsize=16)
        ax[0,1].set_xlabel('X (kpc)', fontsize=14)
        ax[0,1].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(im1, ax=ax[0,1])

        im2 = ax[0,2].scatter(X_regular_grid, Y_regular_grid, c=h2_model_weighted.T, cmap='coolwarm', s = 20, marker='s')
        ax[0,2].set_title('h2 (Model)', fontsize=16)
        ax[0,2].set_xlabel('X (kpc)', fontsize=14)
        ax[0,2].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(im2, ax=ax[0,2])

        im3 = ax[1,0].scatter(X_regular_grid, Y_regular_grid, c=h3_model_weighted.T, cmap='coolwarm', s = 20, marker='s')
        ax[1,0].set_title('h3 (Model)', fontsize=16)
        ax[1,0].set_xlabel('X (kpc)', fontsize=14)
        ax[1,0].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(im3, ax=ax[1,0])

        im4 = ax[1,1].scatter(X_regular_grid, Y_regular_grid, c=h4_model_weighted.T, cmap='coolwarm', s = 20, marker='s')
        ax[1,1].set_title('h4 (Model)', fontsize=16)
        ax[1,1].set_xlabel('X (kpc)', fontsize=14)
        ax[1,1].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(im4, ax=ax[1,1])


        fig, ax = plt.subplots(2, 3, figsize=(24, 10), gridspec_kw={'wspace': 0.5, 'hspace': 0.4,})
        cb = ax[0,0].scatter(X_regular_grid, Y_regular_grid, c=density_mock.T, cmap='viridis', s = 20, marker='s', norm='log')
        ax[0,0].set_title('Surface Density (Mock)', fontsize=16)
        ax[0,0].set_xlabel('X (kpc)', fontsize=14)
        ax[0,0].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(cb, ax=ax[0,0])

        im1 = ax[0,1].scatter(X_regular_grid, Y_regular_grid, c=h1_mock.T, cmap='coolwarm', s = 20, marker='s')
        ax[0,1].set_title('h1 (Mock)', fontsize=16)
        ax[0,1].set_xlabel('X (kpc)', fontsize=14)
        ax[0,1].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(im1, ax=ax[0,1])

        im2 = ax[0,2].scatter(X_regular_grid, Y_regular_grid, c=h2_mock.T, cmap='coolwarm', s = 20, marker='s')
        ax[0,2].set_title('h2 (Mock)', fontsize=16)
        ax[0,2].set_xlabel('X (kpc)', fontsize=14)
        ax[0,2].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(im2, ax=ax[0,2])

        im3 = ax[1,0].scatter(X_regular_grid, Y_regular_grid, c=h3_mock.T, cmap='coolwarm', s = 20, marker='s')
        ax[1,0].set_title('h3 (Mock)', fontsize=16)
        ax[1,0].set_xlabel('X (kpc)', fontsize=14)
        ax[1,0].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(im3, ax=ax[1,0])

        im4 = ax[1,1].scatter(X_regular_grid, Y_regular_grid, c=h4_mock.T, cmap='coolwarm', s = 20, marker='s')
        ax[1,1].set_title('h4 (Mock)', fontsize=16)
        ax[1,1].set_xlabel('X (kpc)', fontsize=14)
        ax[1,1].set_ylabel('Y (kpc)', fontsize=14)
        fig.colorbar(im4, ax=ax[1,1])

        plt.show()
