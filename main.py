
path = '/content/drive/MyDrive/SchwarMAX-MCMC/'

import sys
sys.path.append(path)

from model import *
from likelihoods import *
from utils import *
from sample_from_density import sample_from_density_grid
from CylindricalSpline import get_phi_m, evaluate_phi_axisymmetric

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.numpy.linalg as jnn
import pandas as pd
import numpy as np
import scipy as sp
import pickle

import emcee
import corner
import matplotlib.pyplot as plt

from constants import EPSILON

def get_dict_data(path):

    with open(path + 'mock_Nbody_disc_bulge_XY_withRot.pkl', 'rb') as f:
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

    V_data_err = jnp.where(0.1 * jnp.fabs(V_data) < 10, 10, 0.1 * V_data)
    sigma_data_err = jnp.where(0.1 * jnp.fabs(sigma_data) < 5, 5, 0.1 * sigma_data)
    h1_data_err = jnp.where(0.1 * jnp.fabs(h1_data) < 0.03, 0.03, 0.1 * jnp.fabs(h1_data))
    h2_data_err = jnp.where(0.1 * jnp.fabs(h2_data) < 0.03, 0.03, 0.1 * jnp.fabs(h2_data))
    h3_data_err = jnp.where(0.1 * jnp.fabs(h3_data) < 0.03, 0.03, 0.1 * jnp.fabs(h3_data))
    h4_data_err = jnp.where(0.1 * jnp.fabs(h4_data) < 0.03, 0.03, 0.1 * jnp.fabs(h4_data))

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

if __name__ == "__main__":

    dict_data = get_dict_data(path)

    def log_prior(theta,):
        if (6 < theta[0] < 10) and (8 < theta[1] < 12) and (-1 < theta[2] < 2) and (-1 < theta[3] < 1) and (-1 < theta[4] < 1)\
        and (0 <= theta[5] < jnp.pi) and (0 <= theta[6] < jnp.pi/2) and (0 <= theta[7] < jnp.pi):
            return 0.0  # log(1) = 0 for uniform prior
        return -np.inf  # log(0) = -inf for out-of-bounds

    def log_prob(theta,):
        # print(theta)
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        ll = logl_density(theta, dict_data, dict_data['total_bins'])

        return ll + lp

    ndim = 8
    nwalkers = 16  # must be >= 2 * ndim

    # Initialize walkers around ground truth
    # p0 = np.array([ground_truth[k] for k in param_names])
    p0 = np.array([9.2, 10, 0.3, 0., 0., jnp.pi/4, jnp.pi/4, jnp.pi/4])
    # initial_pos = p0 + 1e-1 * np.random.randn(nwalkers, ndim)
    np.random.seed(42)
    initial_pos = p0 + np.random.uniform(-0.3, 0.3, (nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_pos, 500, progress=True)

    samples = sampler.get_chain(discard=200, flat=True)


    params_bestfit = np.percentile(samples, axis=0, q=50)
    logl_val = logl_density(params_bestfit, dict_data, dict_data['total_bins'])
    print('Best-fit logL projection', logl_val)#

    dict_data['logl_density_max'] = logl_val

    logrho0_best_fit, logM_bulge_best_fit, \
    logRd_disc_best_fit, logHs_disc_best_fit, logRs_bulge_best_fit, \
    alpha_best_fit, beta_best_fit, gamma_best_fit = params_bestfit
    # logMhalo_best_fit, logrho0_best_fit, logM_bulge_best_fit, logRh_disk_best_fit, logRs_disk_best_fit, logHs_disk_best_fit, logRs_bulge_best_fit,\
    #       alpha_best_fit, beta_best_fit, gamma_best_fit, logLM_best_fit = (11.8, 8.8, 10.4, 1.2, 0.45, -0.24, -0.1, 30*np.pi/180, 20*np.pi/180, 0*np.pi/180, 0)

    print('logrho0_best_fit',logrho0_best_fit)
    print('logM_bulge_best_fit',logM_bulge_best_fit)
    print('logRd_disc_best_fit',logRd_disc_best_fit)
    print('logHs_disk_best_fit',logHs_disc_best_fit)
    print('logRs_bulge_best_fit',logRs_bulge_best_fit)
    print('alpha_best_fit',alpha_best_fit * 180 / np.pi)
    print('beta_best_fit',beta_best_fit * 180 / np.pi)
    print('gamma_best_fit',gamma_best_fit * 180 / np.pi)

    params_halo_pot = {
        'logM': 11.8,
        'Rs':19,
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
        'rho0_disc': 10 ** logrho0_best_fit,
        'Rd_disc': 10 ** logRd_disc_best_fit,
        'hz_disc': 10 ** logHs_disc_best_fit,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
        'alpha': alpha_best_fit * 180 / jnp.pi,
        'beta': beta_best_fit * 180 / jnp.pi,
        'gamma': gamma_best_fit * 180 / jnp.pi,
        'light_to_mass_ratio': 1,
        'logM_bulge': logM_bulge_best_fit,
        'Rs_bulge': 10 ** logRs_bulge_best_fit,
    }

    @jax.jit
    def potential_func(x, y, z, dict_phi, params_halo):
        """ Returns Phi(R, z) """
        phi_halo = NFW_potential(x, y, z, params_halo)
        phi_disk = evaluate_phi_axisymmetric(x, y, z, dict_phi)
        return phi_halo + phi_disk

    @jax.jit
    def density_func(x, y, z, params):
        """ Returns Stellar Density nu(R, z) """
        # Double Exponential Disk
        val = DoubleExponentialDisk_density(x, y, z, params) + Hernquist_density(x, y, z, params)
        return val

    bounds = jnp.array(
        [
            [-15.0, 15.0],  # x
            [-15.0, 15.0],  # y
            [-5.0, 5.0],    # z
        ],
        dtype=jnp.float32,
    )

    n_samples = 20_000
    n_x,n_y,n_z = 48, 48, 32

    key = jax.random.PRNGKey(0)
    sample_ic_dict = sample_from_density_grid(
        key,
        density_func,
        params_disk_rho,
        bounds,
        n_samples=n_samples,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
    )
    samples = np.asarray(sample_ic_dict["samples"])
    dict_data['w0'] = samples


    ground_truth = [
        11.5,
        logrho0_best_fit,
        logM_bulge_best_fit,
        jnp.log10(19).item(),
        logRd_disc_best_fit,
        logHs_disc_best_fit,
        logRs_bulge_best_fit,
        alpha_best_fit,
        beta_best_fit,
        gamma_best_fit,
        0.3
    ]
    logL = logl_angular_input(ground_truth, dict_data, dict_data['total_bins'])
    print(logL)

    import time
    start = time.time()
    logL = logl_angular_input(ground_truth, dict_data, dict_data['total_bins'])
    logL.block_until_ready()  # Ensure computation finishes before timing
    end = time.time()
    print('time per logl evaluation', end - start)

    prior_uniform_low =  [
        ground_truth[0] - 3,
        ground_truth[1] - 3,
        ground_truth[2] - 3,
        ground_truth[3]- 1,
        ground_truth[4]- 1,
        ground_truth[5]- 1,
        ground_truth[6]- 1,
        0,
        0,
        0,
        -2
    ]
    prior_uniform_high = [
        ground_truth[0] + 3,
        ground_truth[1] + 3,
        ground_truth[2] + 3,
        ground_truth[3]+ 1,
        ground_truth[4]+ 1,
        ground_truth[5]+ 1,
        ground_truth[6]+ 1,
        jnp.pi,
        jnp.pi/2,
        jnp.pi,
        2
    ]

    def log_prior(params):
        lp = 0
        for i in range (0, ndim):
            if (params[i]<=prior_uniform_low[i]) & (params[i]>=prior_uniform_high[i]):
                lp+= -jnp.inf
        return lp

    def log_prob(theta):
        params = theta
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = float(logl_angular_input(params, dict_data, dict_data['total_bins']))  # convert from JAX array
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    ndim = 11
    nwalkers = 22  # must be >= 2 * ndim

    np.random.seed(42)

    # Initialize walkers around ground truth
    # p0 = np.array([ground_truth[k] for k in param_names])
    p0 = ground_truth
    initial_pos = np.zeros((nwalkers, ndim))
    initial_pos[:, :7] = np.vstack([p0[:7]]*nwalkers) * np.ones((nwalkers, 7)) + np.random.uniform(-0.5, 0.5, (nwalkers, 7))
    initial_pos[:,7:10] = np.vstack([p0[7:10]]*nwalkers) * np.ones((nwalkers, 3)) + np.random.uniform(-0.1, 0.1, (nwalkers, 3))
    initial_pos[:,7:10] = np.clip(initial_pos[:,7:10], a_min=0, a_max=np.vstack([[jnp.pi, jnp.pi/2, jnp.pi]]*nwalkers))
    initial_pos[:,  10] = p0[10] + np.random.uniform(-0.3, 0.3, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_pos, 250, progress=True)

    samples = sampler.get_chain(discard=80, flat=True)

    param_names = ['logM_halo','logM_disk','logM_bulge', 'logRs_halo', 'logRs_disk', 'logHs_disk', 'logRs_bulge', 'alpha', 'beta', 'gamma', 'log_light_to_mass_ratio']
    pd.DataFrame(samples, columns=param_names).to_csv(path+'/test_posterior_0225.csv', index=False)

    samples_raw = sampler.get_chain(discard=0, flat=False)
    with open(path+'/test_posterior_WholeChain_0225.pkl', 'wb') as f:
        pickle.dump(samples_raw, f)