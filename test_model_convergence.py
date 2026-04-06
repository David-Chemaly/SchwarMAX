
data_path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

import sys
sys.path.append(path)

from model_bar import *
from likelihoods_bar import *
from utils import *
from sample_from_density import sample_from_density_grid

import jax
import jax.numpy as jnp

import jax
# jax.config.update("jax_enable_x64", True)
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

filename = 'mock_Nbody_bar_XY_withRot_Nbins1000.pkl'
dict_data = get_dict_data_bootstrap(path, filename)

def log_prior(theta,):
    if (7 < theta[0] < 12) and (8 < theta[1] < 12) and (-1 < theta[2] < 2) and (-1 < theta[3] < 1) and (-1 < theta[4] < 1)\
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
p0 = np.array([10.5, 10, 0.8, 0., 0.5, jnp.pi/4, jnp.pi/4, 3.5*jnp.pi/4])
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

logM_disc_best_fit, logM_bulge_best_fit, \
logRd_disc_best_fit, logHs_disc_best_fit, logRs_bulge_best_fit, \
alpha_best_fit, beta_best_fit, gamma_best_fit = params_bestfit
# logMhalo_best_fit, logrho0_best_fit, logM_bulge_best_fit, logRh_disk_best_fit, logRs_disk_best_fit, logHs_disk_best_fit, logRs_bulge_best_fit,\
#       alpha_best_fit, beta_best_fit, gamma_best_fit, logLM_best_fit = (11.8, 8.8, 10.4, 1.2, 0.45, -0.24, -0.1, 30*np.pi/180, 20*np.pi/180, 0*np.pi/180, 0)

print('logM_disc_best_fit',logM_disc_best_fit)
print('logM_bulge_best_fit',logM_bulge_best_fit)
print('logRd_disc_best_fit',logRd_disc_best_fit)
print('logHs_disc_best_fit',logHs_disc_best_fit)
print('logRs_bulge_best_fit',logRs_bulge_best_fit)
print('alpha_best_fit',alpha_best_fit * 180 / np.pi)
print('beta_best_fit',beta_best_fit * 180 / np.pi)
print('gamma_best_fit',gamma_best_fit * 180 / np.pi)


ground_truth = [
    11.9,
    logM_disc_best_fit,
    logM_bulge_best_fit,
    jnp.log10(19).item(),
    logRd_disc_best_fit,
    logHs_disc_best_fit,
    logRs_bulge_best_fit,
    alpha_best_fit, #  10 * np.pi/180,
    beta_best_fit,
    gamma_best_fit,
    0.,
    1.5,

    -2., # sigma_xy model term
]

import time

# N_max_iteration_ls = [1000, 3000, 5000, 10000, 20000]
# logL_values = []
# times = []
# for N_max in N_max_iteration_ls:
#     time_start = time.time()
#     logL = logl_angular_input_bootstrap_test(ground_truth, dict_data, dict_data['total_bins'], N_max_integration=N_max)
#     logL_values.append(logL)
#     print(f'N_max_integration={N_max}, logL={logL}')
#     time_end = time.time()
#     times.append(time_end - time_start)

# print('Summary of logL values for different N_max_integration:')
# print('N_max_integration  |  logL |  Time (s)')
# for N_max, logL, time_taken in zip(N_max_iteration_ls, logL_values, times):
#     print(f'{N_max:<20} | {logL} | {time_taken:.2f} seconds')
#     print('-'*50)

N_orb_iteration_ls = [1000, 3000, 5000, 10000, 15000]
logL_values = []
times = []
for N_orb in N_orb_iteration_ls:
    time_start = time.time()
    dict_data = get_dict_data_bootstrap(path, filename, N_BOOTSTRAP=100, n_samples=N_orb)
    dict_data['logl_density_max'] = logl_val
    logL = logl_angular_input_bootstrap_test(ground_truth, dict_data, dict_data['total_bins'], N_max_integration=5_000)
    logL_values.append(logL)
    print(f'N_orb={N_orb}, logL={logL}')
    time_end = time.time()
    times.append(time_end - time_start)

print('Summary of logL values for different N_orb:')
print('N_orb  |  logL |  Time (s)')
for N_orb, logL, time_taken in zip(N_orb_iteration_ls, logL_values, times):
    print(f'{N_orb:<20} | {logL} | {time_taken:.2f} seconds')
    print('-'*50)