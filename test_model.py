from model import * 
from sample_from_density import sample_from_density_grid
from likelihoods import logl_density, logl_angular_input
import emcee


import pickle

import jax
import jaxopt
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

import time as tt
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18

import os
import corner
import numpy as np
import multiprocessing as mp

from potentials import *
from integrants_with_binning import *
from ghMoments import *
from utils import *
from constants import EPSILON

from potentials import NFW_acceleration
from densities import MiyamotoNagai_density

from CylindricalSpline import get_phi_m, get_acc, evaluate_phi_axisymmetric

import agama
agama.setUnits(length=1, velocity=1, mass=1)

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

if __name__ == "__main__":
    path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
    dict_data = get_dict_data(path)

    samples = np.array([
        np.random.normal(0, 5, 20000),
        np.random.normal(0, 5, 20000),
        np.random.normal(0, 2, 20000)
    ]).T
    w0 = jnp.array(samples)

    dict_data['w0'] = w0

    logMhalo_best_fit, logrho0_best_fit, logM_bar_best_fit, logRh_disk_best_fit, logRs_disk_best_fit, logHs_disk_best_fit, logRs_bar_best_fit,\
        alpha_best_fit, beta_best_fit, gamma_best_fit, logLM_best_fit, logOmega_bar = (10.07, 9.22, 9.2, 1.95, 0.54, -0.94, 0.35, 0.73, 0.71, 2.45, -0.25, 1.48)  
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

    params_disk_rho = {
        'rho0_disc': 10 ** ground_truth[1],
        'Rd_disc': 10 ** ground_truth[4],
        'hz_disc': 10 ** ground_truth[5],
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
        'p_bar': 0.3,
        'q_bar': 0.3,
        'light_to_mass_ratio': 10 ** ground_truth[10],
        'Omega_bar': 10 ** ground_truth[11],
    }

    surface_density_model = projection(params_disk_rho, dict_data, dict_data['total_bins'])
    surface_density_gt = dict_data['XY_density_data']

    chi2 = jnp.sum((surface_density_gt - surface_density_model)**2 / (0.1 * surface_density_gt)**2)
    _logl_density = -0.5 * chi2 / dict_data['total_bins']
    print(_logl_density)

    time_start = tt.time()
    density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set,\
            density_unity_set, _V_model_unity, _sigma_model_unity, h1_unity_set, h2_unity_set, h3_unity_set, h4_unity_set,\
                weights = model_for_plotting(params_halo_pot, params_disk_rho, dict_data, dict_data['total_bins'])
    V_model.block_until_ready()
    time_end = tt.time()
    print('Model Done', time_end - time_start, 'seconds')

    time_start = tt.time()
    density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set,\
            density_unity_set, _V_model_unity, _sigma_model_unity, h1_unity_set, h2_unity_set, h3_unity_set, h4_unity_set,\
                weights = model_for_plotting(params_halo_pot, params_disk_rho, dict_data, dict_data['total_bins'])
    V_model.block_until_ready()
    time_end = tt.time()
    print('Model Done second', time_end - time_start, 'seconds')

    ground_truth = [10.07, 9.22, 9.2, 1.95, 0.54, -0.94, 0.35, 0.73, 0.71, 2.45, -0.25, 1.48]
    dict_data['logl_density_max'] = -3.2

    start = tt.time()
    logL = logl_angular_input(ground_truth, dict_data, dict_data['total_bins'])
    logL.block_until_ready()  # Ensure computation finishes before timing
    end = tt.time()
    print('logL', logL, 'time per logl evaluation', end - start, 's')

    import time
    start = tt.time()
    logL = logl_angular_input(ground_truth, dict_data, dict_data['total_bins'])
    logL.block_until_ready()  # Ensure computation finishes before timing
    end = tt.time()
    print('logL', logL, 'time per logl evaluation', end - start, 's')