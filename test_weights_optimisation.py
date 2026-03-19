from model import * 


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

import time

from potentials import *
from integrants_with_binning import *
from ghMoments import *
from utils import *
from constants import EPSILON

from potentials import NFW_acceleration
from densities import MiyamotoNagai_density
from sample_from_density import sample_from_density_grid

from CylindricalSpline import get_phi_m, get_acc, evaluate_phi_axisymmetric

# path = '/home/hz420/python_script/SchwarMAX/'
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


# the following commands make plots look better
def plot_prettier(dpi=200, fontsize=11, usetex=False):
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

if __name__ == "__main__":
    plot_prettier(usetex=False)
    dict_data = get_dict_data(path)
    solver = os.environ.get("WEIGHT_SOLVER", "lbfgs").lower()
    solver_maxiter = int(os.environ.get("WEIGHT_SOLVER_MAXITER", "1000"))
    solver_power_iters = int(os.environ.get("WEIGHT_SOLVER_POWER_ITERS", "12"))
    solver_cg_maxiter = int(os.environ.get("WEIGHT_SOLVER_CG_MAXITER", "8"))

    path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
    with open(path_data + 'orbital_library_bar_5.pkl', 'rb') as f:
        orb_lib = pickle.load(f)
    (A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, \
           y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4, \
           sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4) = orb_lib
    
    # print(A_h3, A_h4)
    
    print('Weight solver:', solver)

    time_start = time.time()
    density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set,\
        density_unity_set, V_model_unity, sigma_model_unity, h1_unity_set, h2_unity_set, h3_unity_set, h4_unity_set,\
              weights = get_weights(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
           y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
           sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, dict_data,
           solver=solver,
           solver_maxiter=solver_maxiter,
           solver_power_iters=solver_power_iters,
           solver_cg_maxiter=solver_cg_maxiter)
    V_model.block_until_ready()
    time_end = time.time()
    print('Time taken to get weights:', time_end - time_start, 'seconds')

    density_2DXY, y_xy, sig_xy = density_set
    h1_model, y_h1, sig_A1 = h1_set
    h2_model, y_h2, sig_A2 = h2_set
    h3_model, y_h3, sig_A3 = h3_set
    h4_model, y_h4, sig_A4 = h4_set

    # V_model = jnp.where(jnp.isnan(V_model), 0.0, V_model)
    # sigma_model = jnp.where(jnp.isnan(sigma_model), 0.0, sigma_model)
    h3_model = jnp.where(jnp.isnan(h3_model), 0.0, h3_model)
    h4_model = jnp.where(jnp.isnan(h4_model), 0.0, h4_model)
    h1_data, h1_data_err = y_h1, sig_A1
    h2_data, h2_data_err = y_h2, sig_A2
    h3_data, h3_data_err = y_h3, sig_A3
    h4_data, h4_data_err = y_h4, sig_A4

    res_density = ((density_2DXY - y_xy) / (sig_xy + EPSILON))**2
    res_h1 = ((h1_model - h1_data) / (h1_data_err + 1e-3))**2
    res_h2 = ((h2_model - h2_data) / (h2_data_err + 1e-3))**2
    res_h3 = ((h3_model - h3_data) / (h3_data_err + 1e-3))**2
    res_h4 = ((h4_model - h4_data) / (h4_data_err + 1e-3))**2    

    res_h1 = jnp.where((h1_model < 9.9), res_h1, 0)
    res_h2 = jnp.where((h2_model < 9.9), res_h2, 0)
    res_h3 = jnp.where((h3_model < 9.9), res_h3, 0)
    res_h4 = jnp.where((h4_model < 9.9), res_h4, 0)

    val1 = jnp.nansum( -0.5 * res_density ) / len(density_2DXY)
    val4 = jnp.nansum( -0.5 * res_h1 ) / len(h1_model)
    val5 = jnp.nansum( -0.5 * res_h2 ) / len(h2_model)
    val6 = jnp.nansum( -0.5 * res_h3 ) / len(h3_model)
    val7 = jnp.nansum( -0.5 * res_h4 ) / len(h4_model)

    log_likelihood = 0
    log_likelihood += val1 + val4 + val5 + val6 + val7
    print('log_likelihood:', log_likelihood)

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.hist(jnp.log(weights), range = [-10, 5], bins=30, alpha=0.7, color='blue', edgecolor='black')

    bin_mapping = dict_data['bin_mapping']
    index_remap = bin_mapping[:-1]

    density_2DXY, y_xy, sig_xy = density_set
    h1_model, y_h1, sig_A1 = h1_set
    h2_model, y_h2, sig_A2 = h2_set
    h3_model, y_h3, sig_A3 = h3_set
    h4_model, y_h4, sig_A4 = h4_set

    density_2DXY_weighted = density_2DXY[index_remap]
    V_model_weighted = V_model[index_remap]
    sigma_model_weighted = sigma_model[index_remap]
    h1_model_weighted = h1_model[index_remap]
    h2_model_weighted = h2_model[index_remap]
    h3_model_weighted = h3_model[index_remap]
    h4_model_weighted = h4_model[index_remap]

    density_2DXY_unity, _, _ = density_unity_set
    h1_model_unity, _, _ = h1_unity_set
    h2_model_unity, _, _ = h2_unity_set
    h3_model_unity, _, _ = h3_unity_set
    h4_model_unity, _, _ = h4_unity_set

    density_2DXY_unity = density_2DXY_unity[index_remap]
    V_model_unity = V_model_unity[index_remap]
    sigma_model_unity = sigma_model_unity[index_remap]
    h1_model_unity = h1_model_unity[index_remap]
    h2_model_unity = h2_model_unity[index_remap]
    h3_model_unity = h3_model_unity[index_remap]
    h4_model_unity = h4_model_unity[index_remap]

    surface_density = dict_data['XY_density_data']
    V_data = dict_data['V_data']
    sigma_data = dict_data['sigma_data']
    h1_data = dict_data['h1_data']
    h2_data = dict_data['h2_data']
    h3_data = dict_data['h3_data']
    h4_data = dict_data['h4_data']
    X_regular_grid = dict_data['X_regular_grid']
    Y_regular_grid = dict_data['Y_regular_grid']

    density_2DXY_data = y_xy[index_remap]
    V_model_data = V_data[index_remap]
    sigma_model_data = sigma_data[index_remap]
    h1_model_data = y_h1[index_remap]
    h2_model_data = y_h2[index_remap]
    h3_model_data = y_h3[index_remap]
    h4_model_data = y_h4[index_remap]


    model_batch = (density_2DXY_weighted, V_model_weighted, sigma_model_weighted, h3_model_weighted, h4_model_weighted)
    model_unity_batch = (density_2DXY_unity, V_model_unity, sigma_model_unity, h3_model_unity, h4_model_unity)
    data_batch = (density_2DXY_data, V_model_data, sigma_model_data, h3_model_data, h4_model_data)

    fig_names = [r'$\Sigma_{\rm lum}$ [L$_\odot$/pc$^2$]', r'$V_{\rm los}$ [km/s]', r'$\sigma_{v}$ [km/s]', r'$h_3$', r'$h_4$']
    fig_names_save = ['Surface_density', 'V_los', 'Sigma_V', 'h3', 'h4']
    vmin_ls = [1e-6, -200, 20, -0.2, -0.2]
    vmax_ls = [1e-3, 200, 120, 0.2, 0.2]
    vminmax_ls = [0.5, 20, 20, 0.5, 0.5]

    fig1, ax1 = plt.subplots(len(model_batch),4, figsize = (60,5 * len(model_batch)), gridspec_kw={'hspace':0.5, 'wspace':0.5})

    for i in range(len(model_batch)):

        cb = ax1[i][0].scatter(X_regular_grid, Y_regular_grid, c=model_batch[i],
                        s = 15, cmap='viridis', marker = 's', norm = 'log' if i == 0 else None,
                        vmin = vmin_ls[i], vmax = vmax_ls[i], rasterized = True)
        ax1[i][0].set_title(f'Model', fontsize=15)
        ax1[i][0].set_xlabel('X [kpc]', fontsize=12)
        ax1[i][0].set_ylabel('Y [kpc]', fontsize=12)
        ax1[i][0].set_xlim(-12, 12)
        ax1[i][0].set_ylim(-4, 4)
        cbar = fig1.colorbar(cb, ax=ax1[i][0])
        cbar.set_label(fig_names[i], fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        cb = ax1[i][1].scatter(X_regular_grid, Y_regular_grid, c=model_unity_batch[i],
                        s = 15, cmap='viridis', marker = 's', norm = 'log' if i == 0 else None,
                        vmin = vmin_ls[i], vmax = vmax_ls[i], rasterized = True)
        ax1[i][1].set_title(f'Model', fontsize=15)
        ax1[i][1].set_xlabel('X [kpc]', fontsize=12)
        ax1[i][1].set_ylabel('Y [kpc]', fontsize=12)
        ax1[i][1].set_xlim(-12, 12)
        ax1[i][1].set_ylim(-4, 4)
        cbar = fig1.colorbar(cb, ax=ax1[i][1])
        cbar.set_label(fig_names[i], fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        cb = ax1[i][2].scatter(X_regular_grid, Y_regular_grid, c=data_batch[i],
                        s = 15, cmap='viridis', marker = 's', norm = 'log' if i == 0 else None,
                        vmin = vmin_ls[i], vmax = vmax_ls[i], rasterized = True)
        ax1[i][2].set_title(f'Data', fontsize=15)
        ax1[i][2].set_xlabel('X [kpc]', fontsize=12)
        ax1[i][2].set_ylabel('Y [kpc]', fontsize=12)
        ax1[i][2].set_xlim(-12, 12)
        ax1[i][2].set_ylim(-4, 4)
        cbar = fig1.colorbar(cb, ax=ax1[i][2])
        cbar.set_label(fig_names[i], fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        res = (data_batch[i] - model_batch[i]) / data_batch[i] if i == 0 else (data_batch[i] - model_batch[i])
        cb = ax1[i][3].scatter(X_regular_grid, Y_regular_grid, c=res,
                        s = 15, cmap='coolwarm', marker = 's', vmin = -vminmax_ls[i], vmax = vminmax_ls[i], rasterized = True)
        ax1[i][3].set_title('Residuals (Data - Model) / Data' if i == 0 else 'Residuals (Data - Model)', fontsize=15)
        ax1[i][3].set_xlabel('X [kpc]', fontsize=12)
        ax1[i][3].set_ylabel('Y [kpc]', fontsize=12)
        ax1[i][3].set_xlim(-12, 12)
        ax1[i][3].set_ylim(-4, 4)
        cbar = fig1.colorbar(cb, ax=ax1[i][3])
        cbar.set_label('Residuals', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show()
