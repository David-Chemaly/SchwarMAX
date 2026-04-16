from model_bar import * 
from sample_from_density import sample_from_density_grid
from likelihoods_bar import logl_density, get_dict_data_bootstrap
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
import scipy as sp
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


import cmasher as cmr

from astropy import units as u
from astropy.constants import G

# the following commands make plots look better
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


data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
# data_filename = 'mock_Nbody_bar_XY_withRot_gal2_Nbins1000.pkl'
data_filename = 'mock_data/mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma140_D50_gal2.pkl'
dict_data = get_dict_data_bootstrap(path, data_filename, n_samples = int(7.5e3))

with open(path + data_filename, 'rb') as f:
    bin_dict = pickle.load(f)


DISCARD = 300        # burn-in steps to discard for corner plot
THIN = 1             # thinning factor for corner plot
# CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_gal2_0406.pkl'
CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_0415_beta25_gamma140_D50_gal2.pkl'

figname = data_folder+'/plots/mock_Nbins600_beta25_gamma140_D50_gal2.png'

param_names = ['logM_halo', 'logM_disk', 'logM_bar', 'logRs_halo', 'logRs_disk', 'logHs_disk', 'logRs_bar', 
               r'$\alpha$', r'$\beta$', r'$\gamma$', 
               r'$\log_{10}(L/M)$', r'$\log_{10}(\Omega)$', r'$\log_{10}(\sigma amplifier)$']


posterior, logprob, step = load_checkpoint(CHECKPOINT_FILE)
posterior = posterior[:, logprob[-1, :]>np.amax(logprob[-1, :])-100, :]
posterior = posterior[DISCARD::THIN, :, :]
posterior = posterior.reshape(-1, posterior.shape[-1])


best_fit_param = np.percentile(posterior, 50, axis=0)
print("Best-fit parameters:")
for i, name in enumerate(param_names):
    print(f"{name}: {best_fit_param[i]:.4f}")

dict_data['logl_density_max'] = -0.24
logMhalo_10_best_fit, logMdisk_best_fit, logMbar_best_fit, logC_halo_best_fit, logRs_disk_best_fit, logHs_disk_best_fit, logRs_bar_best_fit,\
      alpha_best_fit, beta_best_fit, gamma_best_fit, logLM_best_fit, logOmega_best_fit, logSigma_amplifier_best_fit = best_fit_param
logMhalo_best_fit, logRh_halo_best_fit = logMenc_logc_to_logM_logRs(logMhalo_10_best_fit, logC_halo_best_fit, r_enc=10.0, Delta=200., rho_crit=277.54)

alpha = alpha_best_fit * 180/np.pi
beta = beta_best_fit * 180/np.pi
gamma = gamma_best_fit * 180/np.pi# gamma_best_fit
ground_truth = [logMhalo_best_fit,
                logMdisk_best_fit,
                logMbar_best_fit,
                logRh_halo_best_fit,
                logRs_disk_best_fit,
                logHs_disk_best_fit,
                logRs_bar_best_fit,   
                alpha,
                beta,
                gamma,
                logLM_best_fit,
                logOmega_best_fit,
                # jnp.log10(15.0).item(),
                logSigma_amplifier_best_fit
]
# ground_truth = [
#     10.8,
#     np.float64(10.650968119218467),
#     np.float64(9.99223738935679),
#     0.8970275,
#     np.float64(0.4951777935492888),
#     np.float64(0.0021520312746454513),
#     np.float64(0.22416550751555658),
#     0.17453292519943295  * 180/np.pi,
#     np.float64(0.3906510255411446)  * 180/np.pi,
#     2.443460952792061  * 180/np.pi,
#     0.,
#     1.5,
#     0.5]
# logMhalo_best_fit, logRh_halo_best_fit = logMenc_logc_to_logM_logRs(ground_truth[0], ground_truth[3], r_enc=10.0, Delta=200., rho_crit=277.54)
# ground_truth[0] = logMhalo_best_fit
# ground_truth[3] = logRh_halo_best_fit

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
    'logM_disc': ground_truth[1],
    'Rs_disc': 10 ** ground_truth[4],
    'Hs_disc': 10 ** ground_truth[5],
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
    'L_bar': 10 ** ground_truth[6],
    'a_bar': 10 ** ground_truth[6] / 5,
    'b_bar': 10 ** ground_truth[5],
    'light_to_mass_ratio': 10 ** ground_truth[10],
    'Omega_bar': 10 ** ground_truth[11],
    'sigma_amplifier': 10 ** ground_truth[12],
}

print('Pattern speed for this model:', params_disk_rho['Omega_bar'])

surface_density_model = projection(params_disk_rho, dict_data, dict_data['total_bins'])
surface_density_gt = dict_data['XY_density_data'] / params_disk_rho['light_to_mass_ratio']

chi2 = jnp.sum((surface_density_gt - surface_density_model)**2 / (0.1 * surface_density_gt)**2)
_logl_density = -0.5 * chi2 / dict_data['total_bins']
print(_logl_density)

X_minmax = dict_data['X_minmax']
Y_minmax = dict_data['Y_minmax']
nX, nY = dict_data['nX_nY']
xy_lim_grid = jnp.array([X_minmax, Y_minmax])
xy_n_grid = jnp.array([nX, nY])

Rmin, Rmax = dict_data['R_minmax']
zmin, zmax = dict_data['z_minmax']
phimin, phimax = dict_data['phi_minmax']
Rzphi_n_tot = dict_data['Rzphi_n_tot']
Rzphi_n_grid = dict_data['Rzphi_n_grid']

logL, density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set,\
        density_unity_set, _V_model_unity, _sigma_model_unity, h1_unity_set, h2_unity_set, h3_unity_set, h4_unity_set,\
              weights = model_for_plotting(params_halo_pot, params_disk_rho, dict_data, dict_data['total_bins'],
                                           Rzphi_n_tot, Rzphi_n_grid, Rzphi_lim_grid=jnp.array([[Rmin, Rmax],[zmin, zmax],[phimin, phimax]]),
                                            xy_lim_grid=xy_lim_grid,
                                            xy_n_grid=xy_n_grid)
V_model.block_until_ready()
print('Model Done')

print('logL:', logL)


mass_unit = 1/((G*u.Msun).to(u.kpc*(u.km/u.s)**2))
w0_data, mass_data = agama.readSnapshot(data_folder + '/Bar_model_TG21/model/t_t0_4')
mass_data = mass_data * mass_unit.value

mask = (mass_data!=np.unique(mass_data)[-1])
w0_data = w0_data[mask]
mass_data = mass_data[mask]

w0_data[:,0] = w0_data[:,0] - np.mean(w0_data[:,0])
w0_data[:,1] = w0_data[:,1] - np.mean(w0_data[:,1])
w0_data[:,2] = w0_data[:,2] - np.mean(w0_data[:,2])
w0_data[:,3] = w0_data[:,3] - np.mean(w0_data[:,3])
w0_data[:,4] = w0_data[:,4] - np.mean(w0_data[:,4])
w0_data[:,5] = w0_data[:,5] - np.mean(w0_data[:,5])

R_mid, bar_angles0, bar_strength0 = bar_angle_bar_strength(w0_data[:,0], w0_data[:,1], R_anulus = np.arange(1,5,0.25))
bar_angle0 = np.mean(bar_angles0[R_mid<4])
rot_angle = -bar_angle0
w0_data = rotate(w0_data, rot_angle)  # rotate to make it anticlockwise

w0_data[:,0] = -w0_data[:,0]
w0_data[:,3] = -w0_data[:,3]


alpha, beta, gamma = bin_dict['orientation']
rot_mat = makeRotationMatrix(alpha, beta, gamma)
x_data, v_data = w0_data[:,:3], w0_data[:,3:]
x_data = (rot_mat @ x_data.T).T
v_data = (rot_mat @ v_data.T).T
w0_data[:,:3] = x_data
w0_data[:,3:] = v_data
x, y = w0_data[:, 0], w0_data[:, 1]
R, phi = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2), np.arctan2(w0_data[:, 1], w0_data[:, 0])
z, vy = w0_data[:, 2], w0_data[:, 4]
XY_stars = np.stack([x, z], axis=-1) # prefect edge-on view

X_min, X_max = -12., 12.
Y_min, Y_max = -4., 4.
nX, nY = 60,40
area_XY = ((X_max - X_min)/nX) * ((Y_max - Y_min)/nY)
X_edge = jnp.linspace(X_min, X_max, nX+1)
Y_edge = jnp.linspace(Y_min, Y_max, nY+1)
X_mids, Y_mids = 0.5 * (X_edge[:-1] + X_edge[1:]), 0.5 * (Y_edge[:-1] + Y_edge[1:])
H, xedge, yedge = np.histogram2d(XY_stars[:, 0], XY_stars[:, 1], bins=[X_edge, Y_edge])
signal = H.flatten()
noise = np.sqrt(signal + 1)
xmid, ymid = 0.5 * (xedge[1:] + xedge[:-1]), 0.5 * (yedge[1:] + yedge[:-1])
H = sp.ndimage.gaussian_filter(H, sigma=0.5)

bin_mapping = dict_data['bin_mapping']
index_remap = bin_mapping[:-1]


density_2DXY, y_xy, sig_xy = density_set
h1_model, y_h1, sig_A1 = h1_set
h2_model, y_h2, sig_A2 = h2_set
h3_model, y_h3, sig_A3 = h3_set
h4_model, y_h4, sig_A4 = h4_set


index_remap = bin_mapping[:-1]

density_2DXY_weighted = density_2DXY[index_remap]
V_model_weighted = V_model[index_remap]
sigma_model_weighted = sigma_model[index_remap]
h1_model_weighted = h1_model[index_remap]
h2_model_weighted = h2_model[index_remap]
h3_model_weighted = h3_model[index_remap]
h4_model_weighted = h4_model[index_remap]

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
data_batch = (density_2DXY_data, V_model_data, sigma_model_data, h3_model_data, h4_model_data)

fig_names = [r'$\Sigma_{\rm *}$ [L$_\odot$/pc$^2$]', r'$V_{\rm los}$ [km/s]', r'$\sigma_{v}$ [km/s]', r'$h_3$', r'$h_4$']
fig_names_save = ['Surface_density', 'V_los', 'Sigma_V', 'h3', 'h4']
color_maps = [cmr.sepia, cmr.iceburn,  cmr.amber, cmr.iceburn, cmr.amber]
vmin_ls = [1e1, -200, 20, -0.2, -0.2]
vmax_ls = [1e4, 200, 150, 0.2, 0.1]
vminmax_ls = [0.15, 25, 25, 0.15, 0.15]

fig1, ax1 = plt.subplots(len(model_batch),3, figsize = (18, 2.5 * len(model_batch)), gridspec_kw={'hspace':0.4, 'wspace':0.3})

for i in range(len(model_batch)):
    
    # fig0, ax0 = plt.subplots(1,3, figsize = (25,3), gridspec_kw={'hspace':0.5, 'wspace':0.3})
    
    # cb = ax0[0].scatter(X_regular_grid, Y_regular_grid, c=model_batch[i],
    #                 s = 22, cmap='viridis', marker = 's', norm = 'log' if i == 0 else None,
    #                 vmin = vmin_ls[i], vmax = vmax_ls[i], rasterized = True)
    # ax0[0].set_title(f'Model', fontsize=15)
    # ax0[0].set_xlabel('X [kpc]', fontsize=12)
    # ax0[0].set_ylabel('Y [kpc]', fontsize=12)
    # ax0[0].set_xlim(-12, 12)
    # ax0[0].set_ylim(-4, 4)
    # cbar = fig0.colorbar(cb, ax=ax0[0])
    # cbar.set_label(fig_names[i], fontsize=18)
    # cbar.ax.tick_params(labelsize=14)

    # cb = ax0[1].scatter(X_regular_grid, Y_regular_grid, c=data_batch[i],
    #                 s = 22, cmap='viridis', marker = 's', norm = 'log' if i == 0 else None,
    #                 vmin = vmin_ls[i], vmax = vmax_ls[i], rasterized = True)
    # ax0[1].set_title(f'Data', fontsize=15)
    # ax0[1].set_xlabel('X [kpc]', fontsize=12)
    # ax0[1].set_ylabel('Y [kpc]', fontsize=12)
    # ax0[1].set_xlim(-12, 12)
    # ax0[1].set_ylim(-4, 4)
    # cbar = fig0.colorbar(cb, ax=ax0[1])
    # cbar.set_label(fig_names[i], fontsize=18)
    # cbar.ax.tick_params(labelsize=14)

    # res = (data_batch[i] - model_batch[i]) / data_batch[i] if i == 0 else (data_batch[i] - model_batch[i])
    # cb = ax0[2].scatter(X_regular_grid, Y_regular_grid, c=res,
    #                 s = 22, cmap='coolwarm', marker = 's', vmin = -vminmax_ls[i], vmax = vminmax_ls[i], rasterized = True)
    # ax0[2].set_title('Residuals (Data - Model) / Data' if i == 0 else 'Residuals (Data - Model)', fontsize=15)
    # ax0[2].set_xlabel('X [kpc]', fontsize=12)
    # ax0[2].set_ylabel('Y [kpc]', fontsize=12)
    # ax0[2].set_xlim(-12, 12)
    # ax0[2].set_ylim(-4, 4)
    # cbar = fig0.colorbar(cb, ax=ax0[2])
    # cbar.set_label('Residuals', fontsize=18)
    # cbar.ax.tick_params(labelsize=14)

    # for j in range (0,3):
    #     ax0[j].contour(xmid, ymid, np.log10(H).T, levels=5, colors='white' if j!=2 else 'grey', linewidths=1)

    # fig0.savefig(f'/data/hz420-2/SchwarMAX/plots/Mock_disc_bulge_{fig_names_save[i]}.png', bbox_inches='tight', dpi=300)

    cb = ax1[i][0].scatter(X_regular_grid, Y_regular_grid, c=model_batch[i],
                    s = 22, cmap=color_maps[i], marker = 's', norm = 'log' if i == 0 else None,
                    vmin = vmin_ls[i], vmax = vmax_ls[i], rasterized = True)
    ax1[i][0].set_title('Model' if i==0 else '', fontsize=18)
    ax1[i][0].set_xlabel('X [kpc]', fontsize=12)
    ax1[i][0].set_ylabel('Y [kpc]', fontsize=12)
    ax1[i][0].set_xlim(X_minmax)
    ax1[i][0].set_ylim(Y_minmax)
    cbar = fig1.colorbar(cb, ax=ax1[i][0])
    cbar.set_label(fig_names[i], fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    cb = ax1[i][1].scatter(X_regular_grid, Y_regular_grid, c=data_batch[i],
                    s = 22, cmap=color_maps[i], marker = 's', norm = 'log' if i == 0 else None,
                    vmin = vmin_ls[i], vmax = vmax_ls[i], rasterized = True)
    ax1[i][1].set_title('Data' if i==0 else '', fontsize=18)
    ax1[i][1].set_xlabel('X [kpc]', fontsize=12)
    ax1[i][1].set_ylabel('Y [kpc]', fontsize=12)
    ax1[i][1].set_xlim(X_minmax)
    ax1[i][1].set_ylim(Y_minmax)
    cbar = fig1.colorbar(cb, ax=ax1[i][1])
    cbar.set_label(fig_names[i], fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # res = (data_batch[i] - model_batch[i]) / data_batch[i] if i == 0 else (data_batch[i] - model_batch[i])
    res = (np.log10(data_batch[i]) - np.log10(model_batch[i])) if i == 0 else (data_batch[i] - model_batch[i])
    cb = ax1[i][2].scatter(X_regular_grid, Y_regular_grid, c=res,
                    s = 22, cmap='coolwarm', marker = 's', vmin = -vminmax_ls[i], vmax = vminmax_ls[i], rasterized = True)
    ax1[i][2].set_title('Residuals' if i == 0 else '', fontsize=18)
    ax1[i][2].set_xlabel('X [kpc]', fontsize=12)
    ax1[i][2].set_ylabel('Y [kpc]', fontsize=12)
    ax1[i][2].set_xlim(X_minmax)
    ax1[i][2].set_ylim(Y_minmax)
    cbar = fig1.colorbar(cb, ax=ax1[i][2])
    cbar.set_label('Data - Model', fontsize=15)
    cbar.ax.tick_params(labelsize=14)

    for j in range (0,3):
        ax1[i][j].contour(xmid, ymid, np.log10(H).T, levels=[2.2, 2.8, 3.1, 3.5, 4.], colors='white' if j!=2 else 'grey', linewidths=1)

fig1.savefig(figname, bbox_inches='tight', dpi=300)
# plt.close()

