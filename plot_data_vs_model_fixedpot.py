from model import * 
from sample_from_density import sample_from_density_grid
from likelihoods import *
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


def get_dict_data_bootstrap(path, filename, N_BOOTSTRAP = 100):

    with open(path + filename, 'rb') as f:
        bin_dict = pickle.load(f)

    X_minmax = jnp.array(bin_dict['X_minmax'])
    Y_minmax = jnp.array(bin_dict['Y_minmax'])
    nX_nY = jnp.array(bin_dict['nX_nY'])

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


    XY_density_data_err = 0.01 * surface_density + EPSILON
    V_data_err = jnp.array(bin_dict['V_mean_err'])
    sigma_data_err = jnp.array(bin_dict['V_sigma_err'])
    h1_data_err = jnp.array(bin_dict['h1_err'])
    h2_data_err = jnp.array(bin_dict['h2_err'])
    h3_data_err = jnp.array(bin_dict['h3_err'])
    h4_data_err = jnp.array(bin_dict['h4_err'])

    '''
    Bootstrap the observation
    '''
    # rng = np.random.default_rng(42)
    # N_BOOTSTRAP = 100
    # y_xy_boot = np.array(surface_density[None, :] + rng.normal(size=(N_BOOTSTRAP, len(surface_density))) * XY_density_data_err[None, :])
    # y_h1_boot = np.array(h1_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h1_data))) * h1_data_err[None, :])
    # y_h2_boot = np.array(h2_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h2_data))) * h2_data_err[None, :])
    # y_h3_boot = np.array(h3_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h3_data))) * h3_data_err[None, :])
    # y_h4_boot = np.array(h4_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h4_data))) * h4_data_err[None, :])
    # V_boot = np.array(V_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(V_data))) * V_data_err[None, :])
    # sigma_boot = np.array(sigma_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(sigma_data))) * sigma_data_err[None, :])
    # # Fix the first one to always be the unperturbed system
    # y_xy_boot[0, :] = surface_density
    # y_h1_boot[0, :] = h1_data
    # y_h2_boot[0, :] = h2_data
    # y_h3_boot[0, :] = h3_data
    # y_h4_boot[0, :] = h4_data
    # V_boot[0, :] = V_data
    # sigma_boot[0, :] = sigma_data
    # y_xy_boot = jnp.array(y_xy_boot)
    # y_h1_boot = jnp.array(y_h1_boot)
    # y_h2_boot = jnp.array(y_h2_boot)
    # y_h3_boot = jnp.array(y_h3_boot)
    # y_h4_boot = jnp.array(y_h4_boot)
    # V_boot = jnp.array(V_boot)
    # sigma_boot = jnp.array(sigma_boot)

    rng = np.random.default_rng(42)
    XY_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(surface_density)))
    h1_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h1_data)))
    h2_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h2_data)))
    h3_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h3_data)))
    h4_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h4_data)))
    V_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(V_data)))
    sigma_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(sigma_data)))
    XY_standard_normal[0, :] = 0.0
    h1_standard_normal[0, :] = 0.0
    h2_standard_normal[0, :] = 0.0
    h3_standard_normal[0, :] = 0.0
    h4_standard_normal[0, :] = 0.0
    V_standard_normal[0, :] = 0.0
    sigma_standard_normal[0, :] = 0.0
    XY_standard_normal = jnp.array(XY_standard_normal)
    h1_standard_normal = jnp.array(h1_standard_normal)
    h2_standard_normal = jnp.array(h2_standard_normal)
    h3_standard_normal = jnp.array(h3_standard_normal)
    h4_standard_normal = jnp.array(h4_standard_normal)
    V_standard_normal = jnp.array(V_standard_normal)
    sigma_standard_normal = jnp.array(sigma_standard_normal)

    from scipy.stats import qmc

    # with open(path + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
    #     Rzphi_density_data = pickle.load(f)
    # R_grid, z_grid, phi_grid = Rzphi_density_data['R_grid'], Rzphi_density_data['z_grid'], Rzphi_density_data['phi_grid']
    # dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    # dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    # dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    # sample_for_integration = Rzphi_density_data['sample_for_integration']

    R_min, R_max =0., 10.
    z_min, z_max = -3., 3.
    n_R, n_z, n_phi =10, 6, 6
    n_tot = int(n_R * n_z * n_phi)
    R_edge = jnp.linspace(R_min, R_max, n_R+1)
    z_edge = jnp.linspace(z_min, z_max, n_z+1)
    phi_edge = jnp.linspace(-jnp.pi, jnp.pi, n_phi+1)
    R_mids, z_mids, phi_mids = 0.5 * (R_edge[:-1] + R_edge[1:]), 0.5 * (z_edge[:-1] + z_edge[1:]), 0.5 * (phi_edge[:-1] + phi_edge[1:])
    dR, dz, dphi = R_edge[1]-R_edge[0], z_edge[1]-z_edge[0], phi_edge[1]-phi_edge[0]
    R_mids_mesh, z_mids_mesh, phi_mids_mesh = jnp.meshgrid(R_mids, z_mids, phi_mids, indexing='ij')
    Rzphi_mid_grid = jnp.stack([R_mids_mesh.ravel(), z_mids_mesh.ravel(), phi_mids_mesh.ravel()], axis=-1)  # (n_R*n_z*n_phi, 3)
    R_grid = Rzphi_mid_grid[:,0]
    z_grid = Rzphi_mid_grid[:,1]
    phi_grid = Rzphi_mid_grid[:,2]
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    Rzphi_minmax=jnp.array([[R_min, R_max],[z_min, z_max],[-jnp.pi, jnp.pi]])
    nRzphi=jnp.array([n_R,n_z,n_phi])
    num_segments_Rzphi=nRzphi.prod()
    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_grid_indices = assign_regular_grid(Rzphi_mid_grid,
                                        grid_min=Rzphi_minmax[:,0],
                                        grid_max=Rzphi_minmax[:,1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)
    _, COUNTS = jnp.unique(Rzphi_grid_indices, return_counts=True)
    argsort = jnp.argsort(Rzphi_grid_indices)
    R_grid = R_grid[argsort]
    z_grid = z_grid[argsort]
    phi_grid = phi_grid[argsort]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample_for_integration = sampler.random_base2(m=10)

    mass_unit = 1/((G*u.Msun).to(u.kpc*(u.km/u.s)**2))
    w0_data, mass_data = agama.readSnapshot(data_folder + '/Bar_model_TG21/model/t_t0_7')
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
    R, phi = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2), np.arctan2(w0_data[:, 1], w0_data[:, 0])
    z = w0_data[:, 2]
    Rzphi_stars = jnp.stack([R, z, phi], axis=-1)
    Rzphi_indices = assign_regular_grid(Rzphi_stars,
                                    grid_min=Rzphi_minmax[:,0],
                                    grid_max=Rzphi_minmax[:,1],
                                    n_bins=nRzphi,
                                    strides=Rzphi_strides)
    mass_in_grid = jax.ops.segment_sum(mass_data, Rzphi_indices, num_segments=num_segments_Rzphi)


    X_regular_grid, Y_regular_grid = bin_dict['X_regular_grid'], bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)

    n_samples = 10_000 #5_000  # Same number as original data
    x_grid = np.linspace(0., 12., 1000)
    logP_xexp = XexpX_pdf_log(x_grid, 4.0)
    key = jax.random.PRNGKey(10086)
    R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.default_rng(42).uniform(0, 2*np.pi, size=n_samples)

    x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)

    x_grid = np.linspace(0, 4., 1000)
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
        'XY_density_data_err': XY_density_data_err,
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

        # 'y_xy_boot': y_xy_boot,
        # 'y_h1_boot': y_h1_boot,
        # 'y_h2_boot': y_h2_boot,
        # 'y_h3_boot': y_h3_boot,
        # 'y_h4_boot': y_h4_boot,
        # 'V_boot': V_boot,
        # 'sigma_boot': sigma_boot,
        'XY_standard_normal': XY_standard_normal,
        'h1_standard_normal': h1_standard_normal,
        'h2_standard_normal': h2_standard_normal,
        'h3_standard_normal': h3_standard_normal,
        'h4_standard_normal': h4_standard_normal,
        'V_standard_normal': V_standard_normal,
        'sigma_standard_normal': sigma_standard_normal,

        'R_grid': R_grid,
        'z_grid': z_grid,
        'phi_grid': phi_grid,
        'R_minmax': [R_min, R_max],
        'z_minmax': [z_min, z_max],
        'phi_minmax': [-jnp.pi, jnp.pi],
        'Rzphi_n_tot': n_tot,
        'Rzphi_n_grid': jnp.array([n_R, n_z, n_phi]),
        'dR': dR,
        'dz': dz,
        'dphi': dphi,
        'sample_for_integration': sample_for_integration,
        'mass_in_grid': mass_in_grid,

        'X_regular_grid': X_regular_grid,
        'Y_regular_grid': Y_regular_grid,
        'dX': dX,
        'dY': dY,
        'sample_for_integration_XY': sample,

        'X_minmax': X_minmax,
        'Y_minmax': Y_minmax,
        'nX_nY': nX_nY,


        'w0': w0,

        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }

    return dict_data

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



data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

figname = data_folder+'/plots/mock_Nbins600_beta25_gamma140_D50_gal2_fixedpotential.png'

# data_filename = 'mock_Nbody_bar_XY_withRot_gal2_Nbins1000.pkl'
data_filename = 'mock_data/mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma140_D50_gal2.pkl'
dict_data = get_dict_data_bootstrap(path, data_filename)
with open(path + data_filename, 'rb') as f:
    bin_dict = pickle.load(f)

filename = f'mock_data/mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma140_D50_gal2.pkl' #_Nbins1000
dict_data = get_dict_data_bootstrap(path, filename)
with open(data_folder + '/Bar_model_TG21/dict_phi_stellar_t_t0_7_centered.pkl', 'rb') as f:
    d = pickle.load(f)
dict_phi = {k: jnp.array(v) for k, v in d.items() if k != '_metadata'}

dict_data['dict_phi'] = dict_phi

girdsearch_result_file = data_folder+f'/grid_search_0420/grid_search_result_0420_beta25_gamma140_D50_gal2.csv'
df_param = pd.read_csv(girdsearch_result_file)
df_param = df_param[np.isfinite(df_param['log_prob'])]
df_param['log_prob'] = df_param['log_prob'] - np.max(df_param['log_prob'])
samples_plot = np.zeros((df_param.shape[0], 2))
samples_plot[:,0] = df_param['log_light_to_mass_ratio'].to_numpy()
samples_plot[:,1] = df_param['log_Omega'].to_numpy()
idx = np.argmax(df_param['log_prob'])
log_light_to_mass_ratio, log_Omega = samples_plot[idx,0], samples_plot[idx,1]

ground_truth = [
    11.88,
    jnp.log10(19.2).item(),
    log_light_to_mass_ratio,
    log_Omega
]

logM_halo = ground_truth[0]
logRs_halo = ground_truth[1]
log_light_to_mass_ratio = ground_truth[2]
log_Omega_bar = ground_truth[3]

params_halo_pot = {
    'logM': logM_halo,
    'Rs': 10 ** logRs_halo,
    'a': 1.0,
    'b': 1.0,
    'c': 1.0,
    'x_origin': 0.0,
    'y_origin': 0.0,
    'z_origin': 0.0,
    'dirx': 0.0,
    'diry': 0.0,
    'dirz': 1.0
}

Omega_bar = 10 ** log_Omega_bar
light_to_mass_ratio = 10 ** log_light_to_mass_ratio
print('Light to mass ratio for this model:', light_to_mass_ratio)
print('Pattern speed for this model:', Omega_bar)

X_minmax = dict_data['X_minmax']
Y_minmax = dict_data['Y_minmax']
nX, nY = dict_data['nX_nY']
xy_lim_grid = jnp.array([X_minmax, Y_minmax])
xy_n_grid = jnp.array([nX, nY])
Rmin, Rmax = dict_data['R_minmax']
zmin, zmax = dict_data['z_minmax']
phimin, phimax = dict_data['phi_minmax']
Rzphi_n_grid = dict_data['Rzphi_n_grid']
Rzphi_n_tot = dict_data['Rzphi_n_tot']

logL, density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set,\
    density_unity_set, _V_model_unity, _sigma_model_unity, h1_unity_set, h2_unity_set, h3_unity_set, h4_unity_set,\
            weights = model_fixed_potential_plotting(params_halo_pot, dict_data['dict_phi'], 
                                Omega_bar, light_to_mass_ratio, 
                                dict_data, dict_data['total_bins'], 
                                Rzphi_n_tot, Rzphi_n_grid, Rzphi_lim_grid=jnp.array([[Rmin, Rmax],[zmin, zmax],[phimin, phimax]]),
                                xy_lim_grid=xy_lim_grid,
                                xy_n_grid=xy_n_grid)

V_model.block_until_ready()
print('Model Done')

print('logL:', logL)

# _, logl_marg, _, _, _, _, _, _, _, logl_all, _ = model_fixed_potential_bootstrap(params_halo_pot, dict_data['dict_phi'], 
#                                 Omega_bar, light_to_mass_ratio, 
#                                 dict_data, dict_data['total_bins'], 
#                                 Rzphi_n_tot, Rzphi_n_grid, Rzphi_lim_grid=jnp.array([[Rmin, Rmax],[zmin, zmax],[phimin, phimax]]),
#                                 xy_lim_grid=xy_lim_grid,
#                                 xy_n_grid=xy_n_grid)
# print('logL (formal_pipeline):', logl_marg)

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

Chi2_density = jnp.sum((density_2DXY - y_xy)**2 / sig_xy**2)
Chi2_h1 = jnp.sum((h1_model - y_h1)**2 / sig_A1**2)
Chi2_h2 = jnp.sum((h2_model - y_h2)**2 / sig_A2**2)
Chi2_h3 = jnp.sum((h3_model - y_h3)**2 / sig_A3**2)
Chi2_h4 = jnp.sum((h4_model - y_h4)**2 / sig_A4**2)


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
chi2_ls = [Chi2_density, Chi2_h1, Chi2_h2, Chi2_h3, Chi2_h4]

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
    ax1[i][2].text(0.05, 0.8, r'$\chi^2 = $'+f'{chi2_ls[i]:.2f}', transform=ax1[i][2].transAxes, fontsize=15)
    cbar = fig1.colorbar(cb, ax=ax1[i][2])
    cbar.set_label('Data - Model', fontsize=15)
    cbar.ax.tick_params(labelsize=14)

    for j in range (0,3):
        ax1[i][j].contour(xmid, ymid, np.log10(H).T, levels=[2.2, 2.8, 3.1, 3.5, 4.], colors='white' if j!=2 else 'grey', linewidths=1)

fig1.savefig(figname, bbox_inches='tight', dpi=300)
# plt.close()

