# Imports
import pickle

import jax
import jaxopt
import jax.nn as jnn
import jax.numpy as jnp
from functools import partial

import time as tt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18

import os
import corner
import numpy as np
import multiprocessing as mp

import dynesty
import dynesty.utils as dyut

from potentials import *
from integrants import *
from utils import *

from potentials import NFW_acceleration
from densities import MiyamotoNagai_density

from CylindricalSpline import get_phi_m, get_acc

def prior_transform(p):
    #ndim = 14
    logM_halo, Rs_halo, \
    logM_disk, Rs_disk, Hs_disk = p 

    logM_halo1 = 10 + 4 * logM_halo
    Rs_halo1   = 10 + 20 * Rs_halo

    logM_disk1 = 7 + 3 * logM_disk
    Rs_disk1   = 1 + 4 * Rs_disk

    return jnp.array([
        logM_halo1,
        Rs_halo1,
        logM_disk1,
        Rs_disk1,
        Hs_disk
    ])

def dynesty_fit(dict_data, logl_fn, prior_fn, ndim, nlive=500):
    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(logl_fn,
                                prior_fn,
                                ndim,
                                logl_args=(dict_data, ),
                                nlive=nlive,
                                sample='rslice',
                                pool=poo,
                                queue_size=nthreads * 2)
        dns.run_nested(n_effective=10000)

    res   = dns.results
    inds  = np.arange(len(res.samples))
    inds  = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl  = res.logl[inds]

    dns_results = {
                    'dns': dns,
                    'samps': samps,
                    'logl': logl,
                    'logz': res.logz,
                    'logzerr': res.logzerr,
                }

    return dns_results

def logl(params, dict_data):
    params_halo_pot = {
        'logM': params[0],
        'Rs':params[1],
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
        'logM': params[2],
        'Rs': params[3],
        'Hs': params[4],
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0
    }

    vr_binned, sigmavr_binned, vphi_binned, sigmavphi_binned, vz_binned, sigmavz_binned = model(params_halo_pot, params_disk_rho, dict_data)

    log_likelihood = jnp.nansum( 1/jnp.sqrt(dict_data['density_data']) * \
                        ( (dict_data['vz_data_binned'] - vr_binned)**2 + (dict_data['vphi_data_binned'] - vphi_binned)**2 + (dict_data['vz_data_binned'] - vz_binned)**2 + \
                            1/jnp.sqrt(2)*( (dict_data['sigmavr_data_binned'] - sigmavr_binned)**2 + (dict_data['sigmavphi_data_binned'] - sigmavphi_binned)**2 + \
                                (dict_data['sigmavz_data_binned'] - sigmavz_binned)**2) ) )

    return -0.5 * log_likelihood

@jax.jit
def model(params_halo_pot, params_disk_rho, dict_data):

    NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-2, 30.0, 1e-3, 15.0, 8.
    Nphi = 200
    N_int = 10_000
    dict_phi = get_phi_m(MiyamotoNagai_density, params_disk_rho, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)

    @jax.jit
    def acc_fn(x, y, z):
        a_halo = NFW_acceleration(x, y, z,  params_halo_pot)
        a_disk = get_acc(x, y, z, dict_phi)
        return a_halo + a_disk

    time = 2. #Gyr
    n_steps = 1000
    dt = time / n_steps
    unroll = False
    initial_time = 0.0

    time, xv = jax.vmap(integrate_leapfrog_traj, in_axes=(0, None, None, None, None, None))(dict_data['w0'], acc_fn, n_steps, dt, initial_time, unroll)
    density_model = jax.vmap(histogram3d)(xv[:, :, :3])

    A = density_model.reshape(len(xv), -1).T.astype(jnp.float32)/n_steps
    y = dict_data['density_data'].reshape(-1).astype(jnp.float32)
    sig = jnp.sqrt(y + 1.0).astype(jnp.float32)  # Poisson noise + 1.0 floor

    weights = solve_lbfgs_softplus(A, y, sig, l2=1e-3, maxiter=300)

    weights_binned = histogram3d(xv[:, :, :3].reshape(-1, 3), jnp.repeat(weights, n_steps).ravel())

    r_model    = jnp.linalg.norm(xv[:, :, :2], axis=-1)
    vr_model   = (xv[:, :, 0] * xv[:, :, 3] + xv[:, :, 1] * xv[:, :, 4]) / r_model
    vr_binned  = histogram3d(xv[:, :, :3].reshape(-1, 3), (vr_model*weights[:, None]).ravel())/(weights_binned+EPSILON)
    vr2_binned  = histogram3d(xv[:, :, :3].reshape(-1, 3), (vr_model**2 * weights[:, None]).ravel())/(weights_binned+EPSILON)
    sigmavr_binned = jnp.sqrt(jnp.clip(vr2_binned - vr_binned**2, a_min=0.0))

    vphi = -(-xv[:, :, 1] * xv[:, :, 3] + xv[:, :, 0] * xv[:, :, 4]) / r_model
    vphi_binned  = histogram3d(xv[:, :, :3].reshape(-1, 3), (vphi*weights[:, None]).ravel())/(weights_binned+EPSILON)
    vphi2_binned  = histogram3d(xv[:, :, :3].reshape(-1, 3), (vphi**2 * weights[:, None]).ravel())/(weights_binned+EPSILON)
    sigmavphi_binned = jnp.sqrt(jnp.clip(vphi2_binned - vphi_binned**2, a_min=0.0))

    vz_binned = histogram3d(xv[:, :, :3].reshape(-1, 3), (xv[:, :, 5]*weights[:, None]).ravel())/(weights_binned+EPSILON)
    vz2_binned = histogram3d(xv[:, :, :3].reshape(-1, 3), (xv[:, :, 5]**2 * weights[:, None]).ravel())/(weights_binned+EPSILON)
    sigmavz_binned = jnp.sqrt(jnp.clip(vz2_binned - vz_binned**2, a_min=0.0))

    return vr_binned, sigmavr_binned, vphi_binned, sigmavphi_binned, vz_binned, sigmavz_binned

@jax.jit
def _nll_z(z, A, y, sig, l2):
    x = jnn.softplus(z)  # strictly positive
    r = (A @ x - y) / sig
    return 0.5 * jnp.dot(r, r) + 0.5 * l2 * jnp.dot(x, x)
_nll_z = jax.value_and_grad(_nll_z)

@jax.jit
def solve_lbfgs_softplus(A, y, sigma, l2=1e-3, maxiter=500, tol=1e-6):
    z0 = jnp.zeros(A.shape[1], A.dtype)
    solver = jaxopt.LBFGS(fun=_nll_z, value_and_grad=True, maxiter=maxiter, tol=tol)
    res = solver.run(z0, A, y, sigma, l2)
    x_hat = jnn.softplus(res.params)
    return x_hat

if __name__ == "__main__":
    #Hyperparameters for dynesty
    ndim = 5
    nlive = 500
    PATH_DATA = f'/data/dc824-2/ScharMAX_first_tests'

    with open('./IC_axisymmetric_disc.pkl', 'rb') as f:
        ic = pickle.load(f)
    w0 = jnp.array([ic['x'], ic['y'], ic['z'], ic['vx'], ic['vy'], ic['vz']]).T

    with open('./axisymmetric_disc.pkl', 'rb') as f:
        data = pickle.load(f)
    w0_data = jnp.array([data['x'], data['y'], data['z'], data['vx'], data['vy'], data['vz']]).T
    density_data = histogram3d(w0_data[:, :3])

    r_data    = jnp.linalg.norm(w0_data[:, :2], axis=-1)
    vr_data   = (w0_data[:, 0] * w0_data[:, 3] + w0_data[:, 1] * w0_data[:, 4]) / r_data
    vr_data_binned  = histogram3d(w0_data[:, :3], vr_data) / density_data
    vr2_data_binned  = histogram3d(w0_data[:, :3], vr_data**2) / density_data
    sigmavr_data_binned = jnp.sqrt(vr2_data_binned - vr_data_binned**2)

    vphi_data = -(-w0_data[:, 1] * w0_data[:, 3] + w0_data[:, 0] * w0_data[:, 4]) / r_data
    vphi_data_binned  = histogram3d(w0_data[:, :3], vphi_data) / density_data
    vphi2_data_binned  = histogram3d(w0_data[:, :3], vphi_data**2) / density_data
    sigmavphi_data_binned = jnp.sqrt(vphi2_data_binned - vphi_data_binned**2)

    vz_data = w0_data[:, 5]
    vz_data_binned = histogram3d(w0_data[:, :3], w0_data[:, 5]) / density_data
    vz2_data_binned = histogram3d(w0_data[:, :3], vz_data**2) / density_data
    sigmavz_data_binned = jnp.sqrt(vz2_data_binned - vz_data_binned**2)

    dict_data = {
        'w0': w0,
        'density_data': density_data,
        'vr_data_binned': vr_data_binned,
        'sigmavr_data_binned': sigmavr_data_binned,
        'vphi_data_binned': vphi_data_binned,
        'sigmavphi_data_binned': sigmavphi_data_binned,
        'vz_data_binned': vz_data_binned,
        'sigmavz_data_binned': sigmavz_data_binned
    }

    print(f'Fitting...')
    dict_results = dynesty_fit(dict_data, logl, prior_transform, ndim, nlive)
    with open(f'{PATH_DATA}/dict_results.pkl', 'wb') as f:
        pickle.dump(dict_results, f)

    ground_truth = [
        jnp.log10(0.8*10**12).item(),
        data['ground_truth']['halo_params']['scaleRadius'],
        jnp.log10(data['ground_truth']['disc_params']['Sigma0']).item(),
        data['ground_truth']['disc_params']['Rd'],
        data['ground_truth']['disc_params']['hz'],
    ]

    # Plot and Save corner plot
    labels = ['logM', 'Rs', 'logm', 'rs', 'hs']
    figure = corner.corner(dict_results['samps'], 
                labels=labels,
                color='blue',
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True, 
                title_kwargs={"fontsize": 16},
                truths=ground_truth,
                truth_color='red',
                )
    figure.savefig(f'{PATH_DATA}/corner_plot.pdf')
    plt.close(figure)