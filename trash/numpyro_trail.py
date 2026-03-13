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

from potentials import *
from integrants import *
from utils import *

from potentials import NFW_acceleration
from densities import MiyamotoNagai_density

from CylindricalSpline import get_phi_m, get_acc

import numpyro
from numpyro_model import numpyro_model


def logl(params, dict_data):


    params_halo_pot = {
        'logM': params['logM_halo'],
        'Rs':10 ** params['logRs_halo'],
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
        'logM': params['logM_disk'],
        'Rs': 10 ** params['logRs_disk'],
        'Hs': 10 ** params['logHs_disk'],
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0
    }

    vr_binned, sigmavr_binned, vphi_binned, sigmavphi_binned, vz_binned, sigmavz_binned = model(params_halo_pot, params_disk_rho, dict_data)

    frac_err = 0.1 

    log_likelihood = 0
    log_likelihood += jnp.nansum( -0.5 * ((dict_data['vz_data_binned'] - vr_binned)**2) / (frac_err * dict_data['vz_data_binned'] + 1e-3) )
    log_likelihood += jnp.nansum( -0.5 * ((dict_data['vphi_data_binned'] - vphi_binned)**2) / (frac_err * dict_data['vphi_data_binned'] + 1e-3) )
    log_likelihood += jnp.nansum( -0.5 * ((dict_data['vz_data_binned'] - vz_binned)**2) / (frac_err * dict_data['vz_data_binned'] + 1e-3) )
    log_likelihood += jnp.nansum( -0.5 * ((dict_data['sigmavr_data_binned'] - sigmavr_binned)**2) / (frac_err * dict_data['sigmavr_data_binned'] + 1e-3) )
    log_likelihood += jnp.nansum( -0.5 * ((dict_data['sigmavphi_data_binned'] - sigmavphi_binned)**2) / (frac_err * dict_data['sigmavphi_data_binned'] + 1e-3) )
    log_likelihood += jnp.nansum( -0.5 * ((dict_data['sigmavz_data_binned'] - sigmavz_binned)**2) / (frac_err * dict_data['sigmavz_data_binned'] + 1e-3) )

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
    density_model = jax.vmap(histogram3d)(xv[:, :, :3], None)

    A = density_model.reshape(len(xv), -1).T.astype(jnp.float32)/n_steps
    y = dict_data['density_data'].reshape(-1).astype(jnp.float32)
    sig = jnp.sqrt(y + 1.0).astype(jnp.floalogt32)  # Poisson noise + 1.0 floor

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
def solve_lbfgs_softplus(A, y, sigma, l2=1e-3, maxiter=200, tol=1e-6):
    z0 = jnp.zeros(A.shape[1], A.dtype)
    solver = jaxopt.LBFGS(fun=_nll_z, value_and_grad=True, maxiter=maxiter, tol=tol)
    res = solver.run(z0, A, y, sigma, l2)
    x_hat = jnn.softplus(res.params)
    return x_hat


#Hyperparameters for dynesty
ndim = 5
nlive = 500
n_particles = 10_000
PATH_DATA = f'/data/hz420-2/'

save_file = PATH_DATA + f'/Schwarzschild/trail_sample.pkl'
save_sampler = PATH_DATA + f'/Schwarzschild/trail_sample_idata.nc'

with open('./IC_axisymmetric_disc.pkl', 'rb') as f:
    ic = pickle.load(f)
index = np.random.choice(n_particles, size=n_particles, replace=False)
w0 = jnp.array([ic['x'][index], ic['y'][index], ic['z'][index], ic['vx'][index], ic['vy'][index], ic['vz'][index]]).T

with open('./axisymmetric_disc.pkl', 'rb') as f:
    data = pickle.load(f)
w0_data = jnp.array([data['x'], data['y'], data['z'], data['vx'], data['vy'], data['vz']]).T
density_data = histogram3d(w0_data[:, :3], None)

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
    'density_data': density_data,
    'vr_data_binned': vr_data_binned,
    'sigmavr_data_binned': sigmavr_data_binned,
    'vphi_data_binned': vphi_data_binned,
    'sigmavphi_data_binned': sigmavphi_data_binned,
    'vz_data_binned': vz_data_binned,
    'sigmavz_data_binned': sigmavz_data_binned
}


ground_truth = {
        'logM_halo': jnp.log10(0.8*10**12).item(),
        'logRs_halo': jnp.log10(data['ground_truth']['halo_params']['scaleRadius']).item(),
        'logM_disk': jnp.log10(data['ground_truth']['disc_params']['Sigma0']).item(),
        'logRs_disk': jnp.log10(data['ground_truth']['disc_params']['Rd']).item(),
        'logHs_disk': jnp.log10(data['ground_truth']['disc_params']['hz']).item(),
}

prior_loc = {
    'logM_halo': ground_truth['logM_halo'],
    'logRs_halo': ground_truth['logRs_halo'],
    'logM_disk': ground_truth['logM_disk'],
    'logRs_disk': ground_truth['logRs_disk'],
    'logHs_disk': ground_truth['logHs_disk'],
}
prior_scales = {
    'logM_halo': 0.5,
    'logRs_halo': 0.5,
    'logM_disk': 0.5,
    'logRs_disk': 0.5,
    'logHs_disk': 0.5,
}

def log_prior(params):
    logp = 0.0
    for key in params.keys():
        scale = prior_scales[key]
        logp += -0.5 * ((params[key] - ground_truth[key]) / scale) ** 2
    return logp


parameters = {
    'logM_halo': numpyro.distributions.Normal(
        loc=prior_loc['logM_halo'],
        scale=prior_scales['logM_halo']
    ),
    'logRs_halo': numpyro.distributions.Normal(
        loc=prior_loc['logRs_halo'],
        scale=prior_scales['logRs_halo'],
    ),
    'logM_disk': numpyro.distributions.Normal(
        loc=prior_loc['logM_disk'],
        scale=prior_scales['logM_disk'],
    ),
    'logRs_disk': numpyro.distributions.Normal(
        loc=prior_loc['logRs_disk'],
        scale=prior_scales['logRs_disk'],
    ),
    'logHs_disk': numpyro.distributions.Normal(
        loc=prior_loc['logHs_disk'],
        scale=prior_scales['logHs_disk'],
    ),
}

init_from_minimiser = True

n_warmup = 200
n_samples = 400
num_chains = 2
max_tree_depth = 6
target_accept_prob = 0.8
step_size = 1e-2
adapt_step_size = True
extra_fields = ('num_steps', 'adapt_state.step_size')
jit_model_args = True

if init_from_minimiser:

    # file_name = path+f'radial_migration_kernel/mock_sample/L_conditioned/minimization_result_withMHgrad_{Nknots}knots4_binned.npy'#_L@10
    # file_name = read_file
    # minimiser_results = np.load(file_name)
    init_guess = ground_truth
    print('Initial guess from minimiser:', init_guess)
    init_strategy=numpyro.infer.initialization.init_to_value(values=init_guess)
else:
    init_strategy=numpyro.infer.init_to_sample()



print('model initialising...')

model = numpyro_model(logl, parameters, dict_data,
                      expand_args=False, log_prior_fn=log_prior)#logL_numpyro, logL_zero

print('model initialised, and start running MCMC...')
model.run_mcmc(num_warmup=n_warmup, num_samples=n_samples, num_chains=num_chains,
                   init_strategy=init_strategy, max_tree_depth=max_tree_depth, step_size=step_size,
                   target_accept_prob=target_accept_prob, adapt_step_size=adapt_step_size,
                   chain_method="vectorized", extra_fields=extra_fields, jit_model_args=jit_model_args) # sequential, vectorized

print('MCMC finished, collecting samples...')
samples = model.samples()
print(samples)
# file = path+f'radial_migration_kernel/mock_sample/L_conditioned/Prior_distribution_{Nknots}knots.pkl'
file = save_file
with open(file,'wb') as f:
    pickle.dump(samples, f)

ef = model.mcmc.get_extra_fields(group_by_chain=True,
                                )
num_steps = ef['num_steps']    # shape (n_chains, n_warmup+n_samples)
print("Avg leapfrog steps per sample:",
      num_steps.mean())
print("Total lnL/grad calls ≃", num_steps.sum())

model.mcmc.print_summary()

import arviz as az


# 2) convert to an ArviZ InferenceData
idata = az.from_numpyro(posterior=model.mcmc, num_chains=num_chains)

# 3) write to NetCDF on disk
idata.to_netcdf(save_sampler)