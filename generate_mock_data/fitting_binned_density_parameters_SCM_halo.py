path = '/home/hz420/python_script/SchwarMAX/'

import sys
sys.path.append(path)

import numpy as np
import scipy as sp
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
from potentials import *
from densities import *
import emcee
import corner
import agama
agama.setUnits(length=1, velocity=1, mass=1)

from astropy.constants import G
import astropy.units as u

from functools import partial


def gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _shift(x, y, z, p):
    # Convert scalar params to arrays matching x,y,z shape
    x0 = jnp.asarray(p["x_origin"])
    y0 = jnp.asarray(p["y_origin"])
    z0 = jnp.asarray(p["z_origin"])

    # Broadcast to match shapes of inputs
    x0 = jnp.broadcast_to(x0, x.shape)
    y0 = jnp.broadcast_to(y0, y.shape)
    z0 = jnp.broadcast_to(z0, z.shape)

    # Stack as a 3-vector field
    return jnp.stack([x - x0, y - y0, z - z0], axis=0)

def _rotate(vec, p):
    # vec: (3, ...)
    R = get_mat(p["dirx"], p["diry"], p["dirz"])  # (3,3)
    # Tensordot over axis: (i,a) * (a,...) -> (i,...)
    return jnp.tensordot(R, vec, axes=[[1],[0]])

def NFW_density(x, y, z, params):
    rin = _shift(x, y, z, params)       # (3, ...)
    rvec = _rotate(rin, params)         # (3, ...)
    rx, ry, rz = rvec + 1e-3

    # Cylindrical R in rotated frame
    r = jnp.sqrt(rx**2 + ry**2 + rz**2)

    val = 10 ** params['logM'] / (4*jnp.pi * r * (params['Rs'] + r)**2)

    return val

@partial(jax.jit,static_argnums=(2))
def sample_from_logP(x_grid, logP, N, key):
    """
    Draw N samples from the distribution defined by logP on the grid x_grid
    using the inverse‐CDF method.
    """
    # 1) Shift & exponentiate for numerical stability
    logP = jnp.asarray(logP)
    logP = logP - jnp.max(logP)
    P = jnp.exp(logP)

    # 2) Normalize to get a proper probability mass on the grid
    P /= P.sum()

    # 3) Build the CDF
    cdf = jnp.cumsum(P)

    # 4) Sample uniforms and invert the CDF via linear interpolation
    # jax_random_key2 = jax.random.PRNGKey(random_seed)
    u = jax.random.uniform(key, shape=(N,))
    samples = jnp.interp(u, cdf, x_grid)
    return samples

if __name__ == "__main__":

    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=False)
    sample_mc = sampler.random_base2(m=10)
    sample_mc_x = jax.scipy.special.ndtri(sample_mc[1:,0]) * 30
    sample_mc_y = jax.scipy.special.ndtri(sample_mc[1:,1]) * 30
    sample_mc_z = jax.scipy.special.ndtri(sample_mc[1:,2]) * 30

    posvel, mass = agama.readSnapshot(f'/data/hz420-2/SchwarMAX/SCM_disc/model/model_halo_final')
    R = np.sqrt(posvel[:,0]**2 + posvel[:,1]**2)
    z = posvel[:,2]
    r = np.sqrt(posvel[:,0]**2 + posvel[:,1]**2 + posvel[:,2]**2)

    mass_factor = 1/((G*u.Msun).to(u.kpc*(u.km/u.s)**2))
    mass = mass * mass_factor.value
 

    rmax, rmin = 40, 0.1
    mask = (r < rmax) & (r > rmin)
    posvel = posvel[mask]
    mass = mass[mask]

    r = r[mask]
    R = R[mask]
    z = z[mask]

    data = {
        'x': posvel[:,0],
        'y': posvel[:,1],
        'z': posvel[:,2],
        'mass': mass
    }

    mass_shell, redge = np.histogram(r, bins=100, range = [rmin, rmax], weights = mass)
    mass2_shell, redge = np.histogram(r, bins=100, range = [rmin, rmax], weights = mass**2)
    mass_err_shell = np.sqrt(mass2_shell)
    r_grid = 0.5 * (redge[:-1] + redge[1:])
    dr = redge[1] - redge[0]
    
    density_grid = mass_shell.ravel() / (4 * np.pi * r_grid**2 * dr)
    density_err_grid = np.sqrt(mass2_shell).ravel() / (4 * np.pi * r_grid**2 * dr)

    mask = (density_grid > 0) & (r_grid > rmin) & (r_grid < rmax)
    r_grid = r_grid[mask]
    density_grid = density_grid[mask]
    density_err_grid = density_err_grid[mask]
    print(True in np.isnan(density_grid))

    def density_func(x, y, z, params, rmax = 50):

        r = np.sqrt(x**2 + y**2 + z**2)

        val = NFW_density(x, y, z, params)
        return np.where((r < rmax) & (r > rmin), val, 0.0)

    def log_prior(theta,):
        if 0 < theta[1] < 2 and 10 < theta[0] < 14:
            return 0.0  # log(1) = 0 for uniform prior
        return -np.inf  # log(0) = -inf for out-of-bounds
    
    def log_prob(theta,):
        # print(theta)
        ll = log_prior(theta)
        if not np.isfinite(ll):
            return -np.inf

        x,y,z = r_grid, np.zeros_like(r_grid), np.zeros_like(r_grid)
        density_data = density_grid

        params = {
            'logM': theta[0],
            'Rs': 10 ** theta[1],
            'x_origin':0.0,
            'y_origin':0.0,
            'z_origin':0.0,
            'dirx':0.0,
            'diry':0.0,
            'dirz':1.0
        }
        density_model = density_func(x,y,z, params, rmax=rmax)#

        # print('nan in density_model:', True in np.isnan(density_model))

        residual = (density_model - density_data) / (density_err_grid + 1e-12)

        # print('nan in residual:', True in np.isnan(residual))

        val = -0.5 * np.sum(residual**2)

        # print('nan in val:', val)
        return val
    
    ndim = 2
    nwalkers = 16  # must be >= 2 * ndim
    p0 = np.array([11.8, 1,])
    initial_pos = p0 + np.random.uniform(-0.5, 0.5, (nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_pos, 500, progress=True)
    samples = sampler.get_chain(discard=200, flat=True)
    final_params = samples
    pd.DataFrame(final_params, columns=['log10_M', 'log10_Rs']).to_csv('/data/hz420-2/SchwarMAX/SCM_disc/model/best_fit_params_binned_halo.csv', index=False)

    # final_params = pd.read_csv('/data/hz420-2/SchwarMAX/SCM_disc/model/best_fit_params.csv').to_numpy()

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    samples_plot = final_params
    samples_plot[:,1] = 10 ** samples_plot[:,1]

    corner.corner(samples_plot, labels=['logM', 'Rs'], 
                show_titles=True, title_fmt='.2f', title_kwargs={"fontsize": 15},
                smooth=True, quantiles=[0.16, 0.5, 0.84], fig=fig)
    ########################
    fig.savefig('/data/hz420-2/SchwarMAX/SCM_disc/model/halo_posterior.png')
    ########################

    best_fit_param = np.percentile(samples_plot, 50, axis=0)
    print("Best-fit parameters:")
    for i, name in enumerate(["logM", "Rs"]):
        print(f"{name}: {best_fit_param[i]:.4f}")
 
    # Sample particles from the best-fit density distribution
    print("\nSampling particles from best-fit density distribution...")
    Rs_best = best_fit_param[1]
    logM_best = best_fit_param[0]
    best_param = {
        'logM': logM_best,
        'Rs': Rs_best,
        'x_origin':0.0,
        'y_origin':0.0,
        'z_origin':0.0,
        'dirx':0.0,
        'diry':0.0,
        'dirz':1.0
    }

    # mass_best = 10 ** logM_best  # Total mass from best-fit parameters
    norm_mc = gaussian(sample_mc_x, 0, 30) * gaussian(sample_mc_y, 0, 30) * gaussian(sample_mc_z, 0, 30)
    mass_best = np.sum(density_func(sample_mc_x, sample_mc_y, sample_mc_z, best_param, rmax=rmax) / norm_mc) / len(sample_mc_x)
    print('best mass:', mass_best)
    
    n_samples = len(posvel)  # Same number as original data
    
    # Sample R from exponential distribution
    # R_samples = np.random.exponential(scale=Rd_best, size=n_samples)
    x_grid = np.linspace(0.3, rmax, 1000)
    y_grid = np.zeros_like(x_grid)
    z_grid = np.zeros_like(x_grid)
    logP_xexp = jnp.log(NFW_density(x_grid, y_grid, z_grid, best_param) * x_grid**2)  # Include Jacobian for spherical coordinates
    key = jax.random.PRNGKey(10086)
    r_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.uniform(0, 2*np.pi, size=n_samples)
    theta_samples = np.arccos(np.random.uniform(-1, 1, size=n_samples))
    x_samples = r_samples * np.sin(theta_samples) * np.cos(phi_samples)
    y_samples = r_samples * np.sin(theta_samples) * np.sin(phi_samples)
    z_samples = r_samples * np.cos(theta_samples)

    mass_per_particle = mass_best / n_samples
    mass_samples = np.ones(n_samples) * mass_per_particle
    sampled_particles = np.column_stack([x_samples, y_samples, z_samples])

    fig, ax = plt.subplots(1,3, figsize=(18,4), gridspec_kw={'wspace':0.4})

    H, xedge, yedge = np.histogram2d(posvel[:,0], posvel[:,1], bins=100, range = [[-30,30],[-30,30]], weights=mass)
    cb = ax[0].imshow(H.T, origin='lower', extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    plt.colorbar(cb, ax=ax[0], label='Mass')


    H_model, xedge, yedge = np.histogram2d(x_samples, y_samples, bins=100, range = [[-30,30],[-30,30]], weights=mass_samples)
    cb = ax[1].imshow(H_model.T, origin='lower', extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    plt.colorbar(cb, ax=ax[1], label='Mass')

    residual = (H - H_model) / H
    cb = ax[2].pcolormesh(xedge, yedge, residual.T, vmin = -0.5, vmax = 0.5, cmap='coolwarm')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_aspect('equal')
    plt.colorbar(cb, ax=ax[2], label='Relative Residual')

    fig, ax = plt.subplots(1,3, figsize=(18,3), gridspec_kw={'wspace':0.4})

    H, xedge, yedge = np.histogram2d(posvel[:,0], posvel[:,2], bins=100, range = [[-30,30],[-30,30]], weights=mass)
    cb = ax[0].imshow(H.T, origin='lower', extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('z')
    plt.colorbar(cb, ax=ax[0], label='Mass')


    H_model, xedge, yedge = np.histogram2d(x_samples, z_samples, bins=100, range = [[-30,30],[-30,30]], weights=mass_samples)
    cb = ax[1].imshow(H_model.T, origin='lower', extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('z')
    plt.colorbar(cb, ax=ax[1], label='Mass')

    residual = (H - H_model) / H
    cb = ax[2].pcolormesh(xedge, yedge, residual.T, vmin = -0.5, vmax = 0.5, cmap='coolwarm')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('z')
    ax[2].set_aspect('equal')
    plt.colorbar(cb, ax=ax[2], label='Relative Residual')

    fig, ax = plt.subplots(1,2, figsize=(12,4))
    H, xedge = np.histogram(r, bins=100, range = [0,50], weights=mass)
    H_den = H / (4 * np.pi * xedge[:-1]**2 * (xedge[1]-xedge[0]))
    ax[0].plot(xedge[:-1], H_den)
    H_model, xedge = np.histogram(r_samples, bins=100, range = [0,50], weights=mass_samples)
    H_model_den = H_model / (4 * np.pi * xedge[:-1]**2 * (xedge[1]-xedge[0]))
    ax[0].plot(xedge[:-1], H_model_den)
    ax[0].set_xlabel('R')
    ax[0].set_ylabel('Mass')
    ax[0].set_yscale('log')

    H, xedge = np.histogram(z, bins=100, range = [-30,30], weights=mass)
    ax[1].plot(xedge[:-1], H)
    H_model, xedge = np.histogram(z_samples, bins=100, range = [-30,30], weights=mass_samples)
    ax[1].plot(xedge[:-1], H_model)
    ax[1].set_xlabel('z')
    ax[1].set_ylabel('Mass')
    ax[1].set_yscale('log')

    plt.show()