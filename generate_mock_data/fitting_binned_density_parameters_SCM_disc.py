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

from multiprocessing import Pool, cpu_count
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

@jax.jit
def DoubleExponentialDisk_density(x, y, z, params):
    """Volume density for a simple exponential disc: rho(R,z) = (Sigma0/(2hz)) e^{-R/Rd} e^{-|z|/hz}."""
    # Shift and rotate coordinates
    rin = _shift(x, y, z, params)       # (3, ...)
    rvec = _rotate(rin, params)         # (3, ...)
    rx, ry, rz = rvec + EPSILON

    # Cylindrical R in rotated frame
    R = jnp.sqrt(rx**2 + ry**2)

    return (params['Sigma0']) * jnp.exp(-R / params['Rd']) * jnp.exp(-jnp.abs(rz) / params['hz'])

def XexpX_pdf_log(x, a):
    """
    Probability density function of the distribution proportional to x * exp(-x/a).
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF. Can be scalar or array.
    a : float
        Scale parameter > 0.
    
    Returns
    -------
    pdf : array_like
        The PDF values at x.
    """
    # Ensure a > 0
    a = jnp.asarray(a)
    # PDF formula: (1/a^2) * x * exp(-x/a)
    pdf = jnp.log(x) - jnp.log(a**2) - (x / a)
    return jnp.where(x >= 0, pdf, -jnp.inf)


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
    sample_mc_x = jax.scipy.special.ndtri(sample_mc[1:,0]) * 8
    sample_mc_y = jax.scipy.special.ndtri(sample_mc[1:,1]) * 8
    sample_mc_z = jax.scipy.special.ndtri(sample_mc[1:,2]) * 3

    posvel, mass = agama.readSnapshot(f'/data/hz420-2/SchwarMAX/SCM_disc/model/model_disc_final')
    R = np.sqrt(posvel[:,0]**2 + posvel[:,1]**2)
    z = posvel[:,2]

    mass_factor = 1/((G*u.Msun).to(u.kpc*(u.km/u.s)**2))
    mass = mass * mass_factor.value

    Rmax, zmax = 20, 5
    xmax = Rmax / np.sqrt(2)
    mask = (R < Rmax) & (np.abs(z) < zmax)
    posvel = posvel[mask]
    mass = mass[mask]

    R = R[mask]
    z = z[mask]

    data = {
        'x': posvel[:,0],
        'y': posvel[:,1],
        'z': posvel[:,2],
        'mass': mass
    }

    mass_cell, (xedge, yedge, zedge) = np.histogramdd((data['x'], data['y'], data['z']), bins=(100, 100, 50), range=[[-xmax,xmax],[-xmax,xmax],[-zmax,zmax]], weights=data['mass'])
    mass2_cell,_ = np.histogramdd((data['x'], data['y'], data['z']), bins=(100, 100, 50), range=[[-xmax,xmax],[-xmax,xmax],[-zmax,zmax]], weights=data['mass']**2)
    mass_err_cell = np.sqrt(mass2_cell)
    dx, dy, dz = xedge[1]-xedge[0], yedge[1]-yedge[0], zedge[1]-zedge[0]
    density_cell = mass_cell / (dx * dy * dz)
    density_err_cell = mass_err_cell / (dx * dy * dz)
    xmid, ymid, zmid = 0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]), 0.5*(zedge[:-1] + zedge[1:])
    x_grid, y_grid, z_grid = np.meshgrid(xmid, ymid, zmid, indexing='ij')

    pos_grid = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
    density_grid = density_cell.ravel()
    density_err_grid = density_err_cell.ravel()

    mask = density_grid > 0
    pos_grid = pos_grid[mask]
    density_grid = density_grid[mask]
    density_err_grid = density_err_grid[mask]
    print(True in np.isnan(density_grid))

    def density_func(x, y, z, params, Rmax = 15, zmax = 3):

        R = np.sqrt(x**2 + y**2)

        val = DoubleExponentialDisk_density(x, y, z, params)
        return np.where((R < Rmax) & (np.abs(z) < zmax), val, 0.0)

    def log_prior(theta,):
        if -1 < theta[0] < 2 and -1 < theta[1] < 1 and 6 < theta[2] < 10:
            return 0.0  # log(1) = 0 for uniform prior
        return -np.inf  # log(0) = -inf for out-of-bounds
    
    def log_prob(theta,):
        # print(theta)
        ll = log_prior(theta)
        if not np.isfinite(ll):
            return -np.inf

        x,y,z = pos_grid[:,0], pos_grid[:,1], pos_grid[:,2]
        density_data = density_grid

        params = {
            'Sigma0': 10**theta[2],
            'Rd': 10 ** theta[0],
            'hz': 10 ** theta[1],
            'x_origin':0.0,
            'y_origin':0.0,
            'z_origin':0.0,
            'dirx':0.0,
            'diry':0.0,
            'dirz':1.0
        }
        density_model = density_func(x,y,z, params, Rmax=Rmax, zmax=zmax)#

        # print('nan in density_model:', True in np.isnan(density_model))

        residual = (density_model - density_data) / (density_err_grid + 1e-12)

        # print('nan in residual:', True in np.isnan(residual))

        val = -0.5 * np.sum(residual**2)

        # print('nan in val:', val)
        return val
    
    ndim = 3
    nwalkers = 16  # must be >= 2 * ndim
    p0 = np.array([0, 0, 8])
    initial_pos = p0 + np.random.uniform(-0.5, 0.5, (nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_pos, 500, progress=True)
    samples = sampler.get_chain(discard=200, flat=True)
    final_params = samples
    pd.DataFrame(final_params, columns=['log10_Rd','log10_hz','mass']).to_csv('/data/hz420-2/SchwarMAX/SCM_disc/model/best_fit_params_binned_disc.csv', index=False)

    # final_params = pd.read_csv('/data/hz420-2/SchwarMAX/SCM_disc/model/best_fit_params.csv').to_numpy()

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    samples_plot = final_params
    samples_plot[:,0] = 10 ** samples_plot[:,0]
    samples_plot[:,1] = 10 ** samples_plot[:,1]
    samples_plot[:,2] = 10 ** samples_plot[:,2]

    corner.corner(samples_plot, labels=['R_d','h_z', 'mass'], 
                show_titles=True, title_fmt='.2f', title_kwargs={"fontsize": 15},
                smooth=True, quantiles=[0.16, 0.5, 0.84], fig=fig)

    best_fit_param = np.percentile(samples_plot, 50, axis=0)
    print("Best-fit parameters:")
    for i, name in enumerate(['R_d','h_z','mass']):
        print(f"{name}: {best_fit_param[i]:.4f}")
 
    # Sample particles from the best-fit density distribution
    print("\nSampling particles from best-fit density distribution...")
    Rd_best = best_fit_param[0]
    hz_best = best_fit_param[1]
    mass_best = best_fit_param[2] * 4 * np.pi * Rd_best**2 * hz_best  # Total mass from best-fit parameters
    
    n_samples = len(posvel)  # Same number as original data
    
    # Sample R from exponential distribution
    # R_samples = np.random.exponential(scale=Rd_best, size=n_samples)
    x_grid = np.linspace(0, Rmax, 1000)
    logP_xexp = XexpX_pdf_log(x_grid, Rd_best)
    key = jax.random.PRNGKey(10086)
    R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.uniform(0, 2*np.pi, size=n_samples)
    z_samples_raw = np.random.exponential(scale=hz_best, size=n_samples)
    z_samples = z_samples_raw * np.random.choice([-1, 1], size=n_samples)
    x_samples = R_samples * np.cos(phi_samples)
    y_samples = R_samples * np.sin(phi_samples)

    mass_per_particle = mass_best / n_samples
    mass_samples = np.ones(n_samples) * mass_per_particle
    sampled_particles = np.column_stack([x_samples, y_samples, z_samples])

    fig, ax = plt.subplots(1,3, figsize=(18,4), gridspec_kw={'wspace':0.4})

    H, xegde, yedge = np.histogram2d(posvel[:,0], posvel[:,1], bins=100, range = [[-15,15],[-15,15]], weights=mass)
    cb = ax[0].imshow(H.T, origin='lower', extent=[xegde[0], xegde[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    plt.colorbar(cb, ax=ax[0], label='Mass')


    H_model, xegde, yedge = np.histogram2d(x_samples, y_samples, bins=100, range = [[-15,15],[-15,15]], weights=mass_samples)
    cb = ax[1].imshow(H_model.T, origin='lower', extent=[xegde[0], xegde[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
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

    H, xegde, yedge = np.histogram2d(posvel[:,0], posvel[:,2], bins=100, range = [[-15,15],[-5,5]], weights=mass)
    cb = ax[0].imshow(H.T, origin='lower', extent=[xegde[0], xegde[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('z')
    plt.colorbar(cb, ax=ax[0], label='Mass')


    H_model, xegde, yedge = np.histogram2d(x_samples, z_samples, bins=100, range = [[-15,15],[-5,5]], weights=mass_samples)
    cb = ax[1].imshow(H_model.T, origin='lower', extent=[xegde[0], xegde[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
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
    H, xegde = np.histogram(R, bins=100, range = [0,15], weights=mass)
    H_den = H / (2 * np.pi * xegde[:-1] * (xegde[1]-xegde[0]))
    ax[0].plot(xegde[:-1], H_den)
    H_model, xegde = np.histogram(R_samples, bins=100, range = [0,15], weights=mass_samples)
    H_model_den = H_model / (2 * np.pi * xegde[:-1] * (xegde[1]-xegde[0]))
    ax[0].plot(xegde[:-1], H_model_den)
    ax[0].set_xlabel('R')
    ax[0].set_ylabel('Mass')
    ax[0].set_yscale('log')

    H, xegde = np.histogram(z, bins=100, range = [-5,5], weights=mass)
    ax[1].plot(xegde[:-1], H)
    H_model, xegde = np.histogram(z_samples, bins=100, range = [-5,5], weights=mass_samples)
    ax[1].plot(xegde[:-1], H_model)
    ax[1].set_xlabel('z')
    ax[1].set_ylabel('Mass')
    ax[1].set_yscale('log')

    plt.show()