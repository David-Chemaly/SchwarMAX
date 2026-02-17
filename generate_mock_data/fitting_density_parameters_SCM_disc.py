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

from multiprocessing import Pool, cpu_count


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

    Rmax, zmax = 25, 8
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

    def density_func(x, y, z, params, Rmax = 15, zmax = 3):

        R = np.sqrt(x**2 + y**2)

        val = DoubleExponentialDisk_density(x, y, z, params)
        return np.where((R < Rmax) & (np.abs(z) < zmax), val, 0.0)

    def log_prior(theta,):
        if -1 < theta[0] < 2 and -1 < theta[1] < 1:
            return 0.0  # log(1) = 0 for uniform prior
        return -np.inf  # log(0) = -inf for out-of-bounds
    
    def log_prob(theta,):

        ll = log_prior(theta)
        if not np.isfinite(ll):
            return -np.inf

        x,y,z = data['x'], data['y'], data['z']
        mass = data['mass']
        wi = mass/np.mean(mass)

        params = {
            'Sigma0': 1,
            'Rd': 10 ** theta[0],
            'hz': 10 ** theta[1],
            'x_origin':0.0,
            'y_origin':0.0,
            'z_origin':0.0,
            'dirx':0.0,
            'diry':0.0,
            'dirz':1.0
        }
        val = density_func(x,y,z, params, Rmax=Rmax, zmax=zmax)

        W = np.sum(wi)
        norm_mc = gaussian(sample_mc_x, 0, 8) * gaussian(sample_mc_y, 0, 8) * gaussian(sample_mc_z, 0, 3)
        mass_tot = np.sum(density_func(sample_mc_x, sample_mc_y, sample_mc_z, params, Rmax=Rmax, zmax=zmax) / norm_mc) / len(sample_mc_x)

        return np.sum(wi * np.log(val + 1e-10)) - W * np.log(mass_tot)
    
    ndim = 2
    nwalkers = 16  # must be >= 2 * ndim
    p0 = np.array([0, 0])
    initial_pos = p0 + np.random.uniform(-0.5, 0.5, (nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_pos, 500, progress=True)
    samples = sampler.get_chain(discard=200, flat=True)
    print("calculating mass normalization...")
    mass_ls = []
    for i in tqdm(range (0, len(samples))):
        params = {
            'Sigma0': 1,
            'Rd': 10 ** samples[i,0],
            'hz': 10 ** samples[i,1],
            'x_origin':0.0,
            'y_origin':0.0,
            'z_origin':0.0,
            'dirx':0.0,
            'diry':0.0,
            'dirz':1.0
        }
        norm_mc = gaussian(sample_mc_x, 0, 8) * gaussian(sample_mc_y, 0, 8) * gaussian(sample_mc_z, 0, 3)
        mass_tot = np.sum(density_func(sample_mc_x, sample_mc_y, sample_mc_z, params, Rmax=Rmax, zmax=zmax) / norm_mc) / len(sample_mc_x)
        mass_ls.append(np.sum(data['mass']) / mass_tot)
    
    final_params = np.hstack([samples, np.array(mass_ls)[:,None]])
    pd.DataFrame(final_params, columns=['log10_Rd','log10_hz','mass']).to_csv('/data/hz420-2/SchwarMAX/SCM_disc/model/best_fit_params.csv', index=False)

    # final_params = pd.read_csv('/data/hz420-2/SchwarMAX/SCM_disc/model/best_fit_params.csv').to_numpy()

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    samples_plot = final_params
    samples_plot[:,0] = 10 ** samples_plot[:,0]
    samples_plot[:,1] = 10 ** samples_plot[:,1]

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
    R_samples = np.random.exponential(scale=Rd_best, size=n_samples)

    phi_samples = np.random.uniform(0, 2*np.pi, size=n_samples)

    z_samples_raw = np.random.exponential(scale=hz_best, size=n_samples)
    z_samples = z_samples_raw * np.random.choice([-1, 1], size=n_samples)

    x_samples = R_samples * np.cos(phi_samples)
    y_samples = R_samples * np.sin(phi_samples)

    mass_per_particle = mass_best / n_samples
    mass_samples = np.ones(n_samples) * mass_per_particle
    sampled_particles = np.column_stack([x_samples, y_samples, z_samples])

    fig, ax = plt.subplots(1,2, figsize=(12,4))

    H, xegde, yedge = np.histogram2d(posvel[:,0], posvel[:,1], bins=100, range = [[-15,15],[-15,15]], weights=mass)
    ax[0].imshow(H.T, origin='lower', extent=[xegde[0], xegde[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')


    H, xegde, yedge = np.histogram2d(posvel[:,0], posvel[:,2], bins=100, range = [[-15,15],[-15,15]], weights=mass)
    ax[1].imshow(H.T, origin='lower', extent=[xegde[0], xegde[-1], yedge[0], yedge[-1]], aspect='equal', norm = 'log')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('z')

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