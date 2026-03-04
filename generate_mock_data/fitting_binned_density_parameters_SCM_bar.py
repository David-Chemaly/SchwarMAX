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
from sample_from_density import sample_from_density_grid
import emcee
import corner
import agama
agama.setUnits(length=1, velocity=1, mass=1)

from astropy.constants import G
import astropy.units as u

from multiprocessing import Pool, cpu_count
from functools import partial
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
from sample_from_density import sample_from_density_grid
import emcee
import corner
import agama
agama.setUnits(length=1, velocity=1, mass=1)

from astropy.constants import G
import astropy.units as u

from multiprocessing import Pool, cpu_count
from functools import partial

def sample_from_density_grid(
    density_fn,
    params,
    bounds,
    n_samples,
    n_x=96,
    n_y=96,
    n_z=64,
):
    """
    JAX-jittable generic sampler from a 3D density via grid-based inverse-CDF.

    Works well for cuspy profiles where naive rejection from a wide uniform box
    becomes inefficient.

    Parameters
    ----------
    key : PRNGKey
    density_fn : callable
        Signature density_fn(x, y, z, params) -> rho.
    params : pytree
        Density parameters.
    bounds : array (3,2)
        [[xmin,xmax], [ymin,ymax], [zmin,zmax]].
    n_samples : int
        Number of particles.
    n_x, n_y, n_z : int
        Grid resolution used to discretize rho.
    """
    bounds = jnp.asarray(bounds, dtype=jnp.float32)

    x_edges = jnp.linspace(bounds[0, 0], bounds[0, 1], n_x + 1)
    y_edges = jnp.linspace(bounds[1, 0], bounds[1, 1], n_y + 1)
    z_edges = jnp.linspace(bounds[2, 0], bounds[2, 1], n_z + 1)

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    dz = z_edges[1] - z_edges[0]

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    X, Y, Z = jnp.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()

    rho = density_fn(X, Y, Z, params)
    rho = jnp.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    rho = jnp.clip(rho, a_min=0.0)

    cell_mass = rho * dx * dy * dz
    flat_mass = cell_mass

    total_mass = flat_mass.sum()
    N_sample_per_cell = int(n_samples / (n_x * n_y * n_z))
    # print(N_sample_per_cell)
    x_samples = []
    y_samples = []
    z_samples = []
    m_samples = []
    for i in tqdm(range(len(flat_mass))):
        x0 = X[i]
        y0 = Y[i]
        z0 = Z[i]

        x_samples.append(x0 + dx * (np.random.uniform(-0.5, 0.5, size=(N_sample_per_cell))))
        y_samples.append(y0 + dy * (np.random.uniform(-0.5, 0.5, size=(N_sample_per_cell))))
        z_samples.append(z0 + dz * (np.random.uniform(-0.5, 0.5, size=(N_sample_per_cell))))

        m_samples.append((flat_mass[i] / N_sample_per_cell) * jnp.ones(N_sample_per_cell))

    x_samples = jnp.concatenate(x_samples)
    y_samples = jnp.concatenate(y_samples)
    z_samples = jnp.concatenate(z_samples)
    m_samples = jnp.concatenate(m_samples)

    samples = jnp.stack([x_samples, y_samples, z_samples], axis=1)
    mass_per_particle = m_samples

    return {
        "samples": samples,
        'mass_per_particle': mass_per_particle,
        "cell_mass_grid": cell_mass,
    }



def gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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

    posvel, mass = agama.readSnapshot(f'/data/hz420-2/SchwarMAX/Bar_model_TG21/model/t_t0_2.5')
    mask = (mass!=np.unique(mass)[-1])
    posvel = posvel[mask]
    mass = mass[mask]

    posvel[:,0] = posvel[:,0] - np.mean(posvel[:,0])
    posvel[:,1] = posvel[:,1] - np.mean(posvel[:,1])
    posvel[:,2] = posvel[:,2] - np.mean(posvel[:,2])
    posvel[:,3] = posvel[:,3] - np.mean(posvel[:,3])
    posvel[:,4] = posvel[:,4] - np.mean(posvel[:,4])
    posvel[:,5] = posvel[:,5] - np.mean(posvel[:,5])

    posvel[:,0] = -posvel[:,0]
    posvel[:,3] = -posvel[:,3]

    R_mid, bar_angles0, bar_strength0 = bar_angle_bar_strength(posvel[:,0], posvel[:,1], R_anulus = np.arange(1,5,0.25))
    bar_angle0 = np.mean(bar_angles0[R_mid<4])
    rot_angle = -bar_angle0
    posvel = rotate(posvel, rot_angle)  # rotate to make it anticlockwise


    R = np.sqrt(posvel[:,0]**2 + posvel[:,1]**2)
    z = posvel[:,2]

    mass_factor = 1/((G*u.Msun).to(u.kpc*(u.km/u.s)**2))
    mass = mass * mass_factor.value

    Rmax, zmax = 15, 5
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

    @jax.jit
    def Dehnen_density(x, y, z, params):

        '''
        Dehnen profile: rho = (M / (4π p q Rs^3)) * (r/Rs)^(-n) * (1 + r/Rs)^(n-4), where n = 2
        '''

        p, q, Rs = params['p_bar'], params['q_bar'], params['Rs_bar']
        M = 10.0 ** params['logM_bar']
        r = jnp.sqrt(x**2 + (y / p)**2 + (z / q)**2)

        val = M / (4 * jnp.pi * p * q * Rs**3) * (r / Rs)**(-2) * (1 + r / Rs)**(-2) * jnp.exp(-(z / 3)**4)

        return val

    def density_func(x, y, z, params):

        R = jnp.sqrt(x**2 + y**2)

        val = DoubleExponentialDisk_density(x, y, z, params) + Dehnen_density(x, y, z, params)#
        return val #jnp.where((R < Rmax) & (jnp.abs(z) < zmax), val, 0.0)

    def log_prior(theta,):
        if (-1 < theta[0] < 2) and (-1 < theta[1] < 1) and (7 < theta[2] < 11) and \
        (8 < theta[3] < 12) and (-1 < theta[4] < 1) and (-2 < theta[5] < 0) and (-2 < theta[6] < 0):
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
            'rho0_disc': 10**theta[2],
            'Rd_disc': 10 ** theta[0],
            'hz_disc': 10 ** theta[1],
            'logM_bar': theta[3],
            'Rs_bar': 10 ** theta[4],
            'p_bar':10 ** theta[5],
            'q_bar':10 ** theta[6],
            'x_origin':0.0,
            'y_origin':0.0,
            'z_origin':0.0,
            'dirx':0.0,
            'diry':0.0,
            'dirz':1.0
        }

        density_model = density_func(x,y,z, params)#

        # print('nan in density_model:', True in np.isnan(density_model))

        residual = (density_model - density_data) / (density_err_grid + 1e-12)

        # print('nan in residual:', True in np.isnan(residual))

        val = -0.5 * np.sum(residual**2)

        # print('nan in val:', val)
        return val

    ndim = 7
    nwalkers = 32  # must be >= 2 * ndim
    p0 = np.array([0, 0, 9, 10, 0, -0.5, -0.5])
    initial_pos = p0 + np.random.uniform(-0.2, 0.2, (nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_pos, 400, progress=True)
    samples = sampler.get_chain(discard=200, flat=True)
    final_params = samples
    pd.DataFrame(
        final_params, columns=['log10_Rd','log10_hz','log10_rho0','log10_M','log10_Rs','log10_p','log10_q']
        ).to_csv('/data/hz420-2/SchwarMAX/Bar_model_TG21/model/best_fit_params_binned_stars.csv', index=False)

    # final_params = pd.read_csv('/data/hz420-2/SchwarMAX/SCM_disc/model/best_fit_params.csv').to_numpy()

    fig, ax = plt.subplots(7, 7, figsize=(35, 35))
    samples_plot = final_params
    samples_plot[:,0] = 10 ** samples_plot[:,0]
    samples_plot[:,1] = 10 ** samples_plot[:,1]
    samples_plot[:,4] = 10 ** samples_plot[:,4]
    samples_plot[:,5] = 10 ** samples_plot[:,5]
    samples_plot[:,6] = 10 ** samples_plot[:,6]

    corner.corner(samples_plot, labels=['R_d','h_z', 'log10 rho0', 'log10 M_bar', 'R_s', 'p', 'q'], 
                show_titles=True, title_fmt='.2f', title_kwargs={"fontsize": 15},
                smooth=True, quantiles=[0.16, 0.5, 0.84], fig=fig)
    ########################
    fig.savefig('/data/hz420-2/SchwarMAX/Bar_model_TG21/model/bar_posterior.png')
    ########################

    best_fit_param = np.percentile(samples_plot, 50, axis=0)
    print("Best-fit parameters:")
    for i, name in enumerate(['R_d','h_z', 'log10 rho0', 'log10 M_bar', 'R_s', 'p', 'q']):
        print(f"{name}: {best_fit_param[i]:.4f}")

    # Sample particles from the best-fit density distribution
    print("\nSampling particles from best-fit density distribution...")
    Rd_best = best_fit_param[0]
    hz_best = best_fit_param[1]
    rho_best = 10**best_fit_param[2]
    logM_best = best_fit_param[3]
    Rs_best = best_fit_param[4]
    p_best = best_fit_param[5]
    q_best = best_fit_param[6]
    params_best = {
            'rho0_disc': rho_best,
            'Rd_disc': Rd_best,
            'hz_disc': hz_best,
            'logM_bar': logM_best,
            'Rs_bar': Rs_best,
            'p_bar': p_best,
            'q_bar': q_best,
            'x_origin':0.0,
            'y_origin':0.0,
            'z_origin':0.0,
            'dirx':0.0,
            'diry':0.0,
            'dirz':1.0
        }
    bounds = jnp.array(
        [
            [-15, 15],  # x
            [-15, 15],  # y
            [-5, 5],    # z
        ],
        dtype=jnp.float32,
    )

    n_samples = 5_000_000
    n_x = 90
    n_y = 90
    n_z = 30
    sample_ic_dict = sample_from_density_grid(
        density_func,
        params_best,
        bounds,
        n_samples=n_samples,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
    )
    sampled_particles = np.asarray(sample_ic_dict["samples"])
    mass_samples = sample_ic_dict['mass_per_particle']
    R_samples = np.sqrt(sampled_particles[:,0]**2 + sampled_particles[:,1]**2)
    z_samples = sampled_particles[:,2]

    sampled_particles = np.asarray(sample_ic_dict["samples"])
    mass_samples = sample_ic_dict['mass_per_particle']
    R_samples = np.sqrt(sampled_particles[:,0]**2 + sampled_particles[:,1]**2)
    z_samples = sampled_particles[:,2]

    fig, ax = plt.subplots(1,3, figsize=(18,4), gridspec_kw={'wspace':0.4})

    H, xedge, yedge = np.histogram2d(posvel[:,0], posvel[:,1], bins=100, range = [[-xmax,xmax],[-xmax,xmax]], weights=mass)
    cb = ax[0].imshow(H.T, origin='lower', extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]], norm = 'log', vmin = 1e6, vmax = 1e9)
    ax[0].contour(xedge[:-1], yedge[:-1], np.log10(H).T, levels=np.arange(6,9.5,0.5), colors='white', linewidths=0.5,)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    plt.colorbar(cb, ax=ax[0], label='Mass')


    H_model, xedge, yedge = np.histogram2d(sampled_particles[:,0], sampled_particles[:,1], bins=100, range = [[-xmax,xmax],[-xmax,xmax]], weights=mass_samples)
    cb = ax[1].imshow(H_model.T, origin='lower', extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]], norm = 'log', vmin = 1e6, vmax = 1e9)
    ax[1].contour(xedge[:-1], yedge[:-1], np.log10(H_model).T, levels=np.arange(6,9.5,0.5), colors='white', linewidths=0.5)
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

    H, xedge, yedge = np.histogram2d(posvel[:,0], posvel[:,2], bins=100, range = [[-xmax,xmax],[-zmax,zmax]], weights=mass)
    cb = ax[0].imshow(H.T, origin='lower', extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]], norm = 'log', vmin = 5e5, vmax = 5e8)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('z')
    plt.colorbar(cb, ax=ax[0], label='Mass')


    H_model, xedge, yedge = np.histogram2d(sampled_particles[:,0], sampled_particles[:,2], bins=100, range = [[-xmax,xmax],[-zmax,zmax]], weights=mass_samples)
    cb = ax[1].imshow(H_model.T, origin='lower', extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]], norm = 'log', vmin = 5e5, vmax = 5e8)
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
    z_bins = [0,1,2,3,]
    R_bins = [0,3,6,9,12]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i in range(len(z_bins)-1):
        mask1 = (np.abs(z) > z_bins[i]) & (np.abs(z) < z_bins[i+1])
        mask2 = (np.abs(z_samples) > z_bins[i]) & (np.abs(z_samples) < z_bins[i+1])

        H, xegde = np.histogram(R[mask1], bins=100, range = [0,Rmax], weights=mass[mask1])
        H_den = H / (2 * np.pi * xegde[:-1] * (xegde[1]-xegde[0]))
        ax[0].plot(xegde[:-1], H_den, color=colors[i], label=f'{z_bins[i]} < |z| < {z_bins[i+1]} (snapshot)')
        H_model, xegde = np.histogram(R_samples[mask2], bins=100, range = [0,Rmax], weights=mass_samples[mask2])
        H_model_den = H_model / (2 * np.pi * xegde[:-1] * (xegde[1]-xegde[0]))
        ax[0].plot(xegde[:-1], H_model_den, color=colors[i], linestyle='dashed', label=f'{z_bins[i]} < |z| < {z_bins[i+1]} (model)')
        ax[0].set_xlabel('R')
        ax[0].set_ylabel('Mass')
        ax[0].set_yscale('log')

    for i in range(len(R_bins)-1):
        mask1 = (R > R_bins[i]) & (R < R_bins[i+1])
        mask2 = (R_samples > R_bins[i]) & (R_samples < R_bins[i+1])
        H, xegde = np.histogram(z[mask1], bins=100, range = [-zmax,zmax], weights=mass[mask1])
        ax[1].plot(xegde[:-1], H, color=colors[i], label=f'{R_bins[i]} < R < {R_bins[i+1]} (snapshot)')
        H_model, xegde = np.histogram(z_samples[mask2], bins=100, range = [-zmax,zmax], weights=mass_samples[mask2])
        ax[1].plot(xegde[:-1], H_model, color=colors[i], linestyle='dashed', label=f'{R_bins[i]} < R < {R_bins[i+1]} (model)')
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('Mass')
        ax[1].set_yscale('log')

    plt.show()