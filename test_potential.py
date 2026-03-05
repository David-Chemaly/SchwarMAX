import agama
from constants import KPCGYR_TO_KMS
agama.setUnits(mass=1, length=1, velocity=KPCGYR_TO_KMS)

from integrants_with_binning import integrate_leapfrog_barred
from sample_from_density import sample_from_density_grid
from densities import *
from potentials import *
from utils import *
from model import *
from CylindricalSpline import get_phi_m, get_acc, evaluate_phi_axisymmetric

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd
import pickle

import time

def getCylindricalFromCartesian_clockwise(x, y, vx, vy):
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    vR = (x*vx + y*vy) / R
    vphi = -(x*vy - y*vx) / R
    return R, phi, vR, vphi

if __name__ == "__main__":

    @jax.jit
    def Dehnen_density(x, y, z, params):

        '''
        Dehnen profile: rho = (M / (4π p q Rs^3)) * (r/Rs)^(-n) * (1 + r/Rs)^(n-4), where n = 2
        '''

        p, q, Rs = params['p_bar'], params['q_bar'], params['Rs_bar']
        M = 10.0 ** params['logM_bar']
        r = jnp.sqrt(x**2 + (y / p)**2 + (z / q)**2)

        val = M / (4 * jnp.pi * p * q * Rs**3) * (r / Rs)**(-2) * (1 + r / Rs)**(-2) * jnp.exp(-(z / 3)**4) * jnp.exp(-(r / 10)**4)

        return val

    def density_func(x, y, z, params):
        return DoubleExponentialDisk_density(x, y, z, params) + Dehnen_density(x, y, z, params)

    def density_func_agama(x):
        return np.array(density_func(x[:, 0], x[:, 1], x[:, 2], params_dict))

    params_dict = {
        'rho0_disc': 1e9,
        'Rd_disc': 3.0,
        'hz_disc': 0.3,
        'logM_bar': 10.,
        'Rs_bar': 5.0,
        'q_bar': 0.3,
        'p_bar':0.3,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
    }

    params_disk_rho = params_dict
    NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 50, 1e-3, 30.0, 1e-3, 15.0, 8.
    Nphi = 500
    N_int = 10_000
    time_start = time.time()
    dict_phi = get_phi_m(density_func, params_disk_rho, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)
    # dict_phi.block_until_ready()  # Ensure computation is complete before timing
    time_end = time.time()
    print(f"Time taken to compute phi_m: {time_end - time_start:.2f} seconds")

    x_c, y_c = np.linspace(-10, 10, 100), np.linspace(-10, 10, 100)
    X_c, Y_c = np.meshgrid(x_c, y_c)
    X_c, Y_c = X_c.flatten(), Y_c.flatten()
    Z_c = np.ones_like(X_c) * 1e-5

    pts = np.array([X_c, Y_c, Z_c]).T
    get_acc_vec = jax.jit(jax.vmap(get_acc, in_axes=(0,0,0,None)))
    # warm-up JIT compile before timing
    _ = get_acc_vec(pts[:,0], pts[:,1], pts[:,2], dict_phi)
    _.block_until_ready()
    time_start = time.time()
    acc_pts_us = get_acc_vec(pts[:,0], pts[:,1], pts[:,2], dict_phi)
    acc_pts_us.block_until_ready()
    time_end = time.time()
    print(f"Time taken to compute acceleration (us): {time_end - time_start:.2f} seconds") 
    R, phi, F_R, F_phi = getCylindricalFromCartesian_clockwise(pts[:,0], pts[:,1], acc_pts_us[:,0], acc_pts_us[:,1])
    acc_pts_us = acc_pts_us.at[:,0].set(F_R)
    acc_pts_us = acc_pts_us.at[:,1].set(F_phi)

    disc_density_agama = agama.Potential(
        type='CylSpline',
        density=density_func_agama,
        symmetry='Triaxial',
        mmax=int(Mmax),
        gridSizeR=int(NR),
        gridSizeZ=int(NZ + 1),  # includes z=0 node
        Rmin=float(Rmin),
        Rmax=float(Rmax),
        zmin=float(Zmin),
        zmax=float(Zmax),
        fixOrder=True,
    )
    time_start = time.time()
    acc_pts_agama = disc_density_agama.force(pts)
    time_end = time.time()
    print(f"Time taken to compute acceleration (agama): {time_end - time_start:.2f} seconds")
    R, phi, F_R, F_phi = getCylindricalFromCartesian_clockwise(pts[:,0], pts[:,1], acc_pts_agama[:,0], acc_pts_agama[:,1])
    acc_pts_agama[:,0] = F_R
    acc_pts_agama[:,1] = F_phi

    eps_rel = 1e-8
    denom = np.fabs(acc_pts_agama)
    rel = np.fabs(acc_pts_agama - acc_pts_us) / np.maximum(denom, eps_rel)

    comp_names = ['R', 'phi', 'z']
    for i, name in enumerate(comp_names):
        # Exclude points where the AGAMA reference value is extremely close to 0.
        mask = denom[:, i] > 1e-6
        comp_rel = rel[mask, i]
        print(f'{name} fractional diff: mean={comp_rel.mean():.4e}, median={np.median(comp_rel):.4e}, p95={np.percentile(comp_rel,95):.4e}')

    _plot = True
    if _plot:
        fig, ax = plt.subplots(3,3, figsize=(25,20), gridspec_kw={'hspace':0.5, 'wspace':0.3})
        cb = ax[0,0].scatter(X_c, Y_c, c=acc_pts_us[:,0], s=3, marker = 's', cmap='viridis')
        plt.colorbar(cb, ax=ax[0,0], label='R acceleration [kpc/Gyr^2]')
        ax[0,0].set_title('R acceleration (us)')
        ax[0,0].set_xlabel('X [kpc]')
        ax[0,0].set_ylabel('Y [kpc]')
        cb = ax[0,1].scatter(X_c, Y_c, c=acc_pts_agama[:,0], s=3, marker = 's', cmap='viridis')
        plt.colorbar(cb, ax=ax[0,1], label='R acceleration [kpc/Gyr^2]')
        ax[0,1].set_title('R acceleration (agama)')
        ax[0,1].set_xlabel('X [kpc]')
        ax[0,1].set_ylabel('Y [kpc]')
        res = (acc_pts_agama[:,0] - acc_pts_us[:,0]) / np.fabs(acc_pts_agama[:,0])
        cb = ax[0,2].scatter(X_c, Y_c, c=res, s=3, marker = 's', cmap='coolwarm', vmin = -0.1, vmax = 0.1)
        plt.colorbar(cb, ax=ax[0,2], label='R acceleration difference [kpc/Gyr^2]')
        ax[0,2].set_title('R acceleration (agama - us)')
        ax[0,2].set_xlabel('X [kpc]')
        ax[0,2].set_ylabel('Y [kpc]')

        cb = ax[1,0].scatter(X_c, Y_c, c=acc_pts_us[:,1], s=3, marker = 's', cmap='viridis')
        plt.colorbar(cb, ax=ax[1,0], label='phi acceleration [kpc/Gyr^2]')
        ax[1,0].set_title('phi acceleration (us)')
        ax[1,0].set_xlabel('X [kpc]')
        ax[1,0].set_ylabel('Y [kpc]')
        cb = ax[1,1].scatter(X_c, Y_c, c=acc_pts_agama[:,1], s=3, marker = 's', cmap='viridis')
        plt.colorbar(cb, ax=ax[1,1], label='phi acceleration [kpc/Gyr^2]')
        ax[1,1].set_title('phi acceleration (agama)')
        ax[1,1].set_xlabel('X [kpc]')
        ax[1,1].set_ylabel('Y [kpc]')
        res = (acc_pts_agama[:,1] - acc_pts_us[:,1]) / np.fabs(acc_pts_agama[:,1])
        cb = ax[1,2].scatter(X_c, Y_c, c=res, s=3, marker = 's', cmap='coolwarm', vmin = -0.1, vmax = 0.1)
        plt.colorbar(cb, ax=ax[1,2], label='phi acceleration difference [kpc/Gyr^2]')
        ax[1,2].set_title('phi acceleration (agama - us)')
        ax[1,2].set_xlabel('X [kpc]')
        ax[1,2].set_ylabel('Y [kpc]')

        cb = ax[2,0].scatter(X_c, Y_c, c=acc_pts_us[:,2], s=3, marker = 's', cmap='viridis')
        plt.colorbar(cb, ax=ax[2,0], label='Z acceleration [kpc/Gyr^2]')
        ax[2,0].set_title('Z acceleration (us)')
        ax[2,0].set_xlabel('X [kpc]')
        ax[2,0].set_ylabel('Y [kpc]')

        cb = ax[2,1].scatter(X_c, Y_c, c=acc_pts_agama[:,2], s=3, marker = 's', cmap='viridis')
        plt.colorbar(cb, ax=ax[2,1], label='Z acceleration [kpc/Gyr^2]')
        ax[2,1].set_title('Z acceleration (agama)')
        ax[2,1].set_xlabel('X [kpc]')
        ax[2,1].set_ylabel('Y [kpc]')

        res = (acc_pts_agama[:,2] - acc_pts_us[:,2]) / np.fabs(acc_pts_agama[:,2])
        cb = ax[2,2].scatter(X_c, Y_c, c=res, s=3, marker = 's', cmap='coolwarm', vmin = -0.4, vmax = 0.4)
        plt.colorbar(cb, ax=ax[2,2], label='Z acceleration difference [kpc/Gyr^2]')
        ax[2,2].set_title('Z acceleration (agama - us)')
        ax[2,2].set_xlabel('X [kpc]')
        ax[2,2].set_ylabel('Y [kpc]')

        plt.show()
