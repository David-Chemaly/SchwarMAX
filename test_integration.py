import agama
agama.setUnits(mass=1, length=1, velocity=1)

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


@jax.jit
def Ferrers_density(x, y, z, params):

    '''
    params include: 'logM_bar', 'Rs_bar', 'q_bar', 'p_bar'
    '''
    p, q, Rs = params['p_bar'], params['q_bar'], params['Rs_bar']
    M = 10.0 ** params['logM_bar']
    r = jnp.sqrt(x**2 + (y / p)**2 + (z / q)**2)
    rho = 105 * M / (32 * jnp.pi * p * q * Rs**3) * (1 - (r / Rs)**2)**2
    rho = jnp.where(r <= Rs, rho, 0.0)
    return rho

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
    return Dehnen_density(x, y, z, params) + DoubleExponentialDisk_density(x, y, z, params)

def density_func_agama(x):
    return np.array(density_func(x[:, 0], x[:, 1], x[:, 2], params_dict))

if __name__ == "__main__":
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

    samples = np.array([
        np.random.normal(0, 5, 10000),
        np.random.normal(0, 5, 10000),
        np.random.normal(0, 2, 10000)
    ]).T


    params_halo_pot = {
        'logM': 11.8,
        'Rs': 16.0,
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

    w0 = jnp.array(samples)
    n_particles = w0.shape[0]
    params_disk_rho = params_dict
    NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-3, 30.0, 1e-3, 15.0, 8.
    Nphi = 300
    N_int = 10_000
    dict_phi = get_phi_m(density_func, params_disk_rho, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)

    #=========================================== GET INITIAL VELOCITY ===================================================
    @jax.jit
    def dPhi_dz(x, y, z, dict_phi, params_halo):
        # Numerical derivative of Potential w.r.t z
        d = 5e-3
        return (potential_func(x, y, z+d, dict_phi, params_halo) - potential_func(x, y, z-d, dict_phi, params_halo)) / (2*d)

    @jax.jit
    def dPhi_dR(x, y, z, dict_phi, params_halo):
        # Numerical derivative of Potential w.r.t R
        d = 5e-3
        R = jnp.sqrt(x**2 + y**2)
        return (potential_func(R+d, 0, z, dict_phi, params_halo) - potential_func(R-d, 0, z, dict_phi, params_halo)) / (2*d)

    @jax.jit
    def get_jeans_moments(x_star, y_star, z_star, dict_phi, params_disc, params_halo, anisotropy_b=1.0):
        """
        Computes (v_mean, sigma_R, sigma_z, sigma_phi) for a star at (R, z).

        Parameters:
        -----------
        x_star, y_star, z_star : float
            Coordinates of the particle.
        density_fn : function
            Function returning stellar density nu(R, z).
        anisotropy_b : float
            sigma_R^2 / sigma_z^2. Default 1.0 (isotropic).
        """
        R_star = jnp.sqrt(x_star**2 + y_star**2)

        # --- Step 1: Compute Sigma_z (Vertical Integration) ---

        # Integrand: nu(R, z) * dPhi/dz
        def integrand(z_prime):
            return density_func(x_star, y_star, z_prime, params_disc) * dPhi_dz(x_star, y_star, z_prime, dict_phi, params_halo)

        # Integrate from z_star to infinity (e.g., 20 kpc)
        # Note: We assume symmetry, so we integrate |z| to infinity
        pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
        dx = pts[1] - pts[0]
        # integrand_val = integrand(pts)
        integrand_val = jax.vmap(integrand, in_axes = (0))(pts)
        integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)

        nu_val = density_func(x_star, y_star, z_star, params_disc)
        # if nu_val <= 0: return 0, 0, 0, 0

        sigma_z2 = (1.0 / nu_val) * integral_val

        sigma_z2 = jnp.maximum(sigma_z2, 0.0)

        sigma_z = jnp.sqrt(sigma_z2)

        # --- Step 2: Compute Sigma_R (Anisotropy assumption) ---
        sigma_R2 = anisotropy_b * sigma_z2
        sigma_R = jnp.sqrt(sigma_R2)

        # --- Step 3: Compute v_phi_total^2 (Radial Equation) ---
        # We need d(nu * sigma_R^2) / dR
        # Since sigma_R^2 = b * sigma_z^2, we need d(b * Integral) / dR

        # Define a helper to calculate the "Vertical Pressure" P_zz = nu * sigma_z^2 at any R
        def vertical_pressure(r_in):
            # We need to re-integrate at radius r_in to get the pressure there
            def integrand_r(z_prime):
                return density_func(r_in, 0, z_prime, params_disc) * dPhi_dz(r_in, 0, z_prime, dict_phi, params_halo)

            pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
            dx = pts[1] - pts[0]
            # integrand_val = integrand(pts)
            integrand_val = jax.vmap(integrand_r, in_axes = (0))(pts)
            integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)
            return integral_val # This is (nu * sigma_z^2)

        # Calculate derivative w.r.t R using central difference
        dR = 5e-3
        Pzz_plus = vertical_pressure(R_star + dR)
        Pzz_minus = vertical_pressure(R_star - dR)
        d_nu_sigR2_dR = anisotropy_b * (Pzz_plus - Pzz_minus) / (2*dR)

        # Radial Jeans Equation:
        # v_phi_sq = sigma_R^2 + (R/nu) * d(nu*sigR^2)/dR + R * dPhi/dR
        term1 = sigma_R2
        term2 = (R_star / nu_val) * d_nu_sigR2_dR
        term3 = R_star * dPhi_dR(x_star, y_star, z_star, dict_phi, params_halo)

        v_phi_total_sq = term1 + term2 + term3

        # --- Step 4: Separate Rotation vs Dispersion ---
        # Assumption: sigma_phi approx sigma_R (Round velocity ellipsoid in plane)
        sigma_phi = sigma_R

        v_streaming_sq = v_phi_total_sq - sigma_phi**2

        v_streaming_sq = jnp.maximum(v_streaming_sq, 0.0)
        v_mean_phi = jnp.sqrt(v_streaming_sq)

        output = jax.lax.cond(nu_val<=0, lambda: (0.0, 0.0, 0.0, 0.0), lambda: (v_mean_phi, sigma_R, sigma_z, sigma_phi))

        return output


    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    # jeans_moments = get_jeans_moments(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot, anisotropy_b=1.0)
    def get_w0_new(w0, key1, key2, key3, n_particles):
        jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], dict_phi, params_disk_rho, params_halo_pot, 1.)
        v_rot, sig_R, sig_z, sig_phi = jeans_moments
        g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
        vR = g1 * sig_R # 2 sigma dispersion
        vz = g2 * sig_z
        vphi = v_rot + g3 * sig_phi
        x, y, vx, vy = getCartesianFromCylindrical_clockwise(jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
        return jnp.array([x, y, w0[:,2], vx, vy, vz]).T
    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    w0_new = get_w0_new(w0, key1, key2, key3, n_particles)


    def _split(w):
        return w[:3], w[3:]

    def _merge(r, v):
        return jnp.concatenate([r, v], axis=0)

    @partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll',))
    def integrate_leapfrog_barred_traj(w0, acc_fn, n_steps, dt = 0.010, t0 = 0.0, Omega = 0.0, unroll=True):
        """Leapfrog (KDK) — returns final time and final state only.

        num_segments_Rzphi: int
            Number of segments for Rzphi bin counting. MUST equal to nRzphi.prod()
        num_segments_XY: int
            Number of segments for XY bin counting. MUST equal to nXY.prod()
        
        v0 and s are arrays of reference velocity and dispersion for each XY cell for the GH coefficent calculation. Length should equal to nXY.prod()
        """

        def step(carry, _):
            t, y = carry
            r, v = _split(y)

            # Gravity half-kick
            a0 = acc_fn(*r)
            v_half = v + 0.5 * dt * a0

            # Exact Omega-subflow for:
            #   xdot = vx + Omega*y, ydot = vy - Omega*x
            #   vxdot = Omega*vy,     vydot = -Omega*vx
            theta = Omega * dt
            c, s_theta = jnp.cos(theta), jnp.sin(theta)

            x_bar = r[0] + dt * v_half[0]
            y_bar = r[1] + dt * v_half[1]
            x_new = c * x_bar + s_theta * y_bar
            y_new = -s_theta * x_bar + c * y_bar
            z_new = r[2] + dt * v_half[2]

            vx_rot = c * v_half[0] + s_theta * v_half[1]
            vy_rot = -s_theta * v_half[0] + c * v_half[1]
            vz_rot = v_half[2]

            r_new = jnp.array([x_new, y_new, z_new])
            v_rot = jnp.array([vx_rot, vy_rot, vz_rot])
            t_new = t + dt

            # Gravity half-kick at updated position
            a1 = acc_fn(*r_new)
            v_new = v_rot + 0.5 * dt * a1
            y_new = _merge(r_new, v_new)
            return (t_new, y_new), (t_new, y_new)

        (_, _), (tN, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

        return tN, wN 

    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (dict_phi, params_halo_pot),
        _z
    )

    N_step_per_orb = 100
    N_dynamical_time = 20
    dt = T_orb / N_step_per_orb
    time_integrate = T_orb * N_dynamical_time
    N_steps = N_step_per_orb * N_dynamical_time
    #=========================================== Integrate orbits =======================================================
    @jax.jit
    def acc_fn(x, y, z):
        a_halo = NFW_acceleration(x, y, z,  params_halo_pot)
        a_disk = get_acc(x, y, z, dict_phi)
        return a_halo + a_disk

    time_start = time.time()
    _integrate_vmap = jax.vmap(integrate_leapfrog_barred_traj, 
                            in_axes=(0, None, None, 0, None, None, None))
    tN, wN = _integrate_vmap(w0_new, acc_fn, N_steps, dt, 0.0, -34.0, False)
    tN.block_until_ready()
    time_end = time.time()
    print(f"Time taken to integrate orbits with reconstructed potential: {time_end - time_start:.2f} seconds")
    x_lf, y_lf, z_lf = wN[:,:,0].flatten(), wN[:,:,1].flatten(), wN[:,:,2].flatten()
    vx_lf, vy_lf, vz_lf = wN[:,:,3].flatten(), wN[:,:,4].flatten(), wN[:,:,5].flatten()
    print("Leapfrog integration done.")


    @jax.jit
    def acc_fn(x, y, z):
        a_halo = NFW_acceleration(x, y, z,  params_halo_pot)
        a_halo = NFW_acceleration(x, y, z,  params_halo_pot)
        a_disk = 0#get_acc(x, y, z, dict_phi)
        return a_halo + a_disk

    time_start = time.time()
    _integrate_vmap = jax.vmap(integrate_leapfrog_barred_traj, 
                            in_axes=(0, None, None, 0, None, None, None))
    tN, wN = _integrate_vmap(w0_new, acc_fn, N_steps, dt, 0.0, -34.0, False)
    tN.block_until_ready()
    time_end = time.time()
    print(f"Time taken to integrate orbits with NFW only: {time_end - time_start:.2f} seconds")
    x_lf, y_lf, z_lf = wN[:,:,0].flatten(), wN[:,:,1].flatten(), wN[:,:,2].flatten()
    vx_lf, vy_lf, vz_lf = wN[:,:,3].flatten(), wN[:,:,4].flatten(), wN[:,:,5].flatten()
    print("Leapfrog integration done.")