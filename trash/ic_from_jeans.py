import numpy as np
from scipy.integrate import quad
from CylindricalSpline import evaluate_phi_axisymmetric, get_phi_m
from potentials import NFW_potential
from densities import MiyamotoNagai_density

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from time import time

@jax.jit
def potential_func(x, y, z, dict_phi, params_halo):
    """ Returns Phi(R, z) """
    r = jnp.sqrt(x**2 + y**2 + z**2)
    
    phi_halo = NFW_potential(x, y, z, params_halo)
        # 1. NFW Halo
    # v_c = 200.0 # km/s
    # r_s = 20.0  # kpc
    # phi_halo = - (v_c**2) * jnp.log(1 + r/r_s) / (r/r_s + 1e-5)
    
    phi_disk = evaluate_phi_axisymmetric(x, y, z, dict_phi)
    
    return phi_halo + phi_disk

@jax.jit
def density_func(x, y, z, params):
    """ Returns Stellar Density nu(R, z) """
    # Double Exponential Disk
    val = MiyamotoNagai_density(x, y, z, params)
    return val


@jax.jit
def dPhi_dz(x, y, z, dict_phi, params_halo):
    # Numerical derivative of Potential w.r.t z
    d = 5e-3
    return (potential_func(x, y, z+d, dict_phi, params_halo) - potential_func(x, y, z-d, dict_phi, params_halo)) / (2*d)

@jax.jit
def dPhi_dR(x, y, z, dict_phi, params_halo ):
    # Numerical derivative of Potential w.r.t R
    d = 5e-3
    R = jnp.sqrt(x**2 + y**2)
    return (potential_func(R+d, 0, z, dict_phi, params_halo) - potential_func(R-d, 0, z, dict_phi, params_halo)) / (2*d)

# ==========================================
# 3. The Jeans Solver (The Core Logic)
# ==========================================

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


params_halo_pot = {
    'logM': 11.9,
    'Rs':10 ** 1.3,
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
    'logM': 9,
    'Rs': 10 ** 0.5,
    'Hs': 10 ** (-0.2),
    'x_origin': 0.0,
    'y_origin': 0.0,
    'z_origin': 0.0,
    'dirx': 0.0,
    'diry': 0.0,
    'dirz': 1.0
}

NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-2, 50.0, 1e-3, 20.0, 8.
Nphi = 200
N_int = 10_000
dict_phi = get_phi_m(MiyamotoNagai_density, params_disk_rho, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)

print(f"Potential is done")


# Example Particle Position
x_p, y_p, z_p = 8.5, 1.2, 0.2 # kpc
R_p = np.sqrt(x_p**2 + y_p**2)
phi_p = np.arctan2(y_p, x_p)

x_p = jnp.array(np.random.uniform(-10,10,10000))
y_p = jnp.array(np.random.uniform(-10,10,10000))
z_p = jnp.array(np.random.uniform(-3,3,10000))
R_p = jnp.sqrt(x_p**2 + y_p**2)
phi_p = jnp.arctan2(y_p, x_p)
# 1. Get Moments


get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
# jeans_moments = get_jeans_moments(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot, anisotropy_b=1.0)
jeans_moments = get_jeans_moments_vmap(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot,1.)
print(jeans_moments)



time_start = time()

get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
# jeans_moments = get_jeans_moments(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot, anisotropy_b=1.0)
jeans_moments = get_jeans_moments_vmap(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot,1.)
print(jeans_moments)
time_end = time()
print(f"Time taken for 10,000 particles: {time_end - time_start:.4f} seconds")
v_rot, sig_R, sig_z, sig_phi = jeans_moments

g1, g2, g3 = np.random.normal(0, 1, 10000), np.random.normal(0, 1, 10000), np.random.normal(0, 1, 10000)
vR = g1 * sig_R
vz = g2 * sig_z
vphi = v_rot + g3 * sig_phi
# print(f"Computed Moments at R={R_p:.2f}, z={z_p:.2f}")
# print(f"Mean V_phi: {v_rot:.2f} km/s")
# print(f"Dispersion: (R={sig_R:.2f}, z={sig_z:.2f}, phi={sig_phi:.2f})")

# # 2. Draw Random Velocity in Cylindrical Coordinates
# vr_sample = np.random.normal(0, sig_R)
# vz_sample = np.random.normal(0, sig_z)
# vphi_sample = np.random.normal(v_rot, sig_phi)

# # 3. Convert to Cartesian (vx, vy, vz)
# # vx = vR cos(phi) - vPhi sin(phi)
# # vy = vR sin(phi) + vPhi cos(phi)
# vx_sample = vr_sample * np.cos(phi_p) - vphi_sample * np.sin(phi_p)
# vy_sample = vr_sample * np.sin(phi_p) + vphi_sample * np.cos(phi_p)
# vz_sample = vz_sample # Same

# print(f"\nSampled Velocities: vx={vx_sample:.1f}, vy={vy_sample:.1f}, vz={vz_sample:.1f}")

def get_rot_curve(R_array, z=0.0, dict_phi=dict_phi, params_halo=params_halo_pot, params_disk=params_disk_rho):
    """ Computes the rotation curve v_c(R) at height z. """
    def v_c_func(R):
        x = R
        y = 0.0
        dPhi_dR_val = dPhi_dR(x, y, z, dict_phi, params_halo)
        v_c_sq = R * dPhi_dR_val
        v_c_sq = jnp.maximum(v_c_sq, 0.0)
        return jnp.sqrt(v_c_sq)
    
    v_c_func_jit = jax.jit(v_c_func)
    v_c_array = jax.vmap(v_c_func_jit)(R_array)
    return v_c_array

print(np.sum(v_rot<5))

import matplotlib.pyplot as plt
print(len(v_rot), len(R_p))
R_list = jnp.linspace(0.1, 15, 200)
v_c_list = get_rot_curve(R_list, z=0.0, dict_phi=dict_phi, params_halo=params_halo_pot, params_disk=params_disk_rho)
plt.plot(R_list, v_c_list)
plt.scatter(R_p, vphi, color='red', label='Sampled Particle', alpha=0.1)
plt.xlabel('R (kpc)')
plt.ylabel('v_c (km/s)')

plt.figure()
plt.hist(z_p[vphi<1], bins=30, alpha=0.7   )
plt.show()