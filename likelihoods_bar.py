import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnn

from constants import EPSILON
from model_bar import model, projection, density_func
from functools import partial

from utils import *
from constants import KPCGYR_TO_KMS

@jax.jit
def mapping_norm_to_scale_uniform(dirx, diry, min=0.5, max=1.5):
    """
    Computes the scale length/height from the direction vector components. Uniform [0.5, 1.5].
    """
    r = jnp.sqrt(dirx**2 + diry**2)
    q = 1 - jnp.exp(-r**2/2)
    q = (max-min)*q + min
    return q

@jax.jit
def mapping_norm_to_scale_normal(dirx, diry, loc=0., norm=1.):
    """
    Computes the scale length/height from the direction vector components. Uniform [0.5, 1.5].
    """
    r = jnp.sqrt(dirx**2 + diry**2)
    q = 1 - jnp.exp(-r**2/2)
    q = jax.scipy.special.ndtri(q)
    q = norm * q + loc
    return q

@jax.jit
def nll_gaussian(z, A, y, sig, l2):
    x = jnn.softplus(z)  # strictly positive
    r = (A @ x - y) / sig
    return 0.5 * jnp.dot(r, r) + 0.5 * l2 * jnp.dot(x, x)
nll_gaussian = jax.value_and_grad(nll_gaussian)

def dynesty_logl(params, dict_data, num_Vbin):
    return logl(params, dict_data, num_Vbin)

# @jax.jit
@partial(jax.jit, static_argnames=('num_Vbin'))
def logl(params, dict_data, num_Vbin):
    """
    Simple axisymmetric MN-only likelihood (no bar).
    Same angular-encoding interface as the original likelihoods.py.
    """

    logM_halo = params[0]
    logM_disc = params[1]
    x_alpha = params[2]
    y_alpha = params[3]
    x_beta = params[4]
    y_beta = params[5]
    x_gamma = params[6]
    y_gamma = params[7]

    alpha = jnp.arctan2(y_alpha, x_alpha) * 180 / jnp.pi
    beta = jnp.arctan2(y_beta, x_beta) * 180 / jnp.pi
    gamma = jnp.arctan2(y_gamma, x_gamma) * 180 / jnp.pi

    logRs_halo = mapping_norm_to_scale_uniform(x_alpha, y_alpha, min=0.5, max=1.5)
    logRs_disk = mapping_norm_to_scale_uniform(x_beta, y_beta, min=0., max=1.0)
    logHs_disk = mapping_norm_to_scale_uniform(x_gamma, y_gamma, min=-1.0, max=0.)

    params_halo_pot = {
        'logM': logM_halo,
        'Rs':10 ** logRs_halo,
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

    # MN disc only — set bar mass to negligible
    Hs_disc = 10 ** logHs_disk
    params_disk_rho = {
        'logM_disc': logM_disc,
        'Rs_disc': 10 ** logRs_disk,
        'Hs_disc': Hs_disc,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'logM_bar': 0.0,  # negligible bar (1 Msun)
        'L_bar': 0.01,
        'a_bar': 0.002,
        'b_bar': Hs_disc,
        'light_to_mass_ratio': 1.0,
        'Omega_bar': 0.0,
    }

    density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set, _, _ = model(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
    density_2DXY, y_xy, sig_xy = density_set
    h1_model, y_h1, sig_A1 = h1_set
    h2_model, y_h2, sig_A2 = h2_set
    h3_model, y_h3, sig_A3 = h3_set
    h4_model, y_h4, sig_A4 = h4_set

    V_model = jnp.where(jnp.isnan(V_model), 0.0, V_model)
    sigma_model = jnp.where(jnp.isnan(sigma_model), 0.0, sigma_model)
    h3_model = jnp.where(jnp.isnan(h3_model), 0.0, h3_model)
    h4_model = jnp.where(jnp.isnan(h4_model), 0.0, h4_model)
    h1_data, h1_data_err = y_h1, sig_A1
    h2_data, h2_data_err = y_h2, sig_A2
    h3_data, h3_data_err = y_h3, sig_A3
    h4_data, h4_data_err = y_h4, sig_A4

    res_density = ((density_2DXY - y_xy) / (sig_xy + EPSILON))**2
    res_h1 = ((h1_model - h1_data) / (h1_data_err + 1e-3))**2
    res_h2 = ((h2_model - h2_data) / (h2_data_err + 1e-3))**2
    res_h3 = ((h3_model - h3_data) / (h3_data_err + 1e-3))**2
    res_h4 = ((h4_model - h4_data) / (h4_data_err + 1e-3))**2

    res_density = jnp.where(res_density<jnp.percentile(res_density, 98.0), res_density, 0)
    res_h1 = jnp.where(res_h1<jnp.percentile(res_h1, 98.0), res_h1, 0)
    res_h2 = jnp.where(res_h2<jnp.percentile(res_h2, 98.0), res_h2, 0)
    res_h3 = jnp.where(res_h3<jnp.percentile(res_h3, 98.0), res_h3, 0)
    res_h4 = jnp.where(res_h4<jnp.percentile(res_h4, 98.0), res_h4, 0)

    val1 = jnp.nansum( -0.5 * res_density )
    val4 = jnp.nansum( -0.5 * res_h1 )
    val5 = jnp.nansum( -0.5 * res_h2 )
    val6 = jnp.nansum( -0.5 * res_h3 )
    val7 = jnp.nansum( -0.5 * res_h4 )

    log_likelihood = 0
    log_likelihood += val1 + val4 + val5 + val6 + val7

    return log_likelihood



# @jax.jit
@partial(jax.jit, static_argnames=('num_Vbin'))
def logl_angular_input(params, dict_data, num_Vbin):
    """
    12-parameter likelihood with MN disc + T3 bar + V4 bulge.

    Parameter mapping:
      0: logM_halo
      1: logM_disc      (MN total mass)
      2: logM_bar        (T3 mass = V4 mass)
      3: logRs_halo
      4: logRs_disk      (MN scale length a)
      5: logHs_disk      (MN scale height b = T3 b_bar)
      6: logL_bar        (T3 half-length L)
      7: alpha            (viewing angle, radians)
      8: beta
      9: gamma
     10: log_light_to_mass
     11: log_Omega_bar
    """

    logM_halo = params[0]
    logM_disc = params[1]
    logM_bar = params[2]
    logRs_halo = params[3]
    logRs_disk = params[4]
    logHs_disk = params[5]
    logL_bar = params[6]
    alpha = params[7]
    beta = params[8]
    gamma = params[9]
    log_light_to_mass_ratio = params[10]
    log_Omega_bar = params[11]

    sigma_density_model = 0. #10**params[12]
    sigma_kine_model = 10**params[12]

    alpha = alpha * 180 / jnp.pi
    beta = beta * 180 / jnp.pi
    gamma = gamma * 180 / jnp.pi

    # Derived bar parameters
    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc  # T3 b_bar = disc scale height

    params_halo_pot = {
        'logM': logM_halo,
        'Rs':10 ** logRs_halo,
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

    params_baryon_rho = {
        'logM_disc': logM_disc,
        'Rs_disc': 10 ** logRs_disk,
        'Hs_disc': Hs_disc,
        'logM_bar': logM_bar,
        'L_bar': L_bar,
        'a_bar': a_bar,
        'b_bar': b_bar,
        'light_to_mass_ratio': 10 ** log_light_to_mass_ratio,
        'Omega_bar': 10 ** log_Omega_bar,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,

        'sigma_density_model': sigma_density_model,
        'sigma_kine_model': sigma_kine_model,
    }

    surface_density_model = projection(params_baryon_rho, dict_data, num_Vbin)
    surface_density_gt = dict_data['XY_density_data'] / params_baryon_rho['light_to_mass_ratio']

    chi2 = jnp.sum((surface_density_gt - surface_density_model)**2 / (0.1 * surface_density_gt)**2)
    logl_density = -0.5 * chi2 / num_Vbin

    logl_density_max = dict_data['logl_density_max']

    def true_func():
        return -jnp.inf
    def false_func():
        density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set, _weights, _logl_marg = model(params_halo_pot, params_baryon_rho, dict_data, num_Vbin)

        def _true_func():
            return -jnp.inf
        def _false_func():
            return _logl_marg

        nan_in_weights = jnp.isnan(_weights).any()
        logl = jax.lax.cond(nan_in_weights, _true_func, _false_func)
        return logl


    val = jax.lax.cond(logl_density < logl_density_max - 1000, true_func, false_func)
    return val


@partial(jax.jit, static_argnames=('num_Vbin'))
def logl_density(params, dict_data, num_Vbin, light_to_mass_ratio=1.0):
    """
    8-parameter density-only likelihood with MN disc + T3 bar + V4 bulge.

    Parameter mapping:
      0: logM_disc      (MN total mass)
      1: logM_bar        (T3 mass = V4 mass)
      2: logRd_disc      (MN scale length a)
      3: loghz_disc      (MN scale height b = T3 b_bar)
      4: logL_bar        (T3 half-length L)
      5: alpha            (radians)
      6: beta
      7: gamma
    """

    logM_disc = params[0]
    logM_bar = params[1]
    logRd_disc = params[2]
    loghz_disc = params[3]
    logL_bar = params[4]
    alpha = params[5] * 180 / jnp.pi
    beta = params[6] * 180 / jnp.pi
    gamma = params[7] * 180 / jnp.pi

    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** loghz_disc
    b_bar = Hs_disc

    density_param = {
        'logM_disc': logM_disc,
        'Rs_disc': 10.0 ** logRd_disc,
        'Hs_disc': Hs_disc,
        'logM_bar': logM_bar,
        'L_bar': L_bar,
        'a_bar': a_bar,
        'b_bar': b_bar,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'light_to_mass_ratio': light_to_mass_ratio,
    }

    surface_density_model = projection(density_param, dict_data, num_Vbin)
    surface_density_gt = dict_data['XY_density_data'] / density_param['light_to_mass_ratio']

    chi2 = jnp.sum((surface_density_gt - surface_density_model)**2 / (0.1 * surface_density_gt)**2)
    logl = -0.5 * chi2 / num_Vbin
    return logl
