import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnn

from constants import EPSILON
from main import model
from functools import partial

@jax.jit
def mapping_norm_to_scale_uniform(dirx, diry, min=0.5, max=1.5):
    """
    Computes the scale length/height from the direction vector components. Uniform [0.5, 1.5].
    """
    r = jnp.sqrt(dirx**2 + diry**2)
    # q = jnp.exp(-r**2/2) * (jnp.sqrt(jnp.pi) * jnp.exp(r**2/2) * jax.scipy.special.erf(r/jnp.sqrt(2)) - jnp.sqrt(2)*r)/jnp.sqrt(jnp.pi)
    q = 1 - jnp.exp(-r**2/2)# * jnp.sqrt(1/jnp.pi/2)
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

    params_disk_rho = {
        'logM': logM_disc,
        'Rs': 10 ** logRs_disk,
        'Hs': 10 ** logHs_disk,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }

    density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set, _ = model(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
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

    val1 = jnp.nansum( -0.5 * res_density ) / len(density_2DXY)
    val4 = jnp.nansum( -0.5 * res_h1 ) / len(h1_model)
    val5 = jnp.nansum( -0.5 * res_h2 ) / len(h2_model)
    val6 = jnp.nansum( -0.5 * res_h3 ) / len(h3_model)
    val7 = jnp.nansum( -0.5 * res_h4 ) / len(h4_model)

    log_likelihood = 0
    log_likelihood += val1 + val4 + val5 + val6 + val7

    return log_likelihood



# @jax.jit
@partial(jax.jit, static_argnames=('num_Vbin'))
def logl_angular_input(params, dict_data, num_Vbin):

    logM_halo = params[0]
    logM_disc = params[1]
    logRs_halo = params[2]
    logRs_disk = params[3]
    logHs_disk = params[4]
    alpha = params[5]
    beta = params[6]
    gamma = params[7]

    alpha = alpha * 180 / jnp.pi
    beta = beta * 180 / jnp.pi
    gamma = gamma * 180 / jnp.pi

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

    params_disk_rho = {
        'logM': logM_disc,
        'Rs': 10 ** logRs_disk,
        'Hs': 10 ** logHs_disk,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }

    density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set, _ = model(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
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

    val1 = jnp.nansum( -0.5 * res_density ) / len(density_2DXY)
    val4 = jnp.nansum( -0.5 * res_h1 ) / len(h1_model)
    val5 = jnp.nansum( -0.5 * res_h2 ) / len(h2_model)
    val6 = jnp.nansum( -0.5 * res_h3 ) / len(h3_model)
    val7 = jnp.nansum( -0.5 * res_h4 ) / len(h4_model)

    log_likelihood = 0
    log_likelihood += val1 + val4 + val5 + val6 + val7

    return log_likelihood