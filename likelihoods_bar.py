import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnn

from constants import EPSILON
from model_bar import model, model_marg, model_bootstrap, model_bootstrap_psf, projection, density_func, model_test_convergence
from functools import partial

from utils import *
from integrants_with_binning import assign_regular_grid
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
def logl_angular_input_marg(params, dict_data, num_Vbin):
    """
    Same as logl_angular_input but uses model_marg() which returns
    logL marginalized over the orbital weight uncertainty ellipse:
        logL_marg = logL_best - (1/2) log det(Q)
    where Q = U^T U + reg*I is the Hessian of the NNLS objective.
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

    sigma_density_model = 0.
    sigma_kine_model = 10**params[12]

    alpha = alpha * 180 / jnp.pi
    beta = beta * 180 / jnp.pi
    gamma = gamma * 180 / jnp.pi

    # Derived bar parameters
    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc

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
        density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set, _weights, _logl_marg = model_marg(params_halo_pot, params_baryon_rho, dict_data, num_Vbin)

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
def logl_angular_input_bootstrap(params, dict_data, num_Vbin):
    """
    Same as logl_angular_input but uses model_bootstrap() which marginalizes
    logL over observational noise via bootstrap resampling:
        - 100 perturbed observation vectors y_i = y + N(0, sig)
        - Shared Cholesky, vmapped ADMM solves
        - logL_marg = log(mean(exp(logL_i)))
    """
    logM_enc = params[0]
    log_c = params[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54)

    # logM_halo = params[0]
    logM_disc = params[1]
    logM_bar = params[2]
    # logRs_halo = params[3]
    logRs_disk = params[4]
    logHs_disk = params[5]
    logL_bar = params[6]
    alpha = params[7]
    beta = params[8]
    gamma = params[9]
    log_light_to_mass_ratio = params[10]
    log_Omega_bar = params[11]

    sigma_density_model = 0#10**params[12]
    sigma_kine_model = 0. 
    sigma_amplifier = 10**params[12]

    alpha = alpha * 180 / jnp.pi
    beta = beta * 180 / jnp.pi
    gamma = gamma * 180 / jnp.pi

    # Derived bar parameters
    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc

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
        'sigma_amplifier': sigma_amplifier,
    }

    surface_density_model = projection(params_baryon_rho, dict_data, num_Vbin)
    surface_density_gt = dict_data['XY_density_data'] / params_baryon_rho['light_to_mass_ratio']

    chi2 = jnp.sum((surface_density_gt - surface_density_model)**2 / (0.1 * surface_density_gt)**2)
    logl_density = -0.5 * chi2 / num_Vbin

    logl_density_max = dict_data['logl_density_max']

    X_minmax = dict_data['X_minmax']
    Y_minmax = dict_data['Y_minmax']
    nX, nY = dict_data['nX_nY']
    xy_lim_grid = jnp.array([X_minmax, Y_minmax])
    xy_n_grid = jnp.array([nX, nY])

    Rmin, Rmax = dict_data['R_minmax']
    zmin, zmax = dict_data['z_minmax']
    phimin, phimax = dict_data['phi_minmax']

    def true_func():
        return -jnp.inf
    def false_func():
        weights_all, _logl_marg, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all, logl_all, _m_eff = \
            model_bootstrap(params_halo_pot, params_baryon_rho, dict_data, num_Vbin,
                            Rzphi_lim_grid=jnp.array([[Rmin, Rmax],[zmin, zmax],[phimin, phimax]]),
                            xy_lim_grid=xy_lim_grid,
                            xy_n_grid=xy_n_grid)

        def _true_func():
            return -jnp.inf
        def _false_func():
            return _logl_marg# - _m_eff

        nan_in_weights = jnp.isnan(weights_all).any()
        logl = jax.lax.cond(nan_in_weights, _true_func, _false_func)
        return logl

    val = jax.lax.cond(logl_density < logl_density_max - 1000, true_func, false_func)
    return val


def logl_angular_input_bootstrap_psf(params, dict_data, num_Vbin):
    """
    Same as logl_angular_input_bootstrap but applies PSF smearing via model_bootstrap_psf.
    Expects dict_data['P_psf'] to contain the (num_Vbin, num_Vbin) PSF transfer matrix.
    """
    logM_enc = params[0]
    log_c = params[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54)

    logM_disc = params[1]
    logM_bar = params[2]
    logRs_disk = params[4]
    logHs_disk = params[5]
    logL_bar = params[6]
    alpha = params[7]
    beta = params[8]
    gamma = params[9]
    log_light_to_mass_ratio = params[10]
    log_Omega_bar = params[11]

    sigma_density_model = 0
    sigma_kine_model = 0.
    sigma_amplifier = 10**params[12]

    alpha = alpha * 180 / jnp.pi
    beta = beta * 180 / jnp.pi
    gamma = gamma * 180 / jnp.pi

    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc

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
        'sigma_amplifier': sigma_amplifier,
    }

    surface_density_model = projection(params_baryon_rho, dict_data, num_Vbin)
    surface_density_gt = dict_data['XY_density_data'] / params_baryon_rho['light_to_mass_ratio']

    chi2 = jnp.sum((surface_density_gt - surface_density_model)**2 / (0.1 * surface_density_gt)**2)
    logl_density = -0.5 * chi2 / num_Vbin

    logl_density_max = dict_data['logl_density_max']

    def true_func():
        return -jnp.inf
    def false_func():
        weights_all, _logl_marg, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all, logl_all, _m_eff = \
            model_bootstrap_psf(params_halo_pot, params_baryon_rho, dict_data, num_Vbin)

        def _true_func():
            return -jnp.inf
        def _false_func():
            return _logl_marg

        nan_in_weights = jnp.isnan(weights_all).any()
        logl = jax.lax.cond(nan_in_weights, _true_func, _false_func)
        return logl

    val = jax.lax.cond(logl_density < logl_density_max - 1000, true_func, false_func)
    return val


@partial(jax.jit, static_argnames=('num_Vbin'))
def logl_angular_input_bootstrap_with_maps(params, dict_data, num_Vbin):
    """
    Same as logl_angular_input but uses model_bootstrap() which marginalizes
    logL over observational noise via bootstrap resampling:
        - 100 perturbed observation vectors y_i = y + N(0, sig)
        - Shared Cholesky, vmapped ADMM solves
        - logL_marg = log(mean(exp(logL_i)))
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

    sigma_density_model = 0#10**params[12]
    sigma_kine_model = 0. 

    alpha = alpha * 180 / jnp.pi
    beta = beta * 180 / jnp.pi
    gamma = gamma * 180 / jnp.pi

    # Derived bar parameters
    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc

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

    weights_all, _logl_marg, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all, logl_all, _m_eff = \
        model_bootstrap(params_halo_pot, params_baryon_rho, dict_data, num_Vbin)
    
    return weights_all, _logl_marg, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all, logl_all, _m_eff, logl_density



@partial(jax.jit, static_argnames=('num_Vbin', 'N_max_integration'))
def logl_angular_input_bootstrap_test(params, dict_data, num_Vbin, N_max_integration):
    """
    Same as logl_angular_input but uses model_bootstrap() which marginalizes
    logL over observational noise via bootstrap resampling:
        - 100 perturbed observation vectors y_i = y + N(0, sig)
        - Shared Cholesky, vmapped ADMM solves
        - logL_marg = log(mean(exp(logL_i)))
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

    sigma_density_model = 0#10**params[12]
    sigma_kine_model = 0. 

    alpha = alpha * 180 / jnp.pi
    beta = beta * 180 / jnp.pi
    gamma = gamma * 180 / jnp.pi

    # Derived bar parameters
    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc

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
        weights_all, _logl_marg, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all, logl_all, _m_eff = \
            model_test_convergence(params_halo_pot, params_baryon_rho, dict_data, num_Vbin, N_max_integration)

        def _true_func():
            return -jnp.inf
        def _false_func():
            return _logl_marg - _m_eff

        nan_in_weights = jnp.isnan(weights_all).any()
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


import pickle
import numpy as np


def get_dict_data(path):

    with open(path + 'mock_Nbody_bar_XY_withRot.pkl', 'rb') as f:
        bin_dict = pickle.load(f)

    # voronoi binning mapping and data
    num_per_bin = jnp.array(bin_dict['num_per_bin'])
    total_bins = jnp.array(bin_dict['total_bins'])
    bin_mapping = jnp.array(bin_dict['bin_mapping'])
    surface_density = jnp.array(bin_dict['surface_density'])
    V_data = jnp.array(bin_dict['V_mean'])
    sigma_data = jnp.array(bin_dict['V_sigma'])
    h1_data = jnp.array(bin_dict['h1'])
    h2_data = jnp.array(bin_dict['h2'])
    h3_data = jnp.array(bin_dict['h3'])
    h4_data = jnp.array(bin_dict['h4'])
    v0 = jnp.array(bin_dict['v0'])
    s = jnp.array(bin_dict['s'])
    alpha, beta, gamma = bin_dict['orientation']

    # V_data_err = jnp.where(0.1 * jnp.fabs(V_data) < 10, 10, 0.1 * V_data)
    # sigma_data_err = jnp.where(0.1 * jnp.fabs(sigma_data) < 5, 5, 0.1 * sigma_data)
    # h1_data_err = jnp.where(0.1 * jnp.fabs(h1_data) < 0.03, 0.03, 0.1 * jnp.fabs(h1_data))
    # h2_data_err = jnp.where(0.1 * jnp.fabs(h2_data) < 0.03, 0.03, 0.1 * jnp.fabs(h2_data))
    # h3_data_err = jnp.where(0.1 * jnp.fabs(h3_data) < 0.03, 0.03, 0.1 * jnp.fabs(h3_data))
    # h4_data_err = jnp.where(0.1 * jnp.fabs(h4_data) < 0.03, 0.03, 0.1 * jnp.fabs(h4_data))
    V_data_err = jnp.array(bin_dict['V_mean_err'])
    sigma_data_err = jnp.array(bin_dict['V_sigma_err'])
    h1_data_err = jnp.array(bin_dict['h1_err'])
    h2_data_err = jnp.array(bin_dict['h2_err'])
    h3_data_err = jnp.array(bin_dict['h3_err'])
    h4_data_err = jnp.array(bin_dict['h4_err'])

    # df_Rzphi_data = pd.read_csv(path + 'mock_axisymmetric_disc_Rzphi.csv')
    # Rzphi_density_data = jnp.array(df_Rzphi_data['mass'].to_numpy()).astype(jnp.float32)
    with open(path + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
        Rzphi_density_data = pickle.load(f)

    R_grid, z_grid, phi_grid = Rzphi_density_data['R_grid'], Rzphi_density_data['z_grid'], Rzphi_density_data['phi_grid']
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    sample_for_integration = Rzphi_density_data['sample_for_integration']

    from scipy.stats import qmc
    X_regular_grid, Y_regular_grid = bin_dict['X_regular_grid'], bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)

    n_samples = 5_000  # Same number as original data
    x_grid = np.linspace(0, 12, 1000)
    logP_xexp = XexpX_pdf_log(x_grid, 4.0)
    key = jax.random.PRNGKey(10086)
    R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.default_rng(54321).uniform(0, 2*np.pi, size=n_samples)

    x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)

    x_grid = np.linspace(0, 4, 1000)
    logP_exp = expX_pdf_log(x_grid, 1.5)
    key = jax.random.PRNGKey(10010)
    z_samples = sample_from_logP(x_grid, logP_exp, n_samples, key)
    w0 = np.array([
        x_samples,
        y_samples,
        z_samples,
    ]).T


    dict_data = {
        # 'w0': w0,
        'v0': v0,
        's': s,

        # 'Rzphi_density_data': Rzphi_density_data,
        'XY_density_data': surface_density,
        'V_data': V_data,
        'V_data_err': V_data_err,
        'sigma_data': sigma_data,
        'sigma_data_err': sigma_data_err,
        'h1_data': h1_data,
        'h1_data_err': h1_data_err,
        'h2_data': h2_data,
        'h2_data_err': h2_data_err,
        'h3_data': h3_data,
        'h3_data_err': h3_data_err,
        'h4_data': h4_data,
        'h4_data_err': h4_data_err,
        'num_per_bin': num_per_bin,
        'bin_mapping': bin_mapping,
        'total_bins': total_bins.item(),

        'R_grid': R_grid,
        'z_grid': z_grid,
        'phi_grid': phi_grid,
        'dR': dR,
        'dz': dz,
        'dphi': dphi,
        'sample_for_integration': sample_for_integration,

        'X_regular_grid': X_regular_grid,
        'Y_regular_grid': Y_regular_grid,
        'dX': dX,
        'dY': dY,
        'sample_for_integration_XY': sample,

        'w0': w0
    }

    return dict_data

def get_dict_data_bootstrap(path, filename, N_BOOTSTRAP = 100, n_samples = 5_000):

    with open(path + filename, 'rb') as f:
        bin_dict = pickle.load(f)

    X_minmax = jnp.array(bin_dict['X_minmax'])
    Y_minmax = jnp.array(bin_dict['Y_minmax'])
    nX_nY = jnp.array(bin_dict['nX_nY'])

    # voronoi binning mapping and data
    num_per_bin = jnp.array(bin_dict['num_per_bin'])
    total_bins = jnp.array(bin_dict['total_bins'])
    bin_mapping = jnp.array(bin_dict['bin_mapping'])
    surface_density = jnp.array(bin_dict['surface_density'])
    V_data = jnp.array(bin_dict['V_mean'])
    sigma_data = jnp.array(bin_dict['V_sigma'])
    h1_data = jnp.array(bin_dict['h1'])
    h2_data = jnp.array(bin_dict['h2'])
    h3_data = jnp.array(bin_dict['h3'])
    h4_data = jnp.array(bin_dict['h4'])
    v0 = jnp.array(bin_dict['v0'])
    s = jnp.array(bin_dict['s'])
    alpha, beta, gamma = bin_dict['orientation']


    XY_density_data_err = 0.01 * surface_density + EPSILON
    V_data_err = jnp.array(bin_dict['V_mean_err'])
    sigma_data_err = jnp.array(bin_dict['V_sigma_err'])
    h1_data_err = jnp.array(bin_dict['h1_err'])
    h2_data_err = jnp.array(bin_dict['h2_err'])
    h3_data_err = jnp.array(bin_dict['h3_err'])
    h4_data_err = jnp.array(bin_dict['h4_err'])

    '''
    Bootstrap the observation
    '''
    # rng = np.random.default_rng(42)
    # N_BOOTSTRAP = 100
    # y_xy_boot = np.array(surface_density[None, :] + rng.normal(size=(N_BOOTSTRAP, len(surface_density))) * XY_density_data_err[None, :])
    # y_h1_boot = np.array(h1_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h1_data))) * h1_data_err[None, :])
    # y_h2_boot = np.array(h2_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h2_data))) * h2_data_err[None, :])
    # y_h3_boot = np.array(h3_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h3_data))) * h3_data_err[None, :])
    # y_h4_boot = np.array(h4_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(h4_data))) * h4_data_err[None, :])
    # V_boot = np.array(V_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(V_data))) * V_data_err[None, :])
    # sigma_boot = np.array(sigma_data[None, :] + rng.normal(size=(N_BOOTSTRAP, len(sigma_data))) * sigma_data_err[None, :])
    # # Fix the first one to always be the unperturbed system
    # y_xy_boot[0, :] = surface_density
    # y_h1_boot[0, :] = h1_data
    # y_h2_boot[0, :] = h2_data
    # y_h3_boot[0, :] = h3_data
    # y_h4_boot[0, :] = h4_data
    # V_boot[0, :] = V_data
    # sigma_boot[0, :] = sigma_data
    # y_xy_boot = jnp.array(y_xy_boot)
    # y_h1_boot = jnp.array(y_h1_boot)
    # y_h2_boot = jnp.array(y_h2_boot)
    # y_h3_boot = jnp.array(y_h3_boot)
    # y_h4_boot = jnp.array(y_h4_boot)
    # V_boot = jnp.array(V_boot)
    # sigma_boot = jnp.array(sigma_boot)

    rng = np.random.default_rng(42)
    XY_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(surface_density)))
    h1_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h1_data)))
    h2_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h2_data)))
    h3_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h3_data)))
    h4_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h4_data)))
    V_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(V_data)))
    sigma_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(sigma_data)))
    XY_standard_normal[0, :] = 0.0
    h1_standard_normal[0, :] = 0.0
    h2_standard_normal[0, :] = 0.0
    h3_standard_normal[0, :] = 0.0
    h4_standard_normal[0, :] = 0.0
    V_standard_normal[0, :] = 0.0
    sigma_standard_normal[0, :] = 0.0
    XY_standard_normal = jnp.array(XY_standard_normal)
    h1_standard_normal = jnp.array(h1_standard_normal)
    h2_standard_normal = jnp.array(h2_standard_normal)
    h3_standard_normal = jnp.array(h3_standard_normal)
    h4_standard_normal = jnp.array(h4_standard_normal)
    V_standard_normal = jnp.array(V_standard_normal)
    sigma_standard_normal = jnp.array(sigma_standard_normal)

    from scipy.stats import qmc

    # with open(path + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
    #     Rzphi_density_data = pickle.load(f)
    # R_grid, z_grid, phi_grid = Rzphi_density_data['R_grid'], Rzphi_density_data['z_grid'], Rzphi_density_data['phi_grid']
    # dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    # dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    # dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    # sample_for_integration = Rzphi_density_data['sample_for_integration']
    R_min, R_max =0., 10.
    z_min, z_max = -3., 3.
    n_R, n_z, n_phi =10, 6, 6
    R_edge = jnp.linspace(R_min, R_max, n_R+1)
    z_edge = jnp.linspace(z_min, z_max, n_z+1)
    phi_edge = jnp.linspace(-jnp.pi, jnp.pi, n_phi+1)
    R_mids, z_mids, phi_mids = 0.5 * (R_edge[:-1] + R_edge[1:]), 0.5 * (z_edge[:-1] + z_edge[1:]), 0.5 * (phi_edge[:-1] + phi_edge[1:])
    dR, dz, dphi = R_edge[1]-R_edge[0], z_edge[1]-z_edge[0], phi_edge[1]-phi_edge[0]
    R_mids_mesh, z_mids_mesh, phi_mids_mesh = jnp.meshgrid(R_mids, z_mids, phi_mids, indexing='ij')
    Rzphi_mid_grid = jnp.stack([R_mids_mesh.ravel(), z_mids_mesh.ravel(), phi_mids_mesh.ravel()], axis=-1)  # (n_R*n_z*n_phi, 3)
    R_grid = Rzphi_mid_grid[:,0]
    z_grid = Rzphi_mid_grid[:,1]
    phi_grid = Rzphi_mid_grid[:,2]
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    Rzphi_minmax=jnp.array([[R_min, R_max],[z_min, z_max],[-jnp.pi, jnp.pi]])
    nRzphi=jnp.array([n_R,n_z,n_phi])
    num_segments_Rzphi=nRzphi.prod()
    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_grid_indices = assign_regular_grid(Rzphi_mid_grid,
                                        grid_min=Rzphi_minmax[:,0],
                                        grid_max=Rzphi_minmax[:,1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)
    _, COUNTS = jnp.unique(Rzphi_grid_indices, return_counts=True)
    argsort = jnp.argsort(Rzphi_grid_indices)
    R_grid = R_grid[argsort]
    z_grid = z_grid[argsort]
    phi_grid = phi_grid[argsort]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample_for_integration = sampler.random_base2(m=10)

    
    X_regular_grid, Y_regular_grid = bin_dict['X_regular_grid'], bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)

    n_samples = n_samples  # Same number as original data
    x_grid = np.linspace(0., 12., 1000)
    logP_xexp = XexpX_pdf_log(x_grid, 4.0)
    key = jax.random.PRNGKey(10086)
    R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.default_rng(42).uniform(0, 2*np.pi, size=n_samples)

    x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)

    x_grid = np.linspace(0, 4, 1000)
    logP_exp = expX_pdf_log(x_grid, 1.5)
    key = jax.random.PRNGKey(10010)
    z_samples = sample_from_logP(x_grid, logP_exp, n_samples, key)
    w0 = np.array([
        x_samples,
        y_samples,
        z_samples,
    ]).T


    dict_data = {
        # 'w0': w0,
        'v0': v0,
        's': s,

        # 'Rzphi_density_data': Rzphi_density_data,
        'XY_density_data': surface_density,
        'XY_density_data_err': XY_density_data_err,
        'V_data': V_data,
        'V_data_err': V_data_err,
        'sigma_data': sigma_data,
        'sigma_data_err': sigma_data_err,
        'h1_data': h1_data,
        'h1_data_err': h1_data_err,
        'h2_data': h2_data,
        'h2_data_err': h2_data_err,
        'h3_data': h3_data,
        'h3_data_err': h3_data_err,
        'h4_data': h4_data,
        'h4_data_err': h4_data_err,
        'num_per_bin': num_per_bin,
        'bin_mapping': bin_mapping,
        'total_bins': total_bins.item(),

        # 'y_xy_boot': y_xy_boot,
        # 'y_h1_boot': y_h1_boot,
        # 'y_h2_boot': y_h2_boot,
        # 'y_h3_boot': y_h3_boot,
        # 'y_h4_boot': y_h4_boot,
        # 'V_boot': V_boot,
        # 'sigma_boot': sigma_boot,
        'XY_standard_normal': XY_standard_normal,
        'h1_standard_normal': h1_standard_normal,
        'h2_standard_normal': h2_standard_normal,
        'h3_standard_normal': h3_standard_normal,
        'h4_standard_normal': h4_standard_normal,
        'V_standard_normal': V_standard_normal,
        'sigma_standard_normal': sigma_standard_normal,

        'R_grid': R_grid,
        'z_grid': z_grid,
        'phi_grid': phi_grid,
        'R_minmax': [R_min, R_max],
        'z_minmax': [z_min, z_max],
        'phi_minmax': [-jnp.pi, jnp.pi],
        'dR': dR,
        'dz': dz,
        'dphi': dphi,
        'sample_for_integration': sample_for_integration,

        'X_regular_grid': X_regular_grid,
        'Y_regular_grid': Y_regular_grid,
        'dX': dX,
        'dY': dY,
        'sample_for_integration_XY': sample,

        'X_minmax': X_minmax,
        'Y_minmax': Y_minmax,
        'nX_nY': nX_nY,

        'w0': w0
    }

    return dict_data