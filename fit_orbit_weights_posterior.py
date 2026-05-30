"""
Two-stage pipeline to sample the posterior over Schwarzschild orbital weights
at a fixed (best-fit) gravitational potential.

Stage `build`
    Load mock data + best-fit potential params (median of an existing MCMC
    checkpoint), replay the orbit-integration block of `model_diagnostic`
    (same setup: Jeans ICs, adaptive leapfrog at -Omega_bar, on-the-fly
    Voronoi + Rzphi binning) to obtain the orbital-library matrices
    A_Rzphi, A_xy, A_h1..A_h4 and the matched data vectors / sigmas, then
    pickle them.

Stage `sample`
    Load the pickle and run NumPyro NUTS in log_w space using the *NNLS
    objective* as the negative log-posterior (block weights sqrt(5)/sqrt(5)/1,
    Rzphi block included, L2 regulariser lambda/n_orb * ||w||^2 with
    lambda=1, with optional change-of-variables Jacobian so that the
    regulariser is a true HalfNormal prior on w).

Run:
    python fit_orbit_weights_posterior.py build
    python fit_orbit_weights_posterior.py sample
"""

import argparse
import pickle
import os

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from constants import EPSILON
from likelihoods_bar import get_dict_data_bootstrap
from model_bar import (
    density_func, potential_func, get_jeans_moments,
    solve_nnls_admm,                    # the un-block-weighted solver, used by plot_data_vs_model.py
    compute_model_and_logl_bootstrap,   # for the logL benchmark printout
)
from utils import (
    logMenc_logc_to_logM_logRs, makeRotationMatrix,
    getCartesianFromCylindrical_clockwise,
    estimate_orbital_timescale, get_rotation_curve,
)
from integrants_with_binning import _integrate_adaptive_batch_chunked_vmap


# =====================================================================
# CONFIG  -- edit here
# =====================================================================
PATH         = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
DATA_FOLDER  = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
DATA_FILE    = 'mock_data/mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma140_D50_gal2.pkl'
# Aligned with plot_data_vs_model.py for benchmark: checkpoint 0415, no Omega override.
CHECKPOINT   = DATA_FOLDER + '/ensemble_checkpoint_0415_beta25_gamma140_D50_gal2.pkl'
MATRICES_OUT = DATA_FOLDER + '/orbit_matrices_benchmark_0415_beta25_gamma140_D50_gal2.pkl'
SAMPLES_OUT  = DATA_FOLDER + '/orbit_weight_samples_benchmark_0415_beta25_gamma140_D50_gal2.pkl'

# Posterior cleanup (matches get_best_orbital_library.py)
DISCARD = 600
THIN    = 1
LOGP_KEEP_WINDOW = 100   # keep chains with last-step logprob > max - this

# Mock-data loader settings (match plot_data_vs_model.py for benchmark)
N_BOOTSTRAP = 10
N_SAMPLES   = 10_000     # number of particles drawn for w0  -> n_orb

# NNLS / sampling
LAMBDA_REG       = 1.0
INCLUDE_JACOBIAN = False  # add sum(log_w) so the L2 reg is a true HalfNormal(w) prior
NUM_WARMUP       = 1000
NUM_SAMPLES      = 1000
NUM_CHAINS       = 4
TARGET_ACCEPT    = 0.8
NUTS_MAX_TREE    = 10
SEED             = 0
INIT_PERTURB_SCALE = 0.1  # std of Gaussian noise added per chain to log_w_init
CHAIN_METHOD       = 'vectorized'  # 'vectorized' (fast on GPU), 'parallel', or 'sequential'

# Extra HMC fields to capture for chain-health diagnostics.
EXTRA_FIELDS = ('potential_energy', 'accept_prob', 'num_steps', 'diverging')

# Laptop-debug overrides (applied when --debug is passed)
DEBUG_N_SAMPLES    = 200
DEBUG_NUM_WARMUP   = 20
DEBUG_NUM_SAMPLES  = 20
DEBUG_NUM_CHAINS   = 4
DEBUG_NUTS_MAX_TREE = 5


# =====================================================================
# Helpers
# =====================================================================
def load_best_fit(checkpoint_file, discard=DISCARD, thin=THIN,
                  logp_window=LOGP_KEEP_WINDOW):
    """Load an emcee/blackjax checkpoint and return the per-parameter median."""
    with open(checkpoint_file, 'rb') as f:
        ckpt = pickle.load(f)
    posterior = np.stack(ckpt['all_samples'], axis=0)
    logprob   = np.stack(ckpt['all_logprob'], axis=0)
    print(f"checkpoint: step={ckpt['step']}  chains={posterior.shape[1]}  "
          f"params={posterior.shape[2]}")
    keep = logprob[-1, :] > np.amax(logprob[-1, :]) - logp_window
    posterior = posterior[:, keep, :]
    posterior = posterior[discard::thin, :, :].reshape(-1, posterior.shape[-1])
    return np.percentile(posterior, 50, axis=0)


def build_param_dicts(best_fit_param):
    """Convert the 13-parameter vector into params_halo_pot / params_disk_rho."""
    (logMhalo_10, logMdisk, logMbar, logC_halo,
     logRs_disk, logHs_disk, logRs_bar,
     alpha, beta, gamma,
     logLM, logOmega, logSigma_amp) = best_fit_param

    logMhalo, logRh_halo = logMenc_logc_to_logM_logRs(
        logMhalo_10, logC_halo, r_enc=10.0, Delta=200., rho_crit=277.54)

    params_halo_pot = {
        'logM': float(logMhalo), 'Rs': 10. ** float(logRh_halo),
        'a': 1.0, 'b': 1.0, 'c': 1.0,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    }
    # makeRotationMatrix() in utils.py applies jnp.radians() internally,
    # so alpha/beta/gamma must be passed in DEGREES (matches plot_data_vs_model.py
    # lines 186-188 which convert from radians to degrees here).
    deg = 180.0 / np.pi
    params_disk_rho = {
        'logM_disc': float(logMdisk),
        'Rs_disc':   10. ** float(logRs_disk),
        'Hs_disc':   10. ** float(logHs_disk),
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
        'alpha': float(alpha) * deg,
        'beta':  float(beta)  * deg,
        'gamma': float(gamma) * deg,
        'logM_bar': float(logMbar),
        'L_bar':  10. ** float(logRs_bar),
        'a_bar':  10. ** float(logRs_bar) / 5.,
        'b_bar':  10. ** float(logHs_disk),
        'light_to_mass_ratio': 10. ** float(logLM),
        'Omega_bar':           10. ** float(logOmega),
        'sigma_amplifier':     10. ** float(logSigma_amp),
        'sigma_density_model': 0.,
        'sigma_kine_model':    0.,
    }
    return params_halo_pot, params_disk_rho


# =====================================================================
# Stage 1 -- build matrices (mirrors model_bar.model_diagnostic up to
# the point where A_*, y_*, sig_* are ready, but does not store
# trajectories and does not call compute_model_and_logl_bootstrap).
# =====================================================================
def build_matrices(params_halo_pot, params_disk_rho, dict_data, num_Vbin,
                   Rzphi_n_tot, Rzphi_n_grid, Rzphi_lim_grid,
                   xy_lim_grid, xy_n_grid, n_realizations=4):
    # n_realizations >=1 ; the integrator collapses jittered realisations into one
    # column per particle, so n_orb == n_particles regardless. Match
    # model_for_plotting / model_bootstrap defaults (4) for matrix smoothness.
    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    v0 = dict_data['v0']; s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']
    Omega_bar = params_disk_rho['Omega_bar']
    rotation_matrix = makeRotationMatrix(
        params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma'])

    params_baryon = {
        'logM_disc': params_disk_rho['logM_disc'],
        'Rs_disc':   params_disk_rho['Rs_disc'],
        'Hs_disc':   params_disk_rho['Hs_disc'],
        'x_origin':  params_disk_rho['x_origin'],
        'y_origin':  params_disk_rho['y_origin'],
        'z_origin':  params_disk_rho['z_origin'],
        'dirx':      params_disk_rho['dirx'],
        'diry':      params_disk_rho['diry'],
        'dirz':      params_disk_rho['dirz'],
        'logM_bar':  params_disk_rho['logM_bar'],
        'L_bar':     params_disk_rho['L_bar'],
        'a_bar':     params_disk_rho['a_bar'],
        'b_bar':     params_disk_rho['b_bar'],
    }

    # ---- Jeans-based initial velocities ----
    get_jeans_vmap = jax.vmap(get_jeans_moments, in_axes=(0, 0, 0, None, None, None, None))
    v_rot, sig_R, sig_z, sig_phi = get_jeans_vmap(
        w0[:, 0], w0[:, 1], w0[:, 2],
        params_baryon, params_disk_rho, params_halo_pot, 1.,
    )
    k1, k2, k3 = (jax.random.PRNGKey(42), jax.random.PRNGKey(109),
                  jax.random.PRNGKey(2026))
    g1 = jax.random.normal(k1, (n_particles,))
    g2 = jax.random.normal(k2, (n_particles,))
    g3 = jax.random.normal(k3, (n_particles,))
    vR  = g1 * sig_R
    vz  = g2 * sig_z
    vph = v_rot + g3 * sig_phi
    R0  = jnp.sqrt(w0[:, 0]**2 + w0[:, 1]**2)
    ph0 = jnp.arctan2(w0[:, 1], w0[:, 0])
    x, y, vx, vy = getCartesianFromCylindrical_clockwise(R0, ph0, vR, vph)
    w0_new = jnp.stack([x, y, w0[:, 2], vx, vy, vz], axis=-1)

    # ---- Orbital timescale ----
    _R = jnp.sqrt(w0_new[:, 0]**2 + w0_new[:, 1]**2)
    _z = w0_new[:, 2]
    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z)

    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = jnp.clip(0.1 * _Vc, a_min=1, a_max=15)
    nx = (jax.random.uniform(keys[0], (n_particles, n_realizations)) - 0.5) * d_scale[:, None]
    ny = (jax.random.uniform(keys[1], (n_particles, n_realizations)) - 0.5) * d_scale[:, None]
    nz = (jax.random.uniform(keys[2], (n_particles, n_realizations)) - 0.5) * d_scale[:, None]
    nvx = (jax.random.uniform(keys[3], (n_particles, n_realizations)) - 0.5) * v_scale[:, None]
    nvy = (jax.random.uniform(keys[4], (n_particles, n_realizations)) - 0.5) * v_scale[:, None]
    nvz = (jax.random.uniform(keys[5], (n_particles, n_realizations)) - 0.5) * v_scale[:, None]
    w0_new_batch = w0_new[:, None, :] + jnp.stack([nx, ny, nz, nvx, nvy, nvz], axis=-1)

    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z)
    T_orb_batch = T_orb[:, None].repeat(n_realizations, axis=1)

    # ---- Orbit integration ----
    @jax.jit
    def acc_fn(x, y, z):
        def _pot(p): return potential_func(p[0], p[1], p[2], params_baryon, params_halo_pot)
        return -jax.grad(_pot)(jnp.array([x, y, z]))

    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    N_step_per_orb = 100
    N_dynamical_time = 50
    N_max = N_step_per_orb * N_dynamical_time
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / N_step_per_orb
    atol, rtol = 1e-7, 1e-4
    dt_min, dt_max = 1e-5, 0.3

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, _ = \
        _integrate_adaptive_batch_chunked_vmap(
            w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
            dt_init_batch, -Omega_bar,
            atol, rtol, dt_min, dt_max,
            num_Vbin, bin_mapping, num_per_bin,
            Rzphi_lim_grid, xy_lim_grid,
            Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
            v0, s, rotation_matrix,
            100,
        )
    A_Rzphi = Rzphi_bin_counts.T
    A_xy    = surface_density.T
    A_h1    = h1.T
    A_h2    = h2.T
    A_h3    = h3.T
    A_h4    = h4.T

    # ---- Build the smooth-density target y_Rzphi ----
    @jax.jit
    def density_func_Rz(R, z, phi, params):
        return density_func(R * jnp.cos(phi), R * jnp.sin(phi), z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_s   = R_grid   + (sample[:, 0] - 0.5) * dR
        z_s   = z_grid   + (sample[:, 1] - 0.5) * dz
        phi_s = phi_grid + (sample[:, 2] - 0.5) * dphi
        rho_s = rho_fct(R_s, z_s, phi_s, dict_params)
        return jnp.sum(rho_s * R_s) / sample.shape[0] * dR * dz * dphi

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
        R_grid, z_grid, phi_grid, density_func_Rz,
        params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration'])

    # ---- Data vectors and sigmas (post light-to-mass + mass-per-orb scaling) ----
    LM = params_disk_rho['light_to_mass_ratio']
    y_xy = dict_data['XY_density_data'].astype(jnp.float32) / LM
    y_h1 = dict_data['h1_data']; y_h2 = dict_data['h2_data']
    y_h3 = dict_data['h3_data']; y_h4 = dict_data['h4_data']

    sig_Rzphi = 0.02 * y_Rzphi + 1e-10
    sig_xy = (dict_data['XY_density_data_err'] + EPSILON) / LM
    sig_A1 = dict_data['h1_data_err'] + EPSILON
    sig_A2 = dict_data['h2_data_err'] + EPSILON
    sig_A3 = dict_data['h3_data_err'] + EPSILON
    sig_A4 = dict_data['h4_data_err'] + EPSILON

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]
    y_xy      = y_xy      / mean_mass_per_orb
    sig_xy    = sig_xy    / mean_mass_per_orb
    y_Rzphi   = y_Rzphi   / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    return dict(
        A_Rzphi=A_Rzphi, A_xy=A_xy, A_h1=A_h1, A_h2=A_h2, A_h3=A_h3, A_h4=A_h4,
        y_Rzphi=y_Rzphi, y_xy=y_xy,
        y_h1=y_h1, y_h2=y_h2, y_h3=y_h3, y_h4=y_h4,
        sig_Rzphi=sig_Rzphi, sig_xy=sig_xy,
        sig_A1=sig_A1, sig_A2=sig_A2, sig_A3=sig_A3, sig_A4=sig_A4,
        mean_mass_per_orb=mean_mass_per_orb,
        Omega_bar=float(Omega_bar),
        rotation_matrix=np.asarray(rotation_matrix),
    )


def stage_build():
    print("[build] loading mock data ...")
    dict_data = get_dict_data_bootstrap(
        PATH, DATA_FILE, N_BOOTSTRAP=N_BOOTSTRAP, n_samples=N_SAMPLES)

    print("[build] loading best-fit parameters ...")
    best_fit_param = load_best_fit(CHECKPOINT)
    param_names = ['logM_10kpc', 'logM_disk', 'logM_bar', 'logC_halo',
                   'logRs_disk', 'logHs_disk', 'logL_bar',
                   'alpha', 'beta', 'gamma',
                   'log_LM', 'log_Omega_bar', 'log_sigma_amp']
    for n, v in zip(param_names, best_fit_param):
        print(f"  {n:>15s} = {v:+.4f}")

    params_halo_pot, params_disk_rho = build_param_dicts(best_fit_param)
    print(f"[build] Omega_bar = {params_disk_rho['Omega_bar']:.4f}")

    Rmin, Rmax = dict_data['R_minmax']
    zmin, zmax = dict_data['z_minmax']
    phimin, phimax = dict_data['phi_minmax']
    X_minmax, Y_minmax = dict_data['X_minmax'], dict_data['Y_minmax']
    nX, nY = dict_data['nX_nY']

    print("[build] running orbit integration ...")
    out = build_matrices(
        params_halo_pot, params_disk_rho, dict_data, dict_data['total_bins'],
        Rzphi_n_tot=int(dict_data['Rzphi_n_tot']),
        Rzphi_n_grid=jnp.array(dict_data['Rzphi_n_grid']),
        Rzphi_lim_grid=jnp.array([[Rmin, Rmax], [zmin, zmax], [phimin, phimax]]),
        xy_lim_grid=jnp.array([X_minmax, Y_minmax]),
        xy_n_grid=jnp.array([nX, nY]),
        n_realizations=4,        # MUST match model_bootstrap / model_for_plotting
    )
    print(f"[build] n_orb = {out['A_xy'].shape[1]}, "
          f"n_xy = {out['A_xy'].shape[0]}, "
          f"n_Rzphi = {out['A_Rzphi'].shape[0]}")

    # Run NNLS once on the unperturbed data to provide an init point for HMC.
    # Use the same solver model_for_plotting calls (no block weighting).
    print("[build] running NNLS (solve_nnls_admm) for HMC init ...")
    weights_init = solve_nnls_admm(
        out['A_Rzphi'], out['A_xy'],
        out['A_h1'], out['A_h2'], out['A_h3'], out['A_h4'],
        out['y_Rzphi'], out['y_xy'],
        out['y_h1'], out['y_h2'], out['y_h3'], out['y_h4'],
        out['sig_Rzphi'], out['sig_xy'],
        out['sig_A1'], out['sig_A2'], out['sig_A3'], out['sig_A4'],
        lambda_reg=LAMBDA_REG, maxiter=500,
    )
    print(f"[build] NNLS init: min={float(weights_init.min()):.3e}  "
          f"median={float(jnp.median(weights_init)):.3e}  "
          f"max={float(weights_init.max()):.3e}  "
          f"frac>0={float((weights_init > 0).mean()):.3f}")

    # ---- Benchmark printout: numbers that should match plot_data_vs_model.py ----
    print("[benchmark] computing logL and per-block chi2 ...")
    sigma_amp     = 10. ** float(best_fit_param[12])
    LM            = 10. ** float(best_fit_param[10])
    mass_per_orb  = float(out['mean_mass_per_orb'])
    v0_arr        = jnp.asarray(dict_data['v0'])
    s_arr         = jnp.asarray(dict_data['s'])

    w_b = jnp.asarray(weights_init)[None, :]
    (_, _, _, _, _, _, _, _, logl_all, _) = compute_model_and_logl_bootstrap(
        w_b,
        out['A_Rzphi'], out['A_xy'],
        out['A_h1'], out['A_h2'], out['A_h3'], out['A_h4'],
        out['y_Rzphi'],
        jnp.asarray(out['y_xy'])[None, :],
        jnp.asarray(out['y_h1'])[None, :], jnp.asarray(out['y_h2'])[None, :],
        jnp.asarray(out['y_h3'])[None, :], jnp.asarray(out['y_h4'])[None, :],
        out['sig_Rzphi'], out['sig_xy'],
        out['sig_A1'], out['sig_A2'], out['sig_A3'], out['sig_A4'],
        v0_arr, s_arr, sigma_amp, LM, mass_per_orb,
    )
    print(f"[benchmark] logL  (compute_model_and_logl_bootstrap, unperturbed)"
          f"  = {float(logl_all[0]):.4f}")
    print("            ^ compare to plot_data_vs_model.py print 'logL: <value>'")

    # Block chi2 in the form plot_data_vs_model.py reports (no sigma_amplifier inflation)
    eps_safe = 1e-8
    y_xy_arr = jnp.asarray(out['y_xy'])
    y_xy_safe = jnp.where(jnp.abs(y_xy_arr) > eps_safe, y_xy_arr, 1.0)
    A_xy_w   = out['A_xy']    @ weights_init
    A_h1_xy  = out['A_h1']  * out['A_xy']
    A_h2_xy  = out['A_h2']  * out['A_xy']
    A_h3_xy  = out['A_h3']  * out['A_xy']
    A_h4_xy  = out['A_h4']  * out['A_xy']
    h1_m = jnp.clip((A_h1_xy @ weights_init) / y_xy_safe, -10., 10.)
    h2_m = jnp.clip((A_h2_xy @ weights_init) / y_xy_safe, -10., 10.)
    h3_m = jnp.clip((A_h3_xy @ weights_init) / y_xy_safe, -10., 10.)
    h4_m = jnp.clip((A_h4_xy @ weights_init) / y_xy_safe, -10., 10.)
    chi2_density = float(jnp.sum((A_xy_w - y_xy_arr)**2 / out['sig_xy']**2))
    chi2_h1 = float(jnp.sum((h1_m - out['y_h1'])**2 / out['sig_A1']**2))
    chi2_h2 = float(jnp.sum((h2_m - out['y_h2'])**2 / out['sig_A2']**2))
    chi2_h3 = float(jnp.sum((h3_m - out['y_h3'])**2 / out['sig_A3']**2))
    chi2_h4 = float(jnp.sum((h4_m - out['y_h4'])**2 / out['sig_A4']**2))
    print(f"[benchmark] Chi2_density = {chi2_density:.2f}")
    print(f"[benchmark] Chi2_h1      = {chi2_h1:.2f}")
    print(f"[benchmark] Chi2_h2      = {chi2_h2:.2f}")
    print(f"[benchmark] Chi2_h3      = {chi2_h3:.2f}")
    print(f"[benchmark] Chi2_h4      = {chi2_h4:.2f}")
    print("            ^ compare to plot_data_vs_model.py Chi2_density / Chi2_h1..h4")

    payload = {k: np.asarray(v) for k, v in out.items()
               if k not in ('Omega_bar',)}
    payload['Omega_bar'] = out['Omega_bar']
    payload['weights_nnls_init'] = np.asarray(weights_init)
    payload['best_fit_param'] = best_fit_param
    payload['lambda_reg'] = LAMBDA_REG

    os.makedirs(os.path.dirname(MATRICES_OUT), exist_ok=True)
    with open(MATRICES_OUT, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[build] saved -> {MATRICES_OUT}")


# =====================================================================
# Stage 2 -- sample log_w with NumPyro NUTS.
#
# The negative log-posterior matches the NNLS objective:
#
#   -2 logp(log_w) = 5 * ||(A_Rzphi w - y_Rzphi)/sig_Rzphi||^2
#                  + 5 * ||(A_xy    w - y_xy)   /sig_xy||^2
#                  + sum_k ||(A_hk * A_xy w / y_xy_safe - y_hk)/sig_Ak||^2
#                  + (lambda/n_orb) * ||w||^2
#                  - 2 * sum(log_w)        [optional Jacobian]
#
# with w = exp(log_w). The Jacobian term makes the L2 regulariser a
# proper HalfNormal(sigma^2 = n_orb/lambda) prior on w; without it the
# implicit prior becomes a 1/w-tilted half-Gaussian and the MAP
# coincides exactly with the NNLS solution.
# =====================================================================
def stage_sample():
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(NUM_CHAINS)

    print(f"[sample] loading matrices <- {MATRICES_OUT}")
    with open(MATRICES_OUT, 'rb') as f:
        d = pickle.load(f)

    A_Rzphi = jnp.asarray(d['A_Rzphi']); A_xy = jnp.asarray(d['A_xy'])
    A_h1 = jnp.asarray(d['A_h1']); A_h2 = jnp.asarray(d['A_h2'])
    A_h3 = jnp.asarray(d['A_h3']); A_h4 = jnp.asarray(d['A_h4'])
    y_Rzphi = jnp.asarray(d['y_Rzphi']); y_xy = jnp.asarray(d['y_xy'])
    y_h1 = jnp.asarray(d['y_h1']); y_h2 = jnp.asarray(d['y_h2'])
    y_h3 = jnp.asarray(d['y_h3']); y_h4 = jnp.asarray(d['y_h4'])
    sig_Rzphi = jnp.asarray(d['sig_Rzphi']); sig_xy = jnp.asarray(d['sig_xy'])
    sig_A1 = jnp.asarray(d['sig_A1']); sig_A2 = jnp.asarray(d['sig_A2'])
    sig_A3 = jnp.asarray(d['sig_A3']); sig_A4 = jnp.asarray(d['sig_A4'])

    n_orb = int(A_xy.shape[1])
    lambda_reg = float(d.get('lambda_reg', LAMBDA_REG))
    print(f"[sample] n_orb={n_orb}  n_xy={A_xy.shape[0]}  "
          f"n_Rzphi={A_Rzphi.shape[0]}  lambda={lambda_reg}  "
          f"jacobian={INCLUDE_JACOBIAN}")

    eps = 1e-8
    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)
    A_h1_xy = A_h1 * A_xy
    A_h2_xy = A_h2 * A_xy
    A_h3_xy = A_h3 * A_xy
    A_h4_xy = A_h4 * A_xy

    # Block weights match solve_nnls_admm in model_bar.py (no block weighting).
    w_rzphi_sq = 1.0
    w_xy_sq    = 1.0
    w_h_sq     = 1.0
    reg_coef   = lambda_reg / n_orb

    def numpyro_model():
        log_w = numpyro.sample(
            "log_w",
            dist.ImproperUniform(constraints.real, (), event_shape=(n_orb,)),
        )
        w = jnp.exp(log_w)

        r_rz = (A_Rzphi @ w - y_Rzphi) / sig_Rzphi
        r_xy = (A_xy    @ w - y_xy)    / sig_xy
        r_h1 = ((A_h1_xy @ w) / y_xy_safe - y_h1) / sig_A1
        r_h2 = ((A_h2_xy @ w) / y_xy_safe - y_h2) / sig_A2
        r_h3 = ((A_h3_xy @ w) / y_xy_safe - y_h3) / sig_A3
        r_h4 = ((A_h4_xy @ w) / y_xy_safe - y_h4) / sig_A4

        chi2 = (w_rzphi_sq * jnp.sum(r_rz ** 2)
                + w_xy_sq * jnp.sum(r_xy ** 2)
                + w_h_sq * (jnp.sum(r_h1 ** 2) + jnp.sum(r_h2 ** 2)
                            + jnp.sum(r_h3 ** 2) + jnp.sum(r_h4 ** 2)))
        reg = reg_coef * jnp.sum(w * w)

        log_post = -0.5 * (chi2 + reg)
        if INCLUDE_JACOBIAN:
            log_post = log_post + jnp.sum(log_w)
        numpyro.factor("log_post", log_post)

    # Init from NNLS solution (clipped away from zero so log is finite),
    # perturbed independently for each chain to enable R-hat / multimodality
    # diagnostics.
    w_init = jnp.asarray(d['weights_nnls_init'])
    w_init = jnp.clip(w_init, a_min=1e-30)
    log_w_init_single = jnp.log(w_init)
    init_key = jax.random.PRNGKey(SEED + 1)
    perturb = jax.random.normal(init_key, (NUM_CHAINS, n_orb)) * INIT_PERTURB_SCALE
    log_w_init = log_w_init_single[None, :] + perturb  # (NUM_CHAINS, n_orb)

    nuts = NUTS(numpyro_model, target_accept_prob=TARGET_ACCEPT,
                max_tree_depth=NUTS_MAX_TREE)
    mcmc = MCMC(nuts, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method=CHAIN_METHOD,
                progress_bar=True)
    rng_key = jax.random.PRNGKey(SEED)
    mcmc.run(rng_key, init_params={"log_w": log_w_init},
             extra_fields=EXTRA_FIELDS)

    samples = mcmc.get_samples(group_by_chain=True)
    extras  = mcmc.get_extra_fields(group_by_chain=True)

    out = {
        'log_w_samples': np.asarray(samples['log_w']),     # (chains, n_samples, n_orb)
        **{k: np.asarray(v) for k, v in extras.items()},   # potential_energy / accept_prob / num_steps / diverging
        'weights_nnls_init': np.asarray(w_init),
        'config': dict(
            lambda_reg=lambda_reg, include_jacobian=INCLUDE_JACOBIAN,
            num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
            num_chains=NUM_CHAINS, target_accept=TARGET_ACCEPT,
            max_tree_depth=NUTS_MAX_TREE, seed=SEED,
            extra_fields=list(EXTRA_FIELDS),
        ),
    }
    with open(SAMPLES_OUT, 'wb') as f:
        pickle.dump(out, f)
    print(f"[sample] saved -> {SAMPLES_OUT}")
    mcmc.print_summary(prob=0.9, exclude_deterministic=True)


# =====================================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['build', 'sample'])
    p.add_argument('--debug', action='store_true',
                   help='use small particle counts and short MCMC for laptop debugging')
    args = p.parse_args()
    if args.debug:
        N_SAMPLES     = DEBUG_N_SAMPLES
        NUM_WARMUP    = DEBUG_NUM_WARMUP
        NUM_SAMPLES   = DEBUG_NUM_SAMPLES
        NUM_CHAINS    = DEBUG_NUM_CHAINS
        NUTS_MAX_TREE = DEBUG_NUTS_MAX_TREE
        MATRICES_OUT  = MATRICES_OUT.replace('.pkl', '_debug.pkl')
        SAMPLES_OUT   = SAMPLES_OUT.replace('.pkl', '_debug.pkl')
        print(f"[debug] N_SAMPLES={N_SAMPLES}  NUM_WARMUP={NUM_WARMUP}  "
              f"NUM_SAMPLES={NUM_SAMPLES}  NUM_CHAINS={NUM_CHAINS}  "
              f"NUTS_MAX_TREE={NUTS_MAX_TREE}")
    if args.stage == 'build':
        stage_build()
    else:
        stage_sample()
