"""BoxCDQP test on orbital_library_bar_4.pkl using paper-style quadratic objective.

Solves
    min_w 0.5 ||U w - y||^2 + 0.5 * lambda_reg * ||w||^2
    s.t.  w >= 0

with jaxopt.BoxCDQP (box-constrained QP), and reports the same log-likelihood
metric used in test_weights_optimisation.py.
"""

from __future__ import annotations

import os
import time
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import BoxCDQP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import qmc

from constants import EPSILON
from ghMoments import h_to_V_sigma


def get_dict_data(path_repo):
    with open(path_repo + 'mock_Nbody_bar_XY_withRot.pkl', 'rb') as f:
        bin_dict = pickle.load(f)

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
    V_data_err = jnp.array(bin_dict['V_mean_err'])
    sigma_data_err = jnp.array(bin_dict['V_sigma_err'])
    h1_data_err = jnp.array(bin_dict['h1_err'])
    h2_data_err = jnp.array(bin_dict['h2_err'])
    h3_data_err = jnp.array(bin_dict['h3_err'])
    h4_data_err = jnp.array(bin_dict['h4_err'])

    with open(path_repo + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
        Rzphi_density_data = pickle.load(f)

    R_grid = Rzphi_density_data['R_grid']
    z_grid = Rzphi_density_data['z_grid']
    phi_grid = Rzphi_density_data['phi_grid']
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    sample_for_integration = Rzphi_density_data['sample_for_integration']

    X_regular_grid = bin_dict['X_regular_grid']
    Y_regular_grid = bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)

    return {
        'v0': v0,
        's': s,
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
    }


def plot_prettier(dpi=200, fontsize=11, usetex=False):
    plt.rcParams['figure.dpi'] = dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern=[2.0, 2.0])
    plt.rc('text', usetex=usetex)
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def compute_projected_observables(w, A_xy, A_h1, A_h2, A_h3, A_h4, y_xy, v0, s):
    A_h1w = A_h1 * A_xy
    A_h2w = A_h2 * A_xy
    A_h3w = A_h3 * A_xy
    A_h4w = A_h4 * A_xy

    density_2DXY = A_xy @ w
    h1_model = (A_h1w @ w) / y_xy
    h2_model = (A_h2w @ w) / y_xy
    h3_model = (A_h3w @ w) / y_xy
    h4_model = (A_h4w @ w) / y_xy

    clip_val = 10.0
    h1_model = jnp.clip(h1_model, -clip_val, clip_val)
    h2_model = jnp.clip(h2_model, -clip_val, clip_val)
    h3_model = jnp.clip(h3_model, -clip_val, clip_val)
    h4_model = jnp.clip(h4_model, -clip_val, clip_val)

    h3_model = jnp.where(jnp.isnan(h3_model), 0.0, h3_model)
    h4_model = jnp.where(jnp.isnan(h4_model), 0.0, h4_model)

    V_model, sigma_model = h_to_V_sigma(h1_model, h2_model, v0, s)
    return density_2DXY, V_model, sigma_model, h3_model, h4_model


def build_weighted_design(
    A_Rzphi,
    A_xy,
    A_h1,
    A_h2,
    A_h3,
    A_h4,
    y_Rzphi,
    y_xy,
    y_h1,
    y_h2,
    y_h3,
    y_h4,
    sig_Rzphi,
    sig_xy,
    sig_A1,
    sig_A2,
    sig_A3,
    sig_A4,
):
    """Build dense U,y for quadratic least-squares objective (no clipping/entropy)."""
    eps = 1e-8

    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    # Match weighting convention from _nll_z.
    w_rzphi = jnp.sqrt(5.0 / A_Rzphi.shape[0])
    w_xy = jnp.sqrt(5.0 / A_xy.shape[0])
    w_h = jnp.sqrt(1.0 / A_h1.shape[0])

    U_rz = w_rzphi * (A_Rzphi / (sig_Rzphi[:, None] + eps))
    y_rz = w_rzphi * (y_Rzphi / (sig_Rzphi + eps))

    U_xy = w_xy * (A_xy / (sig_xy[:, None] + eps))
    y_xy_obs = w_xy * (y_xy / (sig_xy + eps))

    U_h1 = w_h * ((A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps))
    U_h2 = w_h * ((A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps))
    U_h3 = w_h * ((A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps))
    U_h4 = w_h * ((A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps))

    y_h1_obs = w_h * (y_h1 / (sig_A1 + eps))
    y_h2_obs = w_h * (y_h2 / (sig_A2 + eps))
    y_h3_obs = w_h * (y_h3 / (sig_A3 + eps))
    y_h4_obs = w_h * (y_h4 / (sig_A4 + eps))

    U = jnp.vstack([U_rz, U_xy, U_h1, U_h2, U_h3, U_h4])
    y = jnp.concatenate([y_rz, y_xy_obs, y_h1_obs, y_h2_obs, y_h3_obs, y_h4_obs])
    return U, y


def compute_metric_logL(
    w,
    A_xy,
    A_h1,
    A_h2,
    A_h3,
    A_h4,
    y_xy,
    y_h1,
    y_h2,
    y_h3,
    y_h4,
    sig_xy,
    sig_A1,
    sig_A2,
    sig_A3,
    sig_A4,
):
    """Same metric form as test_weights_optimisation.py."""
    A_h1w = A_h1 * A_xy
    A_h2w = A_h2 * A_xy
    A_h3w = A_h3 * A_xy
    A_h4w = A_h4 * A_xy

    density_2DXY = A_xy @ w
    h1_model = (A_h1w @ w) / y_xy
    h2_model = (A_h2w @ w) / y_xy
    h3_model = (A_h3w @ w) / y_xy
    h4_model = (A_h4w @ w) / y_xy

    # Mirror model/get_weights output clipping for comparability.
    clip_val = 10.0
    h1_model = jnp.clip(h1_model, -clip_val, clip_val)
    h2_model = jnp.clip(h2_model, -clip_val, clip_val)
    h3_model = jnp.clip(h3_model, -clip_val, clip_val)
    h4_model = jnp.clip(h4_model, -clip_val, clip_val)

    h3_model = jnp.where(jnp.isnan(h3_model), 0.0, h3_model)
    h4_model = jnp.where(jnp.isnan(h4_model), 0.0, h4_model)

    res_density = ((density_2DXY - y_xy) / (sig_xy + EPSILON)) ** 2
    res_h1 = ((h1_model - y_h1) / (sig_A1 + 1e-3)) ** 2
    res_h2 = ((h2_model - y_h2) / (sig_A2 + 1e-3)) ** 2
    res_h3 = ((h3_model - y_h3) / (sig_A3 + 1e-3)) ** 2
    res_h4 = ((h4_model - y_h4) / (sig_A4 + 1e-3)) ** 2

    # res_h1 = jnp.where((h1_model < 9.9), res_h1, 0)
    # res_h2 = jnp.where((h2_model < 9.9), res_h2, 0)
    # res_h3 = jnp.where((h3_model < 9.9), res_h3, 0)
    # res_h4 = jnp.where((h4_model < 9.9), res_h4, 0)

    val1 = jnp.nansum(-0.5 * res_density) / len(density_2DXY)
    val4 = jnp.nansum(-0.5 * res_h1) / len(h1_model)
    val5 = jnp.nansum(-0.5 * res_h2) / len(h2_model)
    val6 = jnp.nansum(-0.5 * res_h3) / len(h3_model)
    val7 = jnp.nansum(-0.5 * res_h4) / len(h4_model)

    return val1 + val4 + val5 + val6 + val7


def main():
    plot_prettier(usetex=False)

    path_repo = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
    dict_data = get_dict_data(path_repo)

    path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
    pkl = os.path.join(path_data, 'orbital_library_bar_1.pkl')

    lambda_reg = float(os.environ.get('CDQP_LAMBDA', '1'))
    maxiter = int(os.environ.get('CDQP_MAXITER', '50'))
    tol = float(os.environ.get('CDQP_TOL', '1e-1'))

    with open(pkl, 'rb') as f:
        (
            A_Rzphi,
            A_xy,
            A_h1,
            A_h2,
            A_h3,
            A_h4,
            y_Rzphi,
            y_xy,
            y_h1,
            y_h2,
            y_h3,
            y_h4,
            sig_Rzphi,
            sig_xy,
            sig_A1,
            sig_A2,
            sig_A3,
            sig_A4,
        ) = pickle.load(f)

    A_Rzphi = jnp.asarray(A_Rzphi, dtype=jnp.float32)
    A_xy = jnp.asarray(A_xy, dtype=jnp.float32)
    A_h1 = jnp.asarray(A_h1, dtype=jnp.float32)
    A_h2 = jnp.asarray(A_h2, dtype=jnp.float32)
    A_h3 = jnp.asarray(A_h3, dtype=jnp.float32)
    A_h4 = jnp.asarray(A_h4, dtype=jnp.float32)
    y_Rzphi = jnp.asarray(y_Rzphi, dtype=jnp.float32)
    y_xy = jnp.asarray(y_xy, dtype=jnp.float32)
    y_h1 = jnp.asarray(y_h1, dtype=jnp.float32)
    y_h2 = jnp.asarray(y_h2, dtype=jnp.float32)
    y_h3 = jnp.asarray(y_h3, dtype=jnp.float32)
    y_h4 = jnp.asarray(y_h4, dtype=jnp.float32)
    sig_Rzphi = jnp.asarray(sig_Rzphi, dtype=jnp.float32)
    sig_xy = jnp.asarray(sig_xy, dtype=jnp.float32)
    sig_A1 = jnp.asarray(sig_A1, dtype=jnp.float32)
    sig_A2 = jnp.asarray(sig_A2, dtype=jnp.float32)
    sig_A3 = jnp.asarray(sig_A3, dtype=jnp.float32)
    sig_A4 = jnp.asarray(sig_A4, dtype=jnp.float32)

    t_build0 = time.time()
    U, y = build_weighted_design(
        A_Rzphi,
        A_xy,
        A_h1,
        A_h2,
        A_h3,
        A_h4,
        y_Rzphi,
        y_xy,
        y_h1,
        y_h2,
        y_h3,
        y_h4,
        sig_Rzphi,
        sig_xy,
        sig_A1,
        sig_A2,
        sig_A3,
        sig_A4,
    )
    n_orb = U.shape[1]

    # BoxCDQP requires explicit Q and c.
    Q = U.T @ U + (lambda_reg / n_orb) * jnp.eye(n_orb, dtype=U.dtype)
    c = -(U.T @ y)

    l = jnp.zeros((n_orb,), dtype=U.dtype)
    u = jnp.full((n_orb,), jnp.inf, dtype=U.dtype)

    # Reasonable initialization for weights.
    w0 = jnp.ones((n_orb,), dtype=U.dtype) * (jnp.sum(y_Rzphi) / n_orb)
    t_build1 = time.time()

    solver = BoxCDQP(maxiter=maxiter, tol=tol, verbose=False, implicit_diff=False)
    t_solve0 = time.time()
    sol = solver.run(w0, params_obj=(Q, c), params_ineq=(l, u))
    w_hat = sol.params
    w_hat.block_until_ready()
    t_solve1 = time.time()

    logL = compute_metric_logL(
        w_hat,
        A_xy,
        A_h1,
        A_h2,
        A_h3,
        A_h4,
        y_xy,
        y_h1,
        y_h2,
        y_h3,
        y_h4,
        sig_xy,
        sig_A1,
        sig_A2,
        sig_A3,
        sig_A4,
    )

    print('BoxCDQP lambda      :', lambda_reg)
    print('BoxCDQP maxiter/tol :', maxiter, tol)
    print('Build Q/c time [s]  :', float(t_build1 - t_build0))
    print('Solve time [s]      :', float(t_solve1 - t_solve0))
    print('Total time [s]      :', float(t_solve1 - t_build0))
    print('CDQP error          :', float(sol.state.error))
    print('weights sum/min/max :', float(jnp.sum(w_hat)), float(jnp.min(w_hat)), float(jnp.max(w_hat)))
    print('log_likelihood      :', float(logL))

    n_orb = A_Rzphi.shape[1]
    weights_unity = jnp.ones((n_orb,), A_Rzphi.dtype) * (jnp.sum(y_Rzphi) / n_orb)

    v0 = dict_data['v0']
    s = dict_data['s']

    density_2DXY_weighted, V_model_weighted, sigma_model_weighted, h3_model_weighted, h4_model_weighted = compute_projected_observables(
        w_hat, A_xy, A_h1, A_h2, A_h3, A_h4, y_xy, v0, s
    )
    density_2DXY_unity, V_model_unity, sigma_model_unity, h3_model_unity, h4_model_unity = compute_projected_observables(
        weights_unity, A_xy, A_h1, A_h2, A_h3, A_h4, y_xy, v0, s
    )

    bin_mapping = dict_data['bin_mapping']
    index_remap = bin_mapping[:-1]

    density_2DXY_weighted = density_2DXY_weighted[index_remap]
    V_model_weighted = V_model_weighted[index_remap]
    sigma_model_weighted = sigma_model_weighted[index_remap]
    h3_model_weighted = h3_model_weighted[index_remap]
    h4_model_weighted = h4_model_weighted[index_remap]

    density_2DXY_unity = density_2DXY_unity[index_remap]
    V_model_unity = V_model_unity[index_remap]
    sigma_model_unity = sigma_model_unity[index_remap]
    h3_model_unity = h3_model_unity[index_remap]
    h4_model_unity = h4_model_unity[index_remap]

    X_regular_grid = np.asarray(dict_data['X_regular_grid'])
    Y_regular_grid = np.asarray(dict_data['Y_regular_grid'])
    density_2DXY_data = y_xy[index_remap]
    V_model_data = dict_data['V_data'][index_remap]
    sigma_model_data = dict_data['sigma_data'][index_remap]
    h3_model_data = y_h3[index_remap]
    h4_model_data = y_h4[index_remap]

    model_batch = (density_2DXY_weighted, V_model_weighted, sigma_model_weighted, h3_model_weighted, h4_model_weighted)
    model_unity_batch = (density_2DXY_unity, V_model_unity, sigma_model_unity, h3_model_unity, h4_model_unity)
    data_batch = (density_2DXY_data, V_model_data, sigma_model_data, h3_model_data, h4_model_data)

    fig_names = [r'$\Sigma_{\rm lum}$ [L$_\odot$/pc$^2$]', r'$V_{\rm los}$ [km/s]', r'$\sigma_{v}$ [km/s]', r'$h_3$', r'$h_4$']
    vmin_ls = [1e-6, -200, 20, -0.2, -0.2]
    vmax_ls = [1e-3, 200, 120, 0.2, 0.2]
    vminmax_ls = [0.5, 20, 20, 0.5, 0.5]

    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 6))
    ax_hist.hist(np.log(np.asarray(w_hat) + 1e-30), range=[-10, 5], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax_hist.set_xlabel('log(weight)')
    ax_hist.set_ylabel('Count')
    hist_path = os.path.abspath('plots/boxcdqp_weights_hist.png')
    fig_hist.savefig(hist_path, bbox_inches='tight')
    plt.close(fig_hist)

    fig1, ax1 = plt.subplots(len(model_batch), 4, figsize=(45, 5 * len(model_batch)), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

    for i in range(len(model_batch)):
        model_vals = np.asarray(model_batch[i])
        model_unity_vals = np.asarray(model_unity_batch[i])
        data_vals = np.asarray(data_batch[i])

        if i == 0:
            model_vals = np.clip(model_vals, 1e-12, None)
            model_unity_vals = np.clip(model_unity_vals, 1e-12, None)
            data_vals = np.clip(data_vals, 1e-12, None)
            norm = LogNorm(vmin=vmin_ls[i], vmax=vmax_ls[i])
            cb = ax1[i][0].scatter(X_regular_grid, Y_regular_grid, c=model_vals, s=30, cmap='viridis', marker='s', norm=norm, rasterized=True)
            cb = ax1[i][1].scatter(X_regular_grid, Y_regular_grid, c=model_unity_vals, s=30, cmap='viridis', marker='s', norm=norm, rasterized=True)
            cb = ax1[i][2].scatter(X_regular_grid, Y_regular_grid, c=data_vals, s=30, cmap='viridis', marker='s', norm=norm, rasterized=True)
        else:
            cb = ax1[i][0].scatter(X_regular_grid, Y_regular_grid, c=model_vals, s=30, cmap='viridis', marker='s', vmin=vmin_ls[i], vmax=vmax_ls[i], rasterized=True)
            cb = ax1[i][1].scatter(X_regular_grid, Y_regular_grid, c=model_unity_vals, s=30, cmap='viridis', marker='s', vmin=vmin_ls[i], vmax=vmax_ls[i], rasterized=True)
            cb = ax1[i][2].scatter(X_regular_grid, Y_regular_grid, c=data_vals, s=30, cmap='viridis', marker='s', vmin=vmin_ls[i], vmax=vmax_ls[i], rasterized=True)

        ax1[i][0].set_title('Model', fontsize=15)
        ax1[i][1].set_title('Unity Weights', fontsize=15)
        ax1[i][2].set_title('Data', fontsize=15)
        for j in range(3):
            ax1[i][j].set_xlabel('X [kpc]', fontsize=12)
            ax1[i][j].set_ylabel('Y [kpc]', fontsize=12)
            ax1[i][j].set_xlim(-12, 12)
            ax1[i][j].set_ylim(-4, 4)
            cbar = fig1.colorbar(ax1[i][j].collections[-1], ax=ax1[i][j])
            cbar.set_label(fig_names[i], fontsize=18)
            cbar.ax.tick_params(labelsize=14)

        if i == 0:
            res = np.where(np.abs(data_vals) > 1e-12, (data_vals - model_vals) / data_vals, 0.0)
        else:
            res = data_vals - model_vals
        cb = ax1[i][3].scatter(
            X_regular_grid, Y_regular_grid, c=res,
            s=30, cmap='coolwarm', marker='s',
            vmin=-vminmax_ls[i], vmax=vminmax_ls[i], rasterized=True
        )
        ax1[i][3].set_title('Residuals (Data - Model) / Data' if i == 0 else 'Residuals (Data - Model)', fontsize=15)
        ax1[i][3].set_xlabel('X [kpc]', fontsize=12)
        ax1[i][3].set_ylabel('Y [kpc]', fontsize=12)
        ax1[i][3].set_xlim(-12, 12)
        ax1[i][3].set_ylim(-4, 4)
        cbar = fig1.colorbar(cb, ax=ax1[i][3])
        cbar.set_label('Residuals', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

    fig1.tight_layout()
    map_path = os.path.abspath('plots/boxcdqp_maps.png')
    fig1.savefig(map_path, bbox_inches='tight')
    plt.close(fig1)

    print('Saved histogram plot:', hist_path)
    print('Saved map comparison plot:', map_path)


if __name__ == '__main__':
    main()
