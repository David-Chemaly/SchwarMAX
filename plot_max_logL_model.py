"""
Pick the single posterior sample with the highest log-posterior across
all (chain, step) and plot model vs data vs residual maps in the same
5x3 layout as plot_data_vs_model.py.

Requires the samples file to contain `potential_energy` (captured by the
updated fit_orbit_weights_posterior.py).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

from ghMoments import h_to_V_sigma


# =====================================================================
DATA_FOLDER = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
TAG          = '0422_beta25_gamma140_D50_gal2_unity'
MATRICES_IN  = f'{DATA_FOLDER}/orbit_matrices_0422_beta25_gamma140_D50_gal2.pkl'
SAMPLES_IN   = f'{DATA_FOLDER}/orbit_weight_samples_{TAG}.pkl'
SOURCE_DATA  = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/mock_data/mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma140_D50_gal2.pkl'
FIG_OUT      = f'{DATA_FOLDER}/plots/model_vs_data_maxlogL_{TAG}.png'

# Plot style (matches plot_data_vs_model.py)
plt.rcParams['figure.dpi']  = 110
plt.rc('font', size=11)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman'] + plt.rcParams['font.serif']

EPS = 1e-8


# =====================================================================
def main():
    print(f"matrices  <- {MATRICES_IN}")
    with open(MATRICES_IN, 'rb') as f:
        m = pickle.load(f)
    print(f"samples   <- {SAMPLES_IN}")
    with open(SAMPLES_IN, 'rb') as f:
        s = pickle.load(f)
    print(f"source    <- {SOURCE_DATA}")
    with open(SOURCE_DATA, 'rb') as f:
        src = pickle.load(f)

    if 'potential_energy' not in s:
        raise SystemExit("samples file lacks 'potential_energy' -- "
                         "rerun the sampler so it captures HMC extras.")

    log_w = np.asarray(s['log_w_samples'])    # (chains, samples, n_orb)
    pe    = np.asarray(s['potential_energy']) # (chains, samples)
    best_chain, best_step = np.unravel_index(np.argmin(pe), pe.shape)
    logp_max = -float(pe[best_chain, best_step])
    log_w_best = log_w[best_chain, best_step]
    w = np.exp(log_w_best)
    print(f"max-logp sample:  chain={best_chain}  step={best_step}  "
          f"logp={logp_max:.4e}")

    # ---- Pull matrices and matched data vectors / sigmas (normalised units) ----
    A_xy   = np.asarray(m['A_xy'])
    A_h1   = np.asarray(m['A_h1']); A_h2 = np.asarray(m['A_h2'])
    A_h3   = np.asarray(m['A_h3']); A_h4 = np.asarray(m['A_h4'])
    y_xy   = np.asarray(m['y_xy'])
    sig_xy = np.asarray(m['sig_xy'])
    sig_A1 = np.asarray(m['sig_A1']); sig_A2 = np.asarray(m['sig_A2'])
    sig_A3 = np.asarray(m['sig_A3']); sig_A4 = np.asarray(m['sig_A4'])
    y_h1, y_h2 = np.asarray(m['y_h1']), np.asarray(m['y_h2'])
    y_h3, y_h4 = np.asarray(m['y_h3']), np.asarray(m['y_h4'])

    LM           = 10. ** float(np.asarray(m['best_fit_param'])[10])
    mean_mpo     = float(m['mean_mass_per_orb'])
    v0_arr, s_arr = np.asarray(src['v0']), np.asarray(src['s'])

    # ---- Model evaluation (mirrors compute_model_and_logl_bootstrap) ----
    y_xy_safe = np.where(np.abs(y_xy) > EPS, y_xy, 1.0)
    density_norm = A_xy @ w
    h1m = np.clip((A_h1 * A_xy @ w) / y_xy_safe, -10., 10.)
    h2m = np.clip((A_h2 * A_xy @ w) / y_xy_safe, -10., 10.)
    h3m = np.clip((A_h3 * A_xy @ w) / y_xy_safe, -10., 10.)
    h4m = np.clip((A_h4 * A_xy @ w) / y_xy_safe, -10., 10.)
    V_model, sig_model = h_to_V_sigma(h1m, h2m, v0_arr, s_arr)
    V_model   = np.asarray(V_model)
    sig_model = np.asarray(sig_model)

    # ---- Convert density to luminosity units (so the plot_data_vs_model
    # color limits 1e1..1e4 apply directly) ----
    density_lum = density_norm * mean_mpo * LM

    # ---- Per-block chi2 (matches plot_data_vs_model.py lines 359-363) ----
    chi2_density = float(np.sum((density_norm - y_xy)**2 / sig_xy**2))
    chi2_h1 = float(np.sum((h1m - y_h1)**2 / sig_A1**2))
    chi2_h2 = float(np.sum((h2m - y_h2)**2 / sig_A2**2))
    chi2_h3 = float(np.sum((h3m - y_h3)**2 / sig_A3**2))
    chi2_h4 = float(np.sum((h4m - y_h4)**2 / sig_A4**2))
    print(f"per-block chi2: density={chi2_density:.2f}  "
          f"h1={chi2_h1:.2f}  h2={chi2_h2:.2f}  "
          f"h3={chi2_h3:.2f}  h4={chi2_h4:.2f}")

    # ---- Per-pixel maps via bin remap (matches plot_data_vs_model.py) ----
    bin_map     = np.asarray(src['bin_mapping'])
    index_remap = bin_map[:-1]

    # Data per Voronoi bin
    surface_density_data = np.asarray(src['surface_density'])  # already L_sun/pc^2
    V_data    = np.asarray(src['V_mean'])
    sig_data  = np.asarray(src['V_sigma'])
    h3_data   = np.asarray(src['h3'])
    h4_data   = np.asarray(src['h4'])

    model_batch = (density_lum[index_remap], V_model[index_remap],
                   sig_model[index_remap], h3m[index_remap], h4m[index_remap])
    data_batch  = (surface_density_data[index_remap], V_data[index_remap],
                   sig_data[index_remap], h3_data[index_remap], h4_data[index_remap])

    fig_names  = [r'$\Sigma_{\rm *}$ [L$_\odot$/pc$^2$]',
                  r'$V_{\rm los}$ [km/s]', r'$\sigma_{v}$ [km/s]',
                  r'$h_3$', r'$h_4$']
    color_maps = [cmr.sepia, cmr.iceburn, cmr.amber, cmr.iceburn, cmr.amber]
    vmin_ls    = [1e1, -200,  20, -0.2, -0.2]
    vmax_ls    = [1e4,  200, 150,  0.2,  0.1]
    vminmax_ls = [0.15, 25,   25,  0.15, 0.15]
    chi2_ls    = [chi2_density, chi2_h1, chi2_h2, chi2_h3, chi2_h4]

    X_grid   = np.asarray(src['X_regular_grid'])
    Y_grid   = np.asarray(src['Y_regular_grid'])
    X_minmax = np.asarray(src['X_minmax'])
    Y_minmax = np.asarray(src['Y_minmax'])

    fig, ax = plt.subplots(5, 3, figsize=(18, 12.5),
                           gridspec_kw={'hspace': 0.40, 'wspace': 0.30})
    fig.suptitle(
        f'max-logL sample  [chain {best_chain}, step {best_step}]   '
        f'logp = {logp_max:.4e}   [{TAG}]',
        fontsize=14, y=0.998)

    for i in range(5):
        norm = 'log' if i == 0 else None
        # Model
        cb = ax[i, 0].scatter(X_grid, Y_grid, c=model_batch[i], s=22,
                              cmap=color_maps[i], marker='s',
                              norm=norm, vmin=vmin_ls[i], vmax=vmax_ls[i],
                              rasterized=True)
        ax[i, 0].set_title('Model' if i == 0 else '', fontsize=14)
        ax[i, 0].set_xlabel('X [kpc]')
        ax[i, 0].set_ylabel('Y [kpc]')
        ax[i, 0].set_xlim(X_minmax); ax[i, 0].set_ylim(Y_minmax)
        cbar = fig.colorbar(cb, ax=ax[i, 0]); cbar.set_label(fig_names[i])

        # Data
        cb = ax[i, 1].scatter(X_grid, Y_grid, c=data_batch[i], s=22,
                              cmap=color_maps[i], marker='s',
                              norm=norm, vmin=vmin_ls[i], vmax=vmax_ls[i],
                              rasterized=True)
        ax[i, 1].set_title('Data' if i == 0 else '', fontsize=14)
        ax[i, 1].set_xlabel('X [kpc]')
        ax[i, 1].set_ylabel('Y [kpc]')
        ax[i, 1].set_xlim(X_minmax); ax[i, 1].set_ylim(Y_minmax)
        cbar = fig.colorbar(cb, ax=ax[i, 1]); cbar.set_label(fig_names[i])

        # Residuals
        if i == 0:
            res = np.log10(data_batch[i]) - np.log10(model_batch[i])
        else:
            res = data_batch[i] - model_batch[i]
        cb = ax[i, 2].scatter(X_grid, Y_grid, c=res, s=22,
                              cmap='coolwarm', marker='s',
                              vmin=-vminmax_ls[i], vmax=vminmax_ls[i],
                              rasterized=True)
        ax[i, 2].set_title('Residuals' if i == 0 else '', fontsize=14)
        ax[i, 2].set_xlabel('X [kpc]')
        ax[i, 2].set_ylabel('Y [kpc]')
        ax[i, 2].set_xlim(X_minmax); ax[i, 2].set_ylim(Y_minmax)
        ax[i, 2].text(0.05, 0.85, r'$\chi^2 = $' + f'{chi2_ls[i]:.2f}',
                      transform=ax[i, 2].transAxes, fontsize=13)
        cbar = fig.colorbar(cb, ax=ax[i, 2]); cbar.set_label('Data - Model')

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches='tight', dpi=200)
    print(f"saved -> {FIG_OUT}")


if __name__ == '__main__':
    main()
