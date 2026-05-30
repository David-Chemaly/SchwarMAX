import pickle

from animate_nnls_evolution import DATA_FOLDER
import jax
import jaxopt
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

import time as tt
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18

import os
import corner
import numpy as np
import multiprocessing as mp
from scipy.stats import gaussian_kde

from potentials import *
from integrants_with_binning import *
from ghMoments import *
from utils import *
from constants import EPSILON

from potentials import NFW_acceleration
from densities import MiyamotoNagai_density

from likelihoods import mapping_norm_to_scale_uniform

from CylindricalSpline import get_phi_m, get_acc, evaluate_phi_axisymmetric

# path = '/home/hz420/python_script/SchwarMAX/'
# path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
# path = '/data/hz420-2/SchwarMAX/'

plt.rcParams["font.family"]="Times New Roman"
params = {
   'axes.labelsize': 20,
   'axes.titlesize': 20,
   'font.size': 20,
   'legend.fontsize': 20,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   } 
plt.rcParams.update(params)
plt.rcParams["font.family"]="Times New Roman"

import matplotlib.font_manager as fm

# the following commands make plots look better
def plot_prettier(dpi=200, fontsize=20, usetex=False):
    '''
    Make plots look nicer compared to Matplotlib defaults
    Parameters:
        dpi - int, "dots per inch" - controls resolution of PNG images that are produced
                by Matplotlib
        fontsize - int, font size to use overall
        usetex - bool, whether to use LaTeX to render fonds of axes labels
                use False if you don't have LaTeX installed on your system
    '''
    plt.rcParams['figure.dpi']= dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    # if you don't have LaTeX installed on your laptop and this statement
    # generates error, comment it out
    plt.rc('text', usetex=usetex)
    #fm.fontManager.addfont('Serif.ttf')
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plot_prettier(usetex=False)

def logM_logRs_to_logMenc_logc(logM_halo, logRs_halo, r_enc=10.0, Delta=200, rho_crit=277.54):
    M = 10**logM_halo
    Rs = 10**logRs_halo
    x = r_enc / Rs
    M_enc = M * (np.log(1 + x) - x / (1 + x))
    R_vir = (3 * M / (4 * np.pi * Delta * rho_crit))**(1/3)
    c = R_vir / Rs
    return np.log10(M_enc), np.log10(c)

param_names_raw = [
    'logM_bar', 
    'logL_bar', 
    'beta', 
    'gamma',
    'Omega',
]
param_names_plot = [
    r'$\log M_{\rm bar}$',
    r'$\log L_{\rm bar}$', 
    r'$\beta$ - $\beta_{\rm true}$ (deg)', 
    r'$\gamma$ - $\gamma_{\rm true}$ (deg)',
    r'$\Omega$ (km/s/kpc)',
]

# ── Ground truth ─────────────────────────────────────────────────────


def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        ckpt = pickle.load(f)

    all_samples = ckpt['all_samples']   # list of (N_CHAINS, NDIM) arrays
    all_logprob = ckpt['all_logprob']   # list of (N_CHAINS,) arrays
    step = ckpt['step']


    # Stack into (N_STEPS, N_CHAINS, NDIM) and (N_STEPS, N_CHAINS)
    posterior = np.stack(all_samples, axis=0)
    logprob = np.stack(all_logprob, axis=0)
    
    print(f"Loaded checkpoint: {step} steps, "
          f"{posterior.shape[1]} chains, {posterior.shape[2]} params")
    print(f"Chain shape: {posterior.shape}")
    return posterior, logprob, step


def monitor(filepath, true_beta = 25, true_gamma = 140, discard=0):
    posterior, logprob, step = load_checkpoint(filepath)

    posterior[:,:,-2] = 10 ** posterior[:,:,-2] * KPCGYR_TO_KMS # Convert log(Omega) to Omega
    posterior[:,:,-3] = np.log10(1 / 10 ** posterior[:,:,-3])  # Convert log(1/Upsilon) to Upsilon
    posterior[:,:,-4] = posterior[:,:,-4] * 180 / np.pi  # Convert angles from rad to deg
    posterior[:,:,-5] = posterior[:,:,-5] * 180 / np.pi  # Convert angles from rad to deg
    posterior[:,:,-6] = posterior[:,:,-6] * 180 / np.pi  # Convert angles from rad to deg


    posterior = posterior[:, logprob[-1, :]>np.amax(logprob[-1, :])-1000, :]
    logprob = logprob[:, logprob[-1, :]>np.amax(logprob[-1, :])-1000]

    n_steps, n_chains, ndim = posterior.shape
    # ── Best fit ─────────────────────────────────────────────────────
    flat_logprob = logprob.flatten()
    flat_params = posterior.reshape(-1, ndim)
    best_idx = np.argmax(flat_logprob)
    best_logP = flat_logprob[best_idx]
    best_param = flat_params[best_idx]

    logM_bar = posterior[:,:,2]
    logL_bar = posterior[:,:,6] 
    beta = posterior[:,:,8]
    gamma = posterior[:,:,9]
    Omega = posterior[:,:,11]

    beta_corr = beta - true_beta
    gamma_corr = gamma - true_gamma

    pos_sel = np.stack((logM_bar.flatten(), logL_bar.flatten(), beta_corr.flatten(), gamma_corr.flatten(), Omega.flatten()), axis=1)
    logP_sel = logprob.flatten()

    return pos_sel, logP_sel    


if __name__ == '__main__':

        
    FIG_OUT        = (f'{DATA_FOLDER}/plots/'
                  f'posterior_stacked_varying_angles.png')
    FIG_PAPER      = (f'{DATA_FOLDER}/figs_paper/'  
                  f'posterior_stacked_varying_angles.pdf')

    beta_ls = [5, 25, 25, 25, 50]
    gamma_ls = [140, 60, 110, 140, 140]
    colors_ls = ['royalblue', 'green', 'orange', 'aqua', 'purple']

    # ── Configuration ────────────────────────────────────────────────────
    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'


    ndim = len(param_names_raw)

    fig_posterior, axes = plt.subplots(ndim, ndim, figsize=(3*ndim, 3*ndim))

    legend_handles = []
    color_map = {}
    for beta, gamma, color in zip(beta_ls, gamma_ls, colors_ls):
        color_map[(beta, gamma)] = color
    diag_ymax = np.zeros(ndim)  # track tallest KDE per diagonal across all posteriors
    for beta, gamma, color in zip(beta_ls, gamma_ls, colors_ls):
        print(f"beta={beta}, gamma={gamma}")

        ground_truth_dict = {
            'logM_bar': 10.15,
            'logL_bar': np.log10(4.50),
            'beta': 0 ,        # 20 deg → rad
            'gamma': 0 ,      # 140 deg → rad
            'Omega': 25.0,      # Omega=25 → log
        }

        ground_truth = np.array([ground_truth_dict[k] for k in param_names_raw])


        CHECKPOINT_FILE = data_folder+f'/mcmc_checkpoint_0422_beta{beta}_gamma{gamma}_D50_gal2.pkl'
        discard = 300        # burn-in steps to discard for corner plot
        thin = 5             # thinning factor for corner plot (was 1)

        posterior, logprob = monitor(filepath=CHECKPOINT_FILE, true_beta=beta, true_gamma=gamma, discard=discard)

        flat_samples = posterior[discard::thin, :]
        print(f"\nCorner plot: {flat_samples.shape[0]} samples "
            f"(discard={discard}, thin={thin})")

        flat_samples[:,1] = np.random.normal(flat_samples[:,1], 0.05)  # Add small jitter to logL_bar for better visualization

        # mins = [min([flat_samples[:, i].min(), ground_truth[i]-0.02]) for i in range(ndim)]
        # maxs = [max([flat_samples[:, i].max(), ground_truth[i]+0.02]) for i in range(ndim)]
        # ranges = [(mins[i], maxs[i]) for i in range(ndim)]
        ranges = [
            (9.9, 10.22),
            (0.1,0.82),
            (-5, 3),
            (-20, 20),
            (15, 35),
        ]
        corner.corner(
            flat_samples,
            range = ranges,
            labels=param_names_plot,
            truths=ground_truth,
            # show_titles=True,
            # title_fmt='.2f',
            # title_kwargs={"fontsize": 22},
            label_kwargs={"fontsize": 28},   # axis labels along the outside
            color=color,
            smooth=1,
            truth_color='black',
            # quantiles=[0.16, 0.5, 0.84],
            fig=fig_posterior,
            rasterized=True,
            plot_datapoints=False,   # skip per-sample dots -> tiny PDF
            fill_contours=True,      # filled contours instead, looks like density
            plot_density=False,      # skip the density mesh under the contours
            bins=30,
            hist_kwargs = {'density': True, 'lw':0, 'alpha':0,},  # hide hist, KDE replaces it
        )
        # Tick label sizes (corner does not honour rcParams here)
        for ax in fig_posterior.get_axes():
            ax.tick_params(axis='both', which='major', labelsize=20)

        # Replace diagonal histograms with KDE curves.
        for i in range(ndim):
            ax = fig_posterior.axes[i * (ndim + 1)]
            lo, hi = ranges[i]
            s_i = flat_samples[:, i]
            mask = (s_i >= lo) & (s_i <= hi)
            if mask.sum() < 2 or np.std(s_i[mask]) == 0:
                continue
            kde = gaussian_kde(s_i[mask])
            xs = np.linspace(lo, hi, 400)
            ys = kde(xs)
            ax.plot(xs, ys, color=color, lw=2)
            diag_ymax[i] = max(diag_ymax[i], ys.max())

        # ── Legend in the empty upper-right region of the corner plot ──
        from matplotlib.lines import Line2D
        legend_handles.append(
            Line2D([0], [0], color=color,  lw=20, alpha = 0.3,
                label=r'$\beta={beta}^\circ$, $\gamma={gamma}^\circ$'.format(beta=beta, gamma=gamma))
                )
        
    # Set diagonal y-limits to fit the tallest KDE among all overlaid posteriors.
    for i in range(ndim):
        if diag_ymax[i] > 0:
            fig_posterior.axes[i * (ndim + 1)].set_ylim(0, diag_ymax[i] * 1.10)

    # Dedicated axis placed in the upper-right empty triangle: a 3x3 grid
    # of (beta, gamma) cells coloured by the matching posterior contour.
    unique_betas  = sorted({b for b, _ in color_map})   # rows  (y axis)
    unique_gammas = sorted({g for _, g in color_map})   # cols  (x axis)

    legend_ax = fig_posterior.add_axes([0.63, 0.67, 0.28, 0.24])
    n_b, n_g = len(unique_betas), len(unique_gammas)
    for i, b in enumerate(unique_betas):
        for j, g in enumerate(unique_gammas):
            face = color_map.get((b, g), 'white')
            edge = 'black' if (b, g) in color_map else 'black'
            alpha = 0.45 if (b, g) in color_map else 1.0
            legend_ax.add_patch(plt.Rectangle(
                (j, i), 1, 1, facecolor=face, edgecolor=edge,
                alpha=alpha, linewidth=1.5))

    legend_ax.set_xlim(0, n_g)
    legend_ax.set_ylim(0, n_b)
    legend_ax.set_aspect('equal')
    legend_ax.set_xticks(np.arange(n_g) + 0.5)
    legend_ax.set_yticks(np.arange(n_b) + 0.5)
    legend_ax.set_xticklabels([f'{g}' for g in unique_gammas], fontsize=22)
    legend_ax.set_yticklabels([f'{b}' for b in unique_betas], fontsize=22)
    legend_ax.set_xlabel(r'$\gamma_{\rm true}$ (deg)', fontsize=24)
    legend_ax.set_ylabel(r'$\beta_{\rm true}$ (deg)',  fontsize=24)
    legend_ax.tick_params(axis='both', which='both', length=0)
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    # ── Outside-the-grid annotations describing what beta / gamma mean ──
    arrow_kw = dict(arrowstyle='->', lw=2.5, color='black', mutation_scale=25)

    # Two vertical arrows on the right, splitting outward from the centre.
    legend_ax.annotate('', xy=(1.08, 1.0), xytext=(1.08, 0.55),
                       xycoords='axes fraction', arrowprops=arrow_kw)
    legend_ax.annotate('', xy=(1.08, 0.0), xytext=(1.08, 0.45),
                       xycoords='axes fraction', arrowprops=arrow_kw)
    legend_ax.text(1.12, 0.02, 'more\nedge-on disc',
                   rotation=90, va='bottom', ha='left',
                   transform=legend_ax.transAxes, fontsize=21)
    legend_ax.text(1.12, 0.98, 'more\nface-on disc',
                   rotation=90, va='top', ha='left',
                   transform=legend_ax.transAxes, fontsize=21)

    # Two horizontal arrows above the grid, splitting outward from the centre.
    legend_ax.annotate('', xy=(0.0, 1.08), xytext=(0.45, 1.08),
                       xycoords='axes fraction', arrowprops=arrow_kw)
    legend_ax.annotate('', xy=(1.0, 1.08), xytext=(0.55, 1.08),
                       xycoords='axes fraction', arrowprops=arrow_kw)
    legend_ax.text(0.02, 1.12, 'more\nend-on bar',
                   va='bottom', ha='left',
                   transform=legend_ax.transAxes, fontsize=21)
    legend_ax.text(0.98, 1.12, 'more\nside-on bar',
                   va='bottom', ha='right',
                   transform=legend_ax.transAxes, fontsize=21)


    # fig_posterior.tight_layout()
    fig_posterior.savefig(FIG_OUT, dpi=150)
    fig_posterior.savefig(FIG_PAPER, dpi=150)

    plt.close(fig_posterior)
