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
    'logM_10kpc', 'logM_disk', 'logM_bar', 'logc_halo', 'logRs_disk',
    'logHs_disk', 'logL_bar', 'alpha', 'beta', 'gamma',
    'log_mass_to_light', 'Omega',
]
param_names_plot = [
    r'$\log M_{\rm DM, <10}$', r'$\log M_{\rm disk}$', r'$\log M_{\rm bar}$',
    r'$\log C_{\rm halo}$', r'$\log R_{s,\rm disk}$', r'$\log H_{s,\rm disk}$',
    r'$\log L_{\rm bar}$', r'$\alpha$ (deg)', r'$\beta$ (deg)', r'$\gamma$ (deg)',
    r'$\log \Upsilon$', r'$\Omega$ (km/s/kpc)',
]

# ── Ground truth ─────────────────────────────────────────────────────
# Values in the same order as param_names_raw
# Angles in RADIANS (as stored in the chain), not degrees
# ground_truth_dict = {
#     'logM_halo': 11.9,
#     'logM_disk': 10.75,
#     'logM_bar': 10.2,
#     'logRs_halo': np.log10(18.9),
#     'logRs_disk': np.log10(9.0),
#     'logHs_disk': np.log10(0.8),
#     'logL_bar': np.log10(4.87),
#     'alpha': 10 * np.pi / 180,       # 30 deg → rad
#     'beta': 25 * np.pi / 180,        # 20 deg → rad
#     'gamma': 170 * np.pi / 180,      # 140 deg → rad
#     'mass_to_light': 1.,   # Upsilon=1 → log=0
#     'Omega': 32.0,      # Omega=25 → log
#     'log_sigma': 0.5,
# }
ground_truth_dict = {
    'logM_halo': 11.88,
    'logM_disk': 10.75,
    'logM_bar': 10.15,
    'logRs_halo': np.log10(19.2),
    'logRs_disk': np.log10(9.0),
    'logHs_disk': np.log10(0.8),
    'logL_bar': np.log10(4.5),
    'alpha': 40 ,       # 30 deg → rad
    'beta': 25 ,        # 20 deg → rad
    'gamma': 140 ,      # 140 deg → rad
    'log_mass_to_light': 0.,   # Upsilon=1 → log=0
    'Omega': 25.0,      # Omega=25 → log
}
# Ordered array matching param_names_raw
logM_10_gt, logc_halo_gt = logM_logRs_to_logMenc_logc(
    ground_truth_dict['logM_halo'], ground_truth_dict['logRs_halo'])
ground_truth_dict['logM_10kpc'] = logM_10_gt
ground_truth_dict['logc_halo'] = logc_halo_gt

ground_truth = np.array([ground_truth_dict[k] for k in param_names_raw])


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


def monitor(filepath, discard, show_chains=False):
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

    if show_chains:

        print(f"\nBest log-posterior: {best_logP:.2f}")
        print(f"{'Parameter':>25s} {'best-fit':>10s} {'truth':>10s}")
        print("-" * 50)
        for i, name in enumerate(param_names_raw):
            print(f"{name:>25s} {best_param[i]:10.4f} {ground_truth[i]:10.4f}")
            if 'log' in name.lower():
                print(f"{'  10^' + name:>25s} {10**best_param[i]:10.4f} {10**ground_truth[i]:10.4f}")
            elif name in ('alpha', 'beta', 'gamma'):
                print(f"{'  ' + name + ' (deg)':>25s} {best_param[i]*180/np.pi:10.2f} {ground_truth[i]*180/np.pi:10.2f}")
            elif name in ('Upsilon', 'Omega'):
                print(f"{'  ' + name:>25s} {best_param[i]:10.4f} {ground_truth[i]:10.4f}")

        logM_10, logc = logM_logRs_to_logMenc_logc(best_param[0], best_param[3])
        print(f"\n{'logM(<10kpc)':>25s} {logM_10:10.4f} {logM_10_gt:10.4f}")
        print(f"{'log c_halo':>25s} {logc:10.4f} {logc_halo_gt:10.4f}")

        # ── Acceptance rate (approximate from logP changes) ──────────────
        changed = np.diff(logprob, axis=0) != 0  # (n_steps-1, n_chains)
        if changed.size > 0:
            accept_rate = changed.mean()
            print(f"\nApprox acceptance rate: {accept_rate:.3f}")

        # ── Trace plots ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, ndim, figsize=(5 * ndim, 8),
                                gridspec_kw={'hspace': 0.1})
        for i in tqdm(range(ndim), desc="Trace plots"):
            ax = axes[i]
            ylims = np.percentile(posterior[:, :, i].flatten(), [0.5, 99.5])
            ax.set_title(param_names_plot[i], fontsize=25)
            for j in range(n_chains):
                ax.plot(posterior[:, j, i], lw=3, alpha=0.3, color='orange',
                        rasterized=True)
            ax.axhline(ground_truth[i], color='black', ls='--', lw=2)
            if discard < n_steps:
                ax.axvline(discard, color='red', ls=':', lw=2)
            ax.set(xticks=[], ylim=ylims)
            if i == ndim - 1:
                ax.set_xlabel('Iteration', fontsize=20)
        fig.tight_layout()
        plt.show()

        # ── Log-probability trace ────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        for j in range(n_chains):
            ax1.plot(logprob[:, j], lw=3, alpha=0.3, color='darkblue',
                    rasterized=True)
        ax1.axhline(best_logP, color='red', ls='--', lw=2,
                    label=f'best = {best_logP:.1f}')
        ax1.set_title('Log(Posterior)', fontsize=30)
        ax1.set_xlabel('Iteration', fontsize=20)
        ax1.set(ylim = [best_logP - 1000, best_logP + 10])
        ax1.legend(fontsize=15)
        plt.show()

        print(f"\nMax log-posterior: {best_logP:.2f}")
        if n_steps > 0:
            last_logp = logprob[-1, :]
            finite = last_logp[np.isfinite(last_logp)]
            if len(finite) > 0:
                print(f"Last iteration: mean={finite.mean():.1f}, "
                    f"std={finite.std():.1f}")

    return posterior, logprob


if __name__ == '__main__':

        
    FIG_OUT        = (f'{DATA_FOLDER}/plots/'
                  f'posterior_beta25_gamma140_D50_gal2_noh3h4.png')
    FIG_PAPER      = (f'{DATA_FOLDER}/figs_paper/'  
                  f'posterior_beta25_gamma140_D50_gal2_noh3h4.pdf')


    # ── Configuration ────────────────────────────────────────────────────
    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'

    discard = 300        # burn-in steps to discard for corner plot
    thin = 5             # thinning factor for corner plot (was 1)

    ndim = len(param_names_raw)

    fig_posterior, axes = plt.subplots(ndim, ndim, figsize=(3*ndim, 3*ndim))

    CHECKPOINT_FILE = data_folder+'/mcmc_checkpoint_0513_beta25_gamma140_D50_gal2_noh3h4.pkl'
    posterior, logprob = monitor(filepath=CHECKPOINT_FILE, discard=discard)

    flat_samples = posterior[discard::thin, :, :].reshape(-1, ndim+1)
    print(f"\nCorner plot: {flat_samples.shape[0]} samples "
        f"(discard={discard}, thin={thin})")
    flat_samples = flat_samples[:, :-1]  # Exclude log_sigma for corner plot

    flat_samples[:,6] = np.random.normal(flat_samples[:,6], 0.03)  # Add small jitter to logL_bar for better visualization
    flat_samples[:,-2] = np.random.normal(flat_samples[:,-2], 0.03)

    mins = [min([flat_samples[:, i].min(), ground_truth[i]-0.02]) for i in range(ndim)]
    maxs = [max([flat_samples[:, i].max(), ground_truth[i]+0.02]) for i in range(ndim)]
    ranges = [(mins[i], maxs[i]) for i in range(ndim)]
    corner.corner(
        flat_samples,
        range = ranges,
        labels=param_names_plot,
        label_kwargs={"fontsize": 28},   # axis labels along the outside
        color='orange',
        smooth=1.5,
        # quantiles=[0.16, 0.5, 0.84],
        fig=fig_posterior,
        rasterized=True,
        plot_datapoints=False,   # skip per-sample dots -> tiny PDF
        fill_contours=True,      # filled contours instead, looks like density
        plot_density=False,      # skip the density mesh under the contours
        bins=30,
        hist_kwargs = {'density': True, 'lw':2, 'histtype': 'stepfilled', 'alpha': 0.5},
    )

    # CHECKPOINT_FILE = data_folder+'/mcmc_checkpoint_gal2_0407.pkl'
    CHECKPOINT_FILE = data_folder+'/mcmc_checkpoint_0422_beta25_gamma140_D50_gal2.pkl'
    # CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_0418_beta25_gamma140_D50_gal2_fixedbar.pkl'
    # CHECKPOINT_FILE = data_folder+'/ensemble_checkpoint_0423_beta25_gamma110_D50_gal2.pkl'
    posterior, logprob = monitor(filepath=CHECKPOINT_FILE, discard=discard)

    flat_samples = posterior[discard::thin, :, :].reshape(-1, ndim+1)
    print(f"\nCorner plot: {flat_samples.shape[0]} samples "
        f"(discard={discard}, thin={thin})")
    flat_samples = flat_samples[:, :-1]  # Exclude log_sigma for corner plot

    flat_samples[:,6] = np.random.normal(flat_samples[:,6], 0.03)  # Add small jitter to logL_bar for better visualization
    flat_samples[:,-2] = np.random.normal(flat_samples[:,-2], 0.03)

    # mins = [min([flat_samples[:, i].min(), ground_truth[i]-0.02]) for i in range(ndim)]
    # maxs = [max([flat_samples[:, i].max(), ground_truth[i]+0.02]) for i in range(ndim)]
    # ranges = [(mins[i], maxs[i]) for i in range(ndim)]
    corner.corner(
        flat_samples,
        range = ranges,
        labels=param_names_plot,
        truths=ground_truth,
        label_kwargs={"fontsize": 28},   # axis labels along the outside
        color='black',
        smooth=1,
        truth_color='red',
        # quantiles=[0.16, 0.5, 0.84],
        fig=fig_posterior,
        rasterized=True,
        plot_datapoints=False,   # skip per-sample dots -> tiny PDF
        fill_contours=True,      # filled contours instead, looks like density
        plot_density=False,      # skip the density mesh under the contours
        bins=30,
        hist2d_kwargs = {'alpha': 0.5},
        hist_kwargs = {'density': True, 'lw':2, 'histtype': 'stepfilled', 'alpha': 0.25},
    )
    # Tick label sizes (corner does not honour rcParams here)
    for ax in fig_posterior.get_axes():
        ax.tick_params(axis='both', which='major', labelsize=20)

    # ── Legend in the empty upper-right region of the corner plot ──
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='orange',  lw=40, alpha = 0.3,
               label=r'Posterior without $h_3$, $h_4$ measurements'),
        Line2D([0], [0], color='black', lw=40, alpha = 0.3,
               label=r'Posterior with $h_3$, $h_4$ measurements'),
        Line2D([0], [0], color='red',   lw=5, ls='-',
               label=r'Ground truth'),
    ]
    # Dedicated axis placed in the upper-right empty triangle.
    legend_ax = fig_posterior.add_axes([0.55, 0.82, 0.40, 0.12])
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_handles, loc='center',
                     fontsize=50, frameon=True, handlelength=2.5)

    # fig_posterior.tight_layout()
    fig_posterior.savefig(FIG_OUT, dpi=150)
    fig_posterior.savefig(FIG_PAPER, dpi=150)

    plt.close(fig_posterior)
