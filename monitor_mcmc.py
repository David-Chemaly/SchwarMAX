"""
Monitor BlackJAX parallel RMH chains from checkpoint file.
Produces: trace plots, logP evolution, best-fit summary, corner plot.

Usage:
    python monitor_mcmc.py
    # or import and call monitor() from a notebook
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import corner
from tqdm import tqdm


def logM_logRs_to_logMenc_logc(logM_halo, logRs_halo, r_enc=10.0, Delta=200, rho_crit=277.54):
    M = 10**logM_halo
    Rs = 10**logRs_halo
    x = r_enc / Rs
    M_enc = M * (np.log(1 + x) - x / (1 + x))
    R_vir = (3 * M / (4 * np.pi * Delta * rho_crit))**(1/3)
    c = R_vir / Rs
    return np.log10(M_enc), np.log10(c)


def logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0, Delta=200, rho_crit=277.54):
    """Inverse of logM_logRs_to_logMenc_logc.

    Given enclosed mass M_enc(<r_enc) and concentration c, recover total
    NFW mass M and scale radius Rs by solving:
        Rs = (3M / (4 pi Delta rho_crit))^(1/3) / c
        M_enc = M * [ln(1 + r_enc/Rs) - (r_enc/Rs) / (1 + r_enc/Rs)]
    The second equation is one equation in one unknown (M); solved with Brent's method.
    """
    from scipy.optimize import brentq

    M_enc = 10**logM_enc
    c = 10**log_c

    def residual(logM_trial):
        M = 10**logM_trial
        Rs = (3 * M / (4 * np.pi * Delta * rho_crit))**(1/3) / c
        x = r_enc / Rs
        M_enc_trial = M * (np.log(1 + x) - x / (1 + x))
        return M_enc_trial - M_enc

    # M_enc < M always, so search in [logM_enc, logM_enc + 5]
    logM_sol = brentq(residual, logM_enc, logM_enc + 5, xtol=1e-12)
    M_sol = 10**logM_sol
    Rs_sol = (3 * M_sol / (4 * np.pi * Delta * rho_crit))**(1/3) / c
    return logM_sol, np.log10(Rs_sol)


# ── Configuration ────────────────────────────────────────────────────
data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
CHECKPOINT_FILE = data_folder+'/mcmc_checkpoint_parallel.pkl'
DISCARD = 100        # burn-in steps to discard for corner plot
THIN = 1             # thinning factor for corner plot

param_names_raw = [
    'logM_halo', 'logM_disk', 'logM_bar', 'logRs_halo', 'logRs_disk',
    'logHs_disk', 'logL_bar', 'alpha', 'beta', 'gamma',
    'log_light_to_mass_ratio', 'log_Omega', 'log_sigma',
]
param_names_plot = [
    r'$\log M_{\rm halo}$', r'$\log M_{\rm disk}$', r'$\log M_{\rm bar}$',
    r'$\log R_{s,\rm halo}$', r'$\log R_{s,\rm disk}$', r'$\log H_{s,\rm disk}$',
    r'$\log L_{\rm bar}$', r'$\alpha$', r'$\beta$', r'$\gamma$',
    r'$\log(\Upsilon)$', r'$\log(\Omega)$', r'$\log(\sigma)$',
]

# ── Ground truth ─────────────────────────────────────────────────────
# Values in the same order as param_names_raw
# Angles in RADIANS (as stored in the chain), not degrees
ground_truth_dict = {
    'logM_halo': 11.88,
    'logM_disk': 10.75,
    'logM_bar': 10.1,
    'logRs_halo': np.log10(18.8),
    'logRs_disk': np.log10(8.0),
    'logHs_disk': np.log10(0.8),
    'logL_bar': np.log10(4.),
    'alpha': 30 * np.pi / 180,       # 30 deg → rad
    'beta': 20 * np.pi / 180,        # 20 deg → rad
    'gamma': 140 * np.pi / 180,      # 140 deg → rad
    'log_light_to_mass_ratio': np.log10(1.),   # Upsilon=1 → log=0
    'log_Omega': np.log10(25.0),      # Omega=25 → log
    'log_sigma': -4.,
}
# Ordered array matching param_names_raw
ground_truth = np.array([ground_truth_dict[k] for k in param_names_raw])

logM_10_gt, logc_halo_gt = logM_logRs_to_logMenc_logc(
    ground_truth_dict['logM_halo'], ground_truth_dict['logRs_halo'])


def load_checkpoint(filepath=CHECKPOINT_FILE):
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


def monitor(filepath=CHECKPOINT_FILE, discard=DISCARD, thin=THIN, show_corner=True):
    posterior, logprob, step = load_checkpoint(filepath)
    n_steps, n_chains, ndim = posterior.shape

    # ── Best fit ─────────────────────────────────────────────────────
    flat_logprob = logprob.flatten()
    flat_params = posterior.reshape(-1, ndim)
    best_idx = np.argmax(flat_logprob)
    best_logP = flat_logprob[best_idx]
    best_param = flat_params[best_idx]

    print(f"\nBest log-posterior: {best_logP:.2f}")
    print(f"{'Parameter':>25s} {'best-fit':>10s} {'truth':>10s}")
    print("-" * 50)
    for i, name in enumerate(param_names_raw):
        print(f"{name:>25s} {best_param[i]:10.4f} {ground_truth[i]:10.4f}")
        if 'log' in name.lower():
            print(f"{'  10^' + name:>25s} {10**best_param[i]:10.4f} {10**ground_truth[i]:10.4f}")
        elif name in ('alpha', 'beta', 'gamma'):
            print(f"{'  ' + name + ' (deg)':>25s} {best_param[i]*180/np.pi:10.2f} {ground_truth[i]*180/np.pi:10.2f}")

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
    ax1.legend(fontsize=15)
    plt.show()

    print(f"\nMax log-posterior: {best_logP:.2f}")
    if n_steps > 0:
        last_logp = logprob[-1, :]
        finite = last_logp[np.isfinite(last_logp)]
        if len(finite) > 0:
            print(f"Last iteration: mean={finite.mean():.1f}, "
                  f"std={finite.std():.1f}")

    # ── Corner plot ──────────────────────────────────────────────────
    if show_corner and n_steps > discard:
        flat_samples = posterior[discard::thin, :, :].reshape(-1, ndim)
        print(f"\nCorner plot: {flat_samples.shape[0]} samples "
              f"(discard={discard}, thin={thin})")

        fig_corner = corner.corner(
            flat_samples,
            labels=param_names_plot,
            truths=ground_truth,
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={"fontsize": 12},
            color='C0',
            smooth=True,
            truth_color='black',
            quantiles=[0.16, 0.5, 0.84],
        )
        plt.show()

        # Summary stats
        print(f"\n── Posterior summary ({flat_samples.shape[0]} samples) ──")
        print(f"{'Parameter':>25s} {'median':>10s} {'std':>10s} "
              f"{'2.5%':>10s} {'97.5%':>10s} {'truth':>10s}")
        print("-" * 75)
        for i, name in enumerate(param_names_raw):
            s = flat_samples[:, i]
            print(f"{name:>25s} {np.median(s):10.4f} {s.std():10.4f} "
                  f"{np.percentile(s, 2.5):10.4f} "
                  f"{np.percentile(s, 97.5):10.4f} "
                  f"{ground_truth[i]:10.4f}")

    return posterior, logprob


if __name__ == '__main__':
    monitor()
