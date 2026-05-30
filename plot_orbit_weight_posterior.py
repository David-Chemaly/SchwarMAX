"""
Plot the posterior over Schwarzschild orbital weights produced by
fit_orbit_weights_posterior.py:

  * Mean orbit weight per orbit (1D scatter, sorted by R_apo)
  * Covariance matrix of the orbit weights (n_orb x n_orb heatmap,
    rows / columns sorted by R_apo)

R_apo per orbit is estimated from A_Rzphi: the largest R bin where the
orbit deposits at least 1% of its peak mass. The Rzphi grid has shape
(n_R=10, n_z=6, n_phi=10) flattened in the order R-fastest, z-next,
phi-slowest, so the R coordinate of bin k is R_mids[k % n_R].
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import corner

# =====================================================================
DATA_FOLDER = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
TAG          = '0422_beta25_gamma140_D50_gal2'
MATRICES_IN  = f'{DATA_FOLDER}/orbit_matrices_{TAG}.pkl'
SAMPLES_IN   = f'{DATA_FOLDER}/orbit_weight_samples_{TAG}_unity.pkl'

# Rzphi grid (matches likelihoods_bar.get_dict_data_bootstrap)
N_R, N_Z, N_PHI = 10, 6, 10
R_MIN, R_MAX    = 0.0, 10.0

# Plot output
FIG_DIR     = f'{DATA_FOLDER}/plots'
FIG_MEAN    = f'{FIG_DIR}/orbit_weight_mean_{TAG}.png'
FIG_COV     = f'{FIG_DIR}/orbit_weight_cov_{TAG}.png'
FIG_REL_COV = f'{FIG_DIR}/orbit_weight_rel_cov_{TAG}.png'
FIG_diag    = f'{FIG_DIR}/orbit_weight_diag_{TAG}.png'
FIG_errorbar= f'{FIG_DIR}/orbit_weight_errorbar_{TAG}.png'

# R_apo threshold: include Rzphi bins with > THRESH * peak per orbit
R_APO_THRESH = 0.01


# =====================================================================
def estimate_R_apo(A_Rzphi):
    """R_apo per orbit from A_Rzphi (mass per orbit per Rzphi bin)."""
    R_edges = np.linspace(R_MIN, R_MAX, N_R + 1)
    R_mids  = 0.5 * (R_edges[:-1] + R_edges[1:])
    n_bins  = N_R * N_Z * N_PHI
    R_per_bin = R_mids[np.arange(n_bins) % N_R]               # (n_bins,)

    peak = A_Rzphi.max(axis=0)                                # (n_orb,)
    thresh = R_APO_THRESH * np.where(peak > 0, peak, 1.0)
    occupied = A_Rzphi > thresh[None, :]                      # (n_bins, n_orb)
    R_proxy = np.where(occupied, R_per_bin[:, None], 0.0)
    R_apo = R_proxy.max(axis=0)
    return R_apo


def main():
    print(f"loading matrices  <- {MATRICES_IN}")
    with open(MATRICES_IN, 'rb') as f:
        mat = pickle.load(f)
    A_Rzphi = np.asarray(mat['A_Rzphi'])
    n_orb   = A_Rzphi.shape[1]

    print(f"loading samples   <- {SAMPLES_IN}")
    with open(SAMPLES_IN, 'rb') as f:
        smp = pickle.load(f)
    log_w = np.asarray(smp['log_w_samples'])                  # (chains, samples, n_orb)
    n_chains, n_samples, n_orb_s = log_w.shape
    assert n_orb_s == n_orb, f"orbit count mismatch: {n_orb_s} vs {n_orb}"
    print(f"chains={n_chains} samples/chain={n_samples} n_orb={n_orb}")

    # Convert to weights and flatten chains
    w = np.exp(log_w).reshape(-1, n_orb)                      # (chains*samples, n_orb)
    mean_w = w.mean(axis=0)                                   # (n_orb,)

    # R_apo and sort
    R_apo = estimate_R_apo(A_Rzphi)
    order = np.argsort(R_apo)
    R_apo_sorted = R_apo[order]

    # ---- Non-zero filter for the covariance matrix ----
    # NNLS exact zeros stay clipped at log(1e-30) in the HMC chains, so
    # excluding them avoids a huge speckle block in the cov plot.
    nnls_init = np.asarray(mat['weights_nnls_init'])
    # nonzero_mask = nnls_init > 0
    nonzero_mask = np.median(w, axis=0) > 1e-1
    keep_idx = np.where(nonzero_mask[order])[0]               # indices into sorted array
    n_keep = keep_idx.size
    print(f"NNLS-nonzero orbits: {n_keep}/{n_orb} "
          f"({100.*n_keep/n_orb:.1f}%) kept for covariance plot")

    print("computing covariance ...")
    w_kept = w[:, order][:, keep_idx]                          # (chains*samples, n_keep)
    cov_kept = np.cov(w_kept, rowvar=False).astype(np.float32) # (n_keep, n_keep)
    R_apo_kept = R_apo_sorted[keep_idx]

    # ---- Figure 1: mean weights ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter(np.arange(n_orb), mean_w[order],
                    c=R_apo_sorted, s=4, cmap='viridis')
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 1e3)
    ax.set_xlabel('orbit index (sorted by R$_{apo}$)')
    ax.set_ylabel(r'$\langle w \rangle$  (posterior mean)')
    ax.set_title(f'Mean orbit weights  [{TAG}]')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r'R$_{apo}$ proxy [kpc]')
    fig.tight_layout()
    fig.savefig(FIG_MEAN, dpi=150)
    print(f"saved -> {FIG_MEAN}")

    # ---- Figure 2: covariance matrix (NNLS-nonzero orbits only) ----
    vmax = float(np.percentile(np.abs(cov_kept), 99.5))
    vmin = -vmax
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cov_kept, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                   origin='lower', interpolation='nearest', aspect='equal')
    ax.set_title(f'Cov(w$_i$, w$_j$)  -- {n_keep} NNLS-nonzero orbits  [{TAG}]')
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label('covariance')
    ticks = np.linspace(0, n_keep - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{R_apo_kept[t]:.1f}' for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{R_apo_kept[t]:.1f}' for t in ticks])
    ax.set_xlabel(r'R$_{apo}$ proxy [kpc]')
    ax.set_ylabel(r'R$_{apo}$ proxy [kpc]')
    fig.tight_layout()
    fig.savefig(FIG_COV, dpi=150)
    print(f"saved -> {FIG_COV}")

    # ---- Figure 2: covariance matrix (NNLS-nonzero orbits only) ----
    rel_cov = cov_kept / np.sqrt(np.outer(np.diag(cov_kept), np.diag(cov_kept)))
    vmax = float(np.percentile(np.abs(rel_cov), 99.5))
    vmin = -vmax
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(rel_cov, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                   origin='lower', interpolation='nearest', aspect='equal')
    ax.set_title(f'Cov(w$_i$, w$_j$)  -- {n_keep} NNLS-nonzero orbits  [{TAG}]')
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label('covariance')
    ticks = np.linspace(0, n_keep - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{R_apo_kept[t]:.1f}' for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{R_apo_kept[t]:.1f}' for t in ticks])
    ax.set_xlabel(r'R$_{apo}$ proxy [kpc]')
    ax.set_ylabel(r'R$_{apo}$ proxy [kpc]')
    fig.tight_layout()
    fig.savefig(FIG_REL_COV, dpi=150)
    print(f"saved -> {FIG_REL_COV}")

    # ---- Figure 3: posterior standard deviation ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter(np.arange(n_keep), np.sqrt(np.diag(cov_kept)),
                    c=R_apo_kept, s=4, cmap='viridis')
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 1e3)
    ax.set_xlabel('orbit index (sorted by R$_{apo}$)')
    ax.set_ylabel(r'$\sigma_w$  (posterior std)')
    ax.set_title(f'Posterior standard deviation of orbit weights  [{TAG}]')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r'R$_{apo}$ proxy [kpc]')
    fig.tight_layout()
    fig.savefig(FIG_diag, dpi=150)
    print(f"saved -> {FIG_diag}")

    # ---- Figure 1: mean weights ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(np.arange(n_keep), w_kept.mean(axis=0), 
                     yerr=np.percentile(w_kept, [16, 84], axis=0), fmt='o', markersize=4, capsize=2,
                     alpha = 0.3)
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 1e3)
    ax.set_xlabel('orbit index (sorted by R$_{apo}$)')
    ax.set_ylabel(r'$\langle w \rangle$  (posterior mean)')
    ax.set_title(f'Mean orbit weights  [{TAG}]')

    fig.tight_layout()
    fig.savefig(FIG_errorbar, dpi=150)
    print(f"saved -> {FIG_errorbar}")

    sample_plots = w_kept[:, ::50]  # plot a subset of samples for visibility
    fig = corner.corner(sample_plots, labels=[f'w_{i}' for i in range(n_keep)],
                        show_titles=True, title_fmt='.2e', title_kwargs={'fontsize': 8})
    fig.suptitle(f'Corner plot of orbit weights  [{TAG}]', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/orbit_weight_corner_{TAG}.png', dpi=150)
    print(f"saved -> {FIG_DIR}/orbit_weight_corner_{TAG}.png")

if __name__ == '__main__':
    main()
