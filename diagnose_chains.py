"""
Identify and remove chains that landed in different posterior modes.

The previous version only removed *outlier* chains via MAD on the distance
to the median fingerprint. That misses the more common pathology in this
problem: the 32 chains split into several clusters of similar size, each
sitting in its own posterior mode. In a corner plot those show up as
many small islands.

This version clusters the chain fingerprints (mean log_w per orbit, for
the most active orbits only) using hierarchical clustering on pairwise L2
distances, then keeps the LARGEST cluster.

Outputs
-------
* Printed per-chain table with cluster assignment.
* PNG 1: PCA scatter of chain fingerprints, coloured by cluster.
* PNG 2: dendrogram so you can pick a different cut height if you like.
* Filtered samples pickle (chains in the largest cluster only).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
DATA_FOLDER  = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
TAG          = '0422_beta25_gamma140_D50_gal2'
SAMPLES_IN   = f'{DATA_FOLDER}/orbit_weight_samples_{TAG}.pkl'
SAMPLES_OUT  = f'{DATA_FOLDER}/orbit_weight_samples_{TAG}_mode.pkl'
FIG_PCA      = f'{DATA_FOLDER}/plots/chain_clusters_pca_{TAG}.png'
FIG_DENDRO   = f'{DATA_FOLDER}/plots/chain_clusters_dendro_{TAG}.png'

# Clustering: cut the dendrogram at this fraction of the maximum linkage
# distance. 0.5 -> very generous merging (few large clusters);
# 0.25 -> finer split. If you see in the dendrogram that the natural
# cut is elsewhere, just edit this number.
CUT_FRACTION  = 0.5
TOP_K_ORBITS  = 200        # fingerprint uses the top-K most active orbits only
WRITE_CLEAN   = True       # save a samples pickle restricted to the largest cluster

# Dead-chain detection (only used when potential_energy is in the samples file).
# A chain is flagged DEAD if its terminal logp is much worse than the median
# across chains, or if it has very low acceptance / very many divergences.
DEAD_LOGP_DROP   = 50.0    # flagged if logp_terminal < median - DEAD_LOGP_DROP
DEAD_ACCEPT_MIN  = 0.10    # flagged if mean accept_prob < this
DEAD_DIV_FRAC    = 0.25    # flagged if divergence fraction > this
FIG_HEALTH       = f'{DATA_FOLDER}/plots/chain_health_{TAG}.png'


# =====================================================================
def main():
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import pdist

    print(f"loading samples <- {SAMPLES_IN}")
    with open(SAMPLES_IN, 'rb') as f:
        smp = pickle.load(f)

    log_w = np.asarray(smp['log_w_samples'])               # (chains, samples, n_orb)
    n_chains, n_samples, n_orb = log_w.shape
    print(f"chains={n_chains}  samples/chain={n_samples}  n_orb={n_orb}")

    # ---- Per-chain HMC health (only if extras were captured) ----
    pe       = np.asarray(smp['potential_energy']) if 'potential_energy' in smp else None
    accept   = np.asarray(smp['accept_prob'])      if 'accept_prob'      in smp else None
    n_steps  = np.asarray(smp['num_steps'])        if 'num_steps'        in smp else None
    diverge  = np.asarray(smp['diverging'])        if 'diverging'        in smp else None
    have_extras = pe is not None
    if have_extras:
        # logp = -potential_energy (up to constants)
        logp_per_chain     = -pe
        logp_terminal_med  = np.median(logp_per_chain[:, -max(1, n_samples // 10):], axis=1)
        accept_mean        = accept.mean(axis=1)            if accept  is not None else None
        nsteps_mean        = n_steps.mean(axis=1)           if n_steps is not None else None
        diverge_frac       = diverge.mean(axis=1)           if diverge is not None else None
        print(f"\nHMC health summary (terminal logp = median over last 10% of samples):")
        hdr = f"  {'chain':>5s}  {'logp_term':>10s}"
        if accept is not None:  hdr += f"  {'accept':>7s}"
        if n_steps is not None: hdr += f"  {'mean_steps':>10s}"
        if diverge is not None: hdr += f"  {'div_frac':>9s}"
        print(hdr)
        for c in range(n_chains):
            line = f"  {c:>5d}  {float(logp_terminal_med[c]):>10.3e}"
            if accept is not None:  line += f"  {float(accept_mean[c]):>7.3f}"
            if n_steps is not None: line += f"  {float(nsteps_mean[c]):>10.2f}"
            if diverge is not None: line += f"  {float(diverge_frac[c]):>9.3f}"
            print(line)
    else:
        print("\n[note] no potential_energy in samples file -- "
              "rerun the sampler to capture HMC extras.")

    # ---- Fingerprint: per-chain mean log_w over samples ----
    chain_mu = log_w.mean(axis=1)                           # (chains, n_orb)
    if TOP_K_ORBITS is not None and TOP_K_ORBITS < n_orb:
        consensus = np.median(chain_mu, axis=0)             # (n_orb,)
        active = np.argsort(consensus)[-TOP_K_ORBITS:]
        chain_mu_used = chain_mu[:, active]
        print(f"fingerprint restricted to top-{TOP_K_ORBITS} orbits "
              f"(by median chain-mean log_w)")
    else:
        chain_mu_used = chain_mu

    # ---- Hierarchical clustering on pairwise L2 distances ----
    pair_d = pdist(chain_mu_used, metric='euclidean')
    Z      = linkage(pair_d, method='ward')
    max_d  = float(Z[:, 2].max())
    cut_d  = CUT_FRACTION * max_d
    labels = fcluster(Z, t=cut_d, criterion='distance')     # 1-indexed cluster ids
    cluster_ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)                             # largest cluster first
    cluster_ids = cluster_ids[order]
    counts      = counts[order]
    largest     = cluster_ids[0]

    print(f"\nhierarchical clustering (ward, cut at {CUT_FRACTION:.2f}*max_d "
          f"= {cut_d:.3e}):")
    for cid, n in zip(cluster_ids, counts):
        members = np.where(labels == cid)[0].tolist()
        marker = '   <-- LARGEST (kept)' if cid == largest else ''
        print(f"  cluster {cid}:  {n:2d} chains  {members}{marker}")

    # ---- Dead-chain mask from HMC extras ----
    dead_mask = np.zeros(n_chains, dtype=bool)
    dead_reasons = [''] * n_chains
    if have_extras:
        med_logp = float(np.median(logp_terminal_med))
        for c in range(n_chains):
            reasons = []
            if logp_terminal_med[c] < med_logp - DEAD_LOGP_DROP:
                reasons.append(
                    f'logp={logp_terminal_med[c]:.2e} < med-{DEAD_LOGP_DROP:.0f}')
            if accept_mean is not None and accept_mean[c] < DEAD_ACCEPT_MIN:
                reasons.append(f'accept={accept_mean[c]:.3f}')
            if diverge_frac is not None and diverge_frac[c] > DEAD_DIV_FRAC:
                reasons.append(f'divs={diverge_frac[c]:.2f}')
            if reasons:
                dead_mask[c] = True
                dead_reasons[c] = ', '.join(reasons)
        print(f"\nDEAD chains: {dead_mask.sum()}")
        for c in np.where(dead_mask)[0]:
            print(f"  chain {c}:  {dead_reasons[c]}")

    # Combined keep mask: in the largest cluster AND not dead.
    keep_mask = (labels == largest) & (~dead_mask)
    keep_idx  = np.where(keep_mask)[0]
    drop_idx  = np.where(~keep_mask)[0]
    print(f"\nkept {len(keep_idx)} / {n_chains} chains "
          f"(largest cluster minus dead chains)")

    # Per-chain table sorted by distance to consensus of the kept cluster
    consensus_kept = np.median(chain_mu_used[keep_mask], axis=0)
    dists = np.linalg.norm(chain_mu_used - consensus_kept[None, :], axis=1)
    print(f"\n{'chain':>5s}  {'cluster':>7s}  {'dist to kept median':>20s}")
    for c in np.argsort(dists):
        mark = ' [kept]' if keep_mask[c] else ' [drop]'
        print(f"{c:>5d}  {labels[c]:>7d}  {dists[c]:>20.4e}{mark}")

    # ---- PCA plot, coloured by cluster ----
    Xc = chain_mu_used - chain_mu_used.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    var_explained = (S ** 2) / (S ** 2).sum()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = plt.get_cmap('tab10')
    for j, cid in enumerate(cluster_ids):
        m = labels == cid
        marker = 'o' if cid == largest else 'X'
        size   = 90 if cid == largest else 70
        edge   = 'k' if cid == largest else 'C3'
        ax.scatter(pcs[m, 0], pcs[m, 1],
                   c=[cmap(j % 10)], s=size, marker=marker,
                   edgecolor=edge,
                   label=f'cluster {cid} (n={m.sum()})'
                   + (' [kept]' if cid == largest else ''))
    for c in range(n_chains):
        ax.annotate(str(c), (pcs[c, 0], pcs[c, 1]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
    ax.set_title(f'Chain fingerprints clustered (cut at {CUT_FRACTION:.2f}*max_d)')
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PCA), exist_ok=True)
    fig.savefig(FIG_PCA, dpi=150)
    print(f"\nsaved PCA plot       -> {FIG_PCA}")

    # ---- Dendrogram ----
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, labels=[str(c) for c in range(n_chains)], ax=ax)
    ax.axhline(cut_d, color='C3', ls='--',
               label=f'cut = {CUT_FRACTION:.2f}*max_d = {cut_d:.2e}')
    ax.set_xlabel('chain')
    ax.set_ylabel('linkage distance (ward)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DENDRO, dpi=150)
    print(f"saved dendrogram     -> {FIG_DENDRO}")

    # ---- Health plot: logp trace per chain (only if extras available) ----
    if have_extras:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        ax = axes[0, 0]
        for c in range(n_chains):
            color = 'C3' if dead_mask[c] else ('C0' if keep_mask[c] else 'C7')
            lw    = 1.6 if dead_mask[c] else 0.9
            ax.plot(logp_per_chain[c], color=color, lw=lw, alpha=0.85,
                    label=str(c) if (dead_mask[c] or c < 5) else None)
        ax.set_ylabel('log posterior  (= -PE)')
        ax.set_title('logp trace per chain   (red = dead, blue = kept, grey = other)')
        ax.legend(fontsize=7, ncol=2)

        ax = axes[0, 1]
        if accept is not None:
            ax.plot(accept.T, alpha=0.4)
            ax.set_ylabel('accept_prob')
            ax.set_ylim(0, 1.05)
            ax.axhline(DEAD_ACCEPT_MIN, color='C3', ls='--',
                       label=f'dead < {DEAD_ACCEPT_MIN}')
            ax.legend()
        ax = axes[1, 0]
        if diverge is not None:
            cum_div = np.cumsum(diverge.astype(np.int32), axis=1)
            ax.plot(cum_div.T, alpha=0.6)
            ax.set_ylabel('cumulative divergences')
            ax.set_xlabel('sample index')
        ax = axes[1, 1]
        if n_steps is not None:
            ax.plot(n_steps.T, alpha=0.4)
            ax.set_ylabel('NUTS num_steps')
            ax.set_xlabel('sample index')
            ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(FIG_HEALTH, dpi=150)
        print(f"saved health plot    -> {FIG_HEALTH}")

    # ---- Save filtered samples ----
    if WRITE_CLEAN:
        clean = dict(smp)
        clean['log_w_samples']   = log_w[keep_idx]
        clean['kept_chains']     = keep_idx
        clean['flagged_chains']  = drop_idx
        clean['cluster_labels']  = labels
        clean['cluster_largest'] = int(largest)
        with open(SAMPLES_OUT, 'wb') as f:
            pickle.dump(clean, f)
        print(f"saved filtered samples ({len(keep_idx)} chains) -> {SAMPLES_OUT}")


if __name__ == '__main__':
    main()
