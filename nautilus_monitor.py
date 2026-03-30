"""
Monitor a running Nautilus sampler from a separate Colab cell or notebook.

Usage (in a new Colab cell while the sampler is running):
    %run nautilus_monitor.py

Or from terminal:
    python nautilus_monitor.py

This reads the checkpoint file and shows the current posterior estimate.
Safe to run repeatedly — it only reads, never writes.
"""
import sys
import os
import numpy as np

# path = '/content/drive/MyDrive/SchwarMAX-analytic/'
path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
sys.path.append(path)

checkpoint_file = path + 'nautilus_checkpoint.hdf5'
ndim = 13
param_names = [
    'logM_halo', 'logM_disc', 'logM_bar', 'logRs_halo',
    'logRs_disk', 'logHs_disk', 'logL_bar',
    'alpha', 'beta', 'gamma',
    'log_L2M', 'log_Omega', 'log_sigma',
]

# Load prior bounds from saved pickle
import pickle

try:
    with open(path + 'nautilus_posterior_full.pkl', 'rb') as f:
        saved = pickle.load(f)
    prior_low = saved['prior_low']
    prior_high = saved['prior_high']
    print("Loaded prior bounds from saved file")
except (FileNotFoundError, KeyError):
    print("WARNING: Could not load prior bounds. Using fallback.")
    import jax.numpy as jnp
    gt = [11.6, 10.71, 9.96, 1.28, 0.59, 0.02, 0.55,
          0.79, 0.30, 2.36, 0.0, 1.5, -2.0]
    prior_low = np.array([gt[0]-1, gt[1]-0.5, gt[2]-0.5, gt[3]-1, gt[4]-0.5,
                           gt[5]-0.5, gt[6]-0.5, 0, 0, 0, -1, 0.5, -4])
    prior_high = np.array([gt[0]+1, gt[1]+0.5, gt[2]+0.5, gt[3]+1, gt[4]+0.5,
                            gt[5]+0.5, gt[6]+0.5, float(jnp.pi), float(jnp.pi/2),
                            float(jnp.pi), 1, 2, -0.5])

def prior_transform(u):
    return prior_low + (prior_high - prior_low) * u

# ============================================================
if not os.path.exists(checkpoint_file):
    print(f"Checkpoint not found: {checkpoint_file}")
    sys.exit(1)

from nautilus import Sampler

def dummy_likelihood(params):
    return -1e100

sampler = Sampler(
    prior_transform, dummy_likelihood, n_dim=ndim,
    filepath=checkpoint_file, resume=True,
)

print("=" * 70)
print("Nautilus Sampler Status")
print("=" * 70)
print(f"  Checkpoint:  {checkpoint_file}")
print(f"  Total calls: {sampler.n_like:,}")
print(f"  Bounds:      {len(sampler.bounds)}")
print(f"  f_live:      {sampler.f_live}")
print(f"  N_eff:       {sampler.n_eff}")
print(f"  log Z:       {sampler.log_z}")
print()

try:
    points, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()
    n_eff = 1.0 / np.sum(weights**2)

    if n_eff < 2:
        print(f"N_eff = {n_eff:.1f} — not converged yet.")
        print(f"Best logL: {np.max(log_l):.2f}")
        top_idx = np.argsort(log_l)[-10:][::-1]
        n_show = min(13, ndim)
        print(f"\nTop 10 by logL:")
        print(f"  {'logL':>12} | {' | '.join(f'{n:>10}' for n in param_names[:n_show])}")
        for idx in top_idx:
            vals = ' | '.join(f'{points[idx, j]:>10.4f}' for j in range(n_show))
            print(f"  {log_l[idx]:>12.2f} | {vals}")
        raise RuntimeError("Not converged")

    post_mean = np.average(points, weights=weights, axis=0)
    post_std = np.sqrt(np.average((points - post_mean)**2, weights=weights, axis=0))

    print(f"Posterior: {len(points)} points, N_eff = {n_eff:.0f}")
    print()
    print(f"  {'param':>12} | {'mean':>10} | {'std':>10} | {'16%':>10} | {'84%':>10}")
    print(f"  {'-'*58}")

    n_resample = min(50000, max(int(n_eff * 10), 1000))
    idx = np.random.choice(len(points), size=n_resample, p=weights, replace=True)
    resampled = points[idx]

    for i, name in enumerate(param_names):
        q16, q84 = np.percentile(resampled[:, i], [16, 84])
        print(f"  {name:>12} | {post_mean[i]:>10.4f} | {post_std[i]:>10.4f} | {q16:>10.4f} | {q84:>10.4f}")

    print(f"\n  Best logL: {np.max(log_l):.2f}")
    print(f"  log Z:     {sampler.log_z}")

    try:
        import corner
        import matplotlib.pyplot as plt
        fig = corner.corner(resampled, labels=param_names,
                            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f')
        corner_file = path + 'nautilus_corner_intermediate.png'
        fig.savefig(corner_file, dpi=100, bbox_inches='tight')
        print(f"\n  Corner plot saved: {corner_file}")
        plt.show()
        plt.close(fig)
    except ImportError:
        print("\n  Install corner for plots: pip install corner")

except Exception as e:
    print(f"\nCannot show posterior: {e}")
    print("The sampler needs more calls to build a useful posterior.")

progress_log = path + 'nautilus_progress.log'
if os.path.exists(progress_log):
    print()
    print("=" * 70)
    print("Recent progress log:")
    print("=" * 70)
    with open(progress_log) as f:
        lines = f.readlines()
    if len(lines) > 2:
        print(lines[0].rstrip())
        print(lines[1].rstrip())
        for line in lines[-10:]:
            print(line.rstrip())
