"""
Verify PSF transfer matrix P by comparing two pipelines:
  Pipeline A (ground truth): scatter each point by Gaussian PSF, then bin
  Pipeline B (matrix method): bin first, then apply P

Uses the same data setup as test_integration_convergence.py.
Does NOT modify any existing code.
"""

import numpy as np
import jax.numpy as jnp
import pickle
import time
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

# ── Configuration ────────────────────────────────────────────────────
SIGMA_PSF = 0.1            # 100 pc = 0.1 kpc
N_MC = 500                 # MC realisations for Pipeline A (confirmed sufficient at N_MC=2000)
N_TEST_POINTS = 200_000    # random test particles
RNG_SEED = 12345

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'


# ══════════════════════════════════════════════════════════════════════
# Load data (same pickle as test_integration_convergence.py)
# ══════════════════════════════════════════════════════════════════════
with open(path + 'mock_Nbody_bar_XY_withRot.pkl', 'rb') as f:
    bin_dict = pickle.load(f)

nX, nY = bin_dict['nX_nY']
X_min, X_max = bin_dict['X_minmax']
Y_min, Y_max = bin_dict['Y_minmax']
X_regular_grid = np.array(bin_dict['X_regular_grid'])
Y_regular_grid = np.array(bin_dict['Y_regular_grid'])
bin_mapping = np.array(bin_dict['bin_mapping'])
total_bins = int(bin_dict['total_bins'])
num_per_bin = np.array(bin_dict['num_per_bin'])

dX = (X_max - X_min) / nX
dY = (Y_max - Y_min) / nY

print(f"Grid: {nX} x {nY} pixels, FOV: [{X_min},{X_max}] x [{Y_min},{Y_max}] kpc")
print(f"Pixel size: dX={dX:.4f}, dY={dY:.4f} kpc")
print(f"PSF sigma: {SIGMA_PSF} kpc = {SIGMA_PSF/dX:.1f} pixels")
print(f"Voronoi bins: {total_bins}")


# ══════════════════════════════════════════════════════════════════════
# Compute bin centers from regular grid + bin_mapping
# ══════════════════════════════════════════════════════════════════════
index_remap = bin_mapping[:-1]  # drop sentinel entry
bin_centers = np.zeros((total_bins, 2))
for b in range(total_bins):
    mask = (index_remap == b)
    if mask.sum() > 0:
        bin_centers[b, 0] = X_regular_grid[mask].mean()
        bin_centers[b, 1] = Y_regular_grid[mask].mean()

print(f"Bin center range: X=[{bin_centers[:,0].min():.2f}, {bin_centers[:,0].max():.2f}], "
      f"Y=[{bin_centers[:,1].min():.2f}, {bin_centers[:,1].max():.2f}]")


# ══════════════════════════════════════════════════════════════════════
# Function: assign (X, Y) points to Voronoi bins via the regular grid
# (replicates what the integrator does, but in numpy)
# ══════════════════════════════════════════════════════════════════════
def assign_to_voronoi_bins(X, Y):
    """
    Assign (X, Y) points to Voronoi bins using the regular grid + bin_mapping.
    Returns bin IDs; out-of-bounds points get bin_id = total_bins (sentinel).
    """
    # Normalise to [0, 1]
    nx = (X - X_min) / (X_max - X_min)
    ny = (Y - Y_min) / (Y_max - Y_min)

    # Grid pixel indices
    ix = np.floor(nx * nX).astype(int)
    iy = np.floor(ny * nY).astype(int)

    # Out-of-bounds check
    oob = (nx < 0) | (nx >= 1) | (ny < 0) | (ny >= 1)

    # Clip for indexing safety (oob will be overwritten)
    ix = np.clip(ix, 0, nX - 1)
    iy = np.clip(iy, 0, nY - 1)

    # Flat pixel index (same convention as assign_regular_grid with strides=[1, nX])
    flat_idx = ix + iy * nX
    bin_ids = bin_mapping[flat_idx]

    # Mark out-of-bounds as sentinel
    bin_ids[oob] = total_bins
    return bin_ids


# ══════════════════════════════════════════════════════════════════════
# Compute transfer matrix P using grid-based convolution (Option 2)
# ══════════════════════════════════════════════════════════════════════
def compute_transfer_matrix_grid(sigma_psf, grid_res=500):
    """
    Compute P by convolving bin indicator images on a fine grid.
    Uses the Voronoi bin structure from the loaded data.
    """
    print(f"\nBuilding transfer matrix P (grid_res={grid_res})...")
    t0 = time.time()

    # Fine regular grid (higher res than the Voronoi pixel grid)
    x_edges = np.linspace(X_min, X_max, grid_res + 1)
    y_edges = np.linspace(Y_min, Y_max, grid_res + 1)
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    XX, YY = np.meshgrid(x_c, y_c)  # (grid_res, grid_res)

    # Assign fine grid points to Voronoi bins
    fine_bin_ids = assign_to_voronoi_bins(XX.ravel(), YY.ravel()).reshape(grid_res, grid_res)

    # PSF sigma in fine-grid pixel units
    pixel_size = x_edges[1] - x_edges[0]
    sigma_pix = sigma_psf / pixel_size
    print(f"  Fine pixel size: {pixel_size:.5f} kpc, sigma_pix: {sigma_pix:.2f} pixels")

    # Build P by convolving indicator images
    P = np.zeros((total_bins, total_bins))
    for j in range(total_bins):
        indicator_j = (fine_bin_ids == j).astype(float)
        convolved_j = gaussian_filter(indicator_j, sigma=sigma_pix, mode='constant')
        for i in range(total_bins):
            mask_i = (fine_bin_ids == i)
            P[i, j] = convolved_j[mask_i].sum()

    # Column-normalise (flux conservation)
    col_sums = P.sum(axis=0)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    P = P / col_sums[None, :]

    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")
    print(f"  P shape: {P.shape}, sparsity: {(P < 1e-10).mean():.1%} zeros")
    print(f"  Max off-diagonal: {(P - np.diag(np.diag(P))).max():.6f}")
    print(f"  Min diagonal: {np.diag(P).min():.6f}")

    return P


def compute_transfer_matrix_centers(sigma_psf):
    """Option 1: center-to-center Gaussian (for comparison)."""
    from scipy.spatial.distance import cdist
    d = cdist(bin_centers, bin_centers)
    P_raw = np.exp(-0.5 * d**2 / sigma_psf**2)
    P = P_raw / P_raw.sum(axis=0, keepdims=True)
    return P


# ══════════════════════════════════════════════════════════════════════
# Generate test particles
# ══════════════════════════════════════════════════════════════════════
print(f"\nGenerating {N_TEST_POINTS} test particles...")
rng = np.random.default_rng(RNG_SEED)

# Uniform (X, Y) within FOV with some margin
margin = 0.5  # avoid edge effects in the test
X_pts = rng.uniform(X_min + margin, X_max - margin, N_TEST_POINTS)
Y_pts = rng.uniform(Y_min + margin, Y_max - margin, N_TEST_POINTS)

# Flux weight: exponential disk-like profile (more flux in centre)
R_pts = np.sqrt(X_pts**2 + Y_pts**2)
flux_pts = np.exp(-R_pts / 3.0)  # scale length 3 kpc

# Velocity: rotation pattern + dispersion
V_pts = 200.0 * np.sin(np.arctan2(Y_pts, X_pts)) * (1 - np.exp(-R_pts / 2.0))
V_pts += rng.normal(0, 30.0, N_TEST_POINTS)  # add dispersion

# GH-like h3 pattern (antisymmetric, correlated with V)
h3_pts = 0.1 * np.sign(V_pts) * (1 - np.exp(-np.abs(V_pts) / 100.0))

print(f"  Flux range: [{flux_pts.min():.4f}, {flux_pts.max():.4f}]")
print(f"  V range: [{V_pts.min():.1f}, {V_pts.max():.1f}] km/s")


# ══════════════════════════════════════════════════════════════════════
# Pipeline A: scatter then bin (ground truth)
# ══════════════════════════════════════════════════════════════════════
def pipeline_A_scatter_then_bin(X, Y, flux, V, h3, sigma_psf, n_mc, seed):
    """
    Ground truth: add Gaussian PSF scatter to each point, then bin.
    Accumulates flux-weighted quantities.
    """
    print(f"\nPipeline A: scatter-then-bin ({n_mc} MC realisations)...")
    t0 = time.time()
    rng_mc = np.random.default_rng(seed)

    Sigma_acc = np.zeros(total_bins)
    SigmaV_acc = np.zeros(total_bins)
    SigmaV2_acc = np.zeros(total_bins)
    Sigmah3_acc = np.zeros(total_bins)

    for mc in range(n_mc):
        # Scatter by PSF
        dX_scatter = rng_mc.normal(0, sigma_psf, len(X))
        dY_scatter = rng_mc.normal(0, sigma_psf, len(Y))
        X_scattered = X + dX_scatter
        Y_scattered = Y + dY_scatter

        # Bin the scattered points
        bins = assign_to_voronoi_bins(X_scattered, Y_scattered)

        # Accumulate flux-weighted quantities
        for b in range(total_bins):
            mask = (bins == b)
            f_b = flux[mask]
            Sigma_acc[b] += f_b.sum()
            SigmaV_acc[b] += (f_b * V[mask]).sum()
            SigmaV2_acc[b] += (f_b * V[mask]**2).sum()
            Sigmah3_acc[b] += (f_b * h3[mask]).sum()

        if (mc + 1) % 100 == 0:
            print(f"  MC {mc+1}/{n_mc}")

    # Average over MC realisations
    Sigma_A = Sigma_acc / n_mc
    SigmaV_A = SigmaV_acc / n_mc
    SigmaV2_A = SigmaV2_acc / n_mc
    Sigmah3_A = Sigmah3_acc / n_mc

    # Compute observables
    eps = 1e-30
    V_A = SigmaV_A / (Sigma_A + eps)
    sigma_A = np.sqrt(np.maximum(SigmaV2_A / (Sigma_A + eps) - V_A**2, 0))
    h3_A = Sigmah3_A / (Sigma_A + eps)

    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")
    return Sigma_A, V_A, sigma_A, h3_A


# ══════════════════════════════════════════════════════════════════════
# Pipeline B: bin then apply P (matrix method)
# ══════════════════════════════════════════════════════════════════════
def pipeline_B_bin_then_smear(X, Y, flux, V, h3, P):
    """
    Matrix method: bin original (unscattered) points, then apply P.
    """
    print("\nPipeline B: bin-then-smear (matrix P)...")
    t0 = time.time()

    bins = assign_to_voronoi_bins(X, Y)

    # Accumulate raw (unsmeared) binned quantities
    Sigma_raw = np.zeros(total_bins)
    SigmaV_raw = np.zeros(total_bins)
    SigmaV2_raw = np.zeros(total_bins)
    Sigmah3_raw = np.zeros(total_bins)

    for b in range(total_bins):
        mask = (bins == b)
        f_b = flux[mask]
        Sigma_raw[b] = f_b.sum()
        SigmaV_raw[b] = (f_b * V[mask]).sum()
        SigmaV2_raw[b] = (f_b * V[mask]**2).sum()
        Sigmah3_raw[b] = (f_b * h3[mask]).sum()

    # Apply transfer matrix (flux-weighted convolution)
    Sigma_B = P @ Sigma_raw
    SigmaV_B = P @ SigmaV_raw
    SigmaV2_B = P @ SigmaV2_raw
    Sigmah3_B = P @ Sigmah3_raw

    # Compute observables
    eps = 1e-30
    V_B = SigmaV_B / (Sigma_B + eps)
    sigma_B = np.sqrt(np.maximum(SigmaV2_B / (Sigma_B + eps) - V_B**2, 0))
    h3_B = Sigmah3_B / (Sigma_B + eps)

    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")
    return Sigma_B, V_B, sigma_B, h3_B


# ══════════════════════════════════════════════════════════════════════
# Comparison helper
# ══════════════════════════════════════════════════════════════════════
def compare(name, A, B, mask=None):
    """Report differences between Pipeline A and B."""
    if mask is None:
        mask = np.ones(len(A), dtype=bool)
    a, b = A[mask], B[mask]
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    rel_diff = abs_diff / denom
    print(f"  {name:15s}: max_abs={abs_diff.max():.6e}  mean_abs={abs_diff.mean():.6e}  "
          f"max_rel={rel_diff.max():.4f}  mean_rel={rel_diff.mean():.4f}  "
          f"flux_A={a.sum():.4f}  flux_B={b.sum():.4f}")


# ══════════════════════════════════════════════════════════════════════
# Helper: run both pipelines for a given sigma and return mean relative errors
# ══════════════════════════════════════════════════════════════════════
def get_mean_rel_errors(A_vals, B_vals, good_mask):
    """Return mean relative error over good bins."""
    a, b = A_vals[good_mask], B_vals[good_mask]
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    return np.mean(np.abs(a - b) / denom)


def run_one_sigma(sigma_psf, X, Y, flux, V, h3, n_mc, seed):
    """Run both pipelines for one PSF sigma, return error dict."""
    print(f"\n{'#'*70}")
    print(f"# sigma_PSF = {sigma_psf} kpc ({sigma_psf*1000:.0f} pc)")
    print(f"{'#'*70}")

    # Compute P
    P_grid = compute_transfer_matrix_grid(sigma_psf, grid_res=500)
    P_center = compute_transfer_matrix_centers(sigma_psf)

    # Pipeline A
    Sigma_A, V_A, sigma_A, h3_A = pipeline_A_scatter_then_bin(
        X, Y, flux, V, h3, sigma_psf, n_mc, seed
    )

    # Pipeline B (grid P)
    Sigma_Bg, V_Bg, sigma_Bg, h3_Bg = pipeline_B_bin_then_smear(
        X, Y, flux, V, h3, P_grid
    )

    # Pipeline B (center P)
    Sigma_Bc, V_Bc, sigma_Bc, h3_Bc = pipeline_B_bin_then_smear(
        X, Y, flux, V, h3, P_center
    )

    good = Sigma_A > 0.01 * Sigma_A.max()

    return {
        'sigma_psf': sigma_psf,
        'diag_min': np.diag(P_grid).min(),
        'diag_mean': np.diag(P_grid).mean(),
        # Grid P errors
        'grid_Sigma': get_mean_rel_errors(Sigma_A, Sigma_Bg, good),
        'grid_V': get_mean_rel_errors(V_A, V_Bg, good),
        'grid_sigma': get_mean_rel_errors(sigma_A, sigma_Bg, good),
        'grid_h3': get_mean_rel_errors(h3_A, h3_Bg, good),
        # Center P errors
        'ctr_Sigma': get_mean_rel_errors(Sigma_A, Sigma_Bc, good),
        'ctr_V': get_mean_rel_errors(V_A, V_Bc, good),
        'ctr_sigma': get_mean_rel_errors(sigma_A, sigma_Bc, good),
        'ctr_h3': get_mean_rel_errors(h3_A, h3_Bc, good),
    }


# ══════════════════════════════════════════════════════════════════════
# Main: sweep over PSF sigmas
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    SIGMA_LIST = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]  # kpc

    results = []
    for sig in SIGMA_LIST:
        res = run_one_sigma(sig, X_pts, Y_pts, flux_pts, V_pts, h3_pts, N_MC, RNG_SEED + 1)
        results.append(res)

    # ── Print summary table ──────────────────────────────────────────
    print("\n\n" + "=" * 120)
    print("SUMMARY TABLE: Mean relative error (%) — Grid P vs Pipeline A")
    print("=" * 120)
    print(f"{'sigma_PSF':>12s} {'(pc)':>6s} {'sig/dX':>7s} {'diag_min':>9s} {'diag_mean':>10s} "
          f"{'Sigma':>8s} {'V':>8s} {'sigma':>8s} {'h3':>8s}")
    print("-" * 120)
    for r in results:
        sig = r['sigma_psf']
        print(f"{sig:12.3f} {sig*1000:6.0f} {sig/dX:7.2f} {r['diag_min']:9.4f} {r['diag_mean']:10.4f} "
              f"{r['grid_Sigma']*100:7.2f}% {r['grid_V']*100:7.2f}% "
              f"{r['grid_sigma']*100:7.2f}% {r['grid_h3']*100:7.2f}%")

    print("\n" + "=" * 120)
    print("SUMMARY TABLE: Mean relative error (%) — Center P vs Pipeline A")
    print("=" * 120)
    print(f"{'sigma_PSF':>12s} {'(pc)':>6s} {'sig/dX':>7s} "
          f"{'Sigma':>8s} {'V':>8s} {'sigma':>8s} {'h3':>8s}")
    print("-" * 120)
    for r in results:
        sig = r['sigma_psf']
        print(f"{sig:12.3f} {sig*1000:6.0f} {sig/dX:7.2f} "
              f"{r['ctr_Sigma']*100:7.2f}% {r['ctr_V']*100:7.2f}% "
              f"{r['ctr_sigma']*100:7.2f}% {r['ctr_h3']*100:7.2f}%")

    print("\n" + "=" * 120)
    print("SUMMARY TABLE: Grid P vs Center P — which is closer to Pipeline A?")
    print("=" * 120)
    print(f"{'sigma_PSF':>12s} {'(pc)':>6s}  "
          f"{'Sigma(G)':>9s} {'Sigma(C)':>9s}  "
          f"{'V(G)':>9s} {'V(C)':>9s}  "
          f"{'h3(G)':>9s} {'h3(C)':>9s}")
    print("-" * 120)
    for r in results:
        sig = r['sigma_psf']
        print(f"{sig:12.3f} {sig*1000:6.0f}  "
              f"{r['grid_Sigma']*100:8.2f}% {r['ctr_Sigma']*100:8.2f}%  "
              f"{r['grid_V']*100:8.2f}% {r['ctr_V']*100:8.2f}%  "
              f"{r['grid_h3']*100:8.2f}% {r['ctr_h3']*100:8.2f}%")
