# PSF Smearing via Transfer Matrix P

## Motivation

In real IFU observations, the Point Spread Function (PSF) scatters light across neighbouring spatial bins. A star at projected position (X, Y) contributes flux not only to its own Voronoi bin but also to adjacent bins. We model this as a linear operation: a transfer matrix **P** applied to the model's binned outputs before comparing to data.

---

## Definition of P

`P[i,j]` = fraction of flux originating from bin `j` that is observed in bin `i` after PSF convolution.

**Properties:**
- `P[i,j] >= 0` (non-negative)
- `sum_i P[i,j] = 1` for all `j` (flux conservation: all light from bin j ends up somewhere)
- `P` is `(N_bins, N_bins)`, typically ~691×691 — small and cheap to apply
- When `sigma_PSF → 0`, `P → I` (identity matrix, no smearing) — **verified exactly**

---

## Three Methods to Compute P

### Option 1: Center-to-center Gaussian (fast, approximate)

Treat each bin as a point at its luminosity-weighted centre.

```python
from scipy.spatial.distance import cdist
d = cdist(bin_centers, bin_centers)
P_raw = np.exp(-0.5 * d**2 / sigma_psf**2)
P = P_raw / P_raw.sum(axis=0, keepdims=True)
```

**Pros:** One line, fast, only needs bin centers.
**Cons:** Approximate — ignores bin area and shape. Degrades significantly for Sigma at large PSF (~16% error at 500 pc vs ~7% for grid method).
**When to use:** Quick test, or if PSF sigma is a free parameter (cheap to recompute).

### Option 2: Grid-based convolution (exact, recommended) — TESTED

Standard approach in IFU data analysis. Discretise the FOV onto a fine regular grid (500×500), convolve bin indicator images with Gaussian, then map back to bins.

```python
from scipy.ndimage import gaussian_filter

# Assign fine grid points to Voronoi bins using bin_mapping
fine_bin_ids = assign_to_voronoi_bins(XX.ravel(), YY.ravel()).reshape(grid_res, grid_res)
sigma_pix = sigma_psf / pixel_size

P = np.zeros((N_bins, N_bins))
for j in range(N_bins):
    indicator_j = (fine_bin_ids == j).astype(float)
    convolved_j = gaussian_filter(indicator_j, sigma=sigma_pix, mode='constant')
    for i in range(N_bins):
        P[i, j] = convolved_j[fine_bin_ids == i].sum()
P = P / P.sum(axis=0, keepdims=True)  # column-normalise
```

**Pros:** Exact for any bin geometry and Gaussian PSF. Standard IFU method.
**Cons:** ~30s to build for 691 bins at grid_res=500. But only done once.
**When to use:** Production runs. Compute once, store in `dict_data['P_psf']`.

### Option 3: Monte Carlo on orbit points (data-driven) — NOT TESTED

Scatter orbit projected positions by PSF, tally which bins they land in.

**Pros:** Naturally weights by surface brightness.
**Cons:** Stochastic, depends on orbit library (can't pre-compute), very slow.
**When to use:** Only if bin geometry is unavailable.

---

## Applying P to Model Outputs

### Surface density (straightforward)

```python
Sigma_obs = P @ Sigma_model
```

### Kinematics (flux-weighted convolution)

**Critical:** Cannot simply do `V_obs = P @ V_model`. Must convolve flux-weighted moments, then divide by convolved flux.

```python
P = dict_data['P_psf']  # pre-computed (N_bins, N_bins) JAX array

Sigma_obs = P @ Sigma_model
V_obs     = (P @ (Sigma_model * V_model)) / Sigma_obs
sigma_obs = jnp.sqrt((P @ (Sigma_model * (V_model**2 + sigma_model**2))) / Sigma_obs - V_obs**2)
h1_obs    = (P @ (Sigma_model * h1_model)) / Sigma_obs
h2_obs    = (P @ (Sigma_model * h2_model)) / Sigma_obs
h3_obs    = (P @ (Sigma_model * h3_model)) / Sigma_obs
h4_obs    = (P @ (Sigma_model * h4_model)) / Sigma_obs
```

**Note on h3, h4:** The linear flux-weighted averaging is an approximation. Strictly, mixing LOSVDs with different (V, σ, h3, h4) does not produce a new LOSVD whose h3, h4 are simply the flux-weighted averages. For small PSF (< bin size), this is standard practice and sufficiently accurate.

### Summary table

| Component | Before P | After P |
|-----------|----------|---------|
| Surface density Σ | `Sigma` | `P @ Sigma` |
| Mean velocity V | `V` | `(P @ (Σ·V)) / (P @ Σ)` |
| Dispersion σ | `σ` | `sqrt((P @ (Σ·(V²+σ²))) / (P @ Σ) - V_obs²)` |
| h1–h4 | `h_i` | `(P @ (Σ·h_i)) / (P @ Σ)` |

All operations are matrix multiplies + element-wise ops → fully JAX-jittable, no changes to the integrator.

---

## Where to Insert in the Pipeline

```
Current:   orbits → binning → [Sigma, V, sigma, h1-h4] per bin → likelihood
With PSF:  orbits → binning → [Sigma, V, sigma, h1-h4] per bin → apply P → likelihood
```

### PSF sigma as a free parameter

- **Option A:** Pre-compute P for a grid of sigma values, store as `P_grid[k, i, j]`, interpolate at runtime.
- **Option B:** Use Option 1 (center-to-center) which is cheap to recompute on-the-fly each likelihood call.

---

## Verification Results

### Test setup

- **Test script:** `test_psf_smearing.py` (no existing functions modified)
- **Data:** `mock_Nbody_bar_XY_withRot.pkl` — 691 Voronoi bins, 60×40 pixel grid, FOV [-10,10]×[-3,3] kpc
- **Pixel size:** dX=0.333 kpc, dY=0.150 kpc
- **Test particles:** 200,000 random points with exponential disk flux profile and rotation velocity pattern
- **P construction:** Option 2 (grid-based convolution), grid_res=500
- **Pipeline A:** Scatter-then-bin (500 MC realisations) — the ground truth
- **Pipeline B:** Bin-then-smear using matrix P

### Sanity check: sigma_PSF = 0

- `P` is **exactly** the identity matrix: `max|P - I| = 0.0`
- Pipeline B output **exactly** matches raw binning: `max|Sigma_B - Sigma_raw| = 0.0`

### MC convergence check

Errors are identical at N_MC=500 and N_MC=2000 (< 0.02% change), confirming the residual is **systematic** (sub-bin spatial information loss), not MC noise.

### Mean relative error (%) — Grid P (Option 2) vs Pipeline A

| sigma_PSF (kpc) | pc | sig/dX | diag_min | diag_mean | Sigma | V | sigma | h3 |
|---|---|---|---|---|---|---|---|---|
| 0.050 | 50 | 0.15 | 0.813 | 0.857 | 2.5 | 2.9 | 1.4 | 2.8 |
| 0.100 | 100 | 0.30 | 0.631 | 0.713 | 3.3 | 3.0 | 1.6 | 2.9 |
| 0.200 | 200 | 0.60 | 0.357 | 0.480 | 4.3 | 3.1 | 3.4 | 3.1 |
| 0.300 | 300 | 0.90 | 0.207 | 0.335 | 5.6 | 3.9 | 7.1 | 4.3 |
| 0.500 | 500 | 1.50 | 0.088 | 0.190 | 7.1 | 7.4 | 15.7 | 8.9 |

### Mean relative error (%) — Center P (Option 1) vs Pipeline A

| sigma_PSF (kpc) | pc | sig/dX | Sigma | V | sigma | h3 |
|---|---|---|---|---|---|---|
| 0.050 | 50 | 0.15 | 2.4 | 3.8 | 1.7 | 3.7 |
| 0.100 | 100 | 0.30 | 3.8 | 3.7 | 1.8 | 3.6 |
| 0.200 | 200 | 0.60 | 8.4 | 3.0 | 1.2 | 2.7 |
| 0.300 | 300 | 0.90 | 10.9 | 5.0 | 0.9 | 4.5 |
| 0.500 | 500 | 1.50 | 16.0 | 11.4 | 1.7 | 10.5 |

### Key findings

1. **Grid P (Option 2) is consistently better than Center P (Option 1)** for surface density, especially at larger PSF: 7% vs 16% at 500 pc.

2. **~2-3% floor at small PSF** is the irreducible error from the transfer matrix losing sub-bin spatial information — the particle distribution within each bin is not uniform (exponential disk), but P assumes it is.

3. **Velocity dispersion (sigma) is most sensitive** — 15.7% error at 500 pc, because mixing bins with different velocities artificially inflates the second moment.

4. **For realistic IFU PSF (~100-200 pc), Grid P gives 3-4% error** on V, h3, h4 — well within typical observational uncertainties (5-10%).

5. **Flux conservation is exact** for all methods and all sigma values.

### Output files

- `psf_smearing_verification.png` — visual comparison of Pipeline A vs B for all observables
- `psf_transfer_matrix_P.png` — visualisation of P matrices (grid vs center)
- `psf_transfer_matrix.pkl` — saved P matrices for reuse

---

## Implementation

### Computing P: `utils.compute_transfer_matrix()`

Standalone function in `utils.py`. Takes Voronoi bin geometry as input, returns `(num_Vbin, num_Vbin)` numpy array.

```python
from utils import compute_transfer_matrix

P = compute_transfer_matrix(
    sigma_psf=0.1,  # kpc
    nX=nX, nY=nY,
    X_minmax=(X_min, X_max),
    Y_minmax=(Y_min, Y_max),
    bin_mapping=bin_mapping,
    total_bins=total_bins,
    grid_res=500,
)
```

Returns identity when `sigma_psf=0`.

### Applying P in the pipeline: `integrate_adaptive_batch_chunked_psf()`

New function in `integrants_with_binning.py`. Wraps the adaptive BS2(3) chunked integrator and applies PSF smearing to the orbit library matrices.

```python
from integrants_with_binning import integrate_adaptive_batch_chunked_psf

P_psf = jnp.array(P)  # convert numpy -> JAX

A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, valid_count = \
    integrate_adaptive_batch_chunked_psf(
        w0, acc_fn, pot_fn, N_max, T_total, ...,
        P_psf=P_psf,
    )
```

**What it does:**
1. Runs the adaptive integrator with chunked binning (same as `integrate_adaptive_batch_chunked`)
2. Transposes per-orbit outputs into orbit library matrices `A_Rzphi`, `A_xy`, `A_h1`–`A_h4`
3. Flux-weights GH moments: `A_hi = A_hi * A_xy`
4. Applies `P_psf @` to all projected quantities: `A_xy`, `A_h1`–`A_h4`
5. Returns matrices directly for the weight solver

**Key design decisions:**
- `A_Rzphi` (3D density) is NOT smeared — it's intrinsic, not projected
- PSF is applied at the orbit library level (once), not per likelihood call
- Flux-weighted convolution is correct: `h_i_obs = (P @ (Sigma * h_i)) @ w / ((P @ Sigma) @ w)`
- Pass `jnp.eye(num_Vbin)` for no smearing (identity = no-op)
- Vmapped version `_integrate_adaptive_batch_chunked_psf_vmap` available for batched use in `model_bar.py`

---

## Normalisation checks

1. **Flux conservation:** `sum(Sigma_obs) = sum(Sigma_model)` — verified exact.
2. **No negative values:** Guaranteed since P >= 0 and Sigma >= 0.
3. **Edge bins:** Light scattered outside the FOV is lost. Column normalisation redistributes it among in-FOV bins.
