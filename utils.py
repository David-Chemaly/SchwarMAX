import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from constants import G, EPSILON, TWOPI

@jax.jit
def get_mat(x, y, z):
    v1 = jnp.array([0.0, 0.0, 1.0])
    I3 = jnp.eye(3)

    # Create a fixed-shape vector from inputs
    v2 = jnp.array([x, y, z])
    # Normalize v2 in one step
    v2 = v2 / (jnp.linalg.norm(v2) + EPSILON)

    # Compute the angle using a fused dot and clip operation
    angle = jnp.arccos(jnp.clip(jnp.dot(v1, v2), -1.0, 1.0))

    # Compute normalized rotation axis
    v3 = jnp.cross(v1, v2)
    v3 = v3 / (jnp.linalg.norm(v3) + EPSILON)

    # Build the skew-symmetric matrix K for Rodrigues' formula
    K = jnp.array([
        [0, -v3[2], v3[1]],
        [v3[2], 0, -v3[0]],
        [-v3[1], v3[0], 0]
    ])

    sin_angle = jnp.sin(angle)
    cos_angle = jnp.cos(angle)

    # Compute rotation matrix using Rodrigues' formula
    rot_mat = I3 + sin_angle * K + (1 - cos_angle) * jnp.dot(K, K)
    return rot_mat

@jax.jit
def go_to_bar_ref(xv, angle):
    # Rotate contourclockwise with positive angle
    sina, cosa = jnp.sin(angle), jnp.cos(angle)
    x, y, z, vx, vy, vz = xv
    x_new  = x * cosa - y * sina
    y_new  = x * sina + y * cosa
    vx_new = vx * cosa - vy * sina
    vy_new = vx * sina + vy * cosa

    return xv.at[0].set(x_new).at[1].set(y_new).at[3].set(vx_new).at[4].set(vy_new)

@partial(jax.jit, static_argnames=('xlim', 'ylim', 'zlim', 'dx', 'dy', 'dz'))
def histogram3d(x, weights, xlim=(-10, 10), ylim=(-10, 10), zlim=(-3, 3), dx=1.0, dy=1.0, dz=1.0):
    # Define bin edges for each dimension
    x_bins = jnp.arange(xlim[0], xlim[1] + dx, dx)
    y_bins = jnp.arange(ylim[0], ylim[1] + dy, dy)
    z_bins = jnp.arange(zlim[0], zlim[1] + dz, dz)

    bins, _ = jnp.histogramdd(x, bins=[x_bins, y_bins, z_bins], weights=weights)
    return bins

# ---------- helpers ----------
def shift_origin(x, y, z, p):
    # Convert scalar params to arrays matching x,y,z shape
    x0 = jnp.asarray(p["x_origin"])
    y0 = jnp.asarray(p["y_origin"])
    z0 = jnp.asarray(p["z_origin"])

    # Broadcast to match shapes of inputs
    x0 = jnp.broadcast_to(x0, x.shape)
    y0 = jnp.broadcast_to(y0, y.shape)
    z0 = jnp.broadcast_to(z0, z.shape)

    # Stack as a 3-vector field
    return jnp.stack([x - x0, y - y0, z - z0], axis=0)

def rotate_zaxis(vec, p):
    # vec: (3, ...)
    R = get_mat(p["dirx"], p["diry"], p["dirz"])  # (3,3)
    # Tensordot over axis: (i,a) * (a,...) -> (i,...)
    return jnp.tensordot(R, vec, axes=[[1],[0]])

@jax.jit
def getCartesianFromCylindrical_clockwise(R, phi, vR, vphi):
    """
    Reverts R, phi, vR, vphi back to Cartesian x, y, vx, vy.
    Consistent with the 'clockwise' vphi convention provided.
    """
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    
    # 1. Positions
    x = R * cos_phi
    y = R * sin_phi
    
    # 2. Velocities
    # Derived by inverting the linear system from your input function
    vx = vR * cos_phi + vphi * sin_phi
    vy = vR * sin_phi - vphi * cos_phi
    
    return x, y, vx, vy

def Rz(t):
    ct, st = jnp.cos(t), jnp.sin(t)
    return jnp.array([[ct, -st, 0.0],
                     [st,  ct, 0.0],
                     [0.0, 0.0, 1.0]])

def Rx(t):
    ct, st = jnp.cos(t), jnp.sin(t)
    return jnp.array([[1.0, 0.0, 0.0],
                     [0.0,  ct, -st],
                     [0.0,  st,  ct]])

def makeRotationMatrix(alpha, beta, gamma):

    alpha, beta, gamma = jnp.radians(alpha), jnp.radians(beta), jnp.radians(gamma)
    return (Rz(gamma) @ Rx(beta) @ Rz(alpha)).T   # X = R @ x


@partial(jax.jit, static_argnames=("potential_fn",))
def estimate_orbital_timescale(R, potential_fn, potential_args=(), z=0.0, dR=1e-3):
    """
    Order-of-magnitude orbital timescale from a gravitational potential.

    Uses a local circular-orbit estimate:
        Omega^2(R) = (1 / R) * dPhi/dR
        T_orb(R)   = 2*pi / Omega

    Parameters
    ----------
    R : float or array-like
        Cylindrical radius (kpc).
    potential_fn : callable
        Function with signature:
            potential_fn(x, y, z, *potential_args) -> Phi
        and Phi in units of kpc^2 / Gyr^2.
    potential_args : tuple, optional
        Extra arguments forwarded to potential_fn.
    z : float, optional
        Height where dPhi/dR is evaluated (default 0.0).
    dR : float, optional
        Finite-difference step in kpc.

    Returns
    -------
    T_orb : float or jnp.ndarray
        Estimated orbital timescale in Gyr.
    """
    R = jnp.asarray(R)
    R_shape = R.shape
    R_flat = jnp.ravel(R)
    R_safe = jnp.maximum(jnp.abs(R_flat), 2 * dR)
    min_val = 1e-20

    def phi_of_R_scalar(r):
        return potential_fn(r, 0.0, z, *potential_args)

    def dphi_dr_scalar(r):
        return (phi_of_R_scalar(r + dR) - phi_of_R_scalar(r - dR)) / (2.0 * dR)

    dPhi_dR = jax.vmap(dphi_dr_scalar)(R_safe)
    omega2 = jnp.maximum(jnp.abs(dPhi_dR) / R_safe, min_val)
    omega = jnp.sqrt(omega2)

    T_orb = 2 * jnp.pi / omega
    return jnp.reshape(T_orb, R_shape)


@partial(jax.jit, static_argnames=("potential_fn",))
def get_rotation_curve(R, potential_fn, potential_args=(), z=0.0, dR=1e-3):
    """
    Circular speed curve for an axisymmetric potential.

    Uses:
        v_c^2(R, z) = R * dPhi/dR

    Parameters
    ----------
    R : float or array-like
        Cylindrical radius (kpc).
    potential_fn : callable
        Function with signature:
            potential_fn(x, y, z, *potential_args) -> Phi
        and Phi in units of kpc^2 / Gyr^2.
    potential_args : tuple, optional
        Extra arguments forwarded to potential_fn.
    z : float, optional
        Height where dPhi/dR is evaluated (default 0.0).
    dR : float, optional
        Finite-difference step in kpc.

    Returns
    -------
    v_c : float or jnp.ndarray
        Circular speed in kpc / Gyr, with the same shape as R.
    """
    R = jnp.asarray(R)
    R_shape = R.shape
    R_flat = jnp.ravel(R)
    R_safe = jnp.maximum(jnp.abs(R_flat), 2 * dR)

    def phi_of_R_scalar(r):
        return potential_fn(r, 0.0, z, *potential_args)

    def dphi_dr_scalar(r):
        return (phi_of_R_scalar(r + dR) - phi_of_R_scalar(r - dR)) / (2.0 * dR)

    dPhi_dR = jax.vmap(dphi_dr_scalar)(R_safe)
    vc2 = jnp.maximum(R_safe * dPhi_dR, 0.0)
    v_c = jnp.sqrt(vc2)
    return jnp.reshape(v_c, R_shape)

def halo_mass_from_stellar_mass(M_star,
    N=0.0351, log10_M1=11.59, beta=1.376, gamma=0.608,
    mmin=1e9, mmax=3e16, tol=1e-6, max_iter=200):
    """
    Return halo mass M_h [Msun] for a given stellar mass M_star [Msun]
    using the Moster+2013 z=0 SHMR (median relation).
    """
    def mstar_from_mh(Mh):
        x = Mh / (10**log10_M1)
        return 2*N*Mh / (x**(-beta) + x**gamma)

    a, b = mmin, mmax
    for _ in range(max_iter):
        mid = 10**((jnp.log10(a)+jnp.log10(b))/2)
        if mstar_from_mh(mid) > M_star:
            b = mid
        else:
            a = mid
        if abs(jnp.log10(b) - jnp.log10(a)) < tol:
            return 10**((jnp.log10(a)+jnp.log10(b))/2)

    return 10**((jnp.log10(a)+jnp.log10(b))/2)



def XexpX_pdf_log(x, a):
    """
    Probability density function of the distribution proportional to x * exp(-x/a).
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF. Can be scalar or array.
    a : float
        Scale parameter > 0.
    
    Returns
    -------
    pdf : array_like
        The PDF values at x.
    """
    # Ensure a > 0
    a = jnp.asarray(a)
    # PDF formula: (1/a^2) * x * exp(-x/a)
    pdf = jnp.log(x) - jnp.log(a**2) - (x / a)
    return jnp.where(x >= 0, pdf, -jnp.inf)

def expX_pdf_log(x, a):
    """
    Probability density function of the distribution proportional to exp(-x/a).
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF. Can be scalar or array.
    a : float
        Scale parameter > 0.
    
    Returns
    -------
    pdf : array_like
        The PDF values at x.
    """
    # Ensure a > 0
    a = jnp.asarray(a)
    # PDF formula: (1/a^2) * x * exp(-x/a)
    pdf = jnp.log(a) - (x / a)
    return jnp.where(x >= 0, pdf, -jnp.inf)

def compute_transfer_matrix(sigma_psf, nX, nY, X_minmax, Y_minmax,
                             bin_mapping, total_bins, grid_res=500):
    """
    Compute PSF transfer matrix P using grid-based convolution (Option 2).

    P[i,j] = fraction of flux from Voronoi bin j observed in bin i after
    Gaussian PSF convolution. Columns sum to 1 (flux conservation).

    Parameters
    ----------
    sigma_psf : float
        PSF standard deviation in kpc. If 0, returns identity matrix.
    nX, nY : int
        Number of pixels in the regular grid (X and Y directions).
    X_minmax, Y_minmax : tuple of (float, float)
        FOV limits (min, max) in kpc for X and Y.
    bin_mapping : array of int, shape (nX*nY,) or (nX*nY+1,)
        Maps each regular grid pixel to a Voronoi bin ID.
        If length nX*nY+1, the last entry is treated as a sentinel and dropped.
    total_bins : int
        Number of Voronoi bins.
    grid_res : int, optional
        Resolution of the fine grid used for convolution (default 500).

    Returns
    -------
    P : ndarray, shape (total_bins, total_bins)
        PSF transfer matrix.
    """
    from scipy.ndimage import gaussian_filter

    bin_mapping = np.asarray(bin_mapping)
    if len(bin_mapping) == nX * nY + 1:
        bin_mapping = bin_mapping[:-1]  # drop sentinel

    X_min, X_max = X_minmax
    Y_min, Y_max = Y_minmax

    if sigma_psf == 0:
        return np.eye(total_bins)

    # Fine regular grid covering the FOV
    x_edges = np.linspace(X_min, X_max, grid_res + 1)
    y_edges = np.linspace(Y_min, Y_max, grid_res + 1)
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    XX, YY = np.meshgrid(x_c, y_c)  # (grid_res, grid_res)

    # Assign fine grid points to Voronoi bins via the regular grid
    nx = (XX.ravel() - X_min) / (X_max - X_min)
    ny = (YY.ravel() - Y_min) / (Y_max - Y_min)
    ix = np.clip(np.floor(nx * nX).astype(int), 0, nX - 1)
    iy = np.clip(np.floor(ny * nY).astype(int), 0, nY - 1)
    fine_bin_ids = bin_mapping[ix + iy * nX].reshape(grid_res, grid_res)

    # PSF sigma in fine-grid pixel units
    pixel_size = x_edges[1] - x_edges[0]
    sigma_pix = sigma_psf / pixel_size

    # Build P by convolving indicator images
    P = np.zeros((total_bins, total_bins))
    for j in range(total_bins):
        indicator_j = (fine_bin_ids == j).astype(float)
        convolved_j = gaussian_filter(indicator_j, sigma=sigma_pix, mode='constant')
        for i in range(total_bins):
            P[i, j] = convolved_j[fine_bin_ids == i].sum()

    # Column-normalise (flux conservation)
    col_sums = P.sum(axis=0)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    P = P / col_sums[None, :]

    return P


@partial(jax.jit,static_argnums=(2))
def sample_from_logP(x_grid, logP, N, key):
    """
    Draw N samples from the distribution defined by logP on the grid x_grid
    using the inverse‐CDF method.
    """
    # 1) Shift & exponentiate for numerical stability
    logP = jnp.asarray(logP)
    logP = logP - jnp.max(logP)
    P = jnp.exp(logP)

    # 2) Normalize to get a proper probability mass on the grid
    P /= P.sum()

    # 3) Build the CDF
    cdf = jnp.cumsum(P)

    # 4) Sample uniforms and invert the CDF via linear interpolation
    # jax_random_key2 = jax.random.PRNGKey(random_seed)
    u = jax.random.uniform(key, shape=(N,))
    samples = jnp.interp(u, cdf, x_grid)
    return samples


# ── NFW parameter transforms (JAX-compatible) ──────────────────────

@jax.jit
def logM_logRs_to_logMenc_logc(logM_halo, logRs_halo, r_enc=10.0, Delta=200., rho_crit=277.54):
    """NFW (logM, logRs) -> (logM_enc(<r_enc), log_concentration)."""
    M = 10.0 ** logM_halo
    Rs = 10.0 ** logRs_halo
    x = r_enc / Rs
    M_enc = M * (jnp.log(1.0 + x) - x / (1.0 + x))
    R_vir = (3.0 * M / (4.0 * jnp.pi * Delta * rho_crit)) ** (1.0 / 3.0)
    c = R_vir / Rs
    return jnp.log10(M_enc), jnp.log10(c)


@jax.jit
def logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54):
    """Inverse: (logM_enc(<r_enc), log_concentration) -> (logM, logRs).

    Solves M_enc = M * [ln(1+x) - x/(1+x)] where x = r_enc/Rs and
    Rs = R_vir/c = (3M/(4 pi Delta rho_crit))^(1/3) / c.

    Uses Newton's method (6 iterations, quadratic convergence).
    """
    M_enc = 10.0 ** logM_enc
    c = 10.0 ** log_c
    coeff = 4.0 * jnp.pi * Delta * rho_crit / 3.0
    ln10 = jnp.log(10.0)

    def _residual_and_deriv(logM_trial):
        M = 10.0 ** logM_trial
        Rs = (M / coeff) ** (1.0 / 3.0) / c
        x = r_enc / Rs
        f_nfw = jnp.log(1.0 + x) - x / (1.0 + x)
        f = M * f_nfw - M_enc

        # df/d(logM) via chain rule:
        #   d/d(logM) = d/dM * dM/d(logM) = d/dM * M * ln(10)
        #   dRs/dM = Rs / (3M)
        #   dx/dM = -x / (3M)
        #   d(f_nfw)/dx = x / (1+x)^2
        #   d(M*f_nfw)/dM = f_nfw + M * d(f_nfw)/dx * dx/dM
        #                 = f_nfw - x^2 / (3*(1+x)^2)
        df_dM = f_nfw - x * x / (3.0 * (1.0 + x) ** 2)
        df = df_dM * M * ln10
        return f, df

    # Initial guess: logM_enc + 0.5 (M is typically 1–10x M_enc)
    logM = logM_enc + 0.5

    def newton_step(logM, _):
        f, df = _residual_and_deriv(logM)
        return logM - f / df, None

    logM_sol, _ = jax.lax.scan(newton_step, logM, None, length=6)

    M_sol = 10.0 ** logM_sol
    Rs_sol = (M_sol / coeff) ** (1.0 / 3.0) / c
    return logM_sol, jnp.log10(Rs_sol)