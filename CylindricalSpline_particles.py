"""
Particle-based CylindricalSpline potential expansion.

Given N-body particles (positions + masses), compute the gravitational potential
using azimuthal Fourier decomposition on a cylindrical (R, z) grid, followed by
kernel integration with Legendre Q functions and cubic spline interpolation.

This mirrors the algorithm in CylindricalSpline.py but replaces the density-function
evaluation with CIC (cloud-in-cell) particle deposition.

Algorithm:
  1. Convert particles to cylindrical coordinates (R_i, z_i, phi_i).
  2. CIC-deposit mass * exp(-imφ_i) onto (R, z) grid → rho_m(R, z)
  3. Fit cubic splines to rho_m(R, z) on the grid.
  4. Integrate with the cylindrical Green's function kernel Xi_m to get Phi_m(R, z).
  5. Fit cubic splines to Phi_m(R, z).
  6. Evaluate: Phi(x,y,z) = sum_m [Re(Phi_m) cos(mφ) - Im(Phi_m) sin(mφ)] * (2 - δ_m0)

The output dict is fully compatible with CylindricalSpline.evaluate_phi / get_acc.
"""

import jax
import jax.numpy as jnp
import numpy as np
from constants import G
from CubicSpline import jax_precompute_splines
from CylindricalSpline import (
    _Rz_rescale_forward,
    _xieta_to_Rz_jacobian,
    simpson_weights,
    evaluate_phi,
    evaluate_phi_axisymmetric,
    get_acc,
    get_hessian,
    get_density,
    jax_rho_m_eval,
    m_wrapper,
)


# ---------------------------------------------------------------------------
# Particle deposition → azimuthal Fourier density coefficients
# ---------------------------------------------------------------------------

def compute_rho_m_from_particles(positions, masses, R_grid, Z_grid, M_harmonics,
                                  batch_size=4096, smooth_pixels=0.7):
    """
    Compute azimuthal Fourier coefficients of density from particles using
    CIC (cloud-in-cell) deposition in (R, z) with direct Fourier summation in φ.

    Particles are folded to z≥0 with mass halved, since the kernel integration
    (Xi+ + Xi-) doubles the z≥0 contribution to obtain the full potential.

    Parameters
    ----------
    positions     : (N, 3) array — Cartesian (x, y, z)
    masses        : (N,) array — particle masses
    R_grid        : (NR,) array — radial grid nodes
    Z_grid        : (NZ,) array — vertical grid nodes (z ≥ 0)
    M_harmonics   : (NM,) array — harmonic orders [0, 1, ..., Mmax]
    batch_size    : int — particles processed per batch (memory control)
    smooth_pixels : float — Gaussian smoothing width in grid pixels (0 = no smoothing)

    Returns
    -------
    rho_m : (NM, NR, NZ) complex array
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    R_p = np.sqrt(x**2 + y**2)
    phi_p = np.arctan2(y, x)
    z_p = np.abs(z)

    N = len(masses)
    NR, NZ = len(R_grid), len(Z_grid)
    NM = len(M_harmonics)

    masses_half = masses * 0.5

    # Cell edges and volumes
    R_edges = np.zeros(NR + 1)
    R_edges[1:-1] = 0.5 * (R_grid[:-1] + R_grid[1:])
    R_edges[0] = max(0.0, R_grid[0] - (R_edges[1] - R_grid[0]))
    R_edges[-1] = R_grid[-1] + (R_grid[-1] - R_edges[-2])

    Z_edges = np.zeros(NZ + 1)
    Z_edges[1:-1] = 0.5 * (Z_grid[:-1] + Z_grid[1:])
    Z_edges[0] = max(0.0, Z_grid[0] - (Z_edges[1] - Z_grid[0]))
    Z_edges[-1] = Z_grid[-1] + (Z_grid[-1] - Z_edges[-2])

    dR_cells = np.diff(R_edges)
    dz_cells = np.diff(Z_edges)
    cell_vol = 2 * np.pi * R_grid[:, None] * dR_cells[:, None] * dz_cells[None, :]
    cell_vol = np.maximum(cell_vol, 1e-30)

    rho_m = np.zeros((NM, NR, NZ), dtype=np.complex128)

    # CIC bilinear deposition
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        R_b = R_p[start:end]
        z_b = z_p[start:end]
        phi_b = phi_p[start:end]
        m_b = masses_half[start:end]

        iR = np.clip(np.searchsorted(R_grid, R_b) - 1, 0, NR - 2)
        iz = np.clip(np.searchsorted(Z_grid, z_b) - 1, 0, NZ - 2)

        fR = np.clip((R_b - R_grid[iR]) / (R_grid[iR + 1] - R_grid[iR]), 0.0, 1.0)
        fz = np.clip((z_b - Z_grid[iz]) / (Z_grid[iz + 1] - Z_grid[iz]), 0.0, 1.0)

        w00 = (1 - fR) * (1 - fz)
        w10 = fR * (1 - fz)
        w01 = (1 - fR) * fz
        w11 = fR * fz

        exp_mphi = np.exp(-1j * M_harmonics[:, None] * phi_b[None, :])
        weighted = m_b[None, :] * exp_mphi

        for mi in range(NM):
            np.add.at(rho_m[mi], (iR,     iz),     weighted[mi] * w00)
            np.add.at(rho_m[mi], (iR + 1, iz),     weighted[mi] * w10)
            np.add.at(rho_m[mi], (iR,     iz + 1), weighted[mi] * w01)
            np.add.at(rho_m[mi], (iR + 1, iz + 1), weighted[mi] * w11)

    rho_m /= cell_vol[None, :, :]

    if smooth_pixels > 0:
        from scipy.ndimage import gaussian_filter
        for mi in range(NM):
            rho_m[mi].real = gaussian_filter(rho_m[mi].real, sigma=smooth_pixels)
            rho_m[mi].imag = gaussian_filter(rho_m[mi].imag, sigma=smooth_pixels)

    return rho_m


# ---------------------------------------------------------------------------
# Full pipeline: particles → potential dict
# ---------------------------------------------------------------------------

def get_phi_m_from_particles(positions, masses,
                              NR=30, NZ=30,
                              Rmin=0.01, Rmax=50.0,
                              Zmin=0.01, Zmax=20.0,
                              Mmax=0, N_int=64,
                              smooth_pixels=0.7,
                              verbose=False):
    """
    Compute azimuthal-Fourier potential expansion from N-body particles.

    Parameters
    ----------
    positions     : (N, 3) array — Cartesian (x, y, z) in kpc
    masses        : (N,) array — particle masses in Msun
    NR, NZ        : int — grid points in R, z
    Rmin, Rmax    : float — radial grid extent (kpc)
    Zmin, Zmax    : float — vertical grid extent (kpc)
    Mmax          : int — maximum azimuthal harmonic order
    N_int         : int — integration quadrature points
    smooth_pixels : float — Gaussian smoothing width (grid pixels)
    verbose       : bool — print timing info

    Returns
    -------
    dict_phi_m : dict — same format as CylindricalSpline.get_phi_m(),
                 compatible with evaluate_phi(), get_acc(), etc.
    """
    if verbose:
        import time
        t0 = time.time()

    M = jnp.arange(0, Mmax + 1)
    R = np.geomspace(max(Rmin, 1e-3), Rmax, NR)
    Zpos = np.geomspace(max(Zmin, 1e-3), Zmax, NZ)
    Z_nonneg = np.concatenate([[0.0], Zpos])
    R0 = R[NR // 2]

    # Step 1: Particle deposition
    if verbose:
        print("Computing rho_m from particles...")

    rho_m_np = compute_rho_m_from_particles(
        positions, masses, R, Z_nonneg,
        np.arange(0, Mmax + 1, dtype=float),
        smooth_pixels=smooth_pixels
    )

    if verbose:
        t1 = time.time()
        print(f"  Particle deposition: {t1 - t0:.3f} s")

    rho_m = jnp.array(rho_m_np)

    # Step 2: Spline-fit rho_m on rescaled grid
    _R, _z = _Rz_rescale_forward(jnp.array(R), jnp.array(Z_nonneg), R0)
    NM = Mmax + 1
    NZ_total = len(Z_nonneg)

    rho_real = jnp.zeros((NM, NR, NZ_total))
    rho_img = jnp.zeros((NM, NR, NZ_total))
    Mx_real = jnp.zeros((NM, NR, NZ_total))
    My_real = jnp.zeros((NM, NR, NZ_total))
    Mx_img = jnp.zeros((NM, NR, NZ_total))
    My_img = jnp.zeros((NM, NR, NZ_total))

    for m in range(NM):
        rho_real = rho_real.at[m].set(rho_m[m].real)
        M_x, M_y = jax_precompute_splines((_R, _z), rho_m[m].real)
        Mx_real = Mx_real.at[m].set(M_x)
        My_real = My_real.at[m].set(M_y)
        rho_img = rho_img.at[m].set(rho_m[m].imag)
        M_x, M_y = jax_precompute_splines((_R, _z), rho_m[m].imag)
        Mx_img = Mx_img.at[m].set(M_x)
        My_img = My_img.at[m].set(M_y)

    if verbose:
        t2 = time.time()
        print(f"  Density spline fitting: {t2 - t1:.3f} s")

    # Step 3: Kernel integration
    base = max(9, int(np.sqrt(max(16, N_int))))
    base += abs(base % 2 - 1)
    wxi = simpson_weights(base)
    weta = simpson_weights(base)

    xi = jnp.linspace(0.0, 1.0, base)
    eta = jnp.linspace(0.0, 1.0, base)
    XI, ETA = jnp.meshgrid(xi, eta, indexing="ij")

    R_jnp = jnp.array(R)
    Z_jnp = jnp.array(Z_nonneg)

    Rp, zp, Jmap = _xieta_to_Rz_jacobian(
        XI, ETA, jnp.array([R_jnp[1], Rmax, Z_jnp[1], Zmax])
    )
    W2D = jnp.einsum('i,j->ij', wxi, weta)

    phi_m = jax.vmap(
        m_wrapper,
        in_axes=(0, None, None, None, None, None, None,
                 None, None, None, None, None, None, None, None, None)
    )(
        M.astype(int), R_jnp, Z_jnp, Rp, zp,
        _R, _z, rho_real, rho_img, Mx_real, My_real, Mx_img, My_img,
        R0, Jmap, W2D
    )

    if verbose:
        t3 = time.time()
        print(f"  Kernel integration: {t3 - t2:.3f} s")

    # Step 4: Spline-fit the potential
    phi_real = jnp.zeros((NM, NR, NZ_total))
    phi_img = jnp.zeros((NM, NR, NZ_total))
    phi_Mx_real = jnp.zeros((NM, NR, NZ_total))
    phi_My_real = jnp.zeros((NM, NR, NZ_total))
    phi_Mx_img = jnp.zeros((NM, NR, NZ_total))
    phi_My_img = jnp.zeros((NM, NR, NZ_total))

    for m in range(NM):
        phi_real = phi_real.at[m].set(phi_m[m].real)
        M_x, M_y = jax_precompute_splines((_R, _z), phi_m[m].real)
        phi_Mx_real = phi_Mx_real.at[m].set(M_x)
        phi_My_real = phi_My_real.at[m].set(M_y)
        phi_img = phi_img.at[m].set(phi_m[m].imag)
        M_x, M_y = jax_precompute_splines((_R, _z), phi_m[m].imag)
        phi_Mx_img = phi_Mx_img.at[m].set(M_x)
        phi_My_img = phi_My_img.at[m].set(M_y)

    Phi_m_grid = {
        'Rgrid': _R,
        'Zgrid': _z,
        'R0': jnp.array([R0]),
        'm': M,
        'Phi_m_real': phi_real,
        'Phi_m_img': phi_img,
        'Mx_real': phi_Mx_real,
        'My_real': phi_My_real,
        'Mx_img': phi_Mx_img,
        'My_img': phi_My_img,
    }

    if verbose:
        t4 = time.time()
        print(f"  Potential spline fitting: {t4 - t3:.3f} s")
        print(f"  Total: {t4 - t0:.3f} s")

    return Phi_m_grid


# ---------------------------------------------------------------------------
# AGAMA-assisted pipeline: particles → AGAMA CylSpline → sample Phi_m → JAX dict
# ---------------------------------------------------------------------------

def get_phi_m_from_agama(positions, masses,
                          NR=40, NZ=25,
                          Rmin=0.01, Rmax=20.0,
                          Zmin=0.01, Zmax=8.0,
                          Mmax=8, Nphi=64,
                          agama_gridSizeR=40, agama_gridSizeZ=25,
                          verbose=False):
    """
    Build a JAX-compatible potential dict from N-body particles via AGAMA.

    Uses AGAMA's CylSpline (with penalized B-splines) to compute a smooth
    particle-based potential, then samples Phi_m(R, z) Fourier harmonics
    from that smooth potential on our (R, z) grid and fits cubic splines.

    This gives AGAMA-quality accuracy with JAX-compatible evaluation/differentiation.

    Parameters
    ----------
    positions         : (N, 3) array — Cartesian (x, y, z) in kpc
    masses            : (N,) array — particle masses in Msun
    NR, NZ            : int — grid points in R, z for our spline
    Rmin, Rmax        : float — radial grid extent (kpc)
    Zmin, Zmax        : float — vertical grid extent (kpc)
    Mmax              : int — maximum azimuthal harmonic order
    Nphi              : int — azimuthal sampling points for Fourier extraction
    agama_gridSizeR   : int — AGAMA internal grid R
    agama_gridSizeZ   : int — AGAMA internal grid z
    verbose           : bool — print timing info

    Returns
    -------
    dict_phi_m : dict — same format as CylindricalSpline.get_phi_m(),
                 compatible with evaluate_phi(), get_acc(), etc.
    """
    import agama
    from constants import KPCGYR_TO_KMS

    if verbose:
        import time
        t0 = time.time()

    # Step 1: Build AGAMA CylSpline from particles
    # AGAMA needs consistent units — set to (Msun, kpc, kpc/Gyr)
    agama.setUnits(mass=1, length=1, velocity=KPCGYR_TO_KMS)

    pot_agama = agama.Potential(
        type='CylSpline',
        particles=(positions, masses),
        symmetry='Triaxial',
        mmax=Mmax,
        gridSizeR=agama_gridSizeR,
        gridSizeZ=agama_gridSizeZ,
    )

    if verbose:
        t1 = time.time()
        print(f"  AGAMA CylSpline build: {t1 - t0:.3f} s")

    # Step 2: Set up our (R, z) grid
    M = jnp.arange(0, Mmax + 1)
    R = np.geomspace(max(Rmin, 1e-3), Rmax, NR)
    Zpos = np.geomspace(max(Zmin, 1e-3), Zmax, NZ)
    Z_nonneg = np.concatenate([[0.0], Zpos])
    R0 = R[NR // 2]
    NM = Mmax + 1
    NZ_total = len(Z_nonneg)

    # Step 3: Sample AGAMA potential on (R, z, phi) grid → extract Phi_m(R, z)
    phis = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
    cos_mphi = np.array([np.cos(m * phis) for m in range(NM)])  # (NM, Nphi)
    sin_mphi = np.array([np.sin(m * phis) for m in range(NM)])  # (NM, Nphi)

    phi_m_grid = np.zeros((NM, len(R), NZ_total), dtype=np.complex128)

    for iR, Ri in enumerate(R):
        for iz, zi in enumerate(Z_nonneg):
            # Evaluate AGAMA potential at all phi for this (R, z)
            pts = np.column_stack([
                Ri * np.cos(phis),
                Ri * np.sin(phis),
                np.full(Nphi, zi)
            ])
            vals = pot_agama.potential(pts)  # (Nphi,) in (kpc/Gyr)^2

            # Fourier decomposition: Phi(R,z,phi) = sum_m [a_m cos(m*phi) + b_m sin(m*phi)]
            # a_m = (2/Nphi) * sum vals * cos(m*phi),  b_m = (2/Nphi) * sum vals * sin(m*phi)
            # For m=0: a_0 = (1/Nphi) * sum vals
            # Complex: Phi_m = a_m - i*b_m  (so that Re(Phi_m)*cos + Im... works)
            for m in range(NM):
                a_m = np.sum(vals * cos_mphi[m]) / Nphi
                b_m = np.sum(vals * sin_mphi[m]) / Nphi
                if m > 0:
                    a_m *= 2
                    b_m *= 2
                # Store as complex: real=a_m, imag=-b_m
                # because evaluate_phi does: real*cos(m*phi) - imag*sin(m*phi)
                # and we want: a_m*cos(m*phi) + b_m*sin(m*phi)
                # so: imag = -b_m
                phi_m_grid[m, iR, iz] = a_m - 1j * b_m

    if verbose:
        t2 = time.time()
        print(f"  Fourier sampling ({len(R)}x{NZ_total}x{Nphi}): {t2 - t1:.3f} s")

    # Step 4: For m>0, undo the factor of 2 in the stored values,
    # because evaluate_phi multiplies by 2 for m>0 internally.
    for m in range(1, NM):
        phi_m_grid[m] /= 2.0

    # Step 5: Spline-fit the potential on rescaled grid
    _R, _z = _Rz_rescale_forward(jnp.array(R), jnp.array(Z_nonneg), R0)

    phi_real = jnp.zeros((NM, len(R), NZ_total))
    phi_img = jnp.zeros((NM, len(R), NZ_total))
    phi_Mx_real = jnp.zeros((NM, len(R), NZ_total))
    phi_My_real = jnp.zeros((NM, len(R), NZ_total))
    phi_Mx_img = jnp.zeros((NM, len(R), NZ_total))
    phi_My_img = jnp.zeros((NM, len(R), NZ_total))

    for m in range(NM):
        vals_real = jnp.array(phi_m_grid[m].real)
        vals_imag = jnp.array(phi_m_grid[m].imag)
        phi_real = phi_real.at[m].set(vals_real)
        M_x, M_y = jax_precompute_splines((_R, _z), vals_real)
        phi_Mx_real = phi_Mx_real.at[m].set(M_x)
        phi_My_real = phi_My_real.at[m].set(M_y)
        phi_img = phi_img.at[m].set(vals_imag)
        M_x, M_y = jax_precompute_splines((_R, _z), vals_imag)
        phi_Mx_img = phi_Mx_img.at[m].set(M_x)
        phi_My_img = phi_My_img.at[m].set(M_y)

    Phi_m_dict = {
        'Rgrid': _R,
        'Zgrid': _z,
        'R0': jnp.array([R0]),
        'm': M,
        'Phi_m_real': phi_real,
        'Phi_m_img': phi_img,
        'Mx_real': phi_Mx_real,
        'My_real': phi_My_real,
        'Mx_img': phi_Mx_img,
        'My_img': phi_My_img,
    }

    if verbose:
        t3 = time.time()
        print(f"  Spline fitting: {t3 - t2:.3f} s")
        print(f"  Total: {t3 - t0:.3f} s")

    return Phi_m_dict


# ---------------------------------------------------------------------------
# Benchmark against AGAMA
# ---------------------------------------------------------------------------

def benchmark_against_agama(N_particles=50000, seed=42, NR=25, NZ=25, Mmax=0):
    """
    Generate a mock exponential disk, compute potential with both
    CylindricalSpline_particles and AGAMA CylSpline, and compare.
    """
    import time
    from constants import KPCGYR_TO_KMS

    try:
        import agama
    except ImportError:
        print("AGAMA not available — skipping benchmark.")
        return

    np.random.seed(seed)

    # Mock exponential disk
    Rd, hz = 3.0, 0.3  # kpc
    total_mass = 5e10   # Msun
    R = np.random.exponential(Rd, N_particles)
    phi = np.random.uniform(0, 2 * np.pi, N_particles)
    z = np.random.normal(0, hz, N_particles)
    positions = np.column_stack([R * np.cos(phi), R * np.sin(phi), z])
    masses = np.full(N_particles, total_mass / N_particles)

    print(f"=== Benchmark: {N_particles} particles, NR={NR}, NZ={NZ}, Mmax={Mmax} ===\n")

    # AGAMA
    agama.setUnits(length=1, mass=1, velocity=1)
    t0 = time.time()
    pot_agama = agama.Potential(
        type='CylSpline', particles=(positions, masses),
        symmetry='a', gridSizeR=NR, gridSizeZ=NZ, mmax=Mmax
    )
    t_agama = time.time() - t0
    print(f"AGAMA CylSpline: {t_agama:.3f} s")

    # Ours
    t0 = time.time()
    dict_phi = get_phi_m_from_particles(
        positions, masses, NR=NR, NZ=NZ,
        Rmin=0.01, Rmax=50.0, Zmin=0.01, Zmax=20.0,
        Mmax=Mmax, N_int=64, verbose=True
    )
    t_ours = time.time() - t0
    print(f"Our CylSpline:   {t_ours:.3f} s\n")

    # Test points
    test_R = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    zeros = np.zeros_like(test_R)
    test_pts = np.column_stack([test_R, zeros, zeros])

    phi_agama = pot_agama.potential(test_pts)
    x_t, y_t, z_t = jnp.array(test_R), jnp.zeros(len(test_R)), jnp.zeros(len(test_R))
    phi_ours = np.array(evaluate_phi(x_t, y_t, z_t, dict_phi)) * KPCGYR_TO_KMS**2

    print("=== Potential (z=0 plane) ===")
    print(f"{'R':>6s} {'AGAMA':>14s} {'Ours':>14s} {'Rel.err':>10s}")
    for i in range(len(test_R)):
        rel = abs((phi_ours[i] - phi_agama[i]) / (phi_agama[i] + 1e-30))
        print(f"{test_R[i]:6.1f} {phi_agama[i]:14.1f} {phi_ours[i]:14.1f} {rel:10.2%}")

    # Force comparison
    force_agama = pot_agama.force(test_pts)[:, 0]
    acc_ours = np.array(get_acc(x_t, y_t, z_t, dict_phi))[0] * KPCGYR_TO_KMS**2

    print(f"\n=== Radial force (z=0) ===")
    print(f"{'R':>6s} {'AGAMA':>14s} {'Ours':>14s} {'Rel.err':>10s}")
    for i in range(len(test_R)):
        rel = abs((acc_ours[i] - force_agama[i]) / (abs(force_agama[i]) + 1e-30))
        print(f"{test_R[i]:6.1f} {force_agama[i]:14.1f} {acc_ours[i]:14.1f} {rel:10.2%}")

    print(f"\nSpeed: AGAMA = {t_agama:.3f}s, Ours = {t_ours:.3f}s ({t_ours/t_agama:.1f}x slower)")
    print("Note: Force accuracy is limited by interpolating cubic splines on CIC density.")
    print("AGAMA uses penalized B-splines for smoother density representation.")

    return dict_phi, pot_agama


if __name__ == "__main__":
    benchmark_against_agama(N_particles=100000, NR=25, NZ=25)
