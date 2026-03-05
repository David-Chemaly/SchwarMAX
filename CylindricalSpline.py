import jax
import jax.numpy as jnp
import numpy as np
import pickle
from constants import *
from CubicSpline import *
from functools import partial

@jax.jit
def cylindrical_to_cartesian(R, phi, z):
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    return x, y, z

# @partial(jax.jit, static_argnames=['rho_fn'])
# def rho_Rzphi(R, z, phi, rho_fn, params):
#     z = jnp.abs(z)  # even symmetry
#     x, y, zz = cylindrical_to_cartesian(R, phi, z)
#     return rho_fn(x, y, zz, params)

# @partial(jax.jit, static_argnames=['rho_fn'])
# def rho_last(R, z, m, phi, Nphi, rho_fn, params):
#     dphi = (2*jnp.pi) / Nphi
#     vals = rho_Rzphi(R, z, phi, rho_fn, params)
#     exp_ph = jnp.exp(-1j * m * phi)
#     rho_m_stack = vals * exp_ph * dphi / (2.0 * jnp.pi)
#     return rho_m_stack

# @partial(jax.jit, static_argnames=['rho_fn'])
# def rho_phiZRm(R, z, m, phi, rho_fn, params):
#     return jnp.sum(jax.vmap(rho_last, in_axes=(None, None, None, 0, None, None))(R, z, m, phi, rho_fn, params), axis=0)

# @partial(jax.jit, static_argnames=['rho_fn'])
# def compute_rho_m(R, z, m, phi, rho_fn, params):
#     return jax.vmap(rho_phiZRm, in_axes=(None, None, 0, None, None, None))(R, z, m, phi, rho_fn, params)

@partial(jax.jit, static_argnames=["rho_fn"])
def rho_Rzphi(R, z, phi, rho_fn, params):
    """
    R, z: shapes (NR, NZ, 1)
    phi: shape (1, 1, Nφ)
    Output must be (NR, NZ, Nφ)
    """

    z = jnp.abs(z)

    # IMPORTANT: broadcast z to have the φ dimension
    z = jnp.broadcast_to(z, R.shape[:-1] + phi.shape[-1:])   # (NR,NZ,Nφ)

    x = R * jnp.cos(phi)    # (NR,NZ,Nφ)
    y = R * jnp.sin(phi)    # (NR,NZ,Nφ)

    return rho_fn(x, y, z, params)

@partial(jax.jit, static_argnames=['rho_fn'])
def compute_rho_m(R, z, M, phi, rho_fn, params):
    """
    R, z: scalars or broadcastable arrays
    M:   (NM,)
    phi: (Nphi,)
    rho_fn: function  (x,y,z,params) -> density
    params: pytree
    """

    # Compute ρ(R,z,φ) on full φ-grid
    rho_phi = rho_Rzphi(R[..., None],      # (R,z) dims + 1
                        z[..., None],
                        phi[None, None, :], # (1,1,Nphi)
                        rho_fn, params)     # -> (Rdim, Zdim, Nphi)

    # exp(-i m φ)
    exp_mphi = jnp.exp(-1j * M[:, None] * phi[None, :])   # (NM, Nphi)

    # Broadcast to combine:
    # rho   -> (Rdim, Zdim, 1,   Nphi)
    # exp   -> (1,    1,    NM, Nphi)
    rho_b = rho_phi[..., None, :]                # (Rdim, Zdim, 1, Nphi)
    exp_b = exp_mphi[None, None, :, :]           # (1,    1,    NM, Nphi)

    # Integral over φ
    dphi = (2.0 * jnp.pi) / phi.size
    rho_m = jnp.sum(rho_b * exp_b, axis=-1) * dphi / (2.0 * jnp.pi)

    # Output shape: (Rdim, Zdim, NM)
    return rho_m

@jax.jit
def jax_rho_m_eval(m, R, z, Rgrid, Zgrid, rho_real, rho_img, Mx_real, My_real, Mx_img, My_img, R0):
    real_values = rho_real[m]
    imag_values = rho_img[m]
    M_x_real = Mx_real[m]
    M_y_real = My_real[m]
    M_x_imag = Mx_img[m]
    M_y_imag = My_img[m]

    shape = R.shape
    _R, _z = _Rz_rescale_forward(R, jnp.abs(z), R0)
    pts = jnp.column_stack((_R.ravel(), _z.ravel()))

    real_part = cubic_spline_evaluate(pts, (Rgrid, Zgrid), real_values, M_x_real, M_y_real, fill_value=0.0).reshape(shape)
    imag_part = cubic_spline_evaluate(pts, (Rgrid, Zgrid), imag_values, M_x_imag, M_y_imag, fill_value=0.0).reshape(shape)

    return real_part + 1j * imag_part

@jax.jit
def jax_hypergeom_m(m, x):
    """
    m: int,
    x: array-like
    """

    y = 1.0 - x
    y2 = y*y
    z = jnp.log(jnp.where(y > 1e-12, y, 1e-12))

    HYPERGEOM_0_m = HYPERGEOM_0[m]
    HYPERGEOM_I_m = HYPERGEOM_I[m]
    HYPERGEOM_1_m = HYPERGEOM_1[m]

    xA8_1 = x + HYPERGEOM_0_m[8]
    xA6_1 = x + HYPERGEOM_0_m[6] + HYPERGEOM_0_m[7] / xA8_1
    xA4_1 = x + HYPERGEOM_0_m[4] + HYPERGEOM_0_m[5] / xA6_1
    xA2_1 = x + HYPERGEOM_0_m[2] + HYPERGEOM_0_m[3] / xA4_1
    val_1 = HYPERGEOM_0_m[0] + HYPERGEOM_0_m[1] / xA2_1

    xA8_2 = x + HYPERGEOM_I_m[8]
    xA6_2 = x + HYPERGEOM_I_m[6] + HYPERGEOM_I_m[7] / xA8_2
    xA4_2 = x + HYPERGEOM_I_m[4] + HYPERGEOM_I_m[5] / xA6_2
    xA2_2 = x + HYPERGEOM_I_m[2] + HYPERGEOM_I_m[3] / xA4_2
    val_2 = HYPERGEOM_I_m[0] + HYPERGEOM_I_m[1] / xA2_2

    val3 = (HYPERGEOM_1_m[0] + HYPERGEOM_1_m[1]*z +
                (HYPERGEOM_1_m[2] + HYPERGEOM_1_m[3]*z) * y +
                (HYPERGEOM_1_m[4] + HYPERGEOM_1_m[5]*z + 
                (HYPERGEOM_1_m[6] + HYPERGEOM_1_m[7]*z) * y + 
                (HYPERGEOM_1_m[8] + HYPERGEOM_1_m[9]*z) * y2) * y2)

    F = jnp.where(x < X_THRESHOLD1[m],
                    jnp.where(x < X_THRESHOLD0[m], val_1, val_2),
                    val3)

    return F

@jax.jit
def jax_legendreQ(n, x):
    """
    n: float,
    x: array-like
    """

    x = jnp.where(x < 1.0, 1.0, x)
    out = jnp.empty_like(x)
    m = jnp.round(n + 0.5).astype(jnp.int32)

    pref = Q_PREFACTOR[m] / jnp.sqrt(x) / (x**m)
    F = jax_hypergeom_m(m, 1.0/(x*x))
    out = pref * F

    return out

@jax.jit
def jax_kernel_Xi_m(m, R, z, Rp, zp):

    """
    m: int,
    R: float,
    z: float,
    Rp: array-like,
    zp: array-like
    """
    zeros = jnp.zeros_like(Rp, dtype=float)

    val1 = zeros
    val2 = 1.0 / jnp.sqrt(R*R + Rp*Rp + (z - zp)**2)

    val_zero = jax.lax.cond(m>0, lambda: val1, lambda: val2)
    
    Rp_reg = Rp
    dz = (z - zp)
    chi = (R*R + Rp_reg*Rp_reg + dz*dz) / (2.0 * R * Rp_reg)
    chi = jnp.maximum(chi, 1.0)
    Q = jax_legendreQ(m - 0.5, chi)
    val_nonzero = (1.0 / (jnp.pi * jnp.sqrt(R * Rp_reg))) * Q


    val_out = jnp.where(Rp<1e-3, val_zero, val_nonzero)

    out = jax.lax.cond(R < 1e-3, lambda: val_zero, lambda: val_out)

    return out

@partial(jax.jit, static_argnames=['n'])
def simpson_weights(n):
    w = jnp.ones(n)
    w = w.at[1:-1:2].set(4.0)
    w = w.at[2:-1:2].set(2.0)
    w *= (1.0 / (n - 1)) / 3.0   # h = 1/(n-1), scale by h/3
    return w

@jax.jit
def _xieta_to_Rz_jacobian(xi, eta, Rzminmax):
    Rmin_map = Rzminmax[0]
    Rmax_map = Rzminmax[1]
    zmin_map = Rzminmax[2]
    zmax_map = Rzminmax[3]

    # Precompute logs
    LR = jnp.log(1.0 + Rmax_map / Rmin_map)
    LZ = jnp.log(1.0 + zmax_map / zmin_map)

    # Map to physical coordinates
    pR = jnp.power(1.0 + Rmax_map / Rmin_map, xi)
    pZ = jnp.power(1.0 + zmax_map / zmin_map, eta)
    Rp = Rmin_map * (pR - 1.0)
    zp = zmin_map * (pZ - 1.0)

    # Jacobian part from the coordinate transform (no 2πR' here)
    dR_dxi  = LR * (Rmin_map + Rp)
    dz_deta = LZ * (zmin_map + zp)
    J = dR_dxi * dz_deta
    return Rp, zp, J

@jax.jit
def _Rz_rescale_forward(R, z, R0):
    """
    Forward map:
        R̃ = asinh(R/R0)
        z̃ = asinh(z/R0)
    Works with scalars or arrays.
    """
    R_tilde = jnp.arcsinh(R / R0)
    z_tilde = jnp.arcsinh(z / R0)
    return R_tilde, z_tilde

@jax.jit
def _Rz_rescale_inverse(R_tilde, z_tilde, R0):
    """
    Forward map:
        R̃ = asinh(R/R0)
        z̃ = asinh(z/R0)
    Works with scalars or arrays.
    """
    R = R0 * jnp.sinh(R_tilde)
    z = R0 * jnp.sinh(z_tilde)
    return R, z

@jax.jit
def m_wrapper(m, R0, z0, Rp, zp, R, Z_nonneg, rho_real, rho_img, Mx_real, My_real, Mx_img, My_img, R_scale, Jmap, W2D):
    rho_grid = jax_rho_m_eval(m.astype(int), Rp, zp, R, Z_nonneg, rho_real, rho_img, Mx_real, My_real, Mx_img, My_img, R_scale)

    return jax.vmap(R_wrapper, in_axes=(None, 0, None, None, None, None, None, None))(m, R0, z0, Rp, zp, rho_grid, Jmap, W2D)

@jax.jit
def R_wrapper(m, R0, z0, Rp, zp, rho_grid, Jmap, W2D):
    return jax.vmap(Z_wrapper, in_axes=(None, None, 0, None, None, None, None, None))(m, R0, z0, Rp, zp, rho_grid, Jmap, W2D)

@jax.jit
def Z_wrapper(m, R0, z0, Rp, zp, rho_grid, Jmap, W2D):
    Xi_plus  = jax_kernel_Xi_m(m, R0, z0, Rp, zp)
    Xi_minus = jax_kernel_Xi_m(m, R0, z0, Rp, -zp)
    Xi_sum   = Xi_plus + Xi_minus

    F = rho_grid * Xi_sum * (2.0 * jnp.pi) * Rp * Jmap

    I = jnp.sum(W2D * F)

    return -G * I

@partial(jax.jit, static_argnames=['rho_fn', 'NR', 'NZ', 'Rmin', 'Rmax', 'Zmin', 'Zmax', 'Mmax', 'Nphi', 'N_int'])
def get_phi_m(rho_fn, params, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int):
    M = jnp.arange(0, Mmax + 1)

    R = jnp.geomspace(jnp.maximum(Rmin, 1e-3), Rmax, NR)

    Zpos = jnp.geomspace(jnp.maximum(Zmin, 1e-3), Zmax, NZ)
    Z_nonneg = jnp.concatenate([jnp.array([0.0]), Zpos])

    R0 = R[NR//2]

    Rg, Zg = jnp.meshgrid(R, Z_nonneg, indexing="ij")
    phi = jnp.linspace(0.0, 2*jnp.pi, Nphi, endpoint=False)

    # rho_m = jax.vmap(compute_rho_m, in_axes=(0, 0, None, None, None, None))(Rg, Zg, M, phi, rho_fn, params).transpose(1,0,2) 
    rho_m = compute_rho_m(Rg, Zg, M, phi, rho_fn, params).transpose(2, 0 ,1) 

    _R, _z = _Rz_rescale_forward(R, Z_nonneg, R0)

    # print(Rg.shape, Zg.shape, _R.shape, _z.shape, rho_m.shape)

    rho_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    rho_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    Mx_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    My_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    Mx_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    My_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    for m in M.astype(int):
        rho_real = rho_real.at[m].set(rho_m[m].real)
        M_x, M_y = jax_precompute_splines((_R, _z), rho_m[m].real)
        Mx_real = Mx_real.at[m].set(M_x)
        My_real = My_real.at[m].set(M_y)
        rho_img = rho_img.at[m].set(rho_m[m].imag)
        M_x, M_y = jax_precompute_splines((_R, _z), rho_m[m].imag)
        Mx_img = Mx_img.at[m].set(M_x)
        My_img = My_img.at[m].set(M_y)

    base  = np.maximum(9, np.sqrt(np.maximum(16, N_int)).astype(int)).astype(int)
    base += np.abs(base % 2 - 1).astype(int)  # make it odd

    n_xi = base
    n_eta = base
    wxi  = simpson_weights(n_xi)
    weta = simpson_weights(n_eta)

    xi  = jnp.linspace(0.0, 1.0, n_xi)
    eta = jnp.linspace(0.0, 1.0, n_eta)
    XI, ETA = jnp.meshgrid(xi, eta, indexing="ij")
    Rp, zp, Jmap = _xieta_to_Rz_jacobian(XI, ETA, jnp.array([R[1], Rmax, Z_nonneg[1], Zmax])) 
    W2D = jnp.einsum('i,j->ij', wxi, weta)

    phi_m = jax.vmap(m_wrapper, in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)) \
                (M.astype(int), R, Z_nonneg, Rp, zp, _R, _z, rho_real, rho_img, Mx_real, My_real, Mx_img, My_img, R0, Jmap, W2D)

    phi_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    phi_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    phi_Mx_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    phi_My_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    phi_Mx_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    phi_My_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
    for m in M.astype(int):
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

    return Phi_m_grid

@jax.jit
def evaluate_phi(x, y, z, dict_phi_m):

    @jax.jit
    def evaluate_phi_m(m, x, y, z, dict_phi_m):
        R, ph = jnp.sqrt(x*x + y*y), jnp.arctan2(y, x)
        real_values = dict_phi_m['Phi_m_real'][m]
        imag_values = dict_phi_m['Phi_m_img'][m]
        M_x_real    = dict_phi_m['Mx_real'][m]
        M_y_real    = dict_phi_m['My_real'][m]
        M_x_imag    = dict_phi_m['Mx_img'][m]
        M_y_imag    = dict_phi_m['My_img'][m]
        R0 = dict_phi_m['R0'][0]

        shape = R.shape
        _R, _z = _Rz_rescale_forward(R, jnp.abs(z), R0)
        pts = jnp.column_stack((_R.ravel(), _z.ravel()))

        real_part = cubic_spline_evaluate(pts, (dict_phi_m['Rgrid'], dict_phi_m['Zgrid']), real_values, M_x_real, M_y_real, fill_value=0.0).reshape(shape)
        imag_part = cubic_spline_evaluate(pts, (dict_phi_m['Rgrid'], dict_phi_m['Zgrid']), imag_values, M_x_imag, M_y_imag, fill_value=0.0).reshape(shape)

        phi_m = real_part + 1j * imag_part

        val = (phi_m.real * jnp.cos(m*ph) - phi_m.imag * jnp.sin(m*ph))

        val_final = jax.lax.cond(m==0, lambda: val, lambda: val*2)

        return val_final
    
    phi_grid = jax.vmap(evaluate_phi_m, in_axes=(0, None, None, None, None))\
                            (dict_phi_m['m'].astype(int), x, y, z, dict_phi_m)
    
    return jnp.sum(phi_grid, axis=0)

@jax.jit
def evaluate_phi_axisymmetric(x, y, z, dict_phi_m):

    m = 0
    R, ph = jnp.sqrt(x*x + y*y), jnp.arctan2(y, x)
    real_values = dict_phi_m['Phi_m_real'][m]
    imag_values = dict_phi_m['Phi_m_img'][m]
    M_x_real    = dict_phi_m['Mx_real'][m]
    M_y_real    = dict_phi_m['My_real'][m]
    M_x_imag    = dict_phi_m['Mx_img'][m]
    M_y_imag    = dict_phi_m['My_img'][m]
    R0 = dict_phi_m['R0'][0]

    shape = R.shape
    _R, _z = _Rz_rescale_forward(R, jnp.abs(z), R0)
    pts = jnp.column_stack((_R.ravel(), _z.ravel()))

    real_part = cubic_spline_evaluate(pts, (dict_phi_m['Rgrid'], dict_phi_m['Zgrid']), real_values, M_x_real, M_y_real, fill_value=0.0).reshape(shape)
    imag_part = cubic_spline_evaluate(pts, (dict_phi_m['Rgrid'], dict_phi_m['Zgrid']), imag_values, M_x_imag, M_y_imag, fill_value=0.0).reshape(shape)

    phi_m = real_part + 1j * imag_part

    val = (phi_m.real * jnp.cos(m*ph) - phi_m.imag * jnp.sin(m*ph))

    val_final = jax.lax.cond(m==0, lambda: val, lambda: val*2)
    
    return val_final


@jax.jit
def _evaluate_phi_and_scaled_derivs(x, y, z, dict_phi_m):
    """
    Evaluate potential and first derivatives in cylindrical variables.

    Returns:
        phi, dphi_dRscaled, dphi_dzscaled_abs, dphi_dphi, R, R0, z_abs
    where derivatives are taken w.r.t. scaled coordinates:
        Rscaled = asinh(R/R0), zscaled = asinh(|z|/R0)
    """
    R = jnp.sqrt(x*x + y*y)
    ph = jnp.arctan2(y, x)
    z_abs = jnp.abs(z)
    R0 = dict_phi_m['R0'][0]

    shape = R.shape
    _R, _z = _Rz_rescale_forward(R, z_abs, R0)
    pts = jnp.column_stack((_R.ravel(), _z.ravel()))

    def evaluate_mode(m):
        real_values = dict_phi_m['Phi_m_real'][m]
        imag_values = dict_phi_m['Phi_m_img'][m]
        M_x_real = dict_phi_m['Mx_real'][m]
        M_y_real = dict_phi_m['My_real'][m]
        M_x_imag = dict_phi_m['Mx_img'][m]
        M_y_imag = dict_phi_m['My_img'][m]

        real_part, dreal_dRsc, dreal_dzsc = cubic_spline_evaluate_with_derivs(
            pts, (dict_phi_m['Rgrid'], dict_phi_m['Zgrid']),
            real_values, M_x_real, M_y_real, fill_value=0.0)
        imag_part, dimag_dRsc, dimag_dzsc = cubic_spline_evaluate_with_derivs(
            pts, (dict_phi_m['Rgrid'], dict_phi_m['Zgrid']),
            imag_values, M_x_imag, M_y_imag, fill_value=0.0)

        real_part = real_part.reshape(shape)
        imag_part = imag_part.reshape(shape)
        dreal_dRsc = dreal_dRsc.reshape(shape)
        dreal_dzsc = dreal_dzsc.reshape(shape)
        dimag_dRsc = dimag_dRsc.reshape(shape)
        dimag_dzsc = dimag_dzsc.reshape(shape)

        cos_m = jnp.cos(m * ph)
        sin_m = jnp.sin(m * ph)
        factor = jnp.where(m == 0, 1.0, 2.0)

        phi_m = factor * (real_part * cos_m - imag_part * sin_m)
        dphi_dRsc_m = factor * (dreal_dRsc * cos_m - dimag_dRsc * sin_m)
        dphi_dzsc_m = factor * (dreal_dzsc * cos_m - dimag_dzsc * sin_m)
        dphi_dph_m = factor * (-m * real_part * sin_m - m * imag_part * cos_m)

        return phi_m, dphi_dRsc_m, dphi_dzsc_m, dphi_dph_m

    modes = dict_phi_m['m'].astype(int)
    phi_m, dphi_dRsc_m, dphi_dzsc_m, dphi_dph_m = jax.vmap(evaluate_mode)(modes)

    phi = jnp.sum(phi_m, axis=0)
    dphi_dRsc = jnp.sum(dphi_dRsc_m, axis=0)
    dphi_dzsc = jnp.sum(dphi_dzsc_m, axis=0)
    dphi_dph = jnp.sum(dphi_dph_m, axis=0)

    return phi, dphi_dRsc, dphi_dzsc, dphi_dph, R, R0, z_abs



# @jax.jit
# def get_acc(x, y, z, params):
#     def potential_vec(pos):
#         return evaluate_phi(pos[0], pos[1], pos[2], params)
#     grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
#     return -grad_phi

@jax.jit
def get_acc(x, y, z, params, eps=5e-4):
    """
    Return acceleration = -∇phi from analytic derivatives of the spline expansion.
    """
    _, dphi_dRsc, dphi_dzsc, dphi_dph, R, R0, z_abs = _evaluate_phi_and_scaled_derivs(x, y, z, params)

    # Chain rule from scaled coordinates used in interpolation:
    # Rscaled = asinh(R/R0), zscaled = asinh(|z|/R0).
    dRsc_dR = 1.0 / jnp.sqrt(R * R + R0 * R0)
    dzsc_dzabs = 1.0 / jnp.sqrt(z_abs * z_abs + R0 * R0)

    dphi_dR = dphi_dRsc * dRsc_dR
    dphi_dz = dphi_dzsc * dzsc_dzabs * jnp.sign(z)

    tiny = 1e-12
    invR = jnp.where(R > tiny, 1.0 / R, 0.0)
    invR2 = invR * invR

    dRdx = jnp.where(R > tiny, x * invR, 0.0)
    dRdy = jnp.where(R > tiny, y * invR, 0.0)
    dphidx = -y * invR2
    dphidy = x * invR2

    dphidx = jnp.where(R > tiny, dphidx, 0.0)
    dphidy = jnp.where(R > tiny, dphidy, 0.0)

    dphidx_total = dphi_dR * dRdx + dphi_dph * dphidx
    dphidy_total = dphi_dR * dRdy + dphi_dph * dphidy
    dphidz_total = dphi_dz

    return -jnp.array([dphidx_total, dphidy_total, dphidz_total])

@jax.jit
def get_hessian(x, y, z, params):
    def potential_vec(pos):
        return evaluate_phi(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

@jax.jit
def get_density(x, y, z, params):
    H = get_hessian(x, y, z, params)

    # Laplacian = trace of Hessian = ∂²Φ/∂x² + ∂²Φ/∂y² + ∂²Φ/∂z²
    laplacian = jnp.trace(H)

    # Poisson equation → ρ = Laplacian / (4πG)
    return laplacian / (4*jnp.pi*G)
