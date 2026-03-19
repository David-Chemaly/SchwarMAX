"""
JAX-compatible implementation of Dehnen & Aly (2022) disc-bar models.

Provides potential, density, and acceleration for T0-T4, V0-V4, D0-D4, W0-W4
barred disc models. All functions are pure (no mutable state) and compatible
with jax.jit, jax.grad, and jax.vmap.

Reference: Dehnen & Aly (2022), MNRAS 516, 2, 2712-2726

Original code: discBar/discBar.py by Walter Dehnen (2022), GPLv3.
"""

import jax
import jax.numpy as jnp
from functools import partial

from constants import G

# ============================================================================
# Model type indices:
#   T0=0, T1=1, T2=2, T3=3, T4=4, V0=5, V1=6, V2=7, V3=8, V4=9
# ============================================================================
MTYPE_T0, MTYPE_T1, MTYPE_T2, MTYPE_T3, MTYPE_T4 = 0, 1, 2, 3, 4
MTYPE_V0, MTYPE_V1, MTYPE_V2, MTYPE_V3, MTYPE_V4 = 5, 6, 7, 8, 9

MTYPE_NAMES = {
    'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4,
    'V0': 5, 'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9,
}


def make_params(M=1.0, a=None, b=None, s=1.0, q=0.0,
                L=0.0, gamma=0.0, mtype='T1', phi=0.0):
    """
    Create a parameter dictionary for the Dehnen bar model.

    Parameters
    ----------
    M : float
        Total mass.
    a : float, optional
        Scale length. If given with b, overrides s and q.
    b : float, optional
        Scale height. If given with a, overrides s and q.
    s : float
        Scale radius s = a + b.
    q : float
        Flattening q = b / s.
    L : float
        Needle half-length (0 for axisymmetric).
    gamma : float
        Needle slope parameter, -1 <= gamma <= 1.
    mtype : str
        Model type: 'T0'-'T4', 'V0'-'V4'.
    phi : float
        Crossed model angle (0 for single bar).

    Returns
    -------
    dict
        Parameter dictionary with keys: M, a, b, L, gamma, mtype_idx, phi.
    """
    if a is not None and b is not None:
        a, b = float(a), float(b)
    else:
        b = s * q
        a = s - b

    # V models with b=0 reduce to T models
    if mtype in MTYPE_NAMES:
        mtype_idx = MTYPE_NAMES[mtype]
    else:
        raise ValueError(f"Unknown mtype '{mtype}'")

    if mtype_idx >= 5 and b <= 0:
        mtype_idx -= 5  # V -> T

    return {
        'M': jnp.float64(M),
        'a': jnp.float64(a),
        'b': jnp.float64(b),
        'L': jnp.float64(L),
        'gamma': jnp.float64(gamma),
        'mtype_idx': mtype_idx,
        'phi': jnp.float64(phi),
    }


# ============================================================================
# Core building blocks: An, Bn, Cn integrals
# ============================================================================
# These replace the mutable functionACRn class.
# We compute An, Bn at arbitrary order n by starting from the base case
# and iterating with the recurrence:
#   B[n+2] = -Ax / n
#   A[n+2] = ((n-1)*A + x*Ax) / (n*u2)
#   Ax[n+2] = Ax / R2
#
# Base cases:
#   n=1 (odd): A1 = log(x + r), B1 = r, Ax1 = 1/r
#   n=2 (even): A2 = arctan(x/u)/u, B2 = 0.5*log(R2), Ax2 = 1/R2
#   n=3 (odd): A3 = x/(u2*r), B3 = -1/r, Ax3 = 1/R2/r

def _compute_ACRn(x, u2, R2, n):
    """
    Compute An(x, u2), Bn(x, u2), Axn(x, u2) at order n.

    Parameters
    ----------
    x : scalar
    u2 : scalar (y^2 + z^2)
    R2 : scalar (x^2 + u2)
    n : int (static, >= 1)

    Returns
    -------
    A, B, Ax : scalars
    """
    iR2 = 1.0 / R2
    iU2 = 1.0 / u2
    r = jnp.sqrt(R2)

    if (n % 2) == 1:
        # odd: start at n=1
        Ax = 1.0 / r
        A = jnp.log(x + r)
        B = r
        cur_n = 1
        if n >= 3:
            # step to n=3
            B_new = -Ax / cur_n
            A_new = ((cur_n - 1) * A + x * Ax) / (cur_n * u2)
            Ax_new = Ax * iR2
            A, B, Ax = A_new, B_new, Ax_new
            cur_n = 3
        # step by 2 until we reach n
        while cur_n < n:
            B_new = -Ax / cur_n
            A_new = ((cur_n - 1) * A + x * Ax) / (cur_n * u2)
            Ax_new = Ax * iR2
            A, B, Ax = A_new, B_new, Ax_new
            cur_n += 2
    else:
        # even: start at n=2
        iU = jnp.sqrt(iU2)
        Ax = iR2
        A = iU * jnp.arctan(iU * x)
        B = 0.5 * jnp.log(R2)
        cur_n = 2
        while cur_n < n:
            B_new = -Ax / cur_n
            A_new = ((cur_n - 1) * A + x * Ax) / (cur_n * u2)
            Ax_new = Ax * iR2
            A, B, Ax = A_new, B_new, Ax_new
            cur_n += 2

    return A, B, Ax


# ============================================================================
# RFuncAxiN: M/r^n for axisymmetric models
# ============================================================================

def _rfunc_axi_n(M, x, y, z, n):
    """
    Compute M/r^n where r = sqrt(x^2 + y^2 + z^2).

    Parameters
    ----------
    M, x, y, z : scalars
    n : int (static, >= 1)

    Returns
    -------
    scalar
    """
    iR2 = 1.0 / (x * x + y * y + z * z)
    if (n % 2) == 1:
        F = M * jnp.sqrt(iR2)
        cur_n = 1
    else:
        F = M * iR2
        cur_n = 2
    while cur_n < n:
        F *= iR2
        cur_n += 2
    return F


def _rfunc_axi_n_dx(M, x, y, z, n):
    """
    Compute d/dx [M/r^n] = -n * x * M / r^(n+2).
    """
    iR2 = 1.0 / (x * x + y * y + z * z)
    if (n % 2) == 1:
        F = M * jnp.sqrt(iR2)
        cur_n = 1
    else:
        F = M * iR2
        cur_n = 2
    while cur_n < n:
        F *= iR2
        cur_n += 2
    return -n * x * iR2 * F


# ============================================================================
# Convolution for barred models
# ============================================================================

def _convolve_val(func_A_C, M, x, L, gamma, *args):
    """
    Compute the convolution:
        F = c1 [A(x+L, args) - A(x-L, args)]
          + c2 [C(x+L, args) + C(x-L, args) - 2*C(x, args)]
    where c1 = M*(1-gamma)/(2*L), c2 = M*gamma/L^2
    and A, C are computed by func_A_C.

    Parameters
    ----------
    func_A_C : callable
        (x, *args) -> (A, C) tuple
    M : scalar
    x : scalar
    L : scalar (>0)
    gamma : scalar
    *args : additional arguments to func_A_C

    Returns
    -------
    scalar
    """
    c1 = M * (1 - gamma) / (2 * L)
    c2 = M * gamma / (L * L)

    Ap, Cp = func_A_C(x + L, *args)
    Am, Cm = func_A_C(x - L, *args)

    result = c1 * (Ap - Am)
    result += c2 * (Cp + Cm - 2 * func_A_C(x, *args)[1])
    return result


def _convolve_val_dx(func_A_C_dAx, M, x, L, gamma, *args):
    """
    Compute convolution and its x-derivative.

    func_A_C_dAx: (x, *args) -> (A, C, dAx, dCx)
    where dCx = A (since C = x*A - B, dCx/dx = A)

    Returns (val, dval_dx)
    """
    c1 = M * (1 - gamma) / (2 * L)
    c2 = M * gamma / (L * L)

    Ap, Cp, dAxp, dCxp = func_A_C_dAx(x + L, *args)
    Am, Cm, dAxm, dCxm = func_A_C_dAx(x - L, *args)
    A0, C0, dAx0, dCx0 = func_A_C_dAx(x, *args)

    val = c1 * (Ap - Am) + c2 * (Cp + Cm - 2 * C0)
    dvdx = c1 * (dAxp - dAxm) + c2 * (dCxp + dCxm - 2 * dCx0)
    return val, dvdx


# ============================================================================
# RFuncBarN: barred M/r^n via convolution of ACRn
# ============================================================================

def _acrn_A_C(x, y, z, n):
    """
    Compute (A_n, C_n) for given x, y, z at order n.
    C_n = x * A_n - B_n.
    """
    u2 = y * y + z * z
    R2 = x * x + u2
    A, B, Ax = _compute_ACRn(x, u2, R2, n)
    C = x * A - B
    return A, C


def _acrn_A_C_dAx(x, y, z, n):
    """
    Compute (A_n, C_n, dAx_n, dCx_n) for given x, y, z at order n.
    dCx_n = A_n.
    """
    u2 = y * y + z * z
    R2 = x * x + u2
    A, B, Ax = _compute_ACRn(x, u2, R2, n)
    C = x * A - B
    return A, C, Ax, A  # dCx = A


def _rfunc_bar_n(M, x, y, z, n, L, gamma):
    """
    Compute the barred equivalent of M/r^n via needle convolution.
    """
    return _convolve_val(
        lambda xp, y_, z_, n_: _acrn_A_C(xp, y_, z_, n_),
        M, x, L, gamma, y, z, n
    )


def _rfunc_bar_n_dx(M, x, y, z, n, L, gamma):
    """
    Compute the barred M/r^n and its x-derivative via convolution.
    Returns (val, dval_dx).
    """
    return _convolve_val_dx(
        lambda xp, y_, z_, n_: _acrn_A_C_dAx(xp, y_, z_, n_),
        M, x, L, gamma, y, z, n
    )


# ============================================================================
# Unified rfunc: axisymmetric or barred, depending on L
# ============================================================================

def _rfunc_n(M, x, y, z, n, L, gamma):
    """Compute replacement of M/r^n (axisymmetric if L=0, barred if L>0)."""
    return jax.lax.cond(
        L > 0,
        lambda: _rfunc_bar_n(M, x, y, z, n, L, gamma),
        lambda: _rfunc_axi_n(M, x, y, z, n),
    )


# ============================================================================
# LFuncAxiZ: M * ln((r1+z1)/(r2+z2)) for k=0 potential
# ============================================================================

def _lfunc_axi_z(M, x, y, z1, z2):
    """
    Compute M * ln((r1+z1)/(r2+z2)) where ri = sqrt(x^2+y^2+zi^2).
    """
    Rq = x * x + y * y
    r1 = jnp.sqrt(Rq + z1 * z1)
    r2 = jnp.sqrt(Rq + z2 * z2)
    return M * jnp.log((r1 + z1) / (r2 + z2))


# ============================================================================
# functionACLZ: integrals for barred k=0 potential
# ============================================================================

def _aclz_A_C(x, y, z1, z2):
    """
    Compute (A, C) for the LZ convolution integrals.
    A = y*arctan(...) + z1*log(r1+x) - z2*log(r2+x) + x*log((r1+z1)/(r2+z2))
    B = 0.5*(R2*llz + z1*r1 - z2*r2)
    C = x*A - B
    """
    xq = x * x
    yq = y * y
    Rq = xq + yq
    uq1 = yq + z1 * z1
    uq2 = yq + z2 * z2
    r1 = jnp.sqrt(uq1 + xq)
    r2 = jnp.sqrt(uq2 + xq)
    rx1 = r1 + x
    rx2 = r2 + x
    rz1 = r1 + z1
    rz2 = r2 + z2

    llz = jnp.log(rz1 / rz2)
    att = jnp.arctan2(x * y * (r1 * z2 - r2 * z1),
                      yq * r1 * r2 + xq * z1 * z2)

    A = y * att + z1 * jnp.log(rx1) - z2 * jnp.log(rx2) + x * llz
    B = 0.5 * (Rq * llz + z1 * r1 - z2 * r2)
    C = x * A - B
    return A, C


def _lfunc_bar_z(M, x, y, z1, z2, L, gamma):
    """
    Barred version of M*ln((r1+z1)/(r2+z2)) via needle convolution.
    """
    return _convolve_val(
        lambda xp, y_, z1_, z2_: _aclz_A_C(xp, y_, z1_, z2_),
        M, x, L, gamma, y, z1, z2
    )


def _lfunc(M, x, y, z1, z2, L, gamma):
    """Unified log function for k=0 potential."""
    return jax.lax.cond(
        L > 0,
        lambda: _lfunc_bar_z(M, x, y, z1, z2, L, gamma),
        lambda: _lfunc_axi_z(M, x, y, z1, z2),
    )


# ============================================================================
# Density computation (all model types)
# ============================================================================

def _reduced_density(mtype_idx, a, b, M, x, y, z, L, gamma):
    """
    Compute density * zeta^3 / b^2 (the "reduced density").
    This is the core density computation dispatched by model type.

    Parameters
    ----------
    mtype_idx : int (static via lax.switch)
    a, b : scalars (scale length, scale height)
    M, x, y, z : scalars
    L, gamma : scalars (needle params)

    Returns
    -------
    scalar : reduced density
    """
    ze = jnp.sqrt(z * z + b * b)
    iz = 1.0 / ze
    Z = ze + a
    Z2 = Z * Z
    A2 = a * a
    hB2 = 0.5 * b * b
    AZ = a * Z

    def _rn(n):
        return _rfunc_n(M, x, y, n_z, n, L, gamma)

    # We need to evaluate at different z values per model
    # For density: evaluate rfunc at z=ze and z=Z
    n_z = ze  # will be overridden per term

    def density_T0():
        F0_1 = _rfunc_n(M, x, y, ze, 1, L, gamma)
        Fn_1 = _rfunc_n(M, x, y, Z, 1, L, gamma)
        rh = F0_1 - Fn_1
        F0_3 = _rfunc_n(M, x, y, ze, 3, L, gamma)
        Fn_3 = _rfunc_n(M, x, y, Z, 3, L, gamma)
        rh += ze * (ze * F0_3 - Z * Fn_3)
        return (0.25 / (jnp.pi * a)) * rh

    def density_T1():
        Fn_3 = _rfunc_n(M, x, y, Z, 3, L, gamma)
        rh = a * Fn_3
        Fn_5 = _rfunc_n(M, x, y, Z, 5, L, gamma)
        rh += 3 * Z2 * ze * Fn_5
        return (0.25 / jnp.pi) * rh

    def density_T2():
        Fn_5 = _rfunc_n(M, x, y, Z, 5, L, gamma)
        rh = (A2 * a + ze ** 3) * Fn_5
        Fn_7 = _rfunc_n(M, x, y, Z, 7, L, gamma)
        rh += 5 * AZ * Z2 * ze * Fn_7
        return (0.75 / jnp.pi) * rh

    def density_T3():
        Fn_5 = _rfunc_n(M, x, y, Z, 5, L, gamma)
        rh = 3 * ze ** 3 * Fn_5
        Fn_7 = _rfunc_n(M, x, y, Z, 7, L, gamma)
        rh += 5 * a * Z2 * ((3 * ze - 2 * a) * ze + A2) * Fn_7
        Fn_9 = _rfunc_n(M, x, y, Z, 9, L, gamma)
        rh += 35 * A2 * Z2 * Z2 * ze * Fn_9
        return (0.25 / jnp.pi) * rh

    def density_T4():
        Fn_5 = _rfunc_n(M, x, y, Z, 5, L, gamma)
        rh = 3 * Fn_5
        Fn_7 = _rfunc_n(M, x, y, Z, 7, L, gamma)
        rh += 15 * AZ * Fn_7
        rh *= ze ** 3
        Fn_9 = _rfunc_n(M, x, y, Z, 9, L, gamma)
        rh += 7 * A2 * Z * Z2 * ((6 * ze - 3 * a) * ze + A2) * Fn_9
        Fn_11 = _rfunc_n(M, x, y, Z, 11, L, gamma)
        rh += 63 * A2 * AZ * Z2 * Z2 * ze * Fn_11
        return (0.25 / jnp.pi) * rh

    def density_V0():
        F0_1 = _rfunc_n(M, x, y, ze, 1, L, gamma)
        Fn_1 = _rfunc_n(M, x, y, Z, 1, L, gamma)
        rh = F0_1 - Fn_1
        F0_3 = _rfunc_n(M, x, y, ze, 3, L, gamma)
        Fn_3 = _rfunc_n(M, x, y, Z, 3, L, gamma)
        rh += ze * (ze * F0_3 - Z * Fn_3)
        ze2 = ze * ze
        rh *= 3 / ze2
        rh += (Fn_3 - F0_3)
        F0_5 = _rfunc_n(M, x, y, ze, 5, L, gamma)
        Fn_5 = _rfunc_n(M, x, y, Z, 5, L, gamma)
        rh += 3 * (ze2 * F0_5 - Z2 * Fn_5)
        return (0.25 * hB2 / (jnp.pi * a)) * rh

    def density_V1():
        Fn_3 = _rfunc_n(M, x, y, Z, 3, L, gamma)
        rh = a * iz * iz * Fn_3
        Fn_5 = _rfunc_n(M, x, y, Z, 5, L, gamma)
        rh += 3 * AZ * iz * Fn_5
        Fn_7 = _rfunc_n(M, x, y, Z, 7, L, gamma)
        rh += 5 * Z * Z2 * Fn_7
        return (0.375 * b * b / jnp.pi) * rh

    def density_V2():
        Fn_5 = _rfunc_n(M, x, y, Z, 5, L, gamma)
        rh = 3 * a * A2 * iz * iz * Fn_5
        Fn_7 = _rfunc_n(M, x, y, Z, 7, L, gamma)
        rh += 5 * Z2 * (ze * ze - 2 * a * ze + 3 * A2) * iz * Fn_7
        Fn_9 = _rfunc_n(M, x, y, Z, 9, L, gamma)
        rh += 35 * a * Z2 * Z2 * Fn_9
        return (0.375 * b * b / jnp.pi) * rh

    def density_V3():
        Fn_7 = _rfunc_n(M, x, y, Z, 7, L, gamma)
        rh = 3 * ze ** 3 * (1 + (a * iz) ** 5) * Fn_7
        Fn_9 = _rfunc_n(M, x, y, Z, 9, L, gamma)
        rh += 7 * AZ * Z2 * iz * ((3 * ze - 4 * a) * ze + 3 * A2) * Fn_9
        Fn_11 = _rfunc_n(M, x, y, Z, 11, L, gamma)
        rh += 63 * A2 * Z * Z2 * Z2 * Fn_11
        return (0.625 * b * b / jnp.pi) * rh

    def density_V4():
        Fn_7 = _rfunc_n(M, x, y, Z, 7, L, gamma)
        rh = 5 * ze ** 3 * Fn_7
        Fn_9 = _rfunc_n(M, x, y, Z, 9, L, gamma)
        rh += 7 * a * Z2 * (5 * ze * ze - 4 * a * ze + 3 * A2 -
                             2 * a * A2 * iz + A2 * A2 * iz * iz) * Fn_9
        Fn_11 = _rfunc_n(M, x, y, Z, 11, L, gamma)
        rh += 63 * A2 * Z2 * Z2 * (ze + ze - 2 * a + A2 * iz) * Fn_11
        Fn_13 = _rfunc_n(M, x, y, Z, 13, L, gamma)
        rh += 231 * A2 * AZ * Z2 * Z2 * Z * Fn_13
        return (0.375 * b * b / jnp.pi) * rh

    branches = [
        density_T0, density_T1, density_T2, density_T3, density_T4,
        density_V0, density_V1, density_V2, density_V3, density_V4,
    ]
    return jax.lax.switch(mtype_idx, branches)


# ============================================================================
# Potential computation (all model types)
# ============================================================================

def _pot_modKn(mtype_idx, a, b, M, x, y, Z, zeta, L, gamma):
    """
    Gravitational potential for k>=1 models (without sign).
    Returns the potential value (to be negated for physical potential).
    """
    hB2 = 0.5 * b * b
    aZ = a * Z

    def _rn(n):
        return _rfunc_n(M, x, y, Z, n, L, gamma)

    def pot_T1():
        return _rn(1)

    def pot_T2():
        return _rn(1) + aZ * _rn(3)

    def pot_T3():
        ph = _rn(1)
        ph += a * (Z - a / 3) * _rn(3)
        ph += aZ ** 2 * _rn(5)
        return ph

    def pot_T4():
        ph = _rn(1)
        ph += a * (Z - 0.4 * a) * _rn(3)
        ph += 0.6 * a * aZ * (Z + Z - a) * _rn(5)
        ph += aZ ** 3 * _rn(7)
        return ph

    def pot_V1():
        return _rn(1) + (hB2 * Z / zeta) * _rn(3)

    def pot_V2():
        ph = _rn(1)
        ph += (aZ + hB2) * _rn(3)
        ph += (3 * hB2 * aZ * Z / zeta) * _rn(5)
        return ph

    def pot_V3():
        ph = _rn(1)
        ph += (a * (Z - a / 3) + hB2) * _rn(3)
        ph += aZ * (aZ + 3 * hB2) * _rn(5)
        ph += (5 * hB2 * aZ ** 2 * Z / zeta) * _rn(7)
        return ph

    def pot_V4():
        ph = _rn(1)
        ph += (a * (Z - 0.4 * a) + hB2) * _rn(3)
        ph += 3 * a * (0.2 * aZ * (Z + Z - a) +
                        hB2 * (Z - 0.2 * a)) * _rn(5)
        ph += aZ ** 2 * (aZ + 6 * hB2) * _rn(7)
        ph += (7 * hB2 * aZ ** 3 * Z / zeta) * _rn(9)
        return ph

    # k=0 models don't use this path (handled separately)
    def pot_T0():
        return jnp.float64(0.0)

    def pot_V0():
        return jnp.float64(0.0)

    branches = [
        pot_T0, pot_T1, pot_T2, pot_T3, pot_T4,
        pot_V0, pot_V1, pot_V2, pot_V3, pot_V4,
    ]
    return jax.lax.switch(mtype_idx, branches)


def _pot_modK0(mtype_idx, a, b, M, x, y, Z, zeta, L, gamma):
    """
    Gravitational potential for k=0 models (T0 and V0).
    Returns the potential value (to be negated for physical potential).
    """
    hB2 = 0.5 * b * b
    fac = 1.0 / a

    # LFunc: M*fac * ln((rZ + Z)/(r0 + zeta))
    Lf = _lfunc(M * fac, x, y, Z, zeta, L, gamma)
    ps = Lf

    def pot_T0():
        return ps

    def pot_V0():
        F0 = _rfunc_n(M * fac, x, y, zeta, 1, L, gamma)
        Fa = _rfunc_n(M * fac, x, y, Z, 1, L, gamma)
        return ps + (hB2 / zeta) * (F0 - Fa)

    is_V = (mtype_idx >= 5)
    return jax.lax.cond(is_V, pot_V0, pot_T0)


# ============================================================================
# Top-level API
# ============================================================================

def _single_potential(x, y, z, params):
    """
    Compute gravitational potential for a single T/V model.
    Returns scalar potential value.
    """
    M = params['M']
    a = params['a']
    b = params['b']
    L = params['L']
    gamma = params['gamma']
    mtype_idx = params['mtype_idx']

    ze = jnp.sqrt(z * z + b * b)
    Z = ze + a

    is_k0 = (mtype_idx == MTYPE_T0) | (mtype_idx == MTYPE_V0)

    pot_k0 = _pot_modK0(mtype_idx, a, b, M, x, y, Z, ze, L, gamma)
    pot_kn = _pot_modKn(mtype_idx, a, b, M, x, y, Z, ze, L, gamma)

    return -jnp.where(is_k0, pot_k0, pot_kn)


def _single_density(x, y, z, params):
    """
    Compute space density for a single T/V model with finite thickness (b>0).
    """
    M = params['M']
    a = params['a']
    b = params['b']
    L = params['L']
    gamma = params['gamma']
    mtype_idx = params['mtype_idx']

    ze = jnp.sqrt(z * z + b * b)
    iz = 1.0 / ze
    rd = _reduced_density(mtype_idx, a, b, M, x, y, z, L, gamma)
    return (b * b * iz ** 3) * rd


def _crossed_potential(x, y, z, params):
    """
    Potential for crossed model: average of two bars at ±phi.
    """
    phi = params['phi']
    sp = jnp.abs(jnp.sin(phi))
    cp = jnp.abs(jnp.cos(phi))

    # Rotate forward: (x,y) -> (cp*x + sp*y, cp*y - sp*x)
    xf = cp * x + sp * y
    yf = cp * y - sp * x
    pf = _single_potential(xf, yf, z, params)

    # Rotate backward: (x,y) -> (cp*x - sp*y, cp*y + sp*x)
    xb = cp * x - sp * y
    yb = cp * y + sp * x
    pb = _single_potential(xb, yb, z, params)

    return 0.5 * (pf + pb)


def _crossed_density(x, y, z, params):
    """
    Density for crossed model: average of two bars at ±phi.
    """
    phi = params['phi']
    sp = jnp.abs(jnp.sin(phi))
    cp = jnp.abs(jnp.cos(phi))

    xf = cp * x + sp * y
    yf = cp * y - sp * x
    df = _single_density(xf, yf, z, params)

    xb = cp * x - sp * y
    yb = cp * y + sp * x
    db = _single_density(xb, yb, z, params)

    return 0.5 * (df + db)


@jax.jit
def DehnenBar_potential(x, y, z, params):
    """
    Compute Dehnen & Aly (2022) bar potential at (x, y, z).

    Parameters
    ----------
    x, y, z : scalars
        Cartesian coordinates.
    params : dict
        From make_params(). Keys: M, a, b, L, gamma, mtype_idx, phi.

    Returns
    -------
    scalar : gravitational potential (units: kpc^2 / Gyr^2 when M in Msun, lengths in kpc)
    """
    is_crossed = params['phi'] != 0.0
    return G * jax.lax.cond(
        is_crossed,
        lambda: _crossed_potential(x, y, z, params),
        lambda: _single_potential(x, y, z, params),
    )


@jax.jit
def DehnenBar_density(x, y, z, params):
    """
    Compute Dehnen & Aly (2022) bar density at (x, y, z).

    Parameters
    ----------
    x, y, z : scalars
        Cartesian coordinates.
    params : dict
        From make_params(). Keys: M, a, b, L, gamma, mtype_idx, phi.

    Returns
    -------
    scalar : space density
    """
    is_crossed = params['phi'] != 0.0
    return jax.lax.cond(
        is_crossed,
        lambda: _crossed_density(x, y, z, params),
        lambda: _single_density(x, y, z, params),
    )


@jax.jit
def DehnenBar_acceleration(x, y, z, params):
    """
    Compute acceleration (= -grad Phi) for Dehnen & Aly (2022) bar.

    Parameters
    ----------
    x, y, z : scalars
        Cartesian coordinates.
    params : dict
        From make_params().

    Returns
    -------
    array(3,) : [ax, ay, az] acceleration components
    """
    def pot_vec(pos):
        return DehnenBar_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(pot_vec)(jnp.array([x, y, z]))
    return -grad_phi


# ============================================================================
# Compound model support (D and W types)
# ============================================================================

def make_compound_params(M=1.0, a=None, b=None, s=1.0, q=0.0,
                         L=0.0, gamma=0.0, mtype='D2', phi=0.0):
    """
    Create parameter list for compound D/W models.

    D models are linear combinations of T models:
        D0, D1 = T0
        D2 = 2*T0 - T1
        D3 = (8/3)*T0 - (4/3)*T1 - (1/3)*T2
        D4 = 3.2*T0 - 1.6*T1 - 0.4*T2 - 0.2*T3

    W models are the same but with V models:
        W0, W1 = V0
        W2 = 2*V0 - V1
        etc.

    Returns
    -------
    list of (sign, params_dict) tuples where sign is +1 or -1.
    """
    family = mtype[0]
    k = int(mtype[1])

    if family == 'D':
        base = 'T'
    elif family == 'W':
        base = 'V'
    else:
        raise ValueError(f"Unknown compound type '{mtype}'")

    if k <= 1:
        return [(1, make_params(M, a, b, s, q, L, gamma, base + '0', phi))]

    # Compound coefficients
    if k == 2:
        components = [(2 * M, base + '0'), (-M, base + '1')]
    elif k == 3:
        components = [(8 * M / 3, base + '0'),
                      (-4 * M / 3, base + '1'),
                      (-M / 3, base + '2')]
    elif k == 4:
        components = [(3.2 * M, base + '0'),
                      (-1.6 * M, base + '1'),
                      (-0.4 * M, base + '2'),
                      (-0.2 * M, base + '3')]
    else:
        raise ValueError(f"Unsupported k={k} for compound model")

    result = []
    for mass, mt in components:
        sign = 1 if mass > 0 else -1
        p = make_params(abs(mass), a, b, s, q, L, gamma, mt, phi)
        result.append((sign, p))
    return result


def DehnenBar_compound_potential(x, y, z, components):
    """
    Compute potential for compound D/W model.

    Parameters
    ----------
    x, y, z : scalars
    components : list of (sign, params) from make_compound_params()

    Returns
    -------
    scalar : potential
    """
    pot = 0.0
    for sign, p in components:
        pot += sign * DehnenBar_potential(x, y, z, p)
    return pot


def DehnenBar_compound_density(x, y, z, components):
    """
    Compute density for compound D/W model.

    Parameters
    ----------
    x, y, z : scalars
    components : list of (sign, params) from make_compound_params()

    Returns
    -------
    scalar : density
    """
    rho = 0.0
    for sign, p in components:
        rho += sign * DehnenBar_density(x, y, z, p)
    return rho


def DehnenBar_compound_acceleration(x, y, z, components):
    """
    Compute acceleration for compound D/W model.

    Parameters
    ----------
    x, y, z : scalars
    components : list of (sign, params) from make_compound_params()

    Returns
    -------
    array(3,) : acceleration
    """
    acc = jnp.zeros(3)
    for sign, p in components:
        acc += sign * DehnenBar_acceleration(x, y, z, p)
    return acc


# ============================================================================
# FAST PATH: Specialized builders (no lax.switch / lax.cond overhead)
# ============================================================================
# The generic API above uses lax.switch over 10 model branches and lax.cond
# for axisym vs barred. This means jax.grad must differentiate through all
# branches, making acceleration ~5x slower than necessary.
#
# make_dehnen_bar_fns() creates closures specialized to a specific model type
# and barred/axisym mode at Python time, before JIT. The resulting functions
# contain no dynamic dispatch and compile to minimal, efficient XLA code.
# ============================================================================

def _rfunc_axi(M, x, y, z, n):
    """M/r^n for axisymmetric (no needle), n known at trace time."""
    iR2 = 1.0 / (x * x + y * y + z * z)
    if (n % 2) == 1:
        F = M * jnp.sqrt(iR2)
        cur = 1
    else:
        F = M * iR2
        cur = 2
    while cur < n:
        F *= iR2
        cur += 2
    return F


def _rfunc_bar(M, x, y, z, n, L, gamma):
    """Barred M/r^n via needle convolution, n known at trace time."""
    return _convolve_val(
        lambda xp, y_, z_, n_: _acrn_A_C(xp, y_, z_, n_),
        M, x, L, gamma, y, z, n
    )


def _lfunc_axi(M, x, y, z1, z2):
    """Axisymmetric log function for k=0 potential."""
    Rq = x * x + y * y
    r1 = jnp.sqrt(Rq + z1 * z1)
    r2 = jnp.sqrt(Rq + z2 * z2)
    return M * jnp.log((r1 + z1) / (r2 + z2))


def _lfunc_bar(M, x, y, z1, z2, L, gamma):
    """Barred log function for k=0 potential via convolution."""
    return _convolve_val(
        lambda xp, y_, z1_, z2_: _aclz_A_C(xp, y_, z1_, z2_),
        M, x, L, gamma, y, z1, z2
    )


def make_dehnen_bar_fns(M=1.0, a=None, b=None, s=1.0, q=0.0,
                        L=0.0, gamma=0.0, mtype='T1', phi=0.0):
    """
    Build specialized, jit-compiled potential/density/acceleration functions
    for a specific Dehnen & Aly (2022) model.

    This is the FAST PATH for orbit integration. The model type and
    barred/axisymmetric mode are resolved at Python time, so JAX only
    compiles the exact code path needed. This eliminates lax.switch and
    lax.cond overhead, making jax.grad ~3-5x faster than the generic API.

    Parameters
    ----------
    M, a, b, s, q, L, gamma, mtype, phi :
        Same as make_params().

    Returns
    -------
    dict with keys:
        'potential': jitted (x, y, z) -> scalar
        'density':   jitted (x, y, z) -> scalar
        'acceleration': jitted (x, y, z) -> array(3,)
        'potential_batch': jitted (x_arr, y_arr, z_arr) -> array(N,)
        'acceleration_batch': jitted (x_arr, y_arr, z_arr) -> array(N, 3)
        'params': the parameter dict (for reference)
    """
    if a is not None and b is not None:
        a, b = float(a), float(b)
    else:
        b = s * q
        a = s - b

    # Resolve model type
    if mtype in MTYPE_NAMES:
        mtype_idx = MTYPE_NAMES[mtype]
    else:
        raise ValueError(f"Unknown mtype '{mtype}'")
    if mtype_idx >= 5 and b <= 0:
        mtype_idx -= 5

    M = float(M)
    L = float(L)
    gamma = float(gamma)
    phi = float(phi)
    is_barred = L > 0
    is_crossed = phi != 0.0
    is_k0 = mtype_idx in (MTYPE_T0, MTYPE_V0)
    is_V = mtype_idx >= 5
    hB2 = 0.5 * b * b

    # Choose rfunc based on barred/axisym (Python-time decision)
    if is_barred:
        def rn(M_, x, y, z, n):
            return _rfunc_bar(M_, x, y, z, n, L, gamma)
    else:
        def rn(M_, x, y, z, n):
            return _rfunc_axi(M_, x, y, z, n)

    if is_barred:
        def lf(M_, x, y, z1, z2):
            return _lfunc_bar(M_, x, y, z1, z2, L, gamma)
    else:
        def lf(M_, x, y, z1, z2):
            return _lfunc_axi(M_, x, y, z1, z2)

    # ---- Potential (specialized per model type) ----
    def _pot_single(x, y, z):
        ze = jnp.sqrt(z * z + b * b)
        Z = ze + a
        aZ = a * Z

        if is_k0:
            fac = 1.0 / a
            ps = lf(M * fac, x, y, Z, ze)
            if is_V:
                F0 = rn(M * fac, x, y, ze, 1)
                Fa = rn(M * fac, x, y, Z, 1)
                ps += (hB2 / ze) * (F0 - Fa)
            return -ps
        else:
            ph = rn(M, x, y, Z, 1)
            if mtype_idx == MTYPE_T1:
                pass
            elif mtype_idx == MTYPE_T2:
                ph += aZ * rn(M, x, y, Z, 3)
            elif mtype_idx == MTYPE_T3:
                ph += a * (Z - a / 3) * rn(M, x, y, Z, 3)
                ph += aZ ** 2 * rn(M, x, y, Z, 5)
            elif mtype_idx == MTYPE_T4:
                ph += a * (Z - 0.4 * a) * rn(M, x, y, Z, 3)
                ph += 0.6 * a * aZ * (Z + Z - a) * rn(M, x, y, Z, 5)
                ph += aZ ** 3 * rn(M, x, y, Z, 7)
            elif mtype_idx == MTYPE_V1:
                ph += (hB2 * Z / ze) * rn(M, x, y, Z, 3)
            elif mtype_idx == MTYPE_V2:
                ph += (aZ + hB2) * rn(M, x, y, Z, 3)
                ph += (3 * hB2 * aZ * Z / ze) * rn(M, x, y, Z, 5)
            elif mtype_idx == MTYPE_V3:
                ph += (a * (Z - a / 3) + hB2) * rn(M, x, y, Z, 3)
                ph += aZ * (aZ + 3 * hB2) * rn(M, x, y, Z, 5)
                ph += (5 * hB2 * aZ ** 2 * Z / ze) * rn(M, x, y, Z, 7)
            elif mtype_idx == MTYPE_V4:
                ph += (a * (Z - 0.4 * a) + hB2) * rn(M, x, y, Z, 3)
                ph += 3 * a * (0.2 * aZ * (Z + Z - a) +
                               hB2 * (Z - 0.2 * a)) * rn(M, x, y, Z, 5)
                ph += aZ ** 2 * (aZ + 6 * hB2) * rn(M, x, y, Z, 7)
                ph += (7 * hB2 * aZ ** 3 * Z / ze) * rn(M, x, y, Z, 9)
            return -ph

    # ---- Density (specialized per model type) ----
    def _density_single(x, y, z):
        ze = jnp.sqrt(z * z + b * b)
        iz = 1.0 / ze
        Z = ze + a
        Z2 = Z * Z
        A2 = a * a
        AZ = a * Z

        if mtype_idx == MTYPE_T0:
            F0_1 = rn(M, x, y, ze, 1)
            Fn_1 = rn(M, x, y, Z, 1)
            rh = F0_1 - Fn_1
            F0_3 = rn(M, x, y, ze, 3)
            Fn_3 = rn(M, x, y, Z, 3)
            rh += ze * (ze * F0_3 - Z * Fn_3)
            rd = (0.25 / (jnp.pi * a)) * rh
        elif mtype_idx == MTYPE_T1:
            Fn_3 = rn(M, x, y, Z, 3)
            rh = a * Fn_3
            Fn_5 = rn(M, x, y, Z, 5)
            rh += 3 * Z2 * ze * Fn_5
            rd = (0.25 / jnp.pi) * rh
        elif mtype_idx == MTYPE_T2:
            Fn_5 = rn(M, x, y, Z, 5)
            rh = (A2 * a + ze ** 3) * Fn_5
            Fn_7 = rn(M, x, y, Z, 7)
            rh += 5 * AZ * Z2 * ze * Fn_7
            rd = (0.75 / jnp.pi) * rh
        elif mtype_idx == MTYPE_T3:
            Fn_5 = rn(M, x, y, Z, 5)
            rh = 3 * ze ** 3 * Fn_5
            Fn_7 = rn(M, x, y, Z, 7)
            rh += 5 * a * Z2 * ((3 * ze - 2 * a) * ze + A2) * Fn_7
            Fn_9 = rn(M, x, y, Z, 9)
            rh += 35 * A2 * Z2 * Z2 * ze * Fn_9
            rd = (0.25 / jnp.pi) * rh
        elif mtype_idx == MTYPE_T4:
            Fn_5 = rn(M, x, y, Z, 5)
            rh = 3 * Fn_5
            Fn_7 = rn(M, x, y, Z, 7)
            rh += 15 * AZ * Fn_7
            rh *= ze ** 3
            Fn_9 = rn(M, x, y, Z, 9)
            rh += 7 * A2 * Z * Z2 * ((6 * ze - 3 * a) * ze + A2) * Fn_9
            Fn_11 = rn(M, x, y, Z, 11)
            rh += 63 * A2 * AZ * Z2 * Z2 * ze * Fn_11
            rd = (0.25 / jnp.pi) * rh
        elif mtype_idx == MTYPE_V0:
            F0_1 = rn(M, x, y, ze, 1)
            Fn_1 = rn(M, x, y, Z, 1)
            rh = F0_1 - Fn_1
            F0_3 = rn(M, x, y, ze, 3)
            Fn_3 = rn(M, x, y, Z, 3)
            rh += ze * (ze * F0_3 - Z * Fn_3)
            ze2 = ze * ze
            rh *= 3 / ze2
            rh += (Fn_3 - F0_3)
            F0_5 = rn(M, x, y, ze, 5)
            Fn_5 = rn(M, x, y, Z, 5)
            rh += 3 * (ze2 * F0_5 - Z2 * Fn_5)
            rd = (0.25 * hB2 / (jnp.pi * a)) * rh
        elif mtype_idx == MTYPE_V1:
            Fn_3 = rn(M, x, y, Z, 3)
            rh = a * iz * iz * Fn_3
            Fn_5 = rn(M, x, y, Z, 5)
            rh += 3 * AZ * iz * Fn_5
            Fn_7 = rn(M, x, y, Z, 7)
            rh += 5 * Z * Z2 * Fn_7
            rd = (0.375 * b * b / jnp.pi) * rh
        elif mtype_idx == MTYPE_V2:
            Fn_5 = rn(M, x, y, Z, 5)
            rh = 3 * a * A2 * iz * iz * Fn_5
            Fn_7 = rn(M, x, y, Z, 7)
            rh += 5 * Z2 * (ze * ze - 2 * a * ze + 3 * A2) * iz * Fn_7
            Fn_9 = rn(M, x, y, Z, 9)
            rh += 35 * a * Z2 * Z2 * Fn_9
            rd = (0.375 * b * b / jnp.pi) * rh
        elif mtype_idx == MTYPE_V3:
            Fn_7 = rn(M, x, y, Z, 7)
            rh = 3 * ze ** 3 * (1 + (a * iz) ** 5) * Fn_7
            Fn_9 = rn(M, x, y, Z, 9)
            rh += 7 * AZ * Z2 * iz * ((3 * ze - 4 * a) * ze + 3 * A2) * Fn_9
            Fn_11 = rn(M, x, y, Z, 11)
            rh += 63 * A2 * Z * Z2 * Z2 * Fn_11
            rd = (0.625 * b * b / jnp.pi) * rh
        elif mtype_idx == MTYPE_V4:
            Fn_7 = rn(M, x, y, Z, 7)
            rh = 5 * ze ** 3 * Fn_7
            Fn_9 = rn(M, x, y, Z, 9)
            rh += 7 * a * Z2 * (5 * ze * ze - 4 * a * ze + 3 * A2 -
                                 2 * a * A2 * iz + A2 * A2 * iz * iz) * Fn_9
            Fn_11 = rn(M, x, y, Z, 11)
            rh += 63 * A2 * Z2 * Z2 * (ze + ze - 2 * a + A2 * iz) * Fn_11
            Fn_13 = rn(M, x, y, Z, 13)
            rh += 231 * A2 * AZ * Z2 * Z2 * Z * Fn_13
            rd = (0.375 * b * b / jnp.pi) * rh
        else:
            rd = 0.0

        return (b * b * iz ** 3) * rd

    # ---- Handle crossed models ----
    if is_crossed:
        sp = abs(jnp.sin(phi))
        cp = abs(jnp.cos(phi))

        def potential_scalar(x, y, z):
            xf = cp * x + sp * y
            yf = cp * y - sp * x
            pf = _pot_single(xf, yf, z)
            xb = cp * x - sp * y
            yb = cp * y + sp * x
            pb = _pot_single(xb, yb, z)
            return G * 0.5 * (pf + pb)

        def density_scalar(x, y, z):
            xf = cp * x + sp * y
            yf = cp * y - sp * x
            df = _density_single(xf, yf, z)
            xb = cp * x - sp * y
            yb = cp * y + sp * x
            db = _density_single(xb, yb, z)
            return 0.5 * (df + db)
    else:
        def potential_scalar(x, y, z):
            return G * _pot_single(x, y, z)

        def density_scalar(x, y, z):
            return _density_single(x, y, z)

    @jax.jit
    def potential_jit(x, y, z):
        return potential_scalar(x, y, z)

    @jax.jit
    def density_jit(x, y, z):
        return density_scalar(x, y, z)

    @jax.jit
    def acceleration_jit(x, y, z):
        def pot_vec(pos):
            return potential_scalar(pos[0], pos[1], pos[2])
        return -jax.grad(pot_vec)(jnp.array([x, y, z]))

    potential_batch = jax.jit(jax.vmap(potential_scalar, in_axes=(0, 0, 0)))
    density_batch = jax.jit(jax.vmap(density_scalar, in_axes=(0, 0, 0)))
    acceleration_batch = jax.jit(jax.vmap(
        lambda x, y, z: -jax.grad(
            lambda pos: potential_scalar(pos[0], pos[1], pos[2])
        )(jnp.array([x, y, z])),
        in_axes=(0, 0, 0),
    ))

    params = make_params(M=M, a=a, b=b, L=L, gamma=gamma, mtype=mtype, phi=phi)

    return {
        'potential': potential_jit,
        'density': density_jit,
        'acceleration': acceleration_jit,
        'potential_batch': potential_batch,
        'density_batch': density_batch,
        'acceleration_batch': acceleration_batch,
        'params': params,
    }


# ============================================================================
# EXPLICIT PER-MODEL-TYPE FUNCTIONS
# ============================================================================
# Each model type (T0-T4, V0-V4) has its own potential, density, and
# acceleration function with the SAME signature:
#
#   potential(x, y, z, M, a, b, L, gamma) -> scalar
#   density(x, y, z, M, a, b, L, gamma) -> scalar
#   acceleration(x, y, z, M, a, b, L, gamma) -> array(3,)
#
# Model type is resolved at Python time (no lax.switch).
# Barred (L>0) vs axisymmetric (L=0) is handled via lax.cond (cheap binary).
# For crossed models, use the crossed() wrapper.
#
# Usage:
#   from dehnen_bar import T1_potential, T1_density, T1_acceleration
#   phi = T1_potential(x, y, z, M, a, b, L, gamma)
#   rho = T1_density(x, y, z, M, a, b, L, gamma)
#   acc = T1_acceleration(x, y, z, M, a, b, L, gamma)
#
#   # For crossed models:
#   crossed_pot = crossed(T1_potential, phi=0.3)
#   phi_val = crossed_pot(x, y, z, M, a, b, L, gamma)
# ============================================================================


def _rn_cond(M_, x, y, z, n, L, gamma):
    """rfunc with barred/axisym dispatch via lax.cond."""
    return jax.lax.cond(
        L > 0,
        lambda: _rfunc_bar(M_, x, y, z, n, L, gamma),
        lambda: _rfunc_axi(M_, x, y, z, n),
    )


def _lf_cond(M_, x, y, z1, z2, L, gamma):
    """lfunc with barred/axisym dispatch via lax.cond."""
    return jax.lax.cond(
        L > 0,
        lambda: _lfunc_bar(M_, x, y, z1, z2, L, gamma),
        lambda: _lfunc_axi(M_, x, y, z1, z2),
    )


def _make_explicit_fns(mtype_idx):
    """
    Factory: create (potential, density, acceleration) for a given model type.

    The model type is baked in at Python time. Barred vs axisymmetric is
    dispatched at runtime via lax.cond(L > 0).

    Returns (potential_fn, density_fn, acceleration_fn) all jitted.
    """
    is_k0 = mtype_idx in (MTYPE_T0, MTYPE_V0)
    is_V = mtype_idx >= 5

    # ---- Potential ----
    if mtype_idx == MTYPE_T0:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            return -G * _lf_cond(M / a, x, y, Z, ze, L, gamma)

    elif mtype_idx == MTYPE_V0:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            hB2 = 0.5 * b * b
            Mfa = M / a
            ps = _lf_cond(Mfa, x, y, Z, ze, L, gamma)
            F0 = _rn_cond(Mfa, x, y, ze, 1, L, gamma)
            Fa = _rn_cond(Mfa, x, y, Z, 1, L, gamma)
            ps += (hB2 / ze) * (F0 - Fa)
            return -G * ps

    elif mtype_idx == MTYPE_T1:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            return -G * _rn_cond(M, x, y, Z, 1, L, gamma)

    elif mtype_idx == MTYPE_T2:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            aZ = a * Z
            ph = _rn_cond(M, x, y, Z, 1, L, gamma)
            ph += aZ * _rn_cond(M, x, y, Z, 3, L, gamma)
            return -G * ph

    elif mtype_idx == MTYPE_T3:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            aZ = a * Z
            ph = _rn_cond(M, x, y, Z, 1, L, gamma)
            ph += a * (Z - a / 3) * _rn_cond(M, x, y, Z, 3, L, gamma)
            ph += aZ ** 2 * _rn_cond(M, x, y, Z, 5, L, gamma)
            return -G * ph

    elif mtype_idx == MTYPE_T4:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            aZ = a * Z
            ph = _rn_cond(M, x, y, Z, 1, L, gamma)
            ph += a * (Z - 0.4 * a) * _rn_cond(M, x, y, Z, 3, L, gamma)
            ph += 0.6 * a * aZ * (Z + Z - a) * _rn_cond(M, x, y, Z, 5, L, gamma)
            ph += aZ ** 3 * _rn_cond(M, x, y, Z, 7, L, gamma)
            return -G * ph

    elif mtype_idx == MTYPE_V1:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            hB2 = 0.5 * b * b
            ph = _rn_cond(M, x, y, Z, 1, L, gamma)
            ph += (hB2 * Z / ze) * _rn_cond(M, x, y, Z, 3, L, gamma)
            return -G * ph

    elif mtype_idx == MTYPE_V2:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            aZ = a * Z
            hB2 = 0.5 * b * b
            ph = _rn_cond(M, x, y, Z, 1, L, gamma)
            ph += (aZ + hB2) * _rn_cond(M, x, y, Z, 3, L, gamma)
            ph += (3 * hB2 * aZ * Z / ze) * _rn_cond(M, x, y, Z, 5, L, gamma)
            return -G * ph

    elif mtype_idx == MTYPE_V3:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            aZ = a * Z
            hB2 = 0.5 * b * b
            ph = _rn_cond(M, x, y, Z, 1, L, gamma)
            ph += (a * (Z - a / 3) + hB2) * _rn_cond(M, x, y, Z, 3, L, gamma)
            ph += aZ * (aZ + 3 * hB2) * _rn_cond(M, x, y, Z, 5, L, gamma)
            ph += (5 * hB2 * aZ ** 2 * Z / ze) * _rn_cond(M, x, y, Z, 7, L, gamma)
            return -G * ph

    elif mtype_idx == MTYPE_V4:
        def _potential(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            Z = ze + a
            aZ = a * Z
            hB2 = 0.5 * b * b
            ph = _rn_cond(M, x, y, Z, 1, L, gamma)
            ph += (a * (Z - 0.4 * a) + hB2) * _rn_cond(M, x, y, Z, 3, L, gamma)
            ph += 3 * a * (0.2 * aZ * (Z + Z - a) +
                           hB2 * (Z - 0.2 * a)) * _rn_cond(M, x, y, Z, 5, L, gamma)
            ph += aZ ** 2 * (aZ + 6 * hB2) * _rn_cond(M, x, y, Z, 7, L, gamma)
            ph += (7 * hB2 * aZ ** 3 * Z / ze) * _rn_cond(M, x, y, Z, 9, L, gamma)
            return -G * ph

    # ---- Density ----
    if mtype_idx == MTYPE_T0:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            F0_1 = _rn_cond(M, x, y, ze, 1, L, gamma)
            Fn_1 = _rn_cond(M, x, y, Z, 1, L, gamma)
            rh = F0_1 - Fn_1
            F0_3 = _rn_cond(M, x, y, ze, 3, L, gamma)
            Fn_3 = _rn_cond(M, x, y, Z, 3, L, gamma)
            rh += ze * (ze * F0_3 - Z * Fn_3)
            return (b * b * iz ** 3) * (0.25 / (jnp.pi * a)) * rh

    elif mtype_idx == MTYPE_T1:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            Z2 = Z * Z
            Fn_3 = _rn_cond(M, x, y, Z, 3, L, gamma)
            rh = a * Fn_3
            Fn_5 = _rn_cond(M, x, y, Z, 5, L, gamma)
            rh += 3 * Z2 * ze * Fn_5
            return (b * b * iz ** 3) * (0.25 / jnp.pi) * rh

    elif mtype_idx == MTYPE_T2:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            Z2 = Z * Z
            A2 = a * a
            AZ = a * Z
            Fn_5 = _rn_cond(M, x, y, Z, 5, L, gamma)
            rh = (A2 * a + ze ** 3) * Fn_5
            Fn_7 = _rn_cond(M, x, y, Z, 7, L, gamma)
            rh += 5 * AZ * Z2 * ze * Fn_7
            return (b * b * iz ** 3) * (0.75 / jnp.pi) * rh

    elif mtype_idx == MTYPE_T3:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            Z2 = Z * Z
            A2 = a * a
            Fn_5 = _rn_cond(M, x, y, Z, 5, L, gamma)
            rh = 3 * ze ** 3 * Fn_5
            Fn_7 = _rn_cond(M, x, y, Z, 7, L, gamma)
            rh += 5 * a * Z2 * ((3 * ze - 2 * a) * ze + A2) * Fn_7
            Fn_9 = _rn_cond(M, x, y, Z, 9, L, gamma)
            rh += 35 * A2 * Z2 * Z2 * ze * Fn_9
            return (b * b * iz ** 3) * (0.25 / jnp.pi) * rh

    elif mtype_idx == MTYPE_T4:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            Z2 = Z * Z
            A2 = a * a
            AZ = a * Z
            Fn_5 = _rn_cond(M, x, y, Z, 5, L, gamma)
            rh = 3 * Fn_5
            Fn_7 = _rn_cond(M, x, y, Z, 7, L, gamma)
            rh += 15 * AZ * Fn_7
            rh *= ze ** 3
            Fn_9 = _rn_cond(M, x, y, Z, 9, L, gamma)
            rh += 7 * A2 * Z * Z2 * ((6 * ze - 3 * a) * ze + A2) * Fn_9
            Fn_11 = _rn_cond(M, x, y, Z, 11, L, gamma)
            rh += 63 * A2 * AZ * Z2 * Z2 * ze * Fn_11
            return (b * b * iz ** 3) * (0.25 / jnp.pi) * rh

    elif mtype_idx == MTYPE_V0:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            Z2 = Z * Z
            hB2 = 0.5 * b * b
            F0_1 = _rn_cond(M, x, y, ze, 1, L, gamma)
            Fn_1 = _rn_cond(M, x, y, Z, 1, L, gamma)
            rh = F0_1 - Fn_1
            F0_3 = _rn_cond(M, x, y, ze, 3, L, gamma)
            Fn_3 = _rn_cond(M, x, y, Z, 3, L, gamma)
            rh += ze * (ze * F0_3 - Z * Fn_3)
            ze2 = ze * ze
            rh *= 3 / ze2
            rh += (Fn_3 - F0_3)
            F0_5 = _rn_cond(M, x, y, ze, 5, L, gamma)
            Fn_5 = _rn_cond(M, x, y, Z, 5, L, gamma)
            rh += 3 * (ze2 * F0_5 - Z2 * Fn_5)
            return (b * b * iz ** 3) * (0.25 * hB2 / (jnp.pi * a)) * rh

    elif mtype_idx == MTYPE_V1:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            Z2 = Z * Z
            AZ = a * Z
            Fn_3 = _rn_cond(M, x, y, Z, 3, L, gamma)
            rh = a * iz * iz * Fn_3
            Fn_5 = _rn_cond(M, x, y, Z, 5, L, gamma)
            rh += 3 * AZ * iz * Fn_5
            Fn_7 = _rn_cond(M, x, y, Z, 7, L, gamma)
            rh += 5 * Z * Z2 * Fn_7
            return (b * b * iz ** 3) * (0.375 * b * b / jnp.pi) * rh

    elif mtype_idx == MTYPE_V2:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            Z2 = Z * Z
            A2 = a * a
            Fn_5 = _rn_cond(M, x, y, Z, 5, L, gamma)
            rh = 3 * a * A2 * iz * iz * Fn_5
            Fn_7 = _rn_cond(M, x, y, Z, 7, L, gamma)
            rh += 5 * Z2 * (ze * ze - 2 * a * ze + 3 * A2) * iz * Fn_7
            Fn_9 = _rn_cond(M, x, y, Z, 9, L, gamma)
            rh += 35 * a * Z2 * Z2 * Fn_9
            return (b * b * iz ** 3) * (0.375 * b * b / jnp.pi) * rh

    elif mtype_idx == MTYPE_V3:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            Z2 = Z * Z
            A2 = a * a
            AZ = a * Z
            Fn_7 = _rn_cond(M, x, y, Z, 7, L, gamma)
            rh = 3 * ze ** 3 * (1 + (a * iz) ** 5) * Fn_7
            Fn_9 = _rn_cond(M, x, y, Z, 9, L, gamma)
            rh += 7 * AZ * Z2 * iz * ((3 * ze - 4 * a) * ze + 3 * A2) * Fn_9
            Fn_11 = _rn_cond(M, x, y, Z, 11, L, gamma)
            rh += 63 * A2 * Z * Z2 * Z2 * Fn_11
            return (b * b * iz ** 3) * (0.625 * b * b / jnp.pi) * rh

    elif mtype_idx == MTYPE_V4:
        def _density(x, y, z, M, a, b, L, gamma):
            ze = jnp.sqrt(z * z + b * b)
            iz = 1.0 / ze
            Z = ze + a
            Z2 = Z * Z
            A2 = a * a
            AZ = a * Z
            Fn_7 = _rn_cond(M, x, y, Z, 7, L, gamma)
            rh = 5 * ze ** 3 * Fn_7
            Fn_9 = _rn_cond(M, x, y, Z, 9, L, gamma)
            rh += 7 * a * Z2 * (5 * ze * ze - 4 * a * ze + 3 * A2 -
                                 2 * a * A2 * iz + A2 * A2 * iz * iz) * Fn_9
            Fn_11 = _rn_cond(M, x, y, Z, 11, L, gamma)
            rh += 63 * A2 * Z2 * Z2 * (ze + ze - 2 * a + A2 * iz) * Fn_11
            Fn_13 = _rn_cond(M, x, y, Z, 13, L, gamma)
            rh += 231 * A2 * AZ * Z2 * Z2 * Z * Fn_13
            return (b * b * iz ** 3) * (0.375 * b * b / jnp.pi) * rh

    # ---- Acceleration (via jax.grad of potential) ----
    def _acceleration(x, y, z, M, a, b, L, gamma):
        def pot_vec(pos):
            return _potential(pos[0], pos[1], pos[2], M, a, b, L, gamma)
        return -jax.grad(pot_vec)(jnp.array([x, y, z]))

    return (
        jax.jit(_potential),
        jax.jit(_density),
        jax.jit(_acceleration),
    )


# Instantiate all 10 model types at module level
T0_potential, T0_density, T0_acceleration = _make_explicit_fns(MTYPE_T0)
T1_potential, T1_density, T1_acceleration = _make_explicit_fns(MTYPE_T1)
T2_potential, T2_density, T2_acceleration = _make_explicit_fns(MTYPE_T2)
T3_potential, T3_density, T3_acceleration = _make_explicit_fns(MTYPE_T3)
T4_potential, T4_density, T4_acceleration = _make_explicit_fns(MTYPE_T4)
V0_potential, V0_density, V0_acceleration = _make_explicit_fns(MTYPE_V0)
V1_potential, V1_density, V1_acceleration = _make_explicit_fns(MTYPE_V1)
V2_potential, V2_density, V2_acceleration = _make_explicit_fns(MTYPE_V2)
V3_potential, V3_density, V3_acceleration = _make_explicit_fns(MTYPE_V3)
V4_potential, V4_density, V4_acceleration = _make_explicit_fns(MTYPE_V4)

# Convenience dict for programmatic access
EXPLICIT_FNS = {
    'T0': (T0_potential, T0_density, T0_acceleration),
    'T1': (T1_potential, T1_density, T1_acceleration),
    'T2': (T2_potential, T2_density, T2_acceleration),
    'T3': (T3_potential, T3_density, T3_acceleration),
    'T4': (T4_potential, T4_density, T4_acceleration),
    'V0': (V0_potential, V0_density, V0_acceleration),
    'V1': (V1_potential, V1_density, V1_acceleration),
    'V2': (V2_potential, V2_density, V2_acceleration),
    'V3': (V3_potential, V3_density, V3_acceleration),
    'V4': (V4_potential, V4_density, V4_acceleration),
}


def crossed(fn, phi):
    """
    Wrap a potential/density/acceleration function for a crossed model.

    Computes the average of fn evaluated at two bars rotated by ±phi.

    Parameters
    ----------
    fn : callable
        One of the explicit model functions, e.g. T1_potential.
        Must have signature (x, y, z, M, a, b, L, gamma) -> scalar or array.
    phi : float
        Crossed model half-angle (radians).

    Returns
    -------
    callable : crossed version with the same signature.

    Usage
    -----
    >>> crossed_pot = crossed(T1_potential, phi=0.3)
    >>> val = crossed_pot(x, y, z, M, a, b, L, gamma)
    """
    import math
    sp = abs(math.sin(phi))
    cp = abs(math.cos(phi))

    @jax.jit
    def wrapped(x, y, z, M, a, b, L, gamma):
        xf = cp * x + sp * y
        yf = cp * y - sp * x
        vf = fn(xf, yf, z, M, a, b, L, gamma)
        xb = cp * x - sp * y
        yb = cp * y + sp * x
        vb = fn(xb, yb, z, M, a, b, L, gamma)
        return 0.5 * (vf + vb)
    return wrapped
