import jax
import jax.numpy as jnp

from utils import get_mat

from constants import G, EPSILON

# ---------- helpers ----------
@jax.jit
def _shift(x, y, z, p):
    return jnp.array([x - p['x_origin'], y - p['y_origin'], z - p['z_origin']])

@jax.jit
def _rotate(vec, p):
    R = get_mat(p['dirx'], p['diry'], p['dirz'])
    return R @ vec  # matvec

# ---------- (Triaxial) NFW ----------
# Phi = - G M / r * log(1 + r/Rs),  r = sqrt((rx/a)^2 + (ry/b)^2 + (rz/c)^2)
@jax.jit
def NFW_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'a', 'b', 'c', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec
    r = jnp.sqrt((rx/params['a'])**2 + (ry/params['b'])**2 + (rz/params['c'])**2 + EPSILON)
    return -G * 10**params['logM'] * jnp.log(1 + r / params['Rs']) / (r + EPSILON)  # kpc^2 / Gyr^2

@jax.jit
def NFW_acceleration(x, y, z, params):
    def potential_vec(pos):
        return NFW_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def NFW_hessian(x, y, z, params):
    def potential_vec(pos):
        return NFW_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Miyamoto-Nagai Disk ----------
# Phi = - G M / sqrt(R^2 + (Rs + sqrt(z^2 + Hs^2))^2)
@jax.jit
def MiyamotoNagai_potential(x, y, z, params):
    '''
    params: dict with keys 'logM_disc', 'Rs_disc', 'Hs_disc', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec + EPSILON

    R = (rx**2 + ry**2)**0.5

    denom2 = (R**2 + (params['Rs_disc'] + (rz**2 + params['Hs_disc']**2)**0.5)**2)

    Phi = - G * 10**params['logM_disc'] / (denom2**0.5)

    return Phi

@jax.jit
def MiyamotoNagai_acceleration(x, y, z, params):
    def potential_vec(pos):
        return MiyamotoNagai_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def MiyamotoNagai_hessian(x, y, z, params):
    def potential_vec(pos):
        return MiyamotoNagai_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Quadratic Bar ----------
# Phi = A * Rs^3 * (R^2 / (Rs + R)^5) * cos(2*phi_bar) * exp(-|z|/Hs)
# where phi_bar = phi - Omega*(t - t0)
# and phi = arctan(y/x)
# and eps = 2*(t-t0)/(t1-t0) - 1
# and val = 0 if t<t0, 1 if t>t1, and 3/16*eps^5 - 5/8*eps^3 + 15/16*eps + 1/2 if t0<=t<=t1
@jax.jit
def Bar_potential(x, y, z, params):
    '''
    params: dict with keys 'A', 'Rs', 'Hs', 'Omega', 't', 't0', 't1', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)
    rx, ry, rz = rvec + EPSILON

    R   = jnp.sqrt(rx**2 + ry**2)
    phi = jnp.arctan2(ry, rx)

    phi_bar = phi - params['Omega'] * (params['t'] - params['t0'])

    eps = 2.0*(params['t'] - params['t0'])/(params['t1'] - params['t0']) - 1.0
    val = jax.lax.cond(
        params['t'] < params['t0'],
        lambda _: 0.0,
        lambda _: jax.lax.cond(
            params['t'] > params['t1'],
            lambda _: 1.0,
            lambda _: (3.0/16.0)*eps**5 - (5.0/8.0)*eps**3 + (15.0/16.0)*eps + 0.5,
            operand=None
        ),
        operand=None
    )

    amp = params['A'] * params['Rs']**3 * (R**2 / (params['Rs'] + R)**5) * val

    return amp * jnp.cos(2*phi_bar) * (jnp.exp(-(rz/params['Hs'])**2))  # kpc^2 / Gyr^2

@jax.jit
def Bar_acceleration(x, y, z, params):
    def potential_vec(pos):
        return Bar_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def Bar_hessian(x, y, z, params):
    def potential_vec(pos):
        return Bar_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

@jax.jit
def Hernquist_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec + EPSILON

    # Cylindrical R in rotated frame
    r = jnp.sqrt(rx**2 + ry**2 + rz**2)
    
    Phi = - G * 10**params['logM'] / (r + params['Rs'])

    return Phi


# ---------- Dehnen & Aly (2022) Disc-Bar ----------
# Analytic barred disc potential/density pairs from Dehnen & Aly (2022),
# MNRAS 516, 2712. Models T0-T4, V0-V4 (elementary) and D0-D4, W0-W4 (compound).
#
# params dict keys:
#   'logM_bar'       : log10(mass/Msun)
#   'Rs_bar'         : scale length a [kpc]  (same role as Rs in Miyamoto-Nagai)
#   'Hs_bar'         : scale height b [kpc]  (same role as Hs in Miyamoto-Nagai)
#   'L_bar'          : needle half-length [kpc], 0 = axisymmetric
#   'gamma_bar'      : needle slope, -1 <= gamma <= 1
#   'model_bar_idx'  : int model type index (use DEHNEN_MTYPE_NAMES)
#   'phi_bar'        : crossed-bar opening angle [rad], 0 = single bar
#   'x_origin', 'y_origin', 'z_origin' : centre position [kpc]
#   'dirx', 'diry', 'dirz' : orientation (rotation axis direction)

from dehnen_bar import (
    DehnenBar_potential as _db_potential,
    DehnenBar_density as _db_density,
    MTYPE_NAMES as DEHNEN_MTYPE_NAMES,
    _rfunc_axi, _rfunc_bar,
    _lfunc_axi, _lfunc_bar,
    MTYPE_T0, MTYPE_T1, MTYPE_T2, MTYPE_T3, MTYPE_T4,
    MTYPE_V0, MTYPE_V1, MTYPE_V2, MTYPE_V3, MTYPE_V4,
    T3_potential, T3_density, T3_acceleration
)


def _dehnen_bar_inner_params(params):
    """Convert potentials.py-style params to dehnen_bar.py internal format."""
    return {
        'M': 10.0 ** params['logM_bar'],
        'a': params['Rs_bar'],
        'b': params['Hs_bar'],
        'L': params['L_bar'],
        'gamma': params['gamma_bar'],
        'mtype_idx': params['model_bar_idx'],
        'phi': params['phi_bar'],
    }


@jax.jit
def DehnenAlyBar_potential(x, y, z, params):
    rin = _shift(x, y, z, params)
    rvec = _rotate(rin, params)
    rx, ry, rz = rvec
    # bp = _dehnen_bar_inner_params(params)
    M = 10.0 ** params['logM_bar']
    a = params['Rs_bar']
    b = params['Hs_bar']
    L = params['L_bar']
    gamma = params['gamma_bar']
    val = T3_potential(rx, ry, rz, M, a, b, L, gamma)
    return val


@jax.jit
def DehnenAlyBar_acceleration(x, y, z, params):
    def potential_vec(pos):
        return DehnenAlyBar_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi


@jax.jit
def DehnenAlyBar_hessian(x, y, z, params):
    def potential_vec(pos):
        return DehnenAlyBar_potential(pos[0], pos[1], pos[2], params)
    return jax.hessian(potential_vec)(jnp.array([x, y, z]))


def make_DehnenAlyBar_fns(mtype='V3', barred=True, crossed=False):
    """
    Build fast, specialized DehnenAlyBar potential/acceleration/hessian.

    Model type is resolved at Python time (no lax.switch), so jax.grad only
    compiles the single code path needed. ~5-7x faster than the generic
    DehnenAlyBar_acceleration.

    All numerical params (logM_bar, Rs_bar, Hs_bar, L_bar, gamma_bar, phi_bar,
    x_origin, etc.) remain traced — changing their values does NOT trigger
    recompilation.

    Only recompiles if mtype, barred, or crossed change (rare).

    Parameters
    ----------
    mtype : str
        Model type: 'T0'-'T4', 'V0'-'V4'.
    barred : bool
        True if L > 0 (bar), False if L == 0 (axisymmetric disc).
    crossed : bool
        True if phi != 0 (crossed bar).

    Returns
    -------
    dict with 'potential', 'acceleration', 'hessian' functions.
        Each has signature (x, y, z, params) -> scalar or array(3,) or (3,3).
    """
    mtype_idx = DEHNEN_MTYPE_NAMES[mtype]
    is_k0 = mtype_idx in (MTYPE_T0, MTYPE_V0)
    is_V = mtype_idx >= 5

    # Resolve barred/axisym at Python time
    if barred:
        def rn(M, x, y, z, n, L, gamma):
            return _rfunc_bar(M, x, y, z, n, L, gamma)
        def lf(M, x, y, z1, z2, L, gamma):
            return _lfunc_bar(M, x, y, z1, z2, L, gamma)
    else:
        def rn(M, x, y, z, n, L, gamma):
            return _rfunc_axi(M, x, y, z, n)
        def lf(M, x, y, z1, z2, L, gamma):
            return _lfunc_axi(M, x, y, z1, z2)

    # Potential specialized to one model type (Python-time if/elif)
    def _pot_core(x, y, z, M, a, b, L, gamma):
        ze = jnp.sqrt(z * z + b * b)
        Z = ze + a
        aZ = a * Z
        hB2 = 0.5 * b * b

        if is_k0:
            fac = 1.0 / a
            ps = lf(M * fac, x, y, Z, ze, L, gamma)
            if is_V:
                F0 = rn(M * fac, x, y, ze, 1, L, gamma)
                Fa = rn(M * fac, x, y, Z, 1, L, gamma)
                ps += (hB2 / ze) * (F0 - Fa)
            return -ps

        ph = rn(M, x, y, Z, 1, L, gamma)
        if mtype_idx == MTYPE_T1:
            pass
        elif mtype_idx == MTYPE_T2:
            ph += aZ * rn(M, x, y, Z, 3, L, gamma)
        elif mtype_idx == MTYPE_T3:
            ph += a * (Z - a / 3) * rn(M, x, y, Z, 3, L, gamma)
            ph += aZ ** 2 * rn(M, x, y, Z, 5, L, gamma)
        elif mtype_idx == MTYPE_T4:
            ph += a * (Z - 0.4 * a) * rn(M, x, y, Z, 3, L, gamma)
            ph += 0.6 * a * aZ * (Z + Z - a) * rn(M, x, y, Z, 5, L, gamma)
            ph += aZ ** 3 * rn(M, x, y, Z, 7, L, gamma)
        elif mtype_idx == MTYPE_V1:
            ph += (hB2 * Z / ze) * rn(M, x, y, Z, 3, L, gamma)
        elif mtype_idx == MTYPE_V2:
            ph += (aZ + hB2) * rn(M, x, y, Z, 3, L, gamma)
            ph += (3 * hB2 * aZ * Z / ze) * rn(M, x, y, Z, 5, L, gamma)
        elif mtype_idx == MTYPE_V3:
            ph += (a * (Z - a / 3) + hB2) * rn(M, x, y, Z, 3, L, gamma)
            ph += aZ * (aZ + 3 * hB2) * rn(M, x, y, Z, 5, L, gamma)
            ph += (5 * hB2 * aZ ** 2 * Z / ze) * rn(M, x, y, Z, 7, L, gamma)
        elif mtype_idx == MTYPE_V4:
            ph += (a * (Z - 0.4 * a) + hB2) * rn(M, x, y, Z, 3, L, gamma)
            ph += 3 * a * (0.2 * aZ * (Z + Z - a) +
                           hB2 * (Z - 0.2 * a)) * rn(M, x, y, Z, 5, L, gamma)
            ph += aZ ** 2 * (aZ + 6 * hB2) * rn(M, x, y, Z, 7, L, gamma)
            ph += (7 * hB2 * aZ ** 3 * Z / ze) * rn(M, x, y, Z, 9, L, gamma)
        return -ph

    if crossed:
        def _pot_inner(rx, ry, rz, M, a, b, L, gamma, phi):
            sp = jnp.abs(jnp.sin(phi))
            cp = jnp.abs(jnp.cos(phi))
            xf = cp * rx + sp * ry
            yf = cp * ry - sp * rx
            pf = _pot_core(xf, yf, rz, M, a, b, L, gamma)
            xb = cp * rx - sp * ry
            yb = cp * ry + sp * rx
            pb = _pot_core(xb, yb, rz, M, a, b, L, gamma)
            return G * 0.5 * (pf + pb)
    else:
        def _pot_inner(rx, ry, rz, M, a, b, L, gamma, phi):
            return G * _pot_core(rx, ry, rz, M, a, b, L, gamma)

    @jax.jit
    def potential(x, y, z, params):
        rin = _shift(x, y, z, params)
        rvec = _rotate(rin, params)
        rx, ry, rz = rvec
        return _pot_inner(
            rx, ry, rz,
            10.0 ** params['logM_bar'],
            params['Rs_bar'], params['Hs_bar'],
            params['L_bar'], params['gamma_bar'],
            params['phi_bar'],
        )

    @jax.jit
    def acceleration(x, y, z, params):
        def pot_vec(pos):
            return potential(pos[0], pos[1], pos[2], params)
        return -jax.grad(pot_vec)(jnp.array([x, y, z]))

    @jax.jit
    def hessian(x, y, z, params):
        def pot_vec(pos):
            return potential(pos[0], pos[1], pos[2], params)
        return jax.hessian(pot_vec)(jnp.array([x, y, z]))

    return {
        'potential': potential,
        'acceleration': acceleration,
        'hessian': hessian,
    }