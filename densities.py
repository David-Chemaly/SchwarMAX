import jax
import jax.numpy as jnp

from utils import get_mat

from constants import EPSILON

# ---------- helpers ----------
def _shift(x, y, z, p):
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

def _rotate(vec, p):
    # vec: (3, ...)
    R = get_mat(p["dirx"], p["diry"], p["dirz"])  # (3,3)
    # Tensordot over axis: (i,a) * (a,...) -> (i,...)
    return jnp.tensordot(R, vec, axes=[[1],[0]])

# ---------- Miyamoto-Nagai Disk ----------
# rho = (b^2 M / 4π) * [ a R^2 + (a + 3β)(a + β)^2 ] / [ β^3 * (R^2 + (a + β)^2)^(5/2) ] 
# where β = sqrt(z^2 + b^2).
@jax.jit
def MiyamotoNagai_density(x, y, z, params):
    '''
    params: dict with keys 'logM', 'a', 'b', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    # Shift and rotate coordinates
    rin = _shift(x, y, z, params)       # (3, ...)
    rvec = _rotate(rin, params)         # (3, ...)
    rx, ry, rz = rvec + EPSILON

    # Cylindrical R in rotated frame
    R = jnp.sqrt(rx**2 + ry**2)

    # Vertical scale height uses rz (IMPORTANT FIX)
    beta = jnp.sqrt(rz**2 + params["b"]**2)

    D2 = R*R + (params["a"] + beta)**2
    num = params["a"] * R*R + (params["a"] + 3.0*beta) * (params["a"] + beta)**2
    den = beta**3 * D2**2.5

    return (params["b"]**2 * 10.0**params["logM"] / (4 * jnp.pi)) * (num / den)

# ---------- Double Exponential Disk ----------
@jax.jit
def DoubleExponentialDisk_density(x, y, z, params):
    """Volume density for a simple exponential disc: rho(R,z) = (Sigma0/(2hz)) e^{-R/Rd} e^{-|z|/hz}."""
    # Shift and rotate coordinates
    rin = _shift(x, y, z, params)       # (3, ...)
    rvec = _rotate(rin, params)         # (3, ...)
    rx, ry, rz = rvec + EPSILON

    # Cylindrical R in rotated frame
    R = jnp.sqrt(rx**2 + ry**2)

    return (params['Sigma0'] / (2.0 * params['hz'])) * jnp.exp(-R / params['Rd']) * jnp.exp(-jnp.abs(rz) / params['hz'])

# ---------- Ferrers Bar ----------
@jax.jit
def FerrersBar_density(x, y, z, params):
    """
    Ferrers bar: rho = rho0 (1 - m^2)^n  for m^2 = x'^2/a^2 + y'^2/b^2 + z'^2/c^2 < 1, else 0.
    The bar major axis a is rotated in the XY plane by phi_bar (radians) from +x.
    """
    # Shift and rotate coordinates
    rin = _shift(x, y, z, params)       # (3, ...)
    rvec = _rotate(rin, params)         # (3, ...)
    rx, ry, rz = rvec + EPSILON

    cp, sp = jnp.cos(params['phi_bar']), jnp.sin(params['phi_bar'])
    xp =  cp * rx + sp * ry
    yp = -sp * rx + cp * ry
    zp = rz
    m2 = (xp / params['a']) ** 2 + (yp / params['b']) ** 2 + (zp / params['c']) ** 2
    inside = (m2 < 1.0)
    rho = jnp.zeros_like(x, dtype=float)
    # rho = rho.at[inside].set(rho0 * (1.0 - m2[inside]) ** n)
    rho = jnp.where(inside, params['rho0'] * (1.0 - m2) ** params['n'], 0.0)
    return rho

# ---------- Double Exponential Disk x2 + Ferrers Bar  ----------
@jax.jit
def DoubleExponentialDiskx2FerrersBar_density(x, y, z, params):
    """
    Composite test density: thin+thick exponential discs + Ferrers bar.
    Returns rho(x,y,z).
    """

    thin_params = {
        'Sigma0':     params['Sigma0_thin'],
        'Rd':         params['Rd_thin'],
        'hz':         params['hz_thin'],
        'x_origin':   params['x_origin_thin'],
        'y_origin':   params['y_origin_thin'],
        'z_origin':   params['z_origin_thin'],
        'dirx':       params['dirx_thin'],
        'diry':       params['diry_thin'],
        'dirz':       params['dirz_thin'],
    }
    rho_thin  = DoubleExponentialDisk_density(x, y, z, thin_params)

    thick_params = thick_params = {
        'Sigma0':     params['Sigma0_thick'],
        'Rd':         params['Rd_thick'],
        'hz':         params['hz_thick'],
        'x_origin':   params['x_origin_thick'],
        'y_origin':   params['y_origin_thick'],
        'z_origin':   params['z_origin_thick'],
        'dirx':       params['dirx_thick'],
        'diry':       params['diry_thick'],
        'dirz':       params['dirz_thick'],
    }
    rho_thick = DoubleExponentialDisk_density(x, y, z, thick_params)

    bar_params = {
        'a':           params['a_bar'],
        'b':           params['b_bar'],
        'c':           params['c_bar'],
        'rho0':        params['rho0_bar'],
        'n':           params['n_bar'],
        'phi_bar':     params['phi_bar_deg'],
        'x_origin':    params['x_origin'],
        'y_origin':    params['y_origin'],
        'z_origin':    params['z_origin'],
        'dirx':        params['dirx'],
        'diry':        params['diry'],
        'dirz':        params['dirz'],
    }
    rho_bar   = FerrersBar_density(x, y, z, bar_params)

    return rho_thin + rho_thick + rho_bar
