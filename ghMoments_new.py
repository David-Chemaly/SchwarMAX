import jax
import jax.numpy as jnp

# y = (x - m) / s   is the scaled variable

def gauss(x, v, s):
    y = (x-v) / s
    gaussian = (2*jnp.pi)**-0.5 / s * jnp.exp(-0.5 * y**2)
    return y, gaussian

def H0(y):
    return 1.0 + y*0

def H1(y):
    return 2**0.5 * y

def H2(y):
    m=2
    hp = H1(y)
    hpp = H0(y)
    h2 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h2

def H3(y):
    m=3
    hp = H2(y)
    hpp = H1(y)
    h3 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h3

def H4(y):
    m=4
    hp = H3(y)
    hpp = H2(y)
    h4 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h4

def H5(y):
    m=5
    hp = H4(y)
    hpp = H3(y)
    h5 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h5

def H6(y):
    m=6
    hp = H5(y)
    hpp = H4(y)
    h6 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h6


@jax.jit
def h_to_V_sigma(h1, h2, v0, s):
    """
    Convert h1, h2 to actual mean and sigma of the distribution.
    
    Parameters
    ----------
    h1, h2 : Gauss-Hermite coefficients
    v0 : reference velocity
    s : reference dispersion
    
    Returns
    -------
    v_mean, sigma : actual mean and standard deviation
    """
    v_mean = v0 + s * h1
    sigma = s * jnp.sqrt(jnp.clip(1 + jnp.sqrt(2)*h2 - h1**2, 1e-10))
    
    return v_mean, sigma

@jax.jit
def mean_sigma_to_h1h2(V_mean, sigma, v0, s):
    """
    Convert mean and sigma to h1 and h2.
    
    Parameters
    ----------
    V_mean : array or scalar - actual mean velocity
    sigma : array or scalar - actual velocity dispersion
    v0 : array or scalar - reference velocity
    s : array or scalar - reference dispersion
    
    Returns
    -------
    h1, h2 : Gauss-Hermite coefficients
    """
    h1 = (V_mean - v0) / s
    h2 = (sigma**2 / s**2 + h1**2 - 1) / jnp.sqrt(2)
    
    return h1, h2


@jax.jit
def reconstruct_gh_distribution(v, v0, s, h1, h2, h3, h4):
    """
    Reconstruct velocity distribution from Gauss-Hermite coefficients.
    
    Parameters
    ----------
    v : array (N,) - velocity values at which to evaluate the distribution
    v0 : float - reference velocity
    s : float - reference dispersion
    h1, h2, h3, h4 : Gauss-Hermite coefficients
    
    Returns
    -------
    g : array (N,) - reconstructed distribution g(v)
    """
    # Standardized variable
    w = (v - v0) / s
    
    # Normalized Hermite polynomials
    H0 = 1.0
    H1 = w
    H2 = (w**2 - 1) / jnp.sqrt(2)
    H3 = (w**3 - 3*w) / jnp.sqrt(6)
    H4 = (w**4 - 6*w**2 + 3) / jnp.sqrt(24)
    
    # Gauss-Hermite series (h0 = 1 by convention)
    series = H0 + h1*H1 + h2*H2 + h3*H3 + h4*H4
    
    # Gaussian envelope
    gaussian = jnp.exp(-0.5 * w**2) / (jnp.sqrt(2 * jnp.pi) * s)
    
    # Full distribution
    g = gaussian * series
    
    return g