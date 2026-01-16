import jax
import jax.numpy as jnp

@jax.jit
def gauss_hermite_coefficients_binned(v_bin_centers, counts, v0, s):
    """
    Compute Gauss-Hermite coefficients h1-h4 from binned velocity data.
    
    Parameters
    ----------
    v_bin_centers : array (N_bins,) - center of each velocity bin
    counts : array (N_bins,) - counts/histogram values in each bin
    v0 : float - reference velocity (free parameter)
    s : float - reference dispersion (free parameter)
    
    Returns
    -------
    h1, h2, h3, h4 : Gauss-Hermite coefficients
    """
    # Normalize counts to get weights
    weights = counts / jnp.sum(counts)
    
    # Standardized variable with respect to (v0, s)
    w = (v_bin_centers - v0) / s
    
    # Weighted moments of w
    w1 = jnp.sum(weights * w)
    w2 = jnp.sum(weights * w**2)
    w3 = jnp.sum(weights * w**3)
    w4 = jnp.sum(weights * w**4)
    
    # Gauss-Hermite coefficients
    h1 = w1
    h2 = (w2 - 1) / jnp.sqrt(2)
    h3 = (w3 - 3*w1) / jnp.sqrt(6)
    h4 = (w4 - 6*w2 + 3) / jnp.sqrt(24)
    
    return h1, h2, h3, h4

@jax.jit
def gauss_hermite_coefficients(v, weights, v0, s):
    """
    Compute Gauss-Hermite coefficients h1-h4 for a given reference (v0, s).
    
    Parameters
    ----------
    v : array (N,) - velocity samples
    weights : array (N,) - sample weights (will be normalized), weights = all ones if unweighted
    v0 : float - reference velocity (free parameter)
    s : float - reference dispersion (free parameter)
    
    Returns
    -------
    h1, h2, h3, h4 : Gauss-Hermite coefficients
    """
    # Normalize weights
    weights = weights / jnp.sum(weights)
    
    # Standardized variable with respect to (v0, s)
    w = (v - v0) / s
    
    # Raw moments of w
    w1 = jnp.sum(weights * w)
    w2 = jnp.sum(weights * w**2)
    w3 = jnp.sum(weights * w**3)
    w4 = jnp.sum(weights * w**4)
    
    # Gauss-Hermite coefficients (using normalized Hermite polynomials)
    h1 = w1
    h2 = (w2 - 1) / jnp.sqrt(2)
    h3 = (w3 - 3*w1) / jnp.sqrt(6)
    h4 = (w4 - 6*w2 + 3) / jnp.sqrt(24)
    
    return h1, h2, h3, h4


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