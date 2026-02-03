import jax
import jax.numpy as jnp

def prior_transform(p):
    #ndim = 5
    logM, logRs, logm, logrs, loghs = p

    logM1  = 10 + 4 * logM
    logRs1 = 0.5 + logRs

    logm1  = 9 + 3 * logm
    logrs1 = logrs
    loghs1 = -1 + loghs 
    
    return [logM1, logRs1, logm1, logrs1, loghs1]