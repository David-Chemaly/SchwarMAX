"""
Test determinism of the analytic marginalize_weights function.
"""

import pickle
import time
import numpy as np

import jax
import jax.numpy as jnp

from model_bar import solve_nnls_admm, marginalize_weights


path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
lib_file = 'orbital_library_adaptive_Nmax1e4.pkl'

with open(path_data + lib_file, 'rb') as f:
    orb_lib = pickle.load(f)

(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4) = orb_lib[:18]

weights = solve_nnls_admm(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=200,
)
weights.block_until_ready()

# Call 10 times, check determinism
results = []
for i in range(10):
    logl = marginalize_weights(
        weights, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
        lambda_reg=1.0)
    val = float(logl)
    results.append(val)
    print(f"  Call {i}: logL_marg = {val:.6f}")

print(f"\n  Range: {max(results) - min(results):.6f}")
print(f"  All identical: {len(set(results)) == 1}")
