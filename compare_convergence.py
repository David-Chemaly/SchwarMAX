"""
Reproduce the user's exact convergence test:
  - N_dynamical_time = 50 (FIXED)
  - N_step_per_orb varies: 100, 120, 167
  - N_max = N_step_per_orb * N_dynamical_time
  - T_total = T_orb * N_dynamical_time (SAME for all)
  - dt_init = T_orb / N_step_per_orb (varies)

Compare with:
  - N_step_per_orb = 120 (FIXED)
  - N_dynamical_time varies: 50, 83
  - N_max = N_step_per_orb * N_dynamical_time
  - T_total = T_orb * N_dynamical_time (varies)
  - dt_init = T_orb / N_step_per_orb (SAME for all)
"""
import sys
import numpy as np

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
sys.path.insert(0, path)

import jax
import jax.numpy as jnp
import pickle
import time as timer

from constants import *
from utils import *
from potentials import NFW_potential, MiyamotoNagai_potential
from dehnen_bar import T3_potential, V4_potential, T3_density, V4_density
from densities import MiyamotoNagai_density
from integrants_with_binning import _integrate_adaptive_vmap
import jax.scipy as jsp

logM_halo, logM_disc, logM_bar, logRs_halo = 11.8, 10.7, 10.1, 1.2
logRs_disc, logHs_disc, logL_bar = 0.8, -0.24, 0.3
L_bar = 10.0 ** logL_bar; a_bar = L_bar / 5.0
Hs_disc = 10.0 ** logHs_disc; b_bar = Hs_disc
V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0; GAMMA_BAR = 1.0

params_baryon = {
    'logM_disc': logM_disc, 'Rs_disc': 10.0**logRs_disc, 'Hs_disc': Hs_disc,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    'logM_bar': logM_bar, 'L_bar': L_bar, 'a_bar': a_bar, 'b_bar': b_bar,
}
params_halo_pot = {
    'logM': logM_halo, 'Rs': 10**logRs_halo, 'a': 1.0, 'b': 1.0, 'c': 1.0,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
}
params_disk_rho = {**params_baryon,
    'alpha': 30*np.pi/180, 'beta': 20*np.pi/180, 'gamma': 140*np.pi/180,
    'light_to_mass_ratio': 1.0, 'Omega_bar': 10**1.6,
}

@jax.jit
def pot_fn(x, y, z):
    mn_p = {k: params_baryon[k] for k in ['logM_disc','Rs_disc','Hs_disc','x_origin','y_origin','z_origin','dirx','diry','dirz']}
    return (NFW_potential(x, y, z, params_halo_pot) +
            MiyamotoNagai_potential(x, y, z, mn_p) +
            T3_potential(x, y, z, 10.0**logM_bar, a_bar, b_bar, L_bar, GAMMA_BAR) +
            V4_potential(x, y, z, 10.0**logM_bar, V4_A, V4_B, V4_L, V4_GAMMA))

@jax.jit
def acc_fn(x, y, z):
    return -jax.grad(lambda pos: pot_fn(pos[0], pos[1], pos[2]))(jnp.array([x, y, z]))

@jax.jit
def potential_func_full(x, y, z, pb, ph):
    return pot_fn(x, y, z)

@jax.jit
def density_func(x, y, z, params):
    mn_params = {k: params[k] for k in ['logM_disc','Rs_disc','Hs_disc','x_origin','y_origin','z_origin','dirx','diry','dirz']}
    M_bar = 10.0 ** params['logM_bar']
    return (MiyamotoNagai_density(x, y, z, mn_params) +
            T3_density(x, y, z, M_bar, params['a_bar'], params['b_bar'], params['L_bar'], GAMMA_BAR) +
            V4_density(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA))

@jax.jit
def dPhi_dz(x, y, z):
    d = 5e-3
    return (pot_fn(x, y, z+d) - pot_fn(x, y, z-d)) / (2*d)

@jax.jit
def dPhi_dR(R, z):
    d = 5e-3
    return (pot_fn(R+d, 0, z) - pot_fn(R-d, 0, z)) / (2*d)

@jax.jit
def get_jeans_moments(x_star, y_star, z_star):
    R_star = jnp.sqrt(x_star**2 + y_star**2)
    def integrand(z_prime):
        return density_func(x_star, y_star, z_prime, params_disk_rho) * dPhi_dz(x_star, y_star, z_prime)
    pts = jnp.linspace(jnp.abs(z_star), 10.0, 500)
    integrand_val = jax.vmap(integrand)(pts)
    integral_val = jsp.integrate.trapezoid(integrand_val, pts)
    nu_val = density_func(x_star, y_star, z_star, params_disk_rho)
    sigma_z2 = jnp.maximum((1.0 / (nu_val + 1e-30)) * integral_val, 0.0)
    sigma_R2 = sigma_z2
    dR = 5e-3
    def vert_pres(r_in):
        def integ(z_prime):
            return density_func(r_in, 0, z_prime, params_disk_rho) * dPhi_dz(r_in, 0, z_prime)
        pts2 = jnp.linspace(jnp.abs(z_star), 10.0, 500)
        return jsp.integrate.trapezoid(jax.vmap(integ)(pts2), pts2)
    d_nu_sigR2_dR = (vert_pres(R_star+dR) - vert_pres(R_star-dR)) / (2*dR)
    v_phi_total_sq = sigma_R2 + (R_star/(nu_val+1e-30))*d_nu_sigR2_dR + R_star*dPhi_dR(R_star, z_star)
    sigma_phi = jnp.sqrt(sigma_R2)
    v_streaming_sq = jnp.maximum(v_phi_total_sq - sigma_R2, 0.0)
    v_mean_phi = jnp.sqrt(v_streaming_sq)
    return jax.lax.cond(nu_val<=0, lambda: (0.,0.,0.,0.), lambda: (v_mean_phi, jnp.sqrt(sigma_R2), jnp.sqrt(sigma_z2), sigma_phi))

# ---- Generate ICs ----
n_orb = 1000
x_grid = np.linspace(0, 12, 1000)
R_samples = np.array(sample_from_logP(x_grid, XexpX_pdf_log(x_grid, 4.0), n_orb, jax.random.PRNGKey(10086)))
phi_samples = np.random.uniform(0, 2*np.pi, size=n_orb)
x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)
x_grid2 = np.linspace(0, 4, 1000)
z_samples = np.array(sample_from_logP(x_grid2, expX_pdf_log(x_grid2, 1.5), n_orb, jax.random.PRNGKey(10010)))
w0_pos = np.array([x_samples, y_samples, z_samples]).T

print("Computing Jeans moments...")
v_rot, sig_R, sig_z, sig_phi = jax.vmap(get_jeans_moments)(
    jnp.array(w0_pos[:,0]), jnp.array(w0_pos[:,1]), jnp.array(w0_pos[:,2]))

key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
g1, g2, g3 = [np.array(jax.random.normal(k, (n_orb,))) for k in [key1, key2, key3]]
vR, vz = g1*np.array(sig_R), g2*np.array(sig_z)
vphi = np.array(v_rot) + g3*np.array(sig_phi)
R_ic = np.sqrt(w0_pos[:,0]**2 + w0_pos[:,1]**2)
phi_ic = np.arctan2(w0_pos[:,1], w0_pos[:,0])
x_ic, y_ic, vx_ic, vy_ic = [np.array(a) for a in getCartesianFromCylindrical_clockwise(
    jnp.array(R_ic), jnp.array(phi_ic), jnp.array(vR), jnp.array(vphi))]
w0_new = jnp.array(np.array([x_ic, y_ic, w0_pos[:,2], vx_ic, vy_ic, vz]).T)

_R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
_z = w0_new[:,2]
T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0,None,None,0))(
    _R, potential_func_full, (params_baryon, params_halo_pot), _z)

alpha, beta, gamma_angle = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
rotation_matrix = makeRotationMatrix(alpha, beta, gamma_angle)
Omega_bar = params_disk_rho['Omega_bar']

with open(path + 'mock_Nbody_bar_XY_withRot.pkl', 'rb') as f:
    bin_dict = pickle.load(f)
num_per_bin = jnp.array(bin_dict['num_per_bin'])
bin_mapping = jnp.array(bin_dict['bin_mapping'])
num_Vbin = int(jnp.array(bin_dict['total_bins']).item())
v0 = jnp.array(bin_dict['v0'])
s = jnp.array(bin_dict['s'])

Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]])
Rzphi_n_grid = jnp.array([10,6,6])
xy_n_grid = jnp.array([60,40])
Rzphi_n_tot = 360

R_init = np.array(R_ic[:n_orb])
z_init = np.abs(np.array(z_samples[:n_orb]))

# ---- Integration function ----
def run_integration(N_step_per_orb, N_dynamical_time):
    N_max = N_step_per_orb * N_dynamical_time
    T_total = T_orb * N_dynamical_time
    dt_init = T_orb / N_step_per_orb
    atol, rtol = 1e-7, 1e-4

    print(f"  N_step_per_orb={N_step_per_orb}, N_dyn_time={N_dynamical_time}, "
          f"N_max={N_max}, dt_init=T_orb/{N_step_per_orb}")
    t0 = timer.time()
    out = _integrate_adaptive_vmap(
        w0_new, acc_fn, pot_fn, N_max, T_total,
        dt_init, -Omega_bar, atol, rtol, 1e-5, 0.3,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_lim_grid, xy_lim_grid,
        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
        v0, s, rotation_matrix)
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, n_accepted, T_integrated = out
    A_xy = np.array(surface_density)  # (n_orb, n_bins)
    A_h3 = np.array(h3)
    A_h4 = np.array(h4)
    N_dyn = np.array(T_integrated / T_orb)
    valid_np = np.array(valid)
    dt = timer.time() - t0
    n_valid = np.sum(valid_np > 0.5)
    print(f"    Done in {dt:.1f}s. Valid: {n_valid}/{n_orb}, "
          f"N_dyn: median={np.median(N_dyn):.1f}, min={np.min(N_dyn):.1f}")
    return {'A_xy': A_xy, 'A_h3': A_h3, 'A_h4': A_h4,
            'N_dyn': N_dyn, 'valid': valid_np}


def compare(r1, r2, label):
    A1, A2 = r1['A_xy'], r2['A_xy']
    both_valid = (r1['valid'] > 0.5) & (r2['valid'] > 0.5)
    mask = both_valid
    n_bv = np.sum(mask)

    # Per-orbit XY
    od = np.sum(np.abs(A1[mask] - A2[mask]), axis=1)
    ot = np.sum((np.abs(A1[mask]) + np.abs(A2[mask])) / 2, axis=1) + 1e-30
    frac_xy = od / ot

    # Model-level equal weights
    w = np.ones(n_bv)
    m1 = A1[mask].T @ w  # (n_bins,)
    m2 = A2[mask].T @ w
    mask_m = m1 > 1e-10
    frac_model = np.abs(m1[mask_m] - m2[mask_m]) / (m1[mask_m] + 1e-30)

    # h3, h4 model-level
    sd1 = A1[mask].T @ w
    sd2 = A2[mask].T @ w
    h3_1 = (r1['A_h3'][mask] * r1['A_xy'][mask]).T @ w / (sd1 + 1e-30)
    h3_2 = (r2['A_h3'][mask] * r2['A_xy'][mask]).T @ w / (sd2 + 1e-30)
    h4_1 = (r1['A_h4'][mask] * r1['A_xy'][mask]).T @ w / (sd1 + 1e-30)
    h4_2 = (r2['A_h4'][mask] * r2['A_xy'][mask]).T @ w / (sd2 + 1e-30)
    mask_h = sd1 > 1e-10
    dh3 = np.abs(h3_1[mask_h] - h3_2[mask_h])
    dh4 = np.abs(h4_1[mask_h] - h4_2[mask_h])

    flip_in = (~(r1['valid']>0.5)) & (r2['valid']>0.5)
    flip_out = (r1['valid']>0.5) & (~(r2['valid']>0.5))

    print(f"\n  {label}")
    print(f"    Both valid: {n_bv}, flipped: +{np.sum(flip_in)}/-{np.sum(flip_out)}")
    print(f"    Per-orbit XY:  median={np.median(frac_xy):.4f}, p95={np.percentile(frac_xy,95):.4f}, "
          f">10%={np.mean(frac_xy>0.1):.1%}, >20%={np.mean(frac_xy>0.2):.1%}")
    print(f"    Model XY:      median={np.median(frac_model):.4f}, p95={np.percentile(frac_model,95):.4f}")
    print(f"    Model |dh3|:   median={np.median(dh3):.4f}, p95={np.percentile(dh3,95):.4f}")
    print(f"    Model |dh4|:   median={np.median(dh4):.4f}, p95={np.percentile(dh4,95):.4f}")

    # By R
    R_v, z_v = R_init[mask], z_init[mask]
    print(f"    By R:")
    for lo, hi in [(0,2),(2,4),(4,8),(8,12)]:
        m = (R_v>=lo)&(R_v<hi)
        if np.sum(m)<5: continue
        print(f"      R=[{lo},{hi}): med={np.median(frac_xy[m]):.4f}, p95={np.percentile(frac_xy[m],95):.4f}, N={np.sum(m)}")


# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST A: Vary N_step_per_orb (user's original test)")
print("  N_dynamical_time=50 fixed, T_total=50*T_orb fixed")
print("=" * 70)

results_A = {}
for nsp in [100, 120, 167]:
    results_A[nsp] = run_integration(N_step_per_orb=nsp, N_dynamical_time=50)

compare(results_A[100], results_A[120], "N_step=100 vs 120 (N_max=5000 vs 6000)")
compare(results_A[120], results_A[167], "N_step=120 vs 167 (N_max=6000 vs 8350)")
compare(results_A[100], results_A[167], "N_step=100 vs 167 (N_max=5000 vs 8350)")


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST B: Vary N_dynamical_time")
print("  N_step_per_orb=120 fixed, dt_init=T_orb/120 fixed")
print("=" * 70)

results_B = {}
for ndt in [50, 83]:
    results_B[ndt] = run_integration(N_step_per_orb=120, N_dynamical_time=ndt)

compare(results_B[50], results_B[83], "N_dyn=50 vs 83 (N_max=6000 vs 9960)")
