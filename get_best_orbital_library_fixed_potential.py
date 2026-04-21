"""
Build a best-fit orbital library for the fixed-potential model (model.py).

Uses model_fixed_potential_diagnostic() to integrate orbits in a pre-computed
baryonic potential (multipole expansion stored in dict_phi_stellar_t_t0_7.pkl)
+ NFW halo, then stores the NNLS orbit weights and the full 6D trajectories.

Best-fit parameters used here:
    logM_halo           = 11.88
    logRs_halo          = log10(19.2)
    light_to_mass_ratio = 1.0
    Omega_bar           = 25 rad/Gyr
"""
from model import *

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pickle
from tqdm import tqdm
from constants import EPSILON, KPCGYR_TO_KMS

from scipy.interpolate import CubicSpline
from scipy.stats import qmc


def get_dict_data_bootstrap(path, filename, N_BOOTSTRAP=100, n_samples=5_000):

    with open(path + filename, 'rb') as f:
        bin_dict = pickle.load(f)

    X_minmax = jnp.array(bin_dict['X_minmax'])
    Y_minmax = jnp.array(bin_dict['Y_minmax'])
    nX_nY = jnp.array(bin_dict['nX_nY'])

    num_per_bin = jnp.array(bin_dict['num_per_bin'])
    total_bins = jnp.array(bin_dict['total_bins'])
    bin_mapping = jnp.array(bin_dict['bin_mapping'])
    surface_density = jnp.array(bin_dict['surface_density'])
    V_data = jnp.array(bin_dict['V_mean'])
    sigma_data = jnp.array(bin_dict['V_sigma'])
    h1_data = jnp.array(bin_dict['h1'])
    h2_data = jnp.array(bin_dict['h2'])
    h3_data = jnp.array(bin_dict['h3'])
    h4_data = jnp.array(bin_dict['h4'])
    v0 = jnp.array(bin_dict['v0'])
    s = jnp.array(bin_dict['s'])
    alpha, beta, gamma = bin_dict['orientation']

    XY_density_data_err = 0.01 * surface_density + EPSILON
    V_data_err = jnp.array(bin_dict['V_mean_err'])
    sigma_data_err = jnp.array(bin_dict['V_sigma_err'])
    h1_data_err = jnp.array(bin_dict['h1_err'])
    h2_data_err = jnp.array(bin_dict['h2_err'])
    h3_data_err = jnp.array(bin_dict['h3_err'])
    h4_data_err = jnp.array(bin_dict['h4_err'])

    rng = np.random.default_rng(42)
    XY_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(surface_density)))
    h1_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h1_data)))
    h2_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h2_data)))
    h3_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h3_data)))
    h4_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h4_data)))
    V_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(V_data)))
    sigma_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(sigma_data)))
    XY_standard_normal[0, :] = 0.0
    h1_standard_normal[0, :] = 0.0
    h2_standard_normal[0, :] = 0.0
    h3_standard_normal[0, :] = 0.0
    h4_standard_normal[0, :] = 0.0
    V_standard_normal[0, :] = 0.0
    sigma_standard_normal[0, :] = 0.0
    XY_standard_normal = jnp.array(XY_standard_normal)
    h1_standard_normal = jnp.array(h1_standard_normal)
    h2_standard_normal = jnp.array(h2_standard_normal)
    h3_standard_normal = jnp.array(h3_standard_normal)
    h4_standard_normal = jnp.array(h4_standard_normal)
    V_standard_normal = jnp.array(V_standard_normal)
    sigma_standard_normal = jnp.array(sigma_standard_normal)

    R_min, R_max = 0., 10.
    z_min, z_max = -3., 3.
    phi_min, phi_max = -jnp.pi, jnp.pi
    n_R, n_z, n_phi = 10, 6, 20
    n_tot = int(n_R * n_z * n_phi)
    R_edge = jnp.linspace(R_min, R_max, n_R + 1)
    z_edge = jnp.linspace(z_min, z_max, n_z + 1)
    phi_edge = jnp.linspace(phi_min, phi_max, n_phi + 1)
    R_mids, z_mids, phi_mids = 0.5 * (R_edge[:-1] + R_edge[1:]), 0.5 * (z_edge[:-1] + z_edge[1:]), 0.5 * (phi_edge[:-1] + phi_edge[1:])
    R_mids_mesh, z_mids_mesh, phi_mids_mesh = jnp.meshgrid(R_mids, z_mids, phi_mids, indexing='ij')
    Rzphi_mid_grid = jnp.stack([R_mids_mesh.ravel(), z_mids_mesh.ravel(), phi_mids_mesh.ravel()], axis=-1)
    R_grid = Rzphi_mid_grid[:, 0]
    z_grid = Rzphi_mid_grid[:, 1]
    phi_grid = Rzphi_mid_grid[:, 2]
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    Rzphi_minmax = jnp.array([[R_min, R_max], [z_min, z_max], [phi_min, phi_max]])
    nRzphi = jnp.array([n_R, n_z, n_phi])
    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_grid_indices = assign_regular_grid(Rzphi_mid_grid,
                                             grid_min=Rzphi_minmax[:, 0],
                                             grid_max=Rzphi_minmax[:, 1],
                                             n_bins=nRzphi,
                                             strides=Rzphi_strides)
    argsort = jnp.argsort(Rzphi_grid_indices)
    R_grid = R_grid[argsort]
    z_grid = z_grid[argsort]
    phi_grid = phi_grid[argsort]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample_for_integration = sampler.random_base2(m=10)

    X_regular_grid, Y_regular_grid = bin_dict['X_regular_grid'], bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)

    x_grid = np.linspace(0., 12., 1000)
    logP_xexp = XexpX_pdf_log(x_grid, 4.0)
    key = jax.random.PRNGKey(10086)
    R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.default_rng(42).uniform(0, 2 * np.pi, size=n_samples)
    x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)

    x_grid = np.linspace(0, 4, 1000)
    logP_exp = expX_pdf_log(x_grid, 1.5)
    key = jax.random.PRNGKey(10010)
    z_samples = sample_from_logP(x_grid, logP_exp, n_samples, key)
    w0 = np.array([x_samples, y_samples, z_samples]).T

    dict_data = {
        'v0': v0,
        's': s,

        'XY_density_data': surface_density,
        'XY_density_data_err': XY_density_data_err,
        'V_data': V_data,
        'V_data_err': V_data_err,
        'sigma_data': sigma_data,
        'sigma_data_err': sigma_data_err,
        'h1_data': h1_data,
        'h1_data_err': h1_data_err,
        'h2_data': h2_data,
        'h2_data_err': h2_data_err,
        'h3_data': h3_data,
        'h3_data_err': h3_data_err,
        'h4_data': h4_data,
        'h4_data_err': h4_data_err,
        'num_per_bin': num_per_bin,
        'bin_mapping': bin_mapping,
        'total_bins': total_bins.item(),

        'XY_standard_normal': XY_standard_normal,
        'h1_standard_normal': h1_standard_normal,
        'h2_standard_normal': h2_standard_normal,
        'h3_standard_normal': h3_standard_normal,
        'h4_standard_normal': h4_standard_normal,
        'V_standard_normal': V_standard_normal,
        'sigma_standard_normal': sigma_standard_normal,

        'R_grid': R_grid,
        'z_grid': z_grid,
        'phi_grid': phi_grid,
        'R_minmax': [R_min, R_max],
        'z_minmax': [z_min, z_max],
        'phi_minmax': [-jnp.pi, jnp.pi],
        'Rzphi_n_tot': n_tot,
        'Rzphi_n_grid': jnp.array([n_R, n_z, n_phi]),
        'dR': dR,
        'dz': dz,
        'dphi': dphi,
        'sample_for_integration': sample_for_integration,

        'X_regular_grid': X_regular_grid,
        'Y_regular_grid': Y_regular_grid,
        'dX': dX,
        'dY': dY,
        'sample_for_integration_XY': sample,

        'X_minmax': X_minmax,
        'Y_minmax': Y_minmax,
        'nX_nY': nX_nY,

        # required by model_fixed_potential_diagnostic (read from dict_data, not params)
        'alpha': float(alpha),
        'beta': float(beta),
        'gamma': float(gamma),

        'w0': w0,
    }

    return dict_data


if __name__ == '__main__':

    from model import model_fixed_potential_diagnostic

    path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
    data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
    potential_file = data_folder + '/dict_phi_stellar_t_t0_7.pkl'

    # Observational data — same mock as the bar-model script
    data_filename = 'mock_data/mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma140_D50_gal2.pkl'

    output_filename = data_folder + '/best_fit_orbital_library_fixed_potential_t_t0_7.pkl'

    # ── Load observational data ──
    dict_data = get_dict_data_bootstrap(path, data_filename, n_samples=10_000, N_BOOTSTRAP=10)

    # ── Load fixed baryonic potential (multipole expansion) ──
    with open(potential_file, 'rb') as f:
        _d = pickle.load(f)
    dict_phi_baryon = {k: jnp.array(v) for k, v in _d.items() if k != '_metadata'}

    # ── Best-fit parameters ──
    logM_halo = 11.88
    logRs_halo = jnp.log10(19.2).item()
    light_to_mass_ratio = 1.0
    Omega_bar = 25.0

    params_halo_pot = {
        'logM': logM_halo,
        'Rs': 10 ** logRs_halo,
        'a': 1.0,
        'b': 1.0,
        'c': 1.0,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
    }

    print('Halo logM, Rs         :', logM_halo, 10 ** logRs_halo)
    print('light_to_mass_ratio   :', light_to_mass_ratio)
    print('Omega_bar             :', Omega_bar)
    print('Viewing angles (deg)  :', dict_data['alpha'], dict_data['beta'], dict_data['gamma'])

    # ── Run diagnostic ──
    X_minmax = dict_data['X_minmax']
    Y_minmax = dict_data['Y_minmax']
    nX, nY = dict_data['nX_nY']
    xy_lim_grid = jnp.array([X_minmax, Y_minmax])
    xy_n_grid = jnp.array([nX, nY])

    Rmin, Rmax = dict_data['R_minmax']
    zmin, zmax = dict_data['z_minmax']
    phimin, phimax = dict_data['phi_minmax']
    Rzphi_n_tot = dict_data['Rzphi_n_tot']
    Rzphi_n_grid = dict_data['Rzphi_n_grid']

    output_dict = model_fixed_potential_diagnostic(
        params_halo_pot, dict_phi_baryon,
        Omega_bar, light_to_mass_ratio,
        dict_data, dict_data['total_bins'],
        Rzphi_n_tot=Rzphi_n_tot,
        Rzphi_n_grid=Rzphi_n_grid,
        Rzphi_lim_grid=jnp.array([[Rmin, Rmax], [zmin, zmax], [phimin, phimax]]),
        xy_lim_grid=xy_lim_grid,
        xy_n_grid=xy_n_grid,
    )

    logL = output_dict['logl_all'][0]
    print('logL:', logL)
    print('Model Done')

    # ── Repack trajectories on a uniform time grid (same as the bar-model script) ──
    weights = output_dict['weights']   # (n_orbits,)
    o_traj = output_dict['y_traj']     # (n_orbits, n_realizations, N_max, 6)
    t_traj = output_dict['t_traj']     # (n_orbits, n_realizations, N_max)

    weights = np.repeat(weights, o_traj.shape[1])
    o_traj = o_traj.reshape(-1, o_traj.shape[2], 6)
    t_traj = t_traj.reshape(-1, t_traj.shape[2])

    print('weights shape', weights.shape)
    print('orbits shape ', o_traj.shape)

    n_time = 1000
    t_orb, x_orb, y_orb, z_orb, vx_orb, vy_orb, vz_orb = [], [], [], [], [], [], []
    for i in tqdm(range(o_traj.shape[0])):
        t_old = t_traj[i]
        o_old = o_traj[i]

        x_old = o_old[:, 0]
        y_old = o_old[:, 1]
        z_old = o_old[:, 2]
        vx_old = o_old[:, 3]
        vy_old = o_old[:, 4]
        vz_old = o_old[:, 5]

        keep = np.concatenate(([True], np.diff(t_old) > 0))
        t_old = t_old[keep]
        x_old = x_old[keep]
        y_old = y_old[keep]
        z_old = z_old[keep]
        vx_old = vx_old[keep]
        vy_old = vy_old[keep]
        vz_old = vz_old[keep]

        t_new = np.linspace(t_old[0], t_old[-1], n_time)

        try:
            x_interp = CubicSpline(t_old, x_old)
            y_interp = CubicSpline(t_old, y_old)
            z_interp = CubicSpline(t_old, z_old)
            vx_interp = CubicSpline(t_old, vx_old)
            vy_interp = CubicSpline(t_old, vy_old)
            vz_interp = CubicSpline(t_old, vz_old)

            t_orb.append(t_new)
            x_orb.append(x_interp(t_new))
            y_orb.append(y_interp(t_new))
            z_orb.append(z_interp(t_new))
            vx_orb.append(vx_interp(t_new))
            vy_orb.append(vy_interp(t_new))
            vz_orb.append(vz_interp(t_new))
        except Exception as e:
            print(e)
            print('t_old:', t_old)
            zeros = np.zeros(n_time)
            t_orb.append(t_new)
            x_orb.append(zeros)
            y_orb.append(zeros)
            z_orb.append(zeros)
            vx_orb.append(zeros)
            vy_orb.append(zeros)
            vz_orb.append(zeros)

    with open(output_filename, 'wb') as f:
        pickle.dump({
            't_orb': t_orb,
            'x_orb': x_orb,
            'y_orb': y_orb,
            'z_orb': z_orb,
            'vx_orb': vx_orb,
            'vy_orb': vy_orb,
            'vz_orb': vz_orb,
            'weights': weights,
            'rotation_matrix': output_dict['rotation_matrix'],
            'Omega_bar': output_dict['Omega_bar'],
            'logl_all': output_dict['logl_all'],
            'params_halo_pot': params_halo_pot,
            'light_to_mass_ratio': light_to_mass_ratio,
        }, f)

    print('Saved orbital library →', output_filename)
