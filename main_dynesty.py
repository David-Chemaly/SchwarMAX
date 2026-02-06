import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import pickle
import os
import corner
import matplotlib.pyplot as plt

from fit_dynesty import cpu_dynesty_fit, gpu_dynesty_fit
from prior import prior_transform
from likelihoods import dynesty_logl

def get_dict_data(path):
    df_ic = pd.read_csv(path + 'mock_initial_conditions_xyz.csv')
    df_ic = df_ic[np.sqrt(df_ic['x']**2 + df_ic['y']**2) < 10.0]
    df_ic = df_ic[np.fabs(df_ic['z']) < 3.0]

    n_particles =  20_000
    print(n_particles)
    np.random.seed(42)
    index = np.random.choice(len(df_ic['x']), size=n_particles, replace=False)
    df_ic = df_ic.iloc[index]
    # w0 = jnp.array([df_ic['x'], df_ic['y'], df_ic['z'], df_ic['vx'], df_ic['vy'], df_ic['vz']]).T
    w0 = jnp.array(df_ic[['x','y','z']].to_numpy())

    with open(path + 'mock_axisymmetric_disc_XY_withRot.pkl', 'rb') as f:
        bin_dict = pickle.load(f)

    # voronoi binning mapping and data
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

    V_data_err = jnp.where(0.1 * jnp.fabs(V_data) < 10, 10, 0.1 * V_data)
    sigma_data_err = jnp.where(0.1 * jnp.fabs(sigma_data) < 5, 5, 0.1 * sigma_data)
    h1_data_err = jnp.where(0.1 * jnp.fabs(h1_data) < 0.03, 0.03, 0.1 * jnp.fabs(h1_data))
    h2_data_err = jnp.where(0.1 * jnp.fabs(h2_data) < 0.03, 0.03, 0.1 * jnp.fabs(h2_data))
    h3_data_err = jnp.where(0.1 * jnp.fabs(h3_data) < 0.03, 0.03, 0.1 * jnp.fabs(h3_data))
    h4_data_err = jnp.where(0.1 * jnp.fabs(h4_data) < 0.03, 0.03, 0.1 * jnp.fabs(h4_data))

    df_Rzphi_data = pd.read_csv(path + 'mock_axisymmetric_disc_Rzphi.csv')
    Rzphi_density_data = jnp.array(df_Rzphi_data['mass'].to_numpy()).astype(jnp.float32)
    # with open(path + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
    #     Rzphi_density_data = pickle.load(f)

    # R_grid, z_grid, phi_grid = Rzphi_density_data['R_grid'], Rzphi_density_data['z_grid'], Rzphi_density_data['phi_grid']
    # dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    # dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    # dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    # sample_for_integration = Rzphi_density_data['sample_for_integration']
    # Rzphi_density_data = jnp.array([
    #         get_mass(R_grid[i], z_grid[i], phi_grid[i], dR, dz, dphi,
    #                 dict_data['sample_for_integration']) for i in range(len(R_grid))
    #     ]).astype(jnp.float32)


    dict_data = {
        'w0': w0,
        'v0': v0,
        's': s,
        'Rzphi_density_data': Rzphi_density_data,
        'XY_density_data': surface_density,
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
        # 'R_grid': R_grid,
        # 'z_grid': z_grid,
        # 'phi_grid': phi_grid,
        # 'dR': dR,   
        # 'dz': dz,
        # 'dphi': dphi,
        # 'sample_for_integration': sample_for_integration
    }

    return dict_data

if __name__ == "__main__":
    path = '../SchwarMAX/'
    dict_data = get_dict_data(path)

    ndim = 5
    num_Vbin = dict_data['total_bins']
    nlive = 500
    use_cpu = True
    if use_cpu:
        dns_results = cpu_dynesty_fit(dict_data, dynesty_logl, prior_transform, ndim, num_Vbin, nlive=nlive)
    else:
        dns_results = gpu_dynesty_fit(dict_data, dynesty_logl, prior_transform, ndim, num_Vbin, nlive=nlive)
    with open(f'{path}/dict_results_test.pkl', 'wb') as f:
        pickle.dump(dns_results, f)

    with open(path + '/mock_axisymmetric_disc_potential_params.pkl', 'rb') as f:
        gt_params = pickle.load(f)

    ground_truth = {
            'logM_halo': gt_params['halo_params']['logM'].item(),
            'logRs_halo': jnp.log10(gt_params['halo_params']['scaleRadius']).item(),
            'logM_disk': gt_params['disc_params']['logM'].item(),
            'logRs_disk': jnp.log10(gt_params['disc_params']['scaleRadius']).item(),
            'logHs_disk': jnp.log10(gt_params['disc_params']['scaleHeight']).item(),
    }

    # Plot and Save corner plot
    labels = ['logM', 'logRs', 'logm', 'logrs', 'loghs']
    figure = corner.corner(dns_results['samps'], 
                labels=labels,
                color='blue',
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True, 
                title_kwargs={"fontsize": 16},
                truths=[ground_truth['logM_halo'], ground_truth['logRs_halo'], ground_truth['logM_disk'], ground_truth['logRs_disk'], ground_truth['logHs_disk']],
                truth_color='red',
                )
    figure.savefig(f'{path}/corner_plot.pdf')
    plt.close(figure)

