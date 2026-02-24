from likelihoods import *
from utils import *

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnn
import pandas as pd
import numpy as np
import pickle

import emcee
import corner

from constants import EPSILON

def get_dict_data(path):
    df_ic = pd.read_csv(path + 'mock_initial_conditions_xyz.csv')
    df_ic = df_ic[np.sqrt(df_ic['x']**2 + df_ic['y']**2) < 15.0]
    df_ic = df_ic[np.fabs(df_ic['z']) < 4.0]

    n_particles =  20_000
    print(n_particles)
    np.random.seed(42)
    index = np.random.choice(len(df_ic['x']), size=n_particles, replace=False)
    df_ic = df_ic.iloc[index]
    # w0 = jnp.array([df_ic['x'], df_ic['y'], df_ic['z'], df_ic['vx'], df_ic['vy'], df_ic['vz']]).T
    w0 = jnp.array(df_ic[['x','y','z']].to_numpy())

    with open(path + 'mock_SCM_disc_XY_withRot.pkl', 'rb') as f:
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

    # df_Rzphi_data = pd.read_csv(path + 'mock_axisymmetric_disc_Rzphi.csv')
    # Rzphi_density_data = jnp.array(df_Rzphi_data['mass'].to_numpy()).astype(jnp.float32)
    with open(path + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
        Rzphi_density_data = pickle.load(f)

    R_grid, z_grid, phi_grid = Rzphi_density_data['R_grid'], Rzphi_density_data['z_grid'], Rzphi_density_data['phi_grid']
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    sample_for_integration = Rzphi_density_data['sample_for_integration']

    from scipy.stats import qmc
    X_regular_grid, Y_regular_grid = bin_dict['X_regular_grid'], bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)


    dict_data = {
        'w0': w0,
        'v0': v0,
        's': s,

        # 'Rzphi_density_data': Rzphi_density_data,
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

        'R_grid': R_grid,
        'z_grid': z_grid,
        'phi_grid': phi_grid,
        'dR': dR,
        'dz': dz,
        'dphi': dphi,
        'sample_for_integration': sample_for_integration,

        'X_regular_grid': X_regular_grid,
        'Y_regular_grid': Y_regular_grid,
        'dX': dX,
        'dY': dY,
        'sample_for_integration_XY': sample,
    }

    return dict_data

if __name__ == "__main__":
    path = '/home/hz420/python_script/SchwarMAX/'
    dict_data = get_dict_data(path)

    def log_prior(theta,):
        if (6 < theta[0] < 10) and (-1 < theta[1] < 2) and (-1 < theta[2] < 1)\
        and (0 <= theta[3] < jnp.pi) and (0 <= theta[4] < jnp.pi/2) and (0 <= theta[5] < jnp.pi):
            return 0.0  # log(1) = 0 for uniform prior
        return -np.inf  # log(0) = -inf for out-of-bounds

    def log_prob(theta,):
        # print(theta)
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = logl_density(theta, dict_data, dict_data['total_bins'])

        return ll + lp
    
    ndim = 6
    nwalkers = 16  # must be >= 2 * ndim

    # Initialize walkers around ground truth
    # p0 = np.array([ground_truth[k] for k in param_names])
    p0 = np.array([9.2, 0.3, 0., jnp.pi/4, jnp.pi/4, jnp.pi/4])
    # initial_pos = p0 + 1e-1 * np.random.randn(nwalkers, ndim)
    np.random.seed(42)
    initial_pos = p0 + np.random.uniform(-0.3, 0.3, (nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_pos, 500, progress=True)

    samples = sampler.get_chain(discard=200, flat=True)

    params_bestfit = np.percentile(samples, axis=0, q=50)
    logl_val = logl_density(params_bestfit, dict_data, dict_data['total_bins'])
    print('Best-fit logL projection', logl_val)#

    dict_data['logl_density_max'] = logl_val
    logrho0_best_fit, logRs_disk_best_fit, logHs_disk_best_fit, alpha_best_fit, beta_best_fit, gamma_best_fit = params_bestfit

    # dict_data['logl_density_max'] = -0.24
    # logrho0_best_fit, logRs_disk_best_fit, logHs_disk_best_fit, alpha_best_fit, beta_best_fit, gamma_best_fit = (9,0.45,-0.24,0.54,0.36,1.34)
    # disc_mass_tot = 10**logHs_disk_best_fit * 4 * np.pi * 10**(2*logRs_disk_best_fit) * 10**logHs_disk_best_fit  # Total mass from best-fit parameters


    alpha = alpha_best_fit
    beta = beta_best_fit
    gamma = gamma_best_fit
    ground_truth = [11.5,
                    logrho0_best_fit,
                    jnp.log10(19).item(),
                    logRs_disk_best_fit,
                    logHs_disk_best_fit,
                    alpha,
                    beta,
                    gamma,
                    0.
    ]
    logL = logl_angular_input(ground_truth, dict_data, dict_data['total_bins'])
    print(logL)



    prior_uniform_low =  [ground_truth[0] - 3,
                        ground_truth[1] - 3,
                        ground_truth[2]- 1,
                        ground_truth[3]- 1,
                        ground_truth[4]- 1,
                        0,
                        0,
                        0,
                        -2
                        ]
    prior_uniform_high = [ground_truth[0] + 3,
                        ground_truth[1] + 3,
                        ground_truth[2]+ 1,
                        ground_truth[3]+ 1,
                        ground_truth[4]+ 1,
                        jnp.pi,
                        jnp.pi/2,
                        jnp.pi,
                        2
                        ]

    def log_prior(params):
        lp = 0
        for i in range (0, 9):
            if (params[i]<=prior_uniform_low[i]) & (params[i]>=prior_uniform_high[i]):
                lp+= -jnp.inf
        return lp

    def log_prob(theta):
        params = theta
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = float(logl_angular_input(params, dict_data, dict_data['total_bins']))  # convert from JAX array
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
    
    ndim = 9
    nwalkers = 18  # must be >= 2 * ndim

    # Initialize walkers around ground truth
    # p0 = np.array([ground_truth[k] for k in param_names])
    p0 = ground_truth
    # initial_pos = p0 + 1e-1 * np.random.randn(nwalkers, ndim)
    initial_pos = p0 + np.random.uniform(-0.3, 0.3, (nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_pos, 300, progress=True)

    samples = sampler.get_chain(discard=100, flat=True)

    param_names = ['logM_halo','logM_disk', 'logRs_halo', 'logRs_disk', 'logHs_disk', 'alpha', 'beta', 'gamma', 'log_light_to_mass_ratio']
    import pandas as pd
    pd.DataFrame(samples, columns=param_names).to_csv(path+'/test_posterior_0218.csv', index=False)

    samples_raw = sampler.get_chain(discard=0, flat=False)
    with open(path+'/test_posterior_WholeChain_0218.pkl', 'wb') as f:
        pickle.dump(samples_raw, f)