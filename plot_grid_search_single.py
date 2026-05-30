import pickle

import time as tt
import numpy as np
import scipy as sp
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import os
from constants import KPCGYR_TO_KMS

from scipy.interpolate import griddata
# path = '/home/hz420/python_script/SchwarMAX/'
# path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
# path = '/data/hz420-2/SchwarMAX/'

def plot_prettier(dpi=200, fontsize=20, usetex=False):
    '''
    Make plots look nicer compared to Matplotlib defaults
    Parameters:
        dpi - int, "dots per inch" - controls resolution of PNG images that are produced
                by Matplotlib
        fontsize - int, font size to use overall
        usetex - bool, whether to use LaTeX to render fonds of axes labels
                use False if you don't have LaTeX installed on your system
    '''
    plt.rcParams['figure.dpi']= dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    # if you don't have LaTeX installed on your laptop and this statement
    # generates error, comment it out
    plt.rc('text', usetex=usetex)
    #fm.fontManager.addfont('Serif.ttf')
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plot_prettier(usetex=False)

data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
data_folder2 = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/posteriors/'
FIG_NAME = data_folder+'/plots/grid_search_single.png'
FIG_NAME_PAPER = data_folder+'/figs_paper/grid_search_single.pdf'

param_names = ['log_light_to_mass_ratio', 'log_Omega']

ground_truth = {
    'light_to_mass_ratio': 1,
    'Omega': 25.0
}

beta_ls = [25]
gamma_ls = [140]
color_ls = ['royalblue', 'green', 'orange']

fig, ax = plt.subplots(1,1, figsize=(5*len(beta_ls), 5))
for i, beta in enumerate(beta_ls):

    ax.scatter(np.log10(1/ground_truth['light_to_mass_ratio']), ground_truth['Omega'], 
            color='tomato', marker='*', s=100, edgecolor = 'tomato', label='Ground Truth', zorder = 100)
        
    for j, gamma in enumerate(gamma_ls):
        try:
            # df_param = pd.read_csv(data_folder+f'/grid_search_0422/grid_search_result_0422_beta{beta}_gamma{gamma}_D50_gal2.csv')
            df_param = pd.read_csv(data_folder+f'/grid_search_result_0519_beta{beta}_gamma{gamma}_D50_gal2.csv')
            df_param = df_param[np.isfinite(df_param['log_prob'])]
            df_param['log_prob'] = df_param['log_prob'] - np.max(df_param['log_prob'])
        except:
            print(f"No data for beta={beta}, gamma={gamma}")
            continue
            
        samples_plot = np.zeros((df_param.shape[0], len(param_names)))
        samples_plot[:,0] = -df_param['log_light_to_mass_ratio'].to_numpy()
        samples_plot[:,1] = 10 ** df_param['log_Omega'].to_numpy() * KPCGYR_TO_KMS
        
        # cb = ax[i].scatter(samples_plot[:,0], samples_plot[:,1], c=df_param['log_prob'] / 300, 
        #                 cmap='Oranges', vmin = -20, marker='o', s = 20, edgecolor = 'lightgrey', ls = '--', lw = 0.4, alpha = 0.5)
        cb = ax.scatter(samples_plot[:,0], samples_plot[:,1], color = 'lightgrey',
                     marker='o', s = 5, edgecolor = 'lightgrey', ls = '--', lw = 0.4, alpha = 0.5)
        ax.set_xlabel(r'$\log(\Upsilon)$')
        ax.set_ylabel(r'$\Omega$')

        idx = np.argmax(df_param['log_prob'])
        ax.scatter(samples_plot[idx,0], samples_plot[idx,1], 
                    color=color_ls[j], marker='s', s=40, edgecolor = color_ls[j], label=r'Best-fit parameters')
                
        # Draw contour of logL
        xi = np.linspace(samples_plot[:,0].min(), samples_plot[:,0].max(), 100)
        yi = np.linspace(samples_plot[:,1].min(), samples_plot[:,1].max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)

        log_prob_smooth = sp.ndimage.gaussian_filter(df_param['log_prob'].to_numpy(), sigma=0.)
        Zi = griddata((samples_plot[:,0], samples_plot[:,1]), log_prob_smooth / 100, (Xi, Yi), method='cubic')
        ax.contour(Xi, Yi, Zi, levels=[-15, -6, -1.15], colors=color_ls[j], linewidths=1, alpha=0.99, linestyles='-',
                      )
        ax.legend(loc = 'upper right', fontsize = 15)
        ax.set(xlim=[-0.3, 0.3], ylim=[15, 40])
        print(f"beta={beta}, gamma={gamma}", 'Best-fit (L/M, Omega):', 10**samples_plot[np.argmax(df_param['log_prob']),0], samples_plot[np.argmax(df_param['log_prob']),1])

plt.tight_layout()
plt.savefig(FIG_NAME, dpi=300)
plt.savefig(FIG_NAME_PAPER, dpi=300)
plt.show()
plt.close()
