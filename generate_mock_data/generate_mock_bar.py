path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
# path = '/home/hz420/python_script/SchwarMAX/'

data_folder = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
# sim_folder = '/data/hz420-2/SchwarMAX/Bar_model_TG21/model'

import sys
sys.path.append(path)

import numpy as np
import scipy as sp

import pickle
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

from potentials import *
from integrants_with_binning import *
from ghMoments import *
from utils import *
from constants import EPSILON

import agama
agama.setUnits(length=1, velocity=1, mass=1)

from astropy.constants import G
import astropy.units as u

# Load data
import numpy as np
from powerbin import PowerBin

def gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def Rz(t):
    ct, st = jnp.cos(t), jnp.sin(t)
    return jnp.array([[ct, -st, 0.0],
                     [st,  ct, 0.0],
                     [0.0, 0.0, 1.0]])

def Rx(t):
    ct, st = jnp.cos(t), jnp.sin(t)
    return jnp.array([[1.0, 0.0, 0.0],
                     [0.0,  ct, -st],
                     [0.0,  st,  ct]])

def makeRotationMatrix(alpha, beta, gamma):

    alpha, beta, gamma = jnp.radians(alpha), jnp.radians(beta), jnp.radians(gamma)
    return (Rz(gamma) @ Rx(beta) @ Rz(alpha)).T   # X = R @ x

def bar_angle_bar_strength(x, y, R_anulus = np.arange(1,5,0.25)):

    ''' 
    Calculate the angle and strength of the dipole strength in each radial bin.
    Parameters:
        x, y - ndarray, Cartesian coordinates of the stars
        R_anulus - ndarray, edges of the radial bins to use for calculating the dipole angle and strength
    Returns:
        R_mid - ndarray, midpoints of the radial bins
        bar_angles0 - ndarray, angles of the dipole strength in each radial bin
        bar_strength0 - ndarray, strengths of the dipole strength in each radial bin

    '''

    R0,phi0 = np.sqrt(x**2 + y**2), np.arctan2(y,x)

    bar_angles0 = []
    bar_strength0 = []
    R_anulus = R_anulus
    for i in range(len(R_anulus)-1):
        sel0 = (R0>R_anulus[i]) & (R0<R_anulus[i+1])

        r_min_i = (R_anulus[i]+R_anulus[i+1])/2
        
        R0_bin, phi0_bin = R0[sel0], phi0[sel0]

        A2_0 = np.sum(np.cos(2*phi0_bin))/len(phi0_bin)
        B2_0 = np.sum(np.sin(2*phi0_bin))/len(phi0_bin)

        angle_bar0 = 0.5*np.arctan2(B2_0,A2_0)
        bar_strength = np.sqrt(A2_0**2 + B2_0**2)

        bar_angles0.append(angle_bar0)
        bar_strength0.append(bar_strength
                             )
    R_mid = (R_anulus[:-1]+R_anulus[1:])/2
    bar_angles0 = np.array(bar_angles0)
    bar_strength0 = np.array(bar_strength0)

    return R_mid, bar_angles0, bar_strength0

def rotate(posvel, angle):
    # Rotate contourclockwise with positive angle
    x, y, z, vx, vy, vz = posvel.T
    sina, cosa = np.sin(angle), np.cos(angle)
    return np.array([x*cosa-y*sina, x*sina+y*cosa, z, vx*cosa-vy*sina, vx*sina+vy*cosa, vz]).T

# with open('/data/hz420-2/SchwarMAX/mock_axisymmetric_disc.pkl', 'rb') as f:
#     data = pickle.load(f)
# w0_data = np.array([data['x'], data['y'], data['z'], data['vx'], data['vy'], data['vz']]).T
# mass_data = data['mass']

mass_unit = 1/((G*u.Msun).to(u.kpc*(u.km/u.s)**2))
w0_data, mass_data = agama.readSnapshot(data_folder+'/Bar_model_TG21/model/t_t0_7')#snap_t0_3
mass_data = mass_data * mass_unit.value

mask = (mass_data!=np.unique(mass_data)[-1])
w0_data = w0_data[mask]
mass_data = mass_data[mask]
print(np.unique(mass_data, return_counts=True))

R = np.sqrt(w0_data[:,0]**2 + w0_data[:,1]**2)
mask = R < 5
w0_data[:,0] = w0_data[:,0] - np.sum(w0_data[mask,0] * mass_data[mask]) / np.sum(mass_data[mask])
w0_data[:,1] = w0_data[:,1] - np.sum(w0_data[mask,1] * mass_data[mask]) / np.sum(mass_data[mask])
w0_data[:,2] = w0_data[:,2] - np.sum(w0_data[mask,2] * mass_data[mask]) / np.sum(mass_data[mask])
w0_data[:,3] = w0_data[:,3] - np.sum(w0_data[mask,3] * mass_data[mask]) / np.sum(mass_data[mask])
w0_data[:,4] = w0_data[:,4] - np.sum(w0_data[mask,4] * mass_data[mask]) / np.sum(mass_data[mask])
w0_data[:,5] = w0_data[:,5] - np.sum(w0_data[mask,5] * mass_data[mask]) / np.sum(mass_data[mask])

w0_data[:,0] = -w0_data[:,0]
w0_data[:,3] = -w0_data[:,3]

R_mid, bar_angles0, bar_strength0 = bar_angle_bar_strength(w0_data[:,0], w0_data[:,1], R_anulus = np.arange(1,5,0.25))
bar_angle0 = np.mean(bar_angles0[R_mid<4])
rot_angle = -bar_angle0
w0_data = rotate(w0_data, rot_angle)  # rotate to make it anticlockwise


alpha, beta, gamma = 30, 20, 150
rot_mat = makeRotationMatrix(alpha, beta, gamma)
x_data, v_data = w0_data[:,:3], w0_data[:,3:]
x_data = (rot_mat @ x_data.T).T
v_data = (rot_mat @ v_data.T).T
w0_data[:,:3] = x_data
w0_data[:,3:] = v_data


x, y = w0_data[:, 0], w0_data[:, 1]
R, phi = np.sqrt(w0_data[:, 0]**2 + w0_data[:, 1]**2), np.arctan2(w0_data[:, 1], w0_data[:, 0])
z, vy = w0_data[:, 2], w0_data[:, 4]

fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].hist2d(x, y, bins=100, range=[[-10,10],[-10,10]], cmap='plasma', norm = 'log')
ax[0].set_xlabel('X [kpc]')
ax[0].set_ylabel('Y [kpc]')
ax[1].hist2d(x, z, bins=100, range=[[-10,10],[-10,10]], cmap='plasma', norm = 'log')
ax[1].set_xlabel('X [kpc]')
ax[1].set_ylabel('Z [kpc]')
ax[2].hist2d(y, z, bins=100, range=[[-10,10],[-10,10]], cmap='plasma', norm = 'log')
ax[2].set_xlabel('Y [kpc]')
ax[2].set_ylabel('Z [kpc]')

x = np.random.normal(x, 0., size=x.shape)
y = np.random.normal(y, 0., size=y.shape)
z = np.random.normal(z, 0., size=z.shape)

Rzphi_stars = np.stack([R, z, phi], axis=-1)
XY_stars = np.stack([x, z], axis=-1) # prefect edge-on view


X_min, X_max = -10., 10.
Y_min, Y_max = -3., 3.
nX, nY = 60,40
area_XY = ((X_max - X_min)/nX) * ((Y_max - Y_min)/nY)

X_edge = jnp.linspace(X_min, X_max, nX+1)
Y_edge = jnp.linspace(Y_min, Y_max, nY+1)
X_mids, Y_mids = 0.5 * (X_edge[:-1] + X_edge[1:]), 0.5 * (Y_edge[:-1] + Y_edge[1:])

X_mid_mesh, Y_mids_mesh = jnp.meshgrid(X_mids, Y_mids, indexing='ij')
XY_mid_grid = jnp.stack([X_mid_mesh.ravel(), Y_mids_mesh.ravel()], axis=-1)  # (n_X*n_Y, 2)

XY_minmax=jnp.array([[X_min,X_max],[Y_min,Y_max]])
nXY=jnp.array([nX,nY])
num_segments_XY=nXY.prod()
XY_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
XY_grid_indices = assign_regular_grid(XY_mid_grid,
                                    grid_min=XY_minmax[:,0],
                                    grid_max=XY_minmax[:,1],
                                    n_bins=nXY,
                                    strides=XY_strides)
_, COUNTS_XY = jnp.unique(XY_grid_indices, return_counts=True)
print('Total counts in XY bins:', jnp.sum(COUNTS_XY), 'and only 1 per bin:', jnp.all(COUNTS_XY <= 1))

df_XY = pd.DataFrame({
    'id': XY_grid_indices,
    'X_grid': XY_mid_grid[:,0],
    'Y_grid': XY_mid_grid[:,1],
})

XY_strdides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
XY_indices = assign_regular_grid(XY_stars,
                                grid_min=XY_minmax[:,0],
                                grid_max=XY_minmax[:,1],
                                n_bins=nXY,
                                strides=XY_strdides)

index = jnp.arange(num_segments_XY)
counts = jax.ops.segment_sum(mass_data, XY_indices, num_segments=num_segments_XY)
signal = counts / np.mean(mass_data) + 1
noise = np.sqrt(signal + 1)

df_XY_signal = pd.DataFrame({
    'id': index,
    'signal': signal,
    'noise': noise
})
df_XY_merged = df_XY.merge(df_XY_signal, on='id', how='left')

XY_coords = df_XY_merged[['X_grid', 'Y_grid']].to_numpy()
signal = df_XY_merged['signal'].to_numpy()
noise = df_XY_merged['noise'].to_numpy()

mask = signal > 0
XY_coords = XY_coords[mask]
signal = signal[mask]
noise = noise[mask]

target_sn = 30
def capacity_spec(index):
    """Calculates (S/N)^2 for a bin from its pixel indices."""
    # Standard S/N formula for uncorrelated noise
    sn = np.sum(signal[index]) / np.sqrt(np.sum(noise[index]**2))
    # Example for correlated noise (see full example file for details):
    # sn /= 1 + 1.07 * np.log10(len(index))
    return sn**2

# Perform the binning. The target is target_sn**2 to match the capacity definition.
pow = PowerBin(XY_coords, capacity_spec, target_capacity=target_sn**2, regul = True, maxiter=10000)

bin_index_remap = pow.bin_num

plt.figure(figsize=(10,4))
plt.scatter(XY_coords[:, 0], XY_coords[:, 1], c=signal, cmap='viridis', norm='log')
plt.title('Initial grid before binning')

df_XY_merged['Vbin_index'] = bin_index_remap
# df_XY_merged['npix_per_Vbin'] = pow.npix

df_XY_merged = df_XY_merged.sort_values('id') # Very important step !!!

_bin_index_remap = df_XY_merged['Vbin_index'].to_numpy()
_bin_index_remap = jnp.append(_bin_index_remap, -1) # for safety if some stars fall outside the grid, it will be assigned to -1 and ignored

Vbin_index, num_per_bin = np.unique(_bin_index_remap, return_counts=True)
Vbin_index, num_per_bin = Vbin_index[Vbin_index!=-1], num_per_bin[Vbin_index!=-1] # remove the -1 bin
# print(Vbin_index, num_per_bin)

XY_Vbin_index = _bin_index_remap[XY_indices]
mass_in_Vbin = jax.ops.segment_sum(mass_data, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
rho_in_Vbin = mass_in_Vbin / (num_per_bin * area_XY * 1e6) # units: M_sun / pc^2

df_XY_merged['rho_in_Vbin'] = rho_in_Vbin[df_XY_merged['Vbin_index'].to_numpy()]

fig, ax = plt.subplots(figsize=(10,4))
cb = ax.scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=df_XY_merged['rho_in_Vbin'], 
                s = 50, cmap='viridis', marker = 's', norm = 'log')
fig.colorbar(cb, ax=ax, label=r'$\Sigma$ [M$_\odot$/pc$^2$]')
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')


# Create mock data in voronoi bins

percentile_cut_1_99 = np.zeros_like(mass_data)
for i in tqdm(range(0, np.amax(_bin_index_remap)+1)):
    mask = (XY_Vbin_index == i)

    mask_percentile_i = np.where((vy[mask] < np.percentile(vy[mask], 99)) & (vy[mask] > np.percentile(vy[mask], 1)), 1, 0)
    percentile_cut_1_99[mask] = mask_percentile_i
vy = vy[percentile_cut_1_99==1] # remove the outliers beyond 1-99 percentile to make the GH moments more robust
mass_data = mass_data[percentile_cut_1_99==1]
XY_Vbin_index = XY_Vbin_index[percentile_cut_1_99==1]


mass_tot = jax.ops.segment_sum(mass_data, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
V_mean = jax.ops.segment_sum(mass_data * vy, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1) / (mass_tot + EPSILON)
V2_mean = jax.ops.segment_sum(mass_data * vy**2, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1) / (mass_tot + EPSILON)
V_sigma = jnp.sqrt(jnp.clip(V2_mean - V_mean**2, a_min=0.0))

v0 = V_mean# + jnp.array(np.random.uniform(-5., 5., size=V_mean.shape))
s = V_sigma# + jnp.array(np.random.uniform(5., 20., size=V_sigma.shape))

# v0_ls, s_ls = [],[]
# for i in tqdm(range(0, np.amax(_bin_index_remap)+1)):
#     mask = (XY_Vbin_index == i)
#     H_vy, edges = np.histogram(vy[mask], bins=50, range=[V_mean[i]-200,V_mean[i]+200], weights=mass_data[mask])
#     H_vy2 = np.histogram(vy[mask], bins=50, range=[V_mean[i]-200,V_mean[i]+200], weights=mass_data[mask]*vy[mask]**2)[0]
#     H_vy_err = np.sqrt(H_vy2)
#     H_vy_err = H_vy_err / np.sum(H_vy) / ((edges[1] - edges[0])) + 1e-4
#     H_vy = H_vy / np.sum(H_vy) / ((edges[1] - edges[0]))
#     # H_vy_err = H_vy * 0.1 + EPSILON
#     edge_mid = 0.5 * (edges[1:] + edges[:-1])
#     _edge_mid = (edge_mid - V_mean[i]) / (V_sigma[i] + EPSILON)
#     try:
#         popt, _ = sp.optimize.curve_fit(gaussian, edge_mid, H_vy, p0=[V_mean[i], V_sigma[i]], sigma=H_vy_err)
#         v0_ls.append(popt[0])
#         s_ls.append(popt[1])
#     except Exception as e:
#         v0_ls.append(V_mean[i])
#         s_ls.append(V_sigma[i])
#         print(f"Fitting failed for bin {i} with error: {e}. Using mean and sigma instead.")
# v0, s = jnp.array(v0_ls), jnp.array(s_ls)

v0_cell = v0[XY_Vbin_index]
s_cell = s[XY_Vbin_index]

w = (vy - v0_cell) / s_cell # (v_los - v0) / s, where in edge-on case v_los = vy = wN[:,4]

counts = jax.ops.segment_sum(mass_data, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
sum_w1 = jax.ops.segment_sum(w * mass_data, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
sum_w2 = jax.ops.segment_sum(w**2 * mass_data, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
sum_w3 = jax.ops.segment_sum(w**3 * mass_data, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
sum_w4 = jax.ops.segment_sum(w**4 * mass_data, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
eps = EPSILON
norm = counts + eps
w1 = sum_w1 / norm
w2 = sum_w2 / norm
w3 = sum_w3 / norm
w4 = sum_w4 / norm

h1 = w1
h2 = ((w2 - 1) / jnp.sqrt(2))
h3 = ((w3 - 3 * w1) / jnp.sqrt(6))
h4 = ((w4 - 6 * w2 + 3) / jnp.sqrt(24))

df_XY_merged['V_mean'] = V_mean[df_XY_merged['Vbin_index'].to_numpy()]
df_XY_merged['V_sigma'] = V_sigma[df_XY_merged['Vbin_index'].to_numpy()]
df_XY_merged['h1'] = h1[df_XY_merged['Vbin_index'].to_numpy()]
df_XY_merged['h2'] = h2[df_XY_merged['Vbin_index'].to_numpy()]
df_XY_merged['h3'] = h3[df_XY_merged['Vbin_index'].to_numpy()]
df_XY_merged['h4'] = h4[df_XY_merged['Vbin_index'].to_numpy()]

fig1,ax1 = plt.subplots(2,4,figsize=(30,7), gridspec_kw={'hspace':0.5, 'wspace':0.3})

H, xedge, yedge = np.histogram2d(XY_stars[:, 0], XY_stars[:, 1], bins=[X_edge, Y_edge])
signal = H.flatten()
noise = np.sqrt(signal + 1)
xmid, ymid = 0.5 * (xedge[1:] + xedge[:-1]), 0.5 * (yedge[1:] + yedge[:-1])


fig1.suptitle('2D spatial and kinematic maps in Voronoi bins', fontsize=20, y = 1.03)
cb1 = ax1[0,0].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=df_XY_merged['rho_in_Vbin'], 
                s = 20, cmap='viridis', marker = 's', norm = 'log')
fig1.colorbar(cb1, ax=ax1[0,0], label=r'$\Sigma$ [M$_\odot$/pc$^2$]')
ax1[0,0].set_title('Surface density')

cb2 = ax1[0,1].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=df_XY_merged['V_mean'], 
                s = 20, cmap='coolwarm', marker = 's', vmin = -200, vmax=200)
fig1.colorbar(cb2, ax=ax1[0,1], label='<V> [km/s]')
ax1[0,1].set_title('Mean velocity')

cb3 = ax1[0,2].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=df_XY_merged['V_sigma'], 
                s = 20, cmap='viridis', marker = 's')
fig1.colorbar(cb3, ax=ax1[0,2], label=r'$\sigma_V$ [km/s]')
ax1[0,2].set_title(r'$\sigma_V$')

cb4 = ax1[1,0].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=df_XY_merged['h1'],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb4, ax=ax1[1,0], label='h1')
ax1[1,0].set_title('h1')

cb5 = ax1[1,1].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=df_XY_merged['h2'],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb5, ax=ax1[1,1], label='h2')
ax1[1,1].set_title('h2')

cb6 = ax1[1,2].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=df_XY_merged['h3'],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb6, ax=ax1[1,2], label='h3')
ax1[1,2].set_title('h3')

cb7 = ax1[1,3].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=df_XY_merged['h4'],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb7, ax=ax1[1,3], label='h4')
ax1[1,3].set_title('h4')


cb7 = ax1[0,3].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c= v0[df_XY_merged['Vbin_index'].to_numpy()],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb7, ax=ax1[0,3], label='v0')
ax1[0,3].set_title('v0')

fig1.savefig(data_folder+'/plots/mock_Nbody_bar_XY_withRot_gal2_Nbins1000_BarFaceOn.png', bbox_inches='tight')

for i in range (0,2):
    for j in range (0,4):
        ax1[i,j].set_aspect('equal')
        ax1[i,j].set_xlabel('X [kpc]')
        ax1[i,j].set_ylabel('Y [kpc]')
        ax1[i,j].contour(xmid, ymid, np.log10(H).T, levels=10, colors='grey', linewidths=2)

fig2, ax2 = plt.subplots(1, 2, figsize=(12,4))
cb7 = ax2[0].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c= v0[df_XY_merged['Vbin_index'].to_numpy()],
                s = 20, cmap='coolwarm', marker = 's')
fig2.colorbar(cb7, ax=ax2[0], label='v0')
ax2[0].set_title('v0')
ax2[0].set_aspect('equal')
ax2[0].set_xlabel('X [kpc]')
ax2[0].set_ylabel('Y [kpc]')
cb7 = ax2[1].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c= s[df_XY_merged['Vbin_index'].to_numpy()],
                s = 20, cmap='coolwarm', marker = 's')
fig2.colorbar(cb7, ax=ax2[1], label='s')
ax2[1].set_title('s')
ax2[1].set_aspect('equal')
ax2[1].set_xlabel('X [kpc]')
ax2[1].set_ylabel('Y [kpc]')

bin_dict = {
    'nX_nY': [nX, nY],
    'X_minmax': [X_min, X_max],
    'Y_minmax': [Y_min, Y_max],
    'X_regular_grid': df_XY_merged['X_grid'].to_numpy(),
    'Y_regular_grid': df_XY_merged['Y_grid'].to_numpy(),
    'bin_index': np.array(Vbin_index),
    'num_per_bin': np.array(num_per_bin),
    'voronoi_bins_xy':pow.xybin, # list of arrays, each array gives the XY coords of the pixels in that voronoi bin
    'total_bins': np.amax(_bin_index_remap)+1,
    'bin_mapping': np.array(_bin_index_remap), # length = nX*nY, mapping each XY grid to voronoi bin, order is the same as the uniform grid index (0-nX*nY),
    'surface_density': np.array(rho_in_Vbin), # length = number of voronoi bins
    'V_mean': np.array(V_mean),
    'V_sigma': np.array(V_sigma),
    'h1': np.array(h1),
    'h2': np.array(h2),
    'h3': np.array(h3),
    'h4': np.array(h4),
    'v0': np.array(v0),
    's': np.array(s),
    'orientation': (alpha, beta, gamma)
}
bin_dict['V_mean_err'] = np.ones_like(bin_dict['V_mean']) * 5.
bin_dict['V_sigma_err'] = np.ones_like(bin_dict['V_sigma']) * 5.
bin_dict['h1_err'] = bin_dict['V_mean_err'] / (np.sqrt(2) * bin_dict['s'] + EPSILON)
bin_dict['h2_err'] = bin_dict['V_sigma_err'] / (np.sqrt(2) * bin_dict['s'] + EPSILON)
bin_dict['h3_err'] = np.ones_like(bin_dict['h3']) * 0.03 + bin_dict['h3'] * 0.01
bin_dict['h4_err'] = np.ones_like(bin_dict['h4']) * 0.03 + bin_dict['h4'] * 0.01

fig1,ax1 = plt.subplots(2,4,figsize=(30,7), gridspec_kw={'hspace':0.5, 'wspace':0.3})

H, xedge, yedge = np.histogram2d(XY_stars[:, 0], XY_stars[:, 1], bins=[X_edge, Y_edge])
signal = H.flatten()
noise = np.sqrt(signal + 1)
xmid, ymid = 0.5 * (xedge[1:] + xedge[:-1]), 0.5 * (yedge[1:] + yedge[:-1])
fig1.suptitle('2D spatial and kinematic maps in Voronoi bins', fontsize=20, y = 1.03)
cb1 = ax1[0,0].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=bin_dict['surface_density'][df_XY_merged['Vbin_index'].to_numpy()], 
                s = 20, cmap='viridis', marker = 's', norm = 'log')
fig1.colorbar(cb1, ax=ax1[0,0], label=r'$\Sigma$ [M$_\odot$/pc$^2$]')
ax1[0,0].set_title('Surface density')

cb2 = ax1[0,1].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=bin_dict['V_mean_err'][df_XY_merged['Vbin_index'].to_numpy()], 
                s = 20, cmap='coolwarm', marker = 's', vmin = -200, vmax=200)
fig1.colorbar(cb2, ax=ax1[0,1], label='<V> error [km/s]')
ax1[0,1].set_title('Mean velocity error')

cb3 = ax1[0,2].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=bin_dict['V_sigma_err'][df_XY_merged['Vbin_index'].to_numpy()], 
                s = 20, cmap='viridis', marker = 's')
fig1.colorbar(cb3, ax=ax1[0,2], label=r'$\sigma_V$ error [km/s]')
ax1[0,2].set_title(r'$\sigma_V$ error')

cb4 = ax1[1,0].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=bin_dict['h1_err'][df_XY_merged['Vbin_index'].to_numpy()],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb4, ax=ax1[1,0], label='h1 error')
ax1[1,0].set_title('h1 error')

cb5 = ax1[1,1].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=bin_dict['h2_err'][df_XY_merged['Vbin_index'].to_numpy()],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb5, ax=ax1[1,1], label='h2 error')
ax1[1,1].set_title('h2 error')

cb6 = ax1[1,2].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=bin_dict['h3_err'][df_XY_merged['Vbin_index'].to_numpy()],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb6, ax=ax1[1,2], label='h3 error')
ax1[1,2].set_title('h3 error')

cb7 = ax1[1,3].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c=bin_dict['h4_err'][df_XY_merged['Vbin_index'].to_numpy()],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb7, ax=ax1[1,3], label='h4 error')
ax1[1,3].set_title('h4 error')


cb7 = ax1[0,3].scatter(df_XY_merged['X_grid'], df_XY_merged['Y_grid'], c= bin_dict['s'][df_XY_merged['Vbin_index'].to_numpy()],
                s = 20, cmap='coolwarm', marker = 's')
fig1.colorbar(cb7, ax=ax1[0,3], label='s')
ax1[0,3].set_title('s')


with open(path + 'mock_Nbody_bar_XY_withRot_gal2_Nbins1000_BarFaceOn.pkl', 'wb') as f:
    pickle.dump(bin_dict, f)

plt.show()