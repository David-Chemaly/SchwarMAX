
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

path = '/home/hz420/python_script/SchwarMAX/'
# Load data
import numpy as np
from powerbin import PowerBin

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

with open('/data/hz420-2/SchwarMAX/mock_axisymmetric_disc.pkl', 'rb') as f:
    data = pickle.load(f)
w0_data = np.array([data['x'], data['y'], data['z'], data['vx'], data['vy'], data['vz']]).T
mass_data = data['mass']

alpha, beta, gamma = 30, 20, 0
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
ax[0].hist2d(x, y, bins=100, range=[[-50,50],[-50,50]], cmap='plasma', norm = 'log')
ax[0].set_xlabel('X [kpc]')
ax[0].set_ylabel('Y [kpc]')
ax[1].hist2d(x, z, bins=100, range=[[-50,50],[-50,50]], cmap='plasma', norm = 'log')
ax[1].set_xlabel('X [kpc]')
ax[1].set_ylabel('Z [kpc]')
ax[2].hist2d(y, z, bins=100, range=[[-50,50],[-50,50]], cmap='plasma', norm = 'log')
ax[2].set_xlabel('Y [kpc]')
ax[2].set_ylabel('Z [kpc]')

x = np.random.normal(x, 0., size=x.shape)
y = np.random.normal(y, 0., size=y.shape)
z = np.random.normal(z, 0., size=z.shape)

Rzphi_stars = np.stack([R, z, phi], axis=-1)
XY_stars = np.stack([x, z], axis=-1) # prefect edge-on view


X_min, X_max = -12., 12.
Y_min, Y_max = -4., 4.
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
counts = jax.ops.segment_sum(jnp.ones_like(vy), XY_indices, num_segments=num_segments_XY)
signal = counts + 1
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

target_sn = 80
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

mass_tot = jax.ops.segment_sum(mass_data, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
V_mean = jax.ops.segment_sum(mass_data * vy, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1) / (mass_tot + EPSILON)
V2_mean = jax.ops.segment_sum(mass_data * vy**2, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1) / (mass_tot + EPSILON)
V_sigma = jnp.sqrt(jnp.clip(V2_mean - V_mean**2, a_min=0.0))

v0 = V_mean# + jnp.array(np.random.uniform(-5., 5., size=V_mean.shape))
s = V_sigma# + jnp.array(np.random.uniform(5., 20., size=V_sigma.shape))
# s = jnp.where(s < 10.0, 10.0, s)
v0_cell = v0[XY_Vbin_index]
s_cell = s[XY_Vbin_index]

w = (vy - v0_cell) / s_cell # (v_los - v0) / s, where in edge-on case v_los = vy = wN[:,4]
counts = jax.ops.segment_sum(jnp.ones_like(vy), XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
sum_w1 = jax.ops.segment_sum(w, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
sum_w2 = jax.ops.segment_sum(w**2, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
sum_w3 = jax.ops.segment_sum(w**3, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
sum_w4 = jax.ops.segment_sum(w**4, XY_Vbin_index, num_segments=np.amax(_bin_index_remap)+1)
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
fig1.colorbar(cb3, ax=ax1[0,2], label='$\sigma_V$ [km/s]')
ax1[0,2].set_title('$\sigma_V$')

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


for i in range (0,2):
    for j in range (0,4):
        ax1[i,j].set_aspect('equal')
        ax1[i,j].set_xlabel('X [kpc]')
        ax1[i,j].set_ylabel('Y [kpc]')
        ax1[i,j].contour(xmid, ymid, np.log10(H).T, levels=10, colors='grey', linewidths=2)

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

with open(path + 'mock_axisymmetric_disc_XY_withRot.pkl', 'wb') as f:
    pickle.dump(bin_dict, f)

fig1.savefig('/data/hz420-2/SchwarMAX/plots/mock_axisymmetric_disc_XY_withRot.png', bbox_inches='tight')
plt.show()