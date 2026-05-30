"""
Animate the evolution of the orbital weights during ADMM-NNLS iterations.

Reuses the same NNLS setup as model_bar.solve_nnls_admm_bootstrap
(block weights sqrt(5)/sqrt(5)/1 over Rzphi/XY/h_k, L2 regulariser
lambda/n_orb on w, alpha=1.6 over-relaxation, single Cholesky of
Q + rho*I), but records z at every iteration so the evolution can be
animated as `weights vs orbit-index-sorted-by-R_apo`.

Run:
    python animate_nnls_evolution.py
"""
import os
import pickle
import numpy as np

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import cmasher as cmr

from ghMoments import h_to_V_sigma


# Match the look of plot_data_vs_model.py
plt.rcParams['figure.dpi']  = 110
plt.rc('font', size=11)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman'] + plt.rcParams['font.serif']


# =====================================================================
DATA_FOLDER = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
TAG          = 'benchmark_0415_beta25_gamma140_D50_gal2'
MATRICES_IN  = f'{DATA_FOLDER}/orbit_matrices_{TAG}.pkl'
SOURCE_DATA  = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/mock_data/mock_Nbody_bar_XY_withRot_Nbins600_beta25_gamma140_D50_gal2.pkl'
ANIM_OUT     = f'{DATA_FOLDER}/animation/nnls_evolution_{TAG}.mp4'

# ADMM hyperparameters -- match solve_nnls_admm in model_bar.py
# (the solver model_for_plotting / plot_data_vs_model.py uses)
MAXITER     = 500
LAMBDA_REG  = 1.0
ALPHA_OVR   = 1.6        # over-relaxation
W_RZPHI_SQ  = 1.0
W_XY_SQ     = 1.0
W_H_SQ      = 1.0
EPS         = 1e-8

# Rzphi grid (matches likelihoods_bar.get_dict_data_bootstrap)
N_R, N_Z, N_PHI = 10, 6, 10
R_MIN, R_MAX    = 0.0, 10.0
R_APO_THRESH    = 0.01

# Animation timing -- smooth (no discrete transitions).
# Iterations-per-second grows linearly with iteration k, from SPEED_START
# at k=0 to SPEED_END at k=MAXITER. Total wall-time of the animation is
# automatic (set by the integral of 1/speed).
FPS           = 30
SPEED_START   = 7.0    # iterations per second at the beginning  (slow but not stalled)
SPEED_END     = 200.0  # iterations per second at the end        (much faster)
SPEED_POWER   = 1.0    # 1.0 = linear ramp;  >1 ramps up later;  <1 ramps up sooner
DISPLAY_FLOOR = 1e-3   # symlog linear-region threshold for the y axis


# =====================================================================
def estimate_R_apo(A_Rzphi):
    """Mass-weighted mean R per orbit -- a continuous proxy for orbit size.

    The 'max R bin with mass' version was discrete (10 R-bins -> 10 colours).
    This averages R over the orbit's Rzphi mass distribution, so each orbit
    gets a unique floating-point R value -> smooth colour gradient.
    """
    R_edges = np.linspace(R_MIN, R_MAX, N_R + 1)
    R_mids  = 0.5 * (R_edges[:-1] + R_edges[1:])
    n_bins  = N_R * N_Z * N_PHI
    R_per_bin = R_mids[np.arange(n_bins) % N_R]            # (n_bins,)
    mass = np.maximum(A_Rzphi, 0.0)                         # (n_bins, n_orb)
    total = mass.sum(axis=0)
    R_mean = (mass * R_per_bin[:, None]).sum(axis=0) / np.where(total > 0, total, 1.0)
    return R_mean


def admm_with_trajectory(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                         y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                         sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                         lambda_reg=1.0, maxiter=200):
    """Single-RHS ADMM-NNLS recording z at every iteration."""
    y_xy_safe = jnp.where(jnp.abs(y_xy) > EPS, y_xy, 1.0)
    w_rz, w_xy, w_h = jnp.sqrt(W_RZPHI_SQ), jnp.sqrt(W_XY_SQ), jnp.sqrt(W_H_SQ)

    U = jnp.vstack([
        w_rz * A_Rzphi / (sig_Rzphi[:, None] + EPS),
        w_xy * A_xy    / (sig_xy[:, None]    + EPS),
        w_h  * (A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + EPS),
        w_h  * (A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + EPS),
        w_h  * (A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + EPS),
        w_h  * (A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + EPS),
    ])
    y_vec = jnp.concatenate([
        w_rz * y_Rzphi / (sig_Rzphi + EPS),
        w_xy * y_xy    / (sig_xy    + EPS),
        w_h  * y_h1    / (sig_A1    + EPS),
        w_h  * y_h2    / (sig_A2    + EPS),
        w_h  * y_h3    / (sig_A3    + EPS),
        w_h  * y_h4    / (sig_A4    + EPS),
    ])

    n_orb = U.shape[1]
    reg   = lambda_reg / n_orb
    Q     = U.T @ U + reg * jnp.eye(n_orb, dtype=U.dtype)
    rho   = jnp.trace(Q) / n_orb
    L_chol = jnp.linalg.cholesky(Q + rho * jnp.eye(n_orb, dtype=U.dtype))
    c_vec = -(U.T @ y_vec)

    w0 = jnp.full((n_orb,), jnp.sum(y_Rzphi) / n_orb, dtype=U.dtype)
    z0 = w0
    u0 = jnp.zeros(n_orb, dtype=U.dtype)

    def step(carry, _):
        w, z, u = carry
        rhs   = rho * (z - u) - c_vec
        w_new = jax.scipy.linalg.cho_solve((L_chol, True), rhs)
        w_hat = ALPHA_OVR * w_new + (1.0 - ALPHA_OVR) * z
        z_new = jnp.maximum(0.0, w_hat + u)
        u_new = u + w_hat - z_new
        return (w_new, z_new, u_new), z_new

    _, z_traj = jax.lax.scan(step, (w0, z0, u0), xs=None, length=maxiter)
    # Prepend the initial state so frame 0 is the t=0 init, frame k is z after k steps.
    z_traj = jnp.concatenate([z0[None, :], z_traj], axis=0)
    return z_traj  # (maxiter+1, n_orb)


def main():
    print(f"loading matrices <- {MATRICES_IN}")
    with open(MATRICES_IN, 'rb') as f:
        m = pickle.load(f)

    A_Rzphi = jnp.asarray(m['A_Rzphi']); A_xy = jnp.asarray(m['A_xy'])
    A_h1 = jnp.asarray(m['A_h1']); A_h2 = jnp.asarray(m['A_h2'])
    A_h3 = jnp.asarray(m['A_h3']); A_h4 = jnp.asarray(m['A_h4'])
    y_Rzphi = jnp.asarray(m['y_Rzphi']); y_xy = jnp.asarray(m['y_xy'])
    y_h1 = jnp.asarray(m['y_h1']); y_h2 = jnp.asarray(m['y_h2'])
    y_h3 = jnp.asarray(m['y_h3']); y_h4 = jnp.asarray(m['y_h4'])
    sig_Rzphi = jnp.asarray(m['sig_Rzphi']); sig_xy = jnp.asarray(m['sig_xy'])
    sig_A1 = jnp.asarray(m['sig_A1']); sig_A2 = jnp.asarray(m['sig_A2'])
    sig_A3 = jnp.asarray(m['sig_A3']); sig_A4 = jnp.asarray(m['sig_A4'])

    n_orb = int(A_xy.shape[1])
    print(f"n_orb = {n_orb}, running ADMM for {MAXITER} iterations ...")
    z_traj = admm_with_trajectory(
        A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
        lambda_reg=LAMBDA_REG, maxiter=MAXITER,
    )
    z_traj = np.asarray(z_traj)  # (MAXITER, n_orb)
    print(f"z_traj: shape={z_traj.shape}  "
          f"final non-zero fraction={(z_traj[-1] > 0).mean():.3f}  "
          f"max={z_traj.max():.3e}")

    # Sort orbits by R_apo (only affects the scatter; the matrix products use unsorted)
    A_xy_np = np.asarray(A_xy)
    A_h1_np = np.asarray(A_h1); A_h2_np = np.asarray(A_h2)
    A_h3_np = np.asarray(A_h3); A_h4_np = np.asarray(A_h4)
    y_xy_np = np.asarray(y_xy)
    y_xy_safe = np.where(np.abs(y_xy_np) > EPS, y_xy_np, 1.0)
    A_h_xy = [A_h1_np * A_xy_np, A_h2_np * A_xy_np,
              A_h3_np * A_xy_np, A_h4_np * A_xy_np]

    R_apo = estimate_R_apo(np.asarray(A_Rzphi))
    order = np.argsort(R_apo)
    R_apo_sorted = R_apo[order]
    z_traj_sorted = z_traj[:, order]

    # Bin-mapping for 2D maps
    print(f"loading bin mapping <- {SOURCE_DATA}")
    with open(SOURCE_DATA, 'rb') as f:
        src = pickle.load(f)
    X_grid  = np.asarray(src['X_regular_grid'])
    Y_grid  = np.asarray(src['Y_regular_grid'])
    bin_map = np.asarray(src['bin_mapping'])
    index_remap = bin_map[:-1]                        # matches plot_data_vs_model.py
    n_pix = index_remap.size
    valid_pix = index_remap >= 0
    # X/Y arrays and remap may have different lengths; trust X_grid as canonical.
    if X_grid.size != n_pix:
        n_pix = min(X_grid.size, n_pix)
        index_remap = index_remap[:n_pix]
        valid_pix   = valid_pix[:n_pix]

    def per_bin_to_pixels(per_bin):
        out = np.full(n_pix, np.nan, dtype=np.float32)
        out[valid_pix] = per_bin[index_remap[valid_pix]]
        return out

    # Per-iteration model maps in *bin* space, then convert h1,h2 -> V,sigma
    print("evaluating model maps per iteration ...")
    z_unsorted = np.asarray(z_traj)                    # (MAXITER+1, n_orb)
    dens_norm  = z_unsorted @ A_xy_np.T                # (MAXITER+1, n_xy_bins)
    h1_traj    = (z_unsorted @ A_h_xy[0].T) / y_xy_safe[None, :]
    h2_traj    = (z_unsorted @ A_h_xy[1].T) / y_xy_safe[None, :]
    h3_traj    = (z_unsorted @ A_h_xy[2].T) / y_xy_safe[None, :]
    h4_traj    = (z_unsorted @ A_h_xy[3].T) / y_xy_safe[None, :]

    # Convert density to luminosity units (L_sun/pc^2) so the cmr.sepia/LogNorm
    # range from plot_data_vs_model.py applies as-is.
    LM            = 10. ** float(m['best_fit_param'][10])  # log_LM index
    mean_mpo      = float(m['mean_mass_per_orb'])
    dens_lum_traj = dens_norm * (mean_mpo * LM)            # (MAXITER+1, n_xy_bins)

    # Convert h1,h2 -> V,sigma per iteration
    v0 = jnp.asarray(src['v0']); s = jnp.asarray(src['s'])
    h_to_V_sigma_v = jax.vmap(h_to_V_sigma, in_axes=(0, 0, None, None))
    V_traj, sig_traj = h_to_V_sigma_v(jnp.asarray(h1_traj), jnp.asarray(h2_traj), v0, s)
    V_traj   = np.asarray(V_traj)
    sig_traj = np.asarray(sig_traj)

    # Five panels: density, V_LOS, sigma_v, h3, h4  (matches plot_data_vs_model.py)
    fig_names   = [r'$\Sigma_{\rm *}$ [L$_\odot$/pc$^2$]',
                   r'$V_{\rm los}$ [km/s]',
                   r'$\sigma_{v}$ [km/s]',
                   r'$h_3$', r'$h_4$']
    color_maps  = [cmr.sepia, cmr.iceburn, cmr.amber, cmr.iceburn, cmr.amber]
    vmin_ls     = [1e1, -200, 20,  -0.2, -0.2]
    vmax_ls     = [1e4,  200, 150,  0.2,  0.1]
    log_norm    = [True, False, False, False, False]
    map_traj    = [dens_lum_traj, V_traj, sig_traj, h3_traj, h4_traj]

    # Frame schedule: smooth speed(k) = SPEED_START + (SPEED_END - SPEED_START)*(k/MAXITER)^p
    # Total wall-time T = sum_k 1/speed(k); for each frame at fixed FPS find the
    # iteration whose cumulative time has been reached.
    ks_grid    = np.arange(MAXITER + 1)
    speed_grid = SPEED_START + (SPEED_END - SPEED_START) * (ks_grid / MAXITER) ** SPEED_POWER
    t_at_k     = np.concatenate([[0.0], np.cumsum(1.0 / speed_grid[:-1])])
    T_total    = float(t_at_k[-1])
    n_frames_total = int(np.ceil(T_total * FPS)) + 1
    times      = np.arange(n_frames_total) / FPS
    frames     = np.clip(np.searchsorted(t_at_k, times), 0, MAXITER).astype(int)
    print(f"frame schedule: {n_frames_total} frames at {FPS} fps  "
          f"(~{T_total:.1f} s)  speed {SPEED_START:.0f} -> {SPEED_END:.0f} iter/s "
          f"(power={SPEED_POWER})")

    # ---- Figure layout: top = scatter, bottom = 5 maps ----
    fig = plt.figure(figsize=(18, 7.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.4, 1.0],
                          hspace=0.40, wspace=0.45,
                          left=0.05, right=0.985, top=0.94, bottom=0.10)

    ax_sc = fig.add_subplot(gs[0, :])
    vmax_y = float(z_traj_sorted.max()) * 1.1
    sc = ax_sc.scatter(np.arange(n_orb), z_traj_sorted[0],
                       c=R_apo_sorted, s=12, cmap='jet', linewidths=0)
    ax_sc.set_yscale('symlog', linthresh=DISPLAY_FLOOR)
    ax_sc.set_ylim(0, vmax_y)
    ax_sc.set_xlim(-1, n_orb)
    ax_sc.set_xlabel(r'orbit index (sorted by $\langle R \rangle$)', fontsize=12)
    ax_sc.set_ylabel('weight $w$', fontsize=12)
    cb = plt.colorbar(sc, ax=ax_sc, fraction=0.018, pad=0.01)
    cb.set_label(r'$\langle R \rangle$ proxy [kpc]', fontsize=11)
    title = ax_sc.set_title('', loc='left', fontsize=13)

    X_minmax = (float(np.asarray(src['X_minmax'])[0]),
                float(np.asarray(src['X_minmax'])[1]))
    Y_minmax = (float(np.asarray(src['Y_minmax'])[0]),
                float(np.asarray(src['Y_minmax'])[1]))

    map_scs = []
    pix_size = 22  # match plot_data_vs_model.py style
    for k in range(5):
        ax = fig.add_subplot(gs[1, k])
        norm = LogNorm(vmin=vmin_ls[k], vmax=vmax_ls[k]) if log_norm[k] else None
        kw = dict(cmap=color_maps[k], marker='s', linewidths=0,
                  rasterized=True, s=pix_size)
        if norm is not None:
            kw['norm'] = norm
        else:
            kw['vmin'] = vmin_ls[k]; kw['vmax'] = vmax_ls[k]
        pix0 = per_bin_to_pixels(map_traj[k][0])
        scm = ax.scatter(X_grid, Y_grid, c=pix0, **kw)
        ax.set_xlim(X_minmax)
        ax.set_ylim(Y_minmax)
        ax.set_aspect('equal')
        ax.set_xlabel('X [kpc]', fontsize=10)
        if k == 0:
            ax.set_ylabel('Y [kpc]', fontsize=10)
        cbar = plt.colorbar(scm, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label(fig_names[k], fontsize=11)
        cbar.ax.tick_params(labelsize=8)
        map_scs.append(scm)

    def update(i):
        sc.set_offsets(np.column_stack([np.arange(n_orb), z_traj_sorted[i]]))
        title.set_text(f'ADMM-NNLS iteration {i} / {MAXITER}')
        for k in range(5):
            map_scs[k].set_array(per_bin_to_pixels(map_traj[k][i]))
        return [sc, title] + map_scs

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000 / FPS, blit=False)

    os.makedirs(os.path.dirname(ANIM_OUT), exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=FPS, codec='libx264',
        extra_args=['-pix_fmt', 'yuv420p', '-crf', '20'],
    )
    anim.save(ANIM_OUT, writer=writer, dpi=100)
    print(f"saved -> {ANIM_OUT}")


if __name__ == '__main__':
    main()
