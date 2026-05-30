"""
Fair rotation-curve comparison between the N-body ground truth and the
best-fit Schwarzschild model, accounting for the bar non-axisymmetry.

The original plot_rotation_curve.py compared:
  - GT axisymmetrised (agama CylSpline, symmetry='axisymmetric')
  - Model at (x=R, y=0)  -- this is the bar MAJOR axis, not an average.

That's not the same quantity. This script makes three matched comparisons:
  Panel 1 -- Axisymmetric (phi-averaged) Vc
  Panel 2 -- Vc along the bar MAJOR axis  (phi = 0)
  Panel 3 -- Vc along the bar MINOR axis  (phi = pi/2)

For the N-body potential we build two agama representations from the same
particles: a CylSpline with symmetry='axisymmetric' for panel 1, and a
Multipole with symmetry='triaxial' (l_max=m_max=8) for panels 2 and 3.

For the JAX model (NFW + MN disc + T3 bar + V4 bulge), we evaluate the
radial component of -grad(Phi) at (x = R cos(phi), y = R sin(phi), z=0).
For panel 1 we average Vc^2 over a uniform grid of phi (the axisymmetric
average is the RMS Vc over phi).
"""

import os
import pickle
import numpy as np
from tqdm import tqdm

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.constants import G

from utils import logMenc_logc_to_logM_logRs
from constants import KPCGYR_TO_KMS
from potentials import NFW_potential, MiyamotoNagai_potential
from dehnen_bar import T3_potential, V4_potential


# =====================================================================
# CONFIG
# =====================================================================
DATA_FOLDER = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data'
SNAPSHOT    = DATA_FOLDER + '/Bar_model_TG21/model/t_t0_7'
CHECKPOINT  = DATA_FOLDER + '/mcmc_checkpoint_0422_beta25_gamma140_D50_gal2.pkl'
FIG_OUT     = DATA_FOLDER + '/plots/rotation_curve_compare.png'

DISCARD     = 400
THIN        = 1
N_POST_MAX  = 200       # subsample posterior to this many samples
R_KEEP      = 50.0      # only use particles within this 3D radius for the GT pot
N_PHI_AVG   = 32        # phi grid for axisymmetric-average of model
R_MAX_PLOT  = 20.0

# Fixed bar shape constants in the JAX model (must match model_bar.py)
V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
GAMMA_BAR = 1.0


# =====================================================================
# Helpers
# =====================================================================
def bar_angle_bar_strength(x, y, R_anulus=np.arange(1, 5, 0.25)):
    R0 = np.sqrt(x**2 + y**2); phi0 = np.arctan2(y, x)
    angles, strengths = [], []
    for i in range(len(R_anulus) - 1):
        sel = (R0 > R_anulus[i]) & (R0 < R_anulus[i+1])
        if sel.sum() == 0:
            angles.append(0.0); strengths.append(0.0); continue
        ph = phi0[sel]
        A2 = np.sum(np.cos(2*ph)) / len(ph)
        B2 = np.sum(np.sin(2*ph)) / len(ph)
        angles.append(0.5 * np.arctan2(B2, A2))
        strengths.append(np.sqrt(A2**2 + B2**2))
    R_mid = (R_anulus[:-1] + R_anulus[1:]) / 2
    return R_mid, np.array(angles), np.array(strengths)


def rotate(posvel, angle):
    x, y, z, vx, vy, vz = posvel.T
    s, c = np.sin(angle), np.cos(angle)
    return np.array([x*c - y*s, x*s + y*c, z,
                     vx*c - vy*s, vx*s + vy*c, vz]).T


def Vc_agama_at(pot, x, y, z=0.0):
    """Circular Vc at (x_i, y_i, z) for an agama potential.

    Works for any symmetry: Vc^2 = -R * F_R with F_R = (x F_x + y F_y) / R.
    For axisymmetric pots this reduces to -R * F_x at (R, 0, 0); for the
    triaxial pot it depends on phi.
    """
    x = np.atleast_1d(x).astype(float)
    y = np.atleast_1d(y).astype(float)
    R = np.sqrt(x**2 + y**2)
    pos = np.column_stack([x, y, np.full_like(x, z)])
    F = pot.force(pos)
    F_R = (x * F[:, 0] + y * F[:, 1]) / np.where(R > 1e-12, R, 1.0)
    return np.sqrt(np.maximum(-R * F_R, 0.0))


# ---- JAX model potential ----
@jax.jit
def potential_func(x, y, z, params_baryon, params_halo):
    phi_halo = NFW_potential(x, y, z, params_halo)
    mn_params = {k: params_baryon[k] for k in [
        'logM_disc', 'Rs_disc', 'Hs_disc',
        'x_origin', 'y_origin', 'z_origin',
        'dirx', 'diry', 'dirz']}
    phi_mn = MiyamotoNagai_potential(x, y, z, mn_params)
    M_bar  = 10.0 ** params_baryon['logM_bar']
    L_bar  = params_baryon['L_bar']
    a_bar  = params_baryon['a_bar']
    b_bar  = params_baryon['b_bar']
    phi_t3 = T3_potential(x, y, z, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)
    phi_v4 = V4_potential(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)
    return phi_halo + phi_mn + phi_t3 + phi_v4


def _phi_xy(x, y, params_baryon, params_halo):
    return potential_func(x, y, 0.0, params_baryon, params_halo)


def Vc_model_at_phi(r, phi, params_baryon, params_halo):
    """Vc at (R cos phi, R sin phi, 0) for the JAX model.

    V_c^2 = R * dPhi/dR, where dPhi/dR = cos(phi) dPhi/dx + sin(phi) dPhi/dy.
    """
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    dPhi_dx = jax.grad(_phi_xy, 0)(x, y, params_baryon, params_halo)
    dPhi_dy = jax.grad(_phi_xy, 1)(x, y, params_baryon, params_halo)
    dPhi_dR = jnp.cos(phi) * dPhi_dx + jnp.sin(phi) * dPhi_dy
    # JAX potential is in kpc^2/Gyr^2 -> Vc in kpc/Gyr; convert to km/s.
    return jnp.sqrt(jnp.maximum(r * dPhi_dR, 0.0)) # KPCGYR_TO_KMS


PHIS = jnp.linspace(0.0, 2*jnp.pi, N_PHI_AVG, endpoint=False)


def Vc_model_axisymmetric(r, params_baryon, params_halo):
    """RMS-over-phi Vc at radius r (axisymmetric-averaged rotation curve)."""
    Vc2 = jax.vmap(
        lambda p: Vc_model_at_phi(r, p, params_baryon, params_halo) ** 2
    )(PHIS)
    return jnp.sqrt(jnp.mean(Vc2))


@jax.jit
def Vc_curves_for_sample(r_plot, params_baryon, params_halo):
    """Return (Vc_axi, Vc_major, Vc_minor) on r_plot for one posterior sample."""
    Vc_axi   = jax.vmap(Vc_model_axisymmetric,
                        in_axes=(0, None, None))(r_plot, params_baryon, params_halo)
    Vc_major = jax.vmap(lambda r: Vc_model_at_phi(r, 0.0,
                                                  params_baryon, params_halo))(r_plot)
    Vc_minor = jax.vmap(lambda r: Vc_model_at_phi(r, jnp.pi/2,
                                                  params_baryon, params_halo))(r_plot)
    return Vc_axi, Vc_major, Vc_minor


# ---- checkpoint -> physical params (mirrors plot_rotation_curve.load_potential_dict) ----
def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        ckpt = pickle.load(f)
    posterior = np.stack(ckpt['all_samples'], axis=0)
    logprob = np.stack(ckpt['all_logprob'], axis=0)
    print(f"checkpoint: {ckpt['step']} steps, {posterior.shape[1]} chains, "
          f"{posterior.shape[2]} params")
    return posterior, logprob


def load_potential_dict(params):
    logM_enc, log_c = params[0], params[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(
        logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54)

    logM_disc, logM_bar = params[1], params[2]
    logRs_disk, logHs_disk = params[4], params[5]
    logL_bar = params[6]
    log_light_to_mass_ratio = params[10]
    log_Omega_bar = params[11]

    L_bar   = 10.0 ** logL_bar
    a_bar   = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar   = Hs_disc

    params_halo_pot = {
        'logM': float(logM_halo), 'Rs': 10.0 ** float(logRs_halo),
        'a': 1.0, 'b': 1.0, 'c': 1.0,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    }
    params_baryon_rho = {
        'logM_disc': float(logM_disc),
        'Rs_disc':   10.0 ** float(logRs_disk),
        'Hs_disc':   float(Hs_disc),
        'logM_bar':  float(logM_bar),
        'L_bar':     float(L_bar),
        'a_bar':     float(a_bar),
        'b_bar':     float(b_bar),
        'light_to_mass_ratio': 10.0 ** float(log_light_to_mass_ratio),
        'Omega_bar':           10.0 ** float(log_Omega_bar),
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    }
    return params_baryon_rho, params_halo_pot


# ---- N-body prep (matches plot_rotation_curve.py exactly) ----
def prepare_nbody_particles(snapshot, r_keep=50.0):
    mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s) ** 2))
    w0, mass = agama.readSnapshot(snapshot)
    mass = mass * mass_unit.value

    unique_masses = np.unique(mass)
    mask_halo = mass == unique_masses[-1]
    mask_disc = ~mask_halo

    for r_ap in [10.0, 5.0, 3.0, 2.0]:
        R = np.sqrt(w0[:, 0]**2 + w0[:, 1]**2)
        mc = mask_disc & (R < r_ap)
        m_c = mass[mc]
        for col in range(6):
            w0[:, col] -= np.sum(w0[mc, col] * m_c) / np.sum(m_c)

    w0[:, 0] = -w0[:, 0]
    w0[:, 3] = -w0[:, 3]
    R_mid, bar_angles, _ = bar_angle_bar_strength(w0[:, 0], w0[:, 1])
    bar_angle = np.mean(bar_angles[R_mid < 4])
    w0 = rotate(w0, -bar_angle)
    print(f"  bar-angle alignment: rotated by {-bar_angle*180/np.pi:.2f} deg")

    r3 = np.sqrt(w0[:, 0]**2 + w0[:, 1]**2 + w0[:, 2]**2)
    pos  = w0[r3 < r_keep, :3]
    mass = mass[r3 < r_keep]
    print(f"  kept {len(mass)} particles within r < {r_keep} kpc")
    return pos, mass


# =====================================================================
def main():
    pos, mass = prepare_nbody_particles(SNAPSHOT, r_keep=R_KEEP)

    print('building agama CylSpline (axisymmetric) ...')
    pot_axi = agama.Potential(
        type='CylSpline', particles=(pos, mass), symmetry='axisymmetric',
        gridSizeR=30, gridSizez=25, Rmin=0.15, Rmax=50, zmin=0.1, zmax=20,
    )

    print('building agama Multipole (triaxial, lmax=mmax=8) ...')
    pot_tri = agama.Potential(
        type='Multipole', particles=(pos, mass), symmetry='triaxial',
        lmax=8, mmax=8,
        gridSizeR=40, rmin=0.05, rmax=50,
    )

    r_plot = np.linspace(0.05, R_MAX_PLOT, 100)
    r_plot_j = jnp.array(r_plot)
    zeros = np.zeros_like(r_plot)

    print('evaluating ground-truth Vc curves ...')
    Vc_gt_axi   = Vc_agama_at(pot_axi, r_plot, zeros)
    Vc_gt_major = Vc_agama_at(pot_tri, r_plot, zeros)        # phi = 0 -> x = R
    Vc_gt_minor = Vc_agama_at(pot_tri, zeros, r_plot)        # phi = pi/2 -> y = R

    # ---- posterior ----
    print(f'loading posterior <- {CHECKPOINT}')
    posterior, logprob = load_checkpoint(CHECKPOINT)
    posterior = posterior[:, logprob[-1, :] > np.amax(logprob[-1, :]) - 100, :]
    posterior = posterior[DISCARD::THIN, :, :].reshape(-1, posterior.shape[-1])
    if posterior.shape[0] > N_POST_MAX:
        idx = np.random.default_rng(42).choice(
            posterior.shape[0], N_POST_MAX, replace=False)
        posterior = posterior[idx]
    print(f'using {posterior.shape[0]} posterior samples')

    axi_all, maj_all, min_all = [], [], []
    for i in tqdm(range(posterior.shape[0]), desc='model Vc'):
        pb, ph = load_potential_dict(posterior[i])
        a, m, n = Vc_curves_for_sample(r_plot_j, pb, ph)
        axi_all.append(np.asarray(a))
        maj_all.append(np.asarray(m))
        min_all.append(np.asarray(n))
    axi_all = np.stack(axi_all)
    maj_all = np.stack(maj_all)
    min_all = np.stack(min_all)

    def pct(a): return np.percentile(a, [16, 50, 84], axis=0)
    a16, a50, a84 = pct(axi_all)
    m16, m50, m84 = pct(maj_all)
    n16, n50, n84 = pct(min_all)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    panels = [
        ('Axisymmetric (phi-averaged)', Vc_gt_axi,   a16, a50, a84),
        ('Bar major axis ($\\phi=0$)',  Vc_gt_major, m16, m50, m84),
        ('Bar minor axis ($\\phi=\\pi/2$)', Vc_gt_minor, n16, n50, n84),
    ]
    for ax, (title, gt, p16, p50, p84) in zip(axes, panels):
        ax.plot(r_plot, gt, lw=3, color='royalblue', label='N-body (truth)')
        ax.fill_between(r_plot, p16, p84, color='tomato', alpha=0.25,
                        label=r'Model 16-84%')
        ax.plot(r_plot, p50, lw=2, ls='--', color='tomato', label='Model median')
        ax.set_xlim(0, R_MAX_PLOT)
        ax.axvline(10, color='grey', ls=':', alpha=0.6, label='data extent')
        ax.set_xlabel('R [kpc]')
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9, loc='lower right')
    axes[0].set_ylabel(r'$V_c$  [km/s]')
    plt.tight_layout()

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=200, bbox_inches='tight')
    print(f'saved -> {FIG_OUT}')


if __name__ == '__main__':
    main()
