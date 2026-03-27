"""
Benchmark: combined autograd vs separate autograd vs hybrid acceleration.

Compares:
  1. Combined autograd: jax.grad(NFW + MN + T3 + V4)  — 1 backward pass (current)
  2. Separate autograd: NFW_acc + MN_acc + T3_acc + V4_acc  — 4 backward passes
  3. Hybrid: analytic NFW + analytic MN + autograd(T3 + V4)  — 1 backward + 2 analytic

Usage:
    python benchmark_acceleration.py
"""
import sys
import time
import numpy as np

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
sys.path.append(path)

import jax
import jax.numpy as jnp

from potentials import (
    NFW_potential, NFW_acceleration,
    MiyamotoNagai_potential, MiyamotoNagai_acceleration,
)
from dehnen_bar import T3_potential, T3_acceleration, V4_potential, V4_acceleration
from constants import G, EPSILON

# ============================================================
# Setup params
# ============================================================
params_halo_pot = {
    'logM': 11.8, 'Rs': 10**1.24, 'a': 1.0, 'b': 1.0, 'c': 1.0,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
}

logM_disc, logM_bar = 10.71, 9.94
logRs_disk, logHs_disk, logL_bar = 0.62, 0.02, 0.56
L_bar = 10.0**logL_bar
Hs_disc = 10.0**logHs_disk

mn_params = {
    'logM_disc': logM_disc, 'Rs_disc': 10**logRs_disk, 'Hs_disc': Hs_disc,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
}

M_bar = 10.0**logM_bar
a_bar = L_bar / 5.0
b_bar = Hs_disc
GAMMA_BAR = 1.0
V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0

# ============================================================
# Strategy 1: Combined autograd (current approach)
# ============================================================
@jax.jit
def acc_combined(x, y, z):
    def _pot(pos):
        return (NFW_potential(pos[0], pos[1], pos[2], params_halo_pot)
                + MiyamotoNagai_potential(pos[0], pos[1], pos[2], mn_params)
                + T3_potential(pos[0], pos[1], pos[2], M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)
                + V4_potential(pos[0], pos[1], pos[2], M_bar, V4_A, V4_B, V4_L, V4_GAMMA))
    return -jax.grad(_pot)(jnp.array([x, y, z]))

# ============================================================
# Strategy 2: Separate autograd (4 grads)
# ============================================================
@jax.jit
def acc_separate(x, y, z):
    return (NFW_acceleration(x, y, z, params_halo_pot)
            + MiyamotoNagai_acceleration(x, y, z, mn_params)
            + T3_acceleration(x, y, z, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)
            + V4_acceleration(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA))

# ============================================================
# Strategy 3: Hybrid — analytic NFW + MN, autograd T3+V4 combined
# ============================================================
from utils import get_mat

@jax.jit
def NFW_acc_analytic(x, y, z):
    p = params_halo_pot
    rin = jnp.array([x - p['x_origin'], y - p['y_origin'], z - p['z_origin']])
    R = get_mat(p['dirx'], p['diry'], p['dirz'])
    rvec = R @ rin
    rx, ry, rz = rvec
    a, b, c = p['a'], p['b'], p['c']
    sx, sy, sz = rx / a, ry / b, rz / c
    r = jnp.sqrt(sx**2 + sy**2 + sz**2 + EPSILON)
    Rs = p['Rs']
    M = 10.0 ** p['logM']
    log_term = jnp.log(1 + r / Rs)
    dPhi_dr = -G * M * (1.0 / (Rs * (1 + r / Rs) * r) - log_term / r**2)
    dr_drx = sx / (a * r)
    dr_dry = sy / (b * r)
    dr_drz = sz / (c * r)
    grad_rvec = jnp.array([dPhi_dr * dr_drx, dPhi_dr * dr_dry, dPhi_dr * dr_drz])
    return -(R.T @ grad_rvec)

@jax.jit
def MN_acc_analytic(x, y, z):
    p = mn_params
    rin = jnp.array([x - p['x_origin'], y - p['y_origin'], z - p['z_origin']])
    R = get_mat(p['dirx'], p['diry'], p['dirz'])
    rvec = R @ rin
    rx, ry, rz = rvec + EPSILON
    a_mn = p['Rs_disc']
    b_mn = p['Hs_disc']
    M = 10.0 ** p['logM_disc']
    zb = jnp.sqrt(rz**2 + b_mn**2)
    azb = a_mn + zb
    D = jnp.sqrt(rx**2 + ry**2 + azb**2)
    prefac = G * M / D**3
    grad_rvec = jnp.array([prefac * rx, prefac * ry, prefac * rz * azb / zb])
    return -(R.T @ grad_rvec)

@jax.jit
def acc_hybrid(x, y, z):
    acc_nfw = NFW_acc_analytic(x, y, z)
    acc_mn = MN_acc_analytic(x, y, z)
    def _pot_bar(pos):
        return (T3_potential(pos[0], pos[1], pos[2], M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)
                + V4_potential(pos[0], pos[1], pos[2], M_bar, V4_A, V4_B, V4_L, V4_GAMMA))
    acc_bar = -jax.grad(_pot_bar)(jnp.array([x, y, z]))
    return acc_nfw + acc_mn + acc_bar

# ============================================================
# Accuracy check
# ============================================================
x_test, y_test, z_test = 3.0, 2.0, 0.5

r1 = acc_combined(x_test, y_test, z_test)
r2 = acc_separate(x_test, y_test, z_test)
r3 = acc_hybrid(x_test, y_test, z_test)
jax.block_until_ready(r1)
jax.block_until_ready(r2)
jax.block_until_ready(r3)

r1_np, r2_np, r3_np = np.array(r1), np.array(r2), np.array(r3)

print("=" * 60)
print("Accuracy check")
print("=" * 60)
print(f"Combined:  {r1_np}")
print(f"Separate:  {r2_np}")
print(f"Hybrid:    {r3_np}")
for name, rn in [("Separate", r2_np), ("Hybrid", r3_np)]:
    diff = np.abs(rn - r1_np)
    rel = diff / (np.abs(r1_np) + 1e-30)
    print(f"  {name} vs Combined: max|diff|={diff.max():.2e}, max rel={rel.max():.2e}")

# ============================================================
# Timing: scalar
# ============================================================
def bench(fn, x, y, z, n_warmup=5, n_iter=50, label=""):
    for _ in range(n_warmup):
        r = fn(x, y, z)
        jax.block_until_ready(r)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        r = fn(x, y, z)
        jax.block_until_ready(r)
        times.append(time.perf_counter() - t0)
    t_med = np.median(times) * 1000
    return t_med

print()
print("=" * 60)
print("Scalar timing")
print("=" * 60)
t1 = bench(acc_combined, x_test, y_test, z_test)
t2 = bench(acc_separate, x_test, y_test, z_test)
t3 = bench(acc_hybrid, x_test, y_test, z_test)
print(f"Combined (1 grad):                    {t1:.3f} ms")
print(f"Separate (4 grads):                   {t2:.3f} ms  ({t2/t1:.2f}x)")
print(f"Hybrid (analytic NFW+MN, grad T3+V4): {t3:.3f} ms  ({t3/t1:.2f}x)")

# ============================================================
# Timing: vmapped over N points
# ============================================================
acc_combined_vmap = jax.jit(jax.vmap(acc_combined))
acc_separate_vmap = jax.jit(jax.vmap(acc_separate))
acc_hybrid_vmap = jax.jit(jax.vmap(acc_hybrid))

print()
print("=" * 60)
print("Vmapped timing")
print("=" * 60)

all_fns = [
    ("Combined (1 grad)", acc_combined_vmap),
    ("Separate (4 grads)", acc_separate_vmap),
    ("Hybrid (analytic+grad)", acc_hybrid_vmap),
]

for N in [100, 1000, 5000]:
    rng = np.random.default_rng(42)
    xs = jnp.array(rng.uniform(0.1, 10, N))
    ys = jnp.array(rng.uniform(-5, 5, N))
    zs = jnp.array(rng.uniform(-3, 3, N))

    # Warmup
    for _, fn in all_fns:
        r = fn(xs, ys, zs)
        jax.block_until_ready(r)

    print(f"\nN={N}:")
    t_ref = None
    for name, fn in all_fns:
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            r = fn(xs, ys, zs)
            jax.block_until_ready(r)
            times.append(time.perf_counter() - t0)
        t_med = np.median(times) * 1000
        if t_ref is None:
            t_ref = t_med
        print(f"  {name:<30} {t_med:>8.2f} ms  ({t_med/t_ref:.2f}x)")
