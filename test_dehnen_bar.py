"""
Test dehnen_bar.py (JAX) against discBar/discBar.py (NumPy) for correctness.
Also verifies jit, grad, and vmap compatibility.
"""
import sys
import numpy as np
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from dehnen_bar import (
    make_params, DehnenBar_potential, DehnenBar_density,
    DehnenBar_acceleration, make_compound_params,
    DehnenBar_compound_potential, DehnenBar_compound_density,
    DehnenBar_compound_acceleration,
)
from discBar.discBar import makeSingleModel

from constants import G

# Test points (avoid origin and axes where u2=0)
test_points = [
    (1.0, 0.5, 0.3),
    (2.0, 1.0, 0.1),
    (0.5, 2.0, 1.0),
    (3.0, 0.1, 0.5),
    (0.1, 0.1, 2.0),
]

def compare_potential(mtype, M, s, q, L, gamma, phi=0.0, rtol=1e-6):
    """Compare JAX potential against discBar for a given model."""
    a_val = s * (1 - q)
    b_val = s * q

    # Original discBar
    disc = makeSingleModel(mtype=mtype, M=M, s=s, q=q, L=L, gamma=gamma, phi=phi)

    # JAX version
    params = make_params(M=M, s=s, q=q, L=L, gamma=gamma, mtype=mtype, phi=phi)

    errors = []
    for x, y, z in test_points:
        # discBar returns potential without G factor
        pot_orig = disc.potential(x, y, z, forces=False)
        pot_jax = float(DehnenBar_potential(float(x), float(y), float(z), params))

        # discBar potential is -pot_modKn, JAX includes G factor
        # So JAX = G * discBar
        pot_orig_with_G = G * pot_orig

        if abs(pot_orig_with_G) > 1e-15:
            rel_err = abs(pot_jax - pot_orig_with_G) / abs(pot_orig_with_G)
            errors.append(rel_err)
        else:
            errors.append(abs(pot_jax - pot_orig_with_G))

    max_err = max(errors)
    status = "PASS" if max_err < rtol else "FAIL"
    print(f"  {mtype:3s} L={L:.1f} gamma={gamma:.1f} phi={phi:.2f}: "
          f"max rel err = {max_err:.2e} [{status}]")
    return max_err < rtol


def compare_density(mtype, M, s, q, L, gamma, phi=0.0, rtol=1e-6):
    """Compare JAX density against discBar for a given model."""
    disc = makeSingleModel(mtype=mtype, M=M, s=s, q=q, L=L, gamma=gamma, phi=phi)
    params = make_params(M=M, s=s, q=q, L=L, gamma=gamma, mtype=mtype, phi=phi)

    errors = []
    for x, y, z in test_points:
        rho_orig = disc.density(x, y, z)
        rho_jax = float(DehnenBar_density(float(x), float(y), float(z), params))

        if abs(rho_orig) > 1e-15:
            rel_err = abs(rho_jax - rho_orig) / abs(rho_orig)
            errors.append(rel_err)
        else:
            errors.append(abs(rho_jax - rho_orig))

    max_err = max(errors)
    status = "PASS" if max_err < rtol else "FAIL"
    print(f"  {mtype:3s} L={L:.1f} gamma={gamma:.1f} phi={phi:.2f}: "
          f"max rel err = {max_err:.2e} [{status}]")
    return max_err < rtol


def compare_forces(mtype, M, s, q, L, gamma, phi=0.0, rtol=1e-5):
    """Compare JAX acceleration against discBar forces."""
    disc = makeSingleModel(mtype=mtype, M=M, s=s, q=q, L=L, gamma=gamma, phi=phi)
    params = make_params(M=M, s=s, q=q, L=L, gamma=gamma, mtype=mtype, phi=phi)

    errors = []
    for x, y, z in test_points:
        _, forces_orig = disc.potential(x, y, z, forces=True)
        # discBar forces are f = -grad(Phi) without G
        # JAX acceleration = -grad(G * Phi_internal)
        acc_jax = DehnenBar_acceleration(float(x), float(y), float(z), params)

        for i in range(3):
            f_orig = G * forces_orig[i]
            f_jax = float(acc_jax[i])
            if abs(f_orig) > 1e-15:
                rel_err = abs(f_jax - f_orig) / abs(f_orig)
                errors.append(rel_err)

    max_err = max(errors) if errors else 0.0
    status = "PASS" if max_err < rtol else "FAIL"
    print(f"  {mtype:3s} L={L:.1f} gamma={gamma:.1f} phi={phi:.2f}: "
          f"max rel err = {max_err:.2e} [{status}]")
    return max_err < rtol


def test_jit_compilation():
    """Verify that jit compilation succeeds."""
    print("\n=== JIT Compilation Test ===")
    params = make_params(M=1.0, s=1.0, q=0.3, L=0.5, gamma=0.1, mtype='T1')
    x, y, z = 1.0, 0.5, 0.3

    # These should compile without error
    pot = DehnenBar_potential(x, y, z, params)
    rho = DehnenBar_density(x, y, z, params)
    acc = DehnenBar_acceleration(x, y, z, params)

    print(f"  Potential: {float(pot):.6e}")
    print(f"  Density:   {float(rho):.6e}")
    print(f"  Accel:     [{float(acc[0]):.6e}, {float(acc[1]):.6e}, {float(acc[2]):.6e}]")
    print(f"  JIT compilation: PASS")
    return True


def test_vmap():
    """Verify vmap compatibility."""
    print("\n=== vmap Compatibility Test ===")
    params = make_params(M=1.0, s=1.0, q=0.3, L=0.5, gamma=0.1, mtype='T1')

    xs = jnp.array([1.0, 2.0, 3.0])
    ys = jnp.array([0.5, 1.0, 0.1])
    zs = jnp.array([0.3, 0.1, 0.5])

    pot_vmap = jax.vmap(DehnenBar_potential, in_axes=(0, 0, 0, None))
    pots = pot_vmap(xs, ys, zs, params)
    print(f"  vmap potential: {pots}")

    acc_vmap = jax.vmap(DehnenBar_acceleration, in_axes=(0, 0, 0, None))
    accs = acc_vmap(xs, ys, zs, params)
    print(f"  vmap accel shape: {accs.shape}")
    print(f"  vmap: PASS")
    return True


def test_grad_finite():
    """Verify gradients are finite."""
    print("\n=== Gradient Finiteness Test ===")
    params = make_params(M=1.0, s=1.0, q=0.3, L=0.5, gamma=0.1, mtype='V2')
    x, y, z = 1.0, 0.5, 0.3

    acc = DehnenBar_acceleration(x, y, z, params)
    all_finite = jnp.all(jnp.isfinite(acc))
    print(f"  Acceleration: {acc}")
    print(f"  All finite: {bool(all_finite)} [{'PASS' if all_finite else 'FAIL'}]")
    return bool(all_finite)


if __name__ == '__main__':
    all_pass = True

    # ---- Potential comparison ----
    print("=== Potential Comparison (JAX vs discBar) ===")
    configs = [
        # Axisymmetric (L=0)
        ('T1', 1.0, 1.0, 0.3, 0.0, 0.0, 0.0),
        ('T2', 1.0, 1.5, 0.2, 0.0, 0.0, 0.0),
        ('T3', 1.0, 1.0, 0.4, 0.0, 0.0, 0.0),
        ('T4', 1.0, 2.0, 0.1, 0.0, 0.0, 0.0),
        ('V1', 1.0, 1.0, 0.3, 0.0, 0.0, 0.0),
        ('V2', 1.0, 1.5, 0.2, 0.0, 0.0, 0.0),
        ('V3', 1.0, 1.0, 0.4, 0.0, 0.0, 0.0),
        ('V4', 1.0, 2.0, 0.3, 0.0, 0.0, 0.0),
        ('T0', 1.0, 1.0, 0.3, 0.0, 0.0, 0.0),
        ('V0', 1.0, 1.0, 0.3, 0.0, 0.0, 0.0),
        # Barred (L>0)
        ('T1', 1.0, 1.0, 0.3, 0.5, 0.1, 0.0),
        ('T2', 1.0, 1.5, 0.2, 0.3, -0.5, 0.0),
        ('V1', 1.0, 1.0, 0.3, 0.5, 0.1, 0.0),
        ('V2', 1.0, 1.5, 0.2, 0.3, 0.5, 0.0),
        ('V4', 1.0, 1.0, 0.3, 0.8, 0.3, 0.0),
        ('T0', 1.0, 1.0, 0.3, 0.5, 0.1, 0.0),
        ('V0', 1.0, 1.0, 0.3, 0.5, 0.1, 0.0),
        # Crossed (phi != 0)
        ('T1', 1.0, 1.0, 0.3, 0.5, 0.1, 0.3),
        ('V2', 1.0, 1.5, 0.2, 0.3, 0.5, 0.2),
    ]

    for mtype, M, s, q, L, gamma, phi in configs:
        passed = compare_potential(mtype, M, s, q, L, gamma, phi)
        all_pass = all_pass and passed

    # ---- Density comparison ----
    print("\n=== Density Comparison (JAX vs discBar) ===")
    for mtype, M, s, q, L, gamma, phi in configs:
        if q > 0:  # density only for thick models
            passed = compare_density(mtype, M, s, q, L, gamma, phi)
            all_pass = all_pass and passed

    # ---- Force comparison ----
    print("\n=== Force Comparison (JAX grad vs discBar forces) ===")
    force_configs = [
        ('T1', 1.0, 1.0, 0.3, 0.0, 0.0, 0.0),
        ('T2', 1.0, 1.5, 0.2, 0.0, 0.0, 0.0),
        ('V1', 1.0, 1.0, 0.3, 0.0, 0.0, 0.0),
        ('V2', 1.0, 1.5, 0.2, 0.0, 0.0, 0.0),
        ('T1', 1.0, 1.0, 0.3, 0.5, 0.1, 0.0),
        ('V2', 1.0, 1.5, 0.2, 0.3, 0.5, 0.0),
        ('T0', 1.0, 1.0, 0.3, 0.0, 0.0, 0.0),
        ('V0', 1.0, 1.0, 0.3, 0.0, 0.0, 0.0),
        ('T1', 1.0, 1.0, 0.3, 0.5, 0.1, 0.3),
    ]
    for mtype, M, s, q, L, gamma, phi in force_configs:
        passed = compare_forces(mtype, M, s, q, L, gamma, phi)
        all_pass = all_pass and passed

    # ---- JIT, grad, vmap tests ----
    all_pass = test_jit_compilation() and all_pass
    all_pass = test_grad_finite() and all_pass
    all_pass = test_vmap() and all_pass

    print("\n" + "=" * 50)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    sys.exit(0 if all_pass else 1)
