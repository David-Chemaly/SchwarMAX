import jax
import jax.numpy as jnp
from functools import partial
from constants import *

def _split(w):
    return w[:3], w[3:]

def _merge(r, v):
    return jnp.concatenate([r, v], axis=0)

@partial(jax.jit, static_argnames=('n_R', 'n_z', 'n_phi'))
def grid_Rzphi(R_min=0., R_max=10., z_min=-3., z_max=3., n_R=10, n_z=6, n_phi=6):
    """ Create a grid in cylindrical coordinates (R, z, phi) """
    R_mids = jnp.linspace(R_min, R_max, n_R)
    z_mids = jnp.linspace(z_min, z_max, n_z)
    phi_mids = jnp.linspace(-jnp.pi, jnp.pi, n_phi, endpoint=False)
    dR, dz, dphi = R_mids[1]-R_mids[0], z_mids[1]-z_mids[0], phi_mids[1]-phi_mids[0]

    return R_mids, z_mids, phi_mids, dR, dz, dphi

@partial(jax.jit, static_argnames=('n_X', 'n_Y', 'n_Vlos'))
def grid_XYVlos(Xmin, Xmax, Ymin, Ymax, Vlos_min, Vlos_max, n_X = 40, n_Y = 30, n_Vlos = 50):
    """ Create a grid in observed coordinates (X, Y, Vlos) """
    X_mids = jnp.linspace(Xmin, Xmax, n_X)
    Y_mids = jnp.linspace(Ymin, Ymax, n_Y)
    Vlos_mids = jnp.linspace(Vlos_min, Vlos_max, n_Vlos)
    dX, dY, dVlos = X_mids[1]-X_mids[0], Y_mids[1]-Y_mids[0], Vlos_mids[1]-Vlos_mids[0]

    return X_mids, Y_mids, Vlos_mids, dX, dY, dVlos

@jax.jit
def assign_regular_grid(positions, grid_min, grid_max, n_bins, strides):
    """
    Assign positions to regular grid cells.
    
    Parameters
    ----------
    positions : array (N, D) - particle positions
    grid_min : array (D,) - minimum bounds for each dimension
    grid_max : array (D,) - maximum bounds for each dimension
    n_bins : array (D,) - number of bins in each dimension (int)
    
    Returns
    -------
    bin_indices : array (N,) - flattened bin index for each particle
    """
    # Normalize to [0, 1]
    normalized = (positions - grid_min) / (grid_max - grid_min)
    # jax.debug.print("normalized: {normalized}", normalized=normalized)
    
    # Convert to bin indices
    bin_coords = jnp.floor(normalized * n_bins).astype(jnp.int32)
    
    # Clip to valid range [0, n_bins-1]
    bin_coords = jnp.clip(bin_coords, 0, n_bins - 1)
    
    # Flatten to 1D index: idx = i + j*nx + k*nx*ny
    bin_indices = jnp.sum(bin_coords * strides, axis=-1)

    # Check if any normalized coordinate is >= 1.0 or < 0.0 for each particle
    out_of_bounds = jnp.any((normalized >= 1.0) | (normalized < 0.0), axis=-1)

    return jnp.where(out_of_bounds, n_bins.prod(), bin_indices)


@jax.jit
def assign_regular_grid1d(positions, grid_min, grid_max, n_bins):
    """
    Assign positions to regular grid cells.
    
    Parameters
    ----------
    positions : array (N) - particle positions
    grid_min : float - minimum bounds for each dimension
    grid_max : float - maximum bounds for each dimension
    n_bins : int - number of bins in each dimension
    
    Returns
    -------
    bin_indices : array (N,) - flattened bin index for each particle
    """

    dx = (grid_max - grid_min)/n_bins
    index = (positions - grid_min) // dx

    out_of_bounds = (index >= n_bins) | (index < 0.0)
    return jnp.where(out_of_bounds, n_bins, index).astype(jnp.int32)



@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll'))
def integrate_leapfrog_traj(w0, acc_fn, n_steps, dt = 0.010, t0 = 0.0, unroll=True, ):
    """Leapfrog (KDK) — returns final time and final state only."""

    def step(carry, _):
        t, y = carry
        r, v = _split(y)

        # params['t'] = t  # Update time-dependent parameters
        a0 = acc_fn(*r) # , params)
        v_half = v + 0.5 * dt * a0
        r_new = r + dt * v_half
        t_new = t + dt

        # params['t'] = t_new  # Update time-dependent parameters
        a1 = acc_fn(*r_new) # , params)
        v_new = v_half + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return (t_new, y_new), (t_new, y_new)

    (_, _), (tN, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

    return tN, wN 


@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll', 'num_segments_Rzphi', 'num_segments_XY'))
def integrate_leapfrog_grid(w0, acc_fn, n_steps, dt = 0.010, t0 = 0.0, unroll=True, 
                            Rzphi_minmax=jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]), XY_minmax=jnp.array([[-10.,10.],[-2.,2.]]),
                            nRzphi=jnp.array([10,6,6]), nXY=jnp.array([40,30]), num_segments_Rzphi=360, num_segments_XY=1200,
                            v0 = jnp.zeros(1200), s= jnp.ones(1200) * 100.0,
                            ):
    """Leapfrog (KDK) — returns final time and final state only.

    num_segments_Rzphi: int
        Number of segments for Rzphi bin counting. MUST equal to nRzphi.prod()
    num_segments_XY: int
        Number of segments for XY bin counting. MUST equal to nXY.prod()
    
    v0 and s are arrays of reference velocity and dispersion for each XY cell for the GH coefficent calculation. Length should equal to nXY.prod()
    """

    def step(carry, _):
        t, y = carry
        r, v = _split(y)

        # params['t'] = t  # Update time-dependent parameters
        a0 = acc_fn(*r) # , params)
        v_half = v + 0.5 * dt * a0
        r_new = r + dt * v_half
        t_new = t + dt

        # params['t'] = t_new  # Update time-dependent parameters
        a1 = acc_fn(*r_new) # , params)
        v_new = v_half + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return (t_new, y_new), (t_new, y_new)

    (_, _), (tN, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

    # rN, vN = wN[:, :3], wN[:, 3:]
    Rzphi = jnp.array([jnp.sqrt(wN[:,0]**2 + wN[:,1]**2), wN[:,2], jnp.arctan2(wN[:,1], wN[:,0])]).T
    XY = jnp.array([wN[:,0], wN[:,2]]).T

    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_indices = assign_regular_grid(Rzphi,
                                        grid_min=Rzphi_minmax[:,0],
                                        grid_max=Rzphi_minmax[:,1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)

    # jax.debug.print("Rzphi_indices: {Rzphi_indices}", Rzphi_indices=Rzphi_indices)
    Rzphi_bin_counts = jax.ops.segment_sum(jnp.ones_like(Rzphi_indices), Rzphi_indices, num_segments=num_segments_Rzphi)
    
    XY_strdides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    XY_indices = assign_regular_grid(XY,
                                    grid_min=XY_minmax[:,0],
                                    grid_max=XY_minmax[:,1],
                                    n_bins=nXY,
                                    strides=XY_strdides)
    
    # ==========================================================================
    # Compute Gauss-Hermite coefficients in each XY cell

    v0_cell = v0[XY_indices]
    s_cell = s[XY_indices]

    vy = wN[:,4]
    w = (vy - v0_cell) / s_cell # (v_los - v0) / s, where in edge-on case v_los = vy = wN[:,4]
    counts = jax.ops.segment_sum(jnp.ones_like(vy), XY_indices, num_segments=num_segments_XY)
    sum_w1 = jax.ops.segment_sum(w, XY_indices, num_segments=num_segments_XY)
    sum_w2 = jax.ops.segment_sum(w**2, XY_indices, num_segments=num_segments_XY)
    sum_w3 = jax.ops.segment_sum(w**3, XY_indices, num_segments=num_segments_XY)
    sum_w4 = jax.ops.segment_sum(w**4, XY_indices, num_segments=num_segments_XY)
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
    Nxy = counts

    return Rzphi_bin_counts, Nxy, h1, h2, h3, h4


@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_leapfrog_voronoi(w0, acc_fn, n_steps, dt = 0.010, t0 = 0.0, unroll=True, 
                            num_Vbin = 1028, bin_mapping = jnp.zeros(2400, dtype=jnp.int32), num_per_bin = jnp.zeros(1028, dtype=jnp.int32),
                            Rzphi_minmax=jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]), XY_minmax=jnp.array([[-10.,10.],[-2.,2.]]),
                            nRzphi=jnp.array([10,6,6]), nXY=jnp.array([40,30]), num_segments_Rzphi=360,
                            v0 = jnp.zeros(1028), s= jnp.ones(1028) * 5.0,
                            ):
    """Leapfrog (KDK) — returns final time and final state only.

    num_segments_Rzphi: int
        Number of segments for Rzphi bin counting. MUST equal to nRzphi.prod()
    num_segments_XY: int
        Number of segments for XY bin counting. MUST equal to nXY.prod()
    
    v0 and s are arrays of reference velocity and dispersion for each XY cell for the GH coefficent calculation. Length should equal to nXY.prod()
    """

    def step(carry, _):
        t, y = carry
        r, v = _split(y)

        # params['t'] = t  # Update time-dependent parameters
        a0 = acc_fn(*r) # , params)
        v_half = v + 0.5 * dt * a0
        r_new = r + dt * v_half
        t_new = t + dt

        # params['t'] = t_new  # Update time-dependent parameters
        a1 = acc_fn(*r_new) # , params)
        v_new = v_half + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return (t_new, y_new), (t_new, y_new)

    (_, _), (tN, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

    # rN, vN = wN[:, :3], wN[:, 3:]
    Rzphi = jnp.array([jnp.sqrt(wN[:,0]**2 + wN[:,1]**2), wN[:,2], jnp.arctan2(wN[:,1], wN[:,0])]).T
    XY = jnp.array([wN[:,0], wN[:,2]]).T

    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_indices = assign_regular_grid(Rzphi,
                                        grid_min=Rzphi_minmax[:,0],
                                        grid_max=Rzphi_minmax[:,1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)

    # jax.debug.print("Rzphi_indices: {Rzphi_indices}", Rzphi_indices=Rzphi_indices)
    Rzphi_bin_counts = jax.ops.segment_sum(jnp.ones_like(Rzphi_indices), Rzphi_indices, num_segments=num_segments_Rzphi)
    
    XY_strdides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    XY_indices = assign_regular_grid(XY,
                                    grid_min=XY_minmax[:,0],
                                    grid_max=XY_minmax[:,1],
                                    n_bins=nXY,
                                    strides=XY_strdides)
    
    area_pixel = ( (XY_minmax[0,1] - XY_minmax[0,0]) / nXY[0] ) * ( (XY_minmax[1,1] - XY_minmax[1,0]) / nXY[1] ) * 1e6
    
    # ==========================================================================
    # Mapping the regular grid XY_indices to Voronoi bin indices
    Vbin_indices = bin_mapping[XY_indices]


    # ==========================================================================
    # Compute Gauss-Hermite coefficients in each XY cell

    v0_cell = v0[Vbin_indices]
    s_cell = s[Vbin_indices]

    vy = wN[:,4]
    w = (vy - v0_cell) / s_cell # (v_los - v0) / s, where in edge-on case v_los = vy = wN[:,4]
    counts = jax.ops.segment_sum(jnp.ones_like(vy), Vbin_indices, num_segments=num_Vbin)
    sum_w1 = jax.ops.segment_sum(w, Vbin_indices, num_segments=num_Vbin)
    sum_w2 = jax.ops.segment_sum(w**2, Vbin_indices, num_segments=num_Vbin)
    sum_w3 = jax.ops.segment_sum(w**3, Vbin_indices, num_segments=num_Vbin)
    sum_w4 = jax.ops.segment_sum(w**4, Vbin_indices, num_segments=num_Vbin)
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
    # Nxy = counts
    surface_density = counts / (num_per_bin * area_pixel + eps)

    return Rzphi_bin_counts, surface_density, h1, h2, h3, h4


@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_leapfrog_rot(w0, acc_fn, n_steps, dt = 0.010, t0 = 0.0, unroll=True, 
                            num_Vbin = 1028, bin_mapping = jnp.zeros(2400, dtype=jnp.int32), num_per_bin = jnp.zeros(1028, dtype=jnp.int32),
                            Rzphi_minmax=jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]), XY_minmax=jnp.array([[-10.,10.],[-2.,2.]]),
                            nRzphi=jnp.array([10,6,6]), nXY=jnp.array([40,30]), num_segments_Rzphi=360,
                            v0 = jnp.zeros(1028), s= jnp.ones(1028) * 5.0, rotation_matrix = jnp.eye(3)
                            ):
    """Leapfrog (KDK) — returns final time and final state only.

    num_segments_Rzphi: int
        Number of segments for Rzphi bin counting. MUST equal to nRzphi.prod()
    num_segments_XY: int
        Number of segments for XY bin counting. MUST equal to nXY.prod()
    
    v0 and s are arrays of reference velocity and dispersion for each XY cell for the GH coefficent calculation. Length should equal to nXY.prod()
    """

    def step(carry, _):
        t, y = carry
        r, v = _split(y)

        # params['t'] = t  # Update time-dependent parameters
        a0 = acc_fn(*r) # , params)
        v_half = v + 0.5 * dt * a0
        r_new = r + dt * v_half
        t_new = t + dt

        # params['t'] = t_new  # Update time-dependent parameters
        a1 = acc_fn(*r_new) # , params)
        v_new = v_half + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return (t_new, y_new), (t_new, y_new)

    (_, _), (tN, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

    # Get 3D grid before rotation
    Rzphi = jnp.array([jnp.sqrt(wN[:,0]**2 + wN[:,1]**2), wN[:,2], jnp.arctan2(wN[:,1], wN[:,0])]).T

    x = wN[:,:3]
    v = wN[:,3:]
    x_rot = (rotation_matrix @ x.T).T
    v_rot = (rotation_matrix @ v.T).T
    wN = jnp.concatenate([x_rot, v_rot], axis=-1)

    # rN, vN = wN[:, :3], wN[:, 3:]
    XY = jnp.array([wN[:,0], wN[:,2]]).T

    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_indices = assign_regular_grid(Rzphi,
                                        grid_min=Rzphi_minmax[:,0],
                                        grid_max=Rzphi_minmax[:,1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)

    # jax.debug.print("Rzphi_indices: {Rzphi_indices}", Rzphi_indices=Rzphi_indices)
    Rzphi_bin_counts = jax.ops.segment_sum(jnp.ones_like(Rzphi_indices), Rzphi_indices, num_segments=num_segments_Rzphi)
    
    XY_strdides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    XY_indices = assign_regular_grid(XY,
                                    grid_min=XY_minmax[:,0],
                                    grid_max=XY_minmax[:,1],
                                    n_bins=nXY,
                                    strides=XY_strdides)
    
    area_pixel = ( (XY_minmax[0,1] - XY_minmax[0,0]) / nXY[0] ) * ( (XY_minmax[1,1] - XY_minmax[1,0]) / nXY[1] ) * 1e6
    
    # ==========================================================================
    # Mapping the regular grid XY_indices to Voronoi bin indices
    Vbin_indices = bin_mapping[XY_indices]


    # ==========================================================================
    # Compute Gauss-Hermite coefficients in each XY cell

    v0_cell = v0[Vbin_indices]
    s_cell = s[Vbin_indices]

    vy = wN[:,4]
    w = (vy - v0_cell) / s_cell # (v_los - v0) / s, where in edge-on case v_los = vy = wN[:,4]
    counts = jax.ops.segment_sum(jnp.ones_like(vy), Vbin_indices, num_segments=num_Vbin)
    sum_w1 = jax.ops.segment_sum(w, Vbin_indices, num_segments=num_Vbin)
    sum_w2 = jax.ops.segment_sum(w**2, Vbin_indices, num_segments=num_Vbin)
    sum_w3 = jax.ops.segment_sum(w**3, Vbin_indices, num_segments=num_Vbin)
    sum_w4 = jax.ops.segment_sum(w**4, Vbin_indices, num_segments=num_Vbin)
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
    # Nxy = counts
    surface_density = counts / (num_per_bin * area_pixel + eps)

    return Rzphi_bin_counts, surface_density, h1, h2, h3, h4


@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'n_steps', 'unroll', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_leapfrog_barred(w0, acc_fn, pot_fn, n_steps, dt = 0.010, t0 = 0.0, Omega = 0.0, unroll=True, 
                            num_Vbin = 1028, bin_mapping = jnp.zeros(2400, dtype=jnp.int32), num_per_bin = jnp.zeros(1028, dtype=jnp.int32),
                            Rzphi_minmax=jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]), XY_minmax=jnp.array([[-10.,10.],[-2.,2.]]),
                            nRzphi=jnp.array([10,6,6]), nXY=jnp.array([40,30]), num_segments_Rzphi=360,
                            v0 = jnp.zeros(1028), s= jnp.ones(1028) * 5.0, rotation_matrix = jnp.eye(3)
                            ):
    """Leapfrog (KDK) — returns final time and final state only.

    num_segments_Rzphi: int
        Number of segments for Rzphi bin counting. MUST equal to nRzphi.prod()
    num_segments_XY: int
        Number of segments for XY bin counting. MUST equal to nXY.prod()
    
    v0 and s are arrays of reference velocity and dispersion for each XY cell for the GH coefficent calculation. Length should equal to nXY.prod()
    """

    def step(carry, _):
        t, y = carry
        r, v = _split(y)

        # Gravity half-kick
        a0 = acc_fn(*r)
        v_half = v + 0.5 * dt * a0

        # Exact Omega-subflow for:
        #   xdot = vx + Omega*y, ydot = vy - Omega*x
        #   vxdot = Omega*vy,     vydot = -Omega*vx
        theta = Omega * dt
        c, s_theta = jnp.cos(theta), jnp.sin(theta)

        x_bar = r[0] + dt * v_half[0]
        y_bar = r[1] + dt * v_half[1]
        x_new = c * x_bar + s_theta * y_bar
        y_new = -s_theta * x_bar + c * y_bar
        z_new = r[2] + dt * v_half[2]

        vx_rot = c * v_half[0] + s_theta * v_half[1]
        vy_rot = -s_theta * v_half[0] + c * v_half[1]
        vz_rot = v_half[2]

        r_new = jnp.array([x_new, y_new, z_new])
        v_rot = jnp.array([vx_rot, vy_rot, vz_rot])
        t_new = t + dt

        # Gravity half-kick at updated position
        a1 = acc_fn(*r_new)
        v_new = v_rot + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return (t_new, y_new), (t_new, y_new)

    (_, _), (tN, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

    x_orb = wN[:,0]
    y_orb = wN[:,1]
    z_orb = wN[:,2]
    vx_orb = wN[:,3]
    vy_orb = wN[:,4]
    vz_orb = wN[:,5]
    Lz_orb = (vx_orb * y_orb - vy_orb * x_orb)

    E_pot = pot_fn(x_orb, y_orb, z_orb)
    E_kin = 0.5 * (vx_orb**2 + vy_orb**2 + vz_orb**2)
    E_J = E_pot + E_kin - Omega * Lz_orb
    delta_EJ = jnp.fabs(E_J[-1]/E_J[0] - 1)

    # wN0 = wN
    # wN1 = jnp.array([ wN[:,0],  wN[:,1], -wN[:,2],  wN[:,3],  wN[:,4], -wN[:,5]]).T # Symmetry: x, y, z, vx, vy, vz -> x, y, -z, vx, vy, -vz
    # wN2 = jnp.array([-wN[:,0], -wN[:,1],  wN[:,2], -wN[:,3], -wN[:,4],  wN[:,5]]).T # Symmetry: -x, -y, z, -vx, -vy, vz -> -x, -y, z, -vx, -vy, vz
    # wN3 = jnp.array([-wN[:,0], -wN[:,1], -wN[:,2], -wN[:,3], -wN[:,4], -wN[:,5]]).T # Symmetry: -x, -y, -z, -vx, -vy, -vz -> -x, -y, -z, -vx, -vy, -vz
    # wN = jnp.concatenate([wN0, wN1, wN2, wN3], axis=0)

    # Get 3D grid before rotation
    def good_orb(wN):
        Rzphi = jnp.array([jnp.sqrt(wN[:,0]**2 + wN[:,1]**2), wN[:,2], jnp.arctan2(wN[:,1], wN[:,0])]).T

        x = wN[:,:3]
        v = wN[:,3:]
        x_rot = (rotation_matrix @ x.T).T
        v_rot = (rotation_matrix @ v.T).T
        wN = jnp.concatenate([x_rot, v_rot], axis=-1)

        # rN, vN = wN[:, :3], wN[:, 3:]
        XY = jnp.array([wN[:,0], wN[:,2]]).T

        Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
        Rzphi_indices = assign_regular_grid(Rzphi,
                                            grid_min=Rzphi_minmax[:,0],
                                            grid_max=Rzphi_minmax[:,1],
                                            n_bins=nRzphi,
                                            strides=Rzphi_strides)

        # jax.debug.print("Rzphi_indices: {Rzphi_indices}", Rzphi_indices=Rzphi_indices)
        Rzphi_bin_counts = jax.ops.segment_sum(jnp.ones_like(Rzphi_indices), Rzphi_indices, num_segments=num_segments_Rzphi)
        
        XY_strdides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
        XY_indices = assign_regular_grid(XY,
                                        grid_min=XY_minmax[:,0],
                                        grid_max=XY_minmax[:,1],
                                        n_bins=nXY,
                                        strides=XY_strdides)
        
        area_pixel = ( (XY_minmax[0,1] - XY_minmax[0,0]) / nXY[0] ) * ( (XY_minmax[1,1] - XY_minmax[1,0]) / nXY[1] ) * 1e6
        
        # ==========================================================================
        # Mapping the regular grid XY_indices to Voronoi bin indices
        Vbin_indices = bin_mapping[XY_indices]


        # ==========================================================================
        # Compute Gauss-Hermite coefficients in each XY cell

        v0_cell = v0[Vbin_indices]
        s_cell = s[Vbin_indices]

        vy = wN[:,4] * KPCGYR_TO_KMS
        w = (vy - v0_cell) / s_cell # (v_los - v0) / s, where in edge-on case v_los = vy = wN[:,4]
        counts = jax.ops.segment_sum(jnp.ones_like(vy), Vbin_indices, num_segments=num_Vbin)
        sum_w1 = jax.ops.segment_sum(w, Vbin_indices, num_segments=num_Vbin)
        sum_w2 = jax.ops.segment_sum(w**2, Vbin_indices, num_segments=num_Vbin)
        sum_w3 = jax.ops.segment_sum(w**3, Vbin_indices, num_segments=num_Vbin)
        sum_w4 = jax.ops.segment_sum(w**4, Vbin_indices, num_segments=num_Vbin)
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
        # Nxy = counts
        surface_density = counts / (num_per_bin * area_pixel + eps)

        return Rzphi_bin_counts.astype(jnp.float32), surface_density, h1, h2, h3, h4, 1.
    
    def bad_orb(wN):
        Rzphi_bin_counts = jnp.zeros(num_segments_Rzphi)
        surface_density = jnp.zeros(num_Vbin)
        h1 = jnp.zeros(num_Vbin)
        h2 = jnp.zeros(num_Vbin)
        h3 = jnp.zeros(num_Vbin)
        h4 = jnp.zeros(num_Vbin)
        return Rzphi_bin_counts, surface_density, h1, h2, h3, h4, 0.
    
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid = jax.lax.cond(delta_EJ<0.5, good_orb, bad_orb, wN)
    return Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid



@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll',))
def integrate_leapfrog_barred_traj(w0, acc_fn, n_steps, dt = 0.010, t0 = 0.0, Omega = 0.0, unroll=True, 
                            ):
    """Leapfrog (KDK) — returns final time and final state only.

    num_segments_Rzphi: int
        Number of segments for Rzphi bin counting. MUST equal to nRzphi.prod()
    num_segments_XY: int
        Number of segments for XY bin counting. MUST equal to nXY.prod()
    
    v0 and s are arrays of reference velocity and dispersion for each XY cell for the GH coefficent calculation. Length should equal to nXY.prod()
    """

    def step(carry, _):
        t, y = carry
        r, v = _split(y)

        # Gravity half-kick
        a0 = acc_fn(*r)
        v_half = v + 0.5 * dt * a0

        # Exact Omega-subflow for:
        #   xdot = vx + Omega*y, ydot = vy - Omega*x
        #   vxdot = Omega*vy,     vydot = -Omega*vx
        theta = Omega * dt
        c, s_theta = jnp.cos(theta), jnp.sin(theta)

        x_bar = r[0] + dt * v_half[0]
        y_bar = r[1] + dt * v_half[1]
        x_new = c * x_bar + s_theta * y_bar
        y_new = -s_theta * x_bar + c * y_bar
        z_new = r[2] + dt * v_half[2]

        vx_rot = c * v_half[0] + s_theta * v_half[1]
        vy_rot = -s_theta * v_half[0] + c * v_half[1]
        vz_rot = v_half[2]

        r_new = jnp.array([x_new, y_new, z_new])
        v_rot = jnp.array([vx_rot, vy_rot, vz_rot])
        t_new = t + dt

        # Gravity half-kick at updated position
        a1 = acc_fn(*r_new)
        v_new = v_rot + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return (t_new, y_new), (t_new, y_new)

    (_, _), (tN, wN) = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

    return tN, wN 


_integrate_barred_vmap = jax.vmap(integrate_leapfrog_barred,
                    in_axes=(
                            0, None, None, None, 0, None, None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None))


# ==========================================================================
# Fused version: accumulate binning inside the scan, no trajectory storage.
# ==========================================================================
@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'n_steps', 'unroll', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_leapfrog_barred_fused(w0, acc_fn, pot_fn, n_steps, dt=0.010, t0=0.0, Omega=0.0, unroll=False,
                                    num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
                                    num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
                                    Rzphi_minmax=jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]),
                                    XY_minmax=jnp.array([[-10.,10.],[-2.,2.]]),
                                    nRzphi=jnp.array([10,6,6]), nXY=jnp.array([40,30]),
                                    num_segments_Rzphi=360,
                                    v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
                                    rotation_matrix=jnp.eye(3)):
    """Leapfrog (KDK) with fused binning — no trajectory stored.

    Identical physics and outputs to integrate_leapfrog_barred, but accumulates
    Rzphi bin counts and Gauss-Hermite moment sums inside the scan carry,
    eliminating the (n_steps, 6) trajectory buffer.

    Memory per orbit: O(num_segments_Rzphi + num_Vbin) instead of O(n_steps * 6).
    """
    # Precompute grid constants
    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    XY_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    area_pixel = ((XY_minmax[0,1] - XY_minmax[0,0]) / nXY[0]) * \
                 ((XY_minmax[1,1] - XY_minmax[1,0]) / nXY[1]) * 1e6
    Rzphi_grid_min = Rzphi_minmax[:, 0]
    Rzphi_grid_range = Rzphi_minmax[:, 1] - Rzphi_minmax[:, 0]
    XY_grid_min = XY_minmax[:, 0]
    XY_grid_range = XY_minmax[:, 1] - XY_minmax[:, 0]

    # Initial accumulators
    init_Rzphi = jnp.zeros(num_segments_Rzphi)
    init_counts = jnp.zeros(num_Vbin)
    init_sw1 = jnp.zeros(num_Vbin)
    init_sw2 = jnp.zeros(num_Vbin)
    init_sw3 = jnp.zeros(num_Vbin)
    init_sw4 = jnp.zeros(num_Vbin)

    # For Jacobi energy check: track E_J at first and last step
    init_EJ_first = 0.0
    init_EJ_last = 0.0

    def _compute_EJ(r, v):
        """Compute Jacobi energy at a single phase-space point."""
        Lz = v[0] * r[1] - v[1] * r[0]
        E_pot = pot_fn(r[0][jnp.newaxis], r[1][jnp.newaxis], r[2][jnp.newaxis])[0]
        E_kin = 0.5 * (v[0]**2 + v[1]**2 + v[2]**2)
        return E_pot + E_kin - Omega * Lz

    def _bin_one_point(r_new, v_new, Rzphi_acc, cnt_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc):
        """Accumulate binning statistics for one timestep."""
        # ── Rzphi bin (pre-rotation frame) ──
        R_val = jnp.sqrt(r_new[0]**2 + r_new[1]**2)
        phi_val = jnp.arctan2(r_new[1], r_new[0])
        Rzphi_pos = jnp.array([R_val, r_new[2], phi_val])
        Rzphi_norm = (Rzphi_pos - Rzphi_grid_min) / Rzphi_grid_range
        Rzphi_bin = jnp.floor(Rzphi_norm * nRzphi).astype(jnp.int32)
        Rzphi_bin = jnp.clip(Rzphi_bin, 0, nRzphi - 1)
        Rzphi_idx = jnp.sum(Rzphi_bin * Rzphi_strides)
        Rzphi_in = jnp.all((Rzphi_norm >= 0.0) & (Rzphi_norm < 1.0))
        Rzphi_acc = Rzphi_acc.at[Rzphi_idx].add(jnp.where(Rzphi_in, 1.0, 0.0))

        # ── XY bin (after rotation) ──
        xyz_rot = rotation_matrix @ r_new
        v_rot_full = rotation_matrix @ v_new
        XY_pos = jnp.array([xyz_rot[0], xyz_rot[2]])
        XY_norm = (XY_pos - XY_grid_min) / XY_grid_range
        XY_bin = jnp.floor(XY_norm * nXY).astype(jnp.int32)
        XY_bin = jnp.clip(XY_bin, 0, nXY - 1)
        XY_idx = jnp.sum(XY_bin * XY_strides)
        XY_in = jnp.all((XY_norm >= 0.0) & (XY_norm < 1.0))

        # Map to Voronoi bin
        Vbin_idx = bin_mapping[XY_idx]

        # GH moment accumulation
        vy_los = v_rot_full[1] * KPCGYR_TO_KMS
        w_val = (vy_los - v0[Vbin_idx]) / s[Vbin_idx]
        one = jnp.where(XY_in, 1.0, 0.0)
        cnt_acc = cnt_acc.at[Vbin_idx].add(one)
        sw1_acc = sw1_acc.at[Vbin_idx].add(one * w_val)
        sw2_acc = sw2_acc.at[Vbin_idx].add(one * w_val**2)
        sw3_acc = sw3_acc.at[Vbin_idx].add(one * w_val**3)
        sw4_acc = sw4_acc.at[Vbin_idx].add(one * w_val**4)

        return Rzphi_acc, cnt_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc

    def step(carry, _):
        (t, y, Rzphi_acc, cnt_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc,
         EJ_first, EJ_last, step_idx) = carry
        r, v = _split(y)

        # ── Leapfrog KDK with rotating frame ──
        # Gravity half-kick
        a0 = acc_fn(*r)
        v_half = v + 0.5 * dt * a0

        # Exact Omega-subflow
        theta = Omega * dt
        c, s_theta = jnp.cos(theta), jnp.sin(theta)

        x_bar = r[0] + dt * v_half[0]
        y_bar = r[1] + dt * v_half[1]
        x_new = c * x_bar + s_theta * y_bar
        y_new = -s_theta * x_bar + c * y_bar
        z_new = r[2] + dt * v_half[2]

        vx_rot = c * v_half[0] + s_theta * v_half[1]
        vy_rot = -s_theta * v_half[0] + c * v_half[1]
        vz_rot = v_half[2]

        r_new = jnp.array([x_new, y_new, z_new])
        v_rot = jnp.array([vx_rot, vy_rot, vz_rot])
        t_new = t + dt

        # Gravity half-kick at updated position
        a1 = acc_fn(*r_new)
        v_new = v_rot + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)

        # ── Fused binning for this timestep ──
        Rzphi_acc, cnt_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc = \
            _bin_one_point(r_new, v_new, Rzphi_acc, cnt_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc)

        # ── Track Jacobi energy at first and last step ──
        EJ_now = _compute_EJ(r_new, v_new)
        EJ_first = jnp.where(step_idx == 0, EJ_now, EJ_first)
        EJ_last = EJ_now

        return (t_new, y_new, Rzphi_acc, cnt_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc,
                EJ_first, EJ_last, step_idx + 1), None  # <-- No per-step output!

    init_carry = (t0, w0, init_Rzphi, init_counts, init_sw1, init_sw2, init_sw3, init_sw4,
                  init_EJ_first, init_EJ_last, 0)

    (_, _, Rzphi_acc, cnt_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc,
     EJ_first, EJ_last, _), _ = jax.lax.scan(step, init_carry, xs=None, length=n_steps, unroll=unroll)

    # ── Jacobi energy check ──
    delta_EJ = jnp.fabs(EJ_last / EJ_first - 1)

    # ── Finalize GH moments ──
    eps = EPSILON
    norm = cnt_acc + eps
    w1 = sw1_acc / norm
    w2 = sw2_acc / norm
    w3 = sw3_acc / norm
    w4 = sw4_acc / norm

    h1 = w1
    h2 = (w2 - 1) / jnp.sqrt(2.0)
    h3 = (w3 - 3 * w1) / jnp.sqrt(6.0)
    h4 = (w4 - 6 * w2 + 3) / jnp.sqrt(24.0)
    surface_density = cnt_acc / (num_per_bin * area_pixel + eps)

    # ── Invalidate bad orbits (same logic as original) ──
    valid = jnp.where(delta_EJ < 0.5, 1.0, 0.0)
    Rzphi_acc = jnp.where(valid > 0.5, Rzphi_acc, jnp.zeros_like(Rzphi_acc))
    surface_density = jnp.where(valid > 0.5, surface_density, jnp.zeros_like(surface_density))
    h1 = jnp.where(valid > 0.5, h1, jnp.zeros_like(h1))
    h2 = jnp.where(valid > 0.5, h2, jnp.zeros_like(h2))
    h3 = jnp.where(valid > 0.5, h3, jnp.zeros_like(h3))
    h4 = jnp.where(valid > 0.5, h4, jnp.zeros_like(h4))

    return Rzphi_acc, surface_density, h1, h2, h3, h4, valid


_integrate_barred_fused_vmap = jax.vmap(integrate_leapfrog_barred_fused,
                    in_axes=(
                            0, None, None, None, 0, None, None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None))

# ==========================================================================
# V2: same scan-then-bin as original, with three optimizations:
#   1. Jacobi energy: only evaluate pot_fn at first & last point (not all n_steps)
#   2. Replace lax.cond with jnp.where (avoids evaluating both branches under vmap)
#   3. Precompute cos/sin of Omega*dt before the scan
# ==========================================================================
@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'n_steps', 'unroll', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_leapfrog_barred_v2(w0, acc_fn, pot_fn, n_steps, dt=0.010, t0=0.0, Omega=0.0, unroll=False,
                                 num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
                                 num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
                                 Rzphi_minmax=jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]),
                                 XY_minmax=jnp.array([[-10.,10.],[-2.,2.]]),
                                 nRzphi=jnp.array([10,6,6]), nXY=jnp.array([40,30]),
                                 num_segments_Rzphi=360,
                                 v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
                                 rotation_matrix=jnp.eye(3)):
    """Leapfrog (KDK) with barred potential — optimized v2.

    Same physics and outputs as integrate_leapfrog_barred, but:
    - Jacobi energy check uses only 2 pot_fn evaluations instead of n_steps
    - No lax.cond (uses jnp.where masking instead, faster under vmap)
    - cos/sin of rotation angle precomputed once
    """
    # Fix 3: precompute cos/sin outside the scan
    theta = Omega * dt
    c_theta = jnp.cos(theta)
    s_theta = jnp.sin(theta)

    def step(carry, _):
        t, y = carry
        r, v = _split(y)

        # Gravity half-kick
        a0 = acc_fn(*r)
        v_half = v + 0.5 * dt * a0

        # Exact Omega-subflow (uses precomputed cos/sin)
        x_bar = r[0] + dt * v_half[0]
        y_bar = r[1] + dt * v_half[1]
        x_new = c_theta * x_bar + s_theta * y_bar
        y_new = -s_theta * x_bar + c_theta * y_bar
        z_new = r[2] + dt * v_half[2]

        vx_rot = c_theta * v_half[0] + s_theta * v_half[1]
        vy_rot = -s_theta * v_half[0] + c_theta * v_half[1]
        vz_rot = v_half[2]

        r_new = jnp.array([x_new, y_new, z_new])
        v_rot = jnp.array([vx_rot, vy_rot, vz_rot])
        t_new = t + dt

        # Gravity half-kick at updated position
        a1 = acc_fn(*r_new)
        v_new = v_rot + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return (t_new, y_new), y_new  # only store phase-space, not time

    (_, _), wN = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)

    # Fix 1: Jacobi energy check — only first and last point
    def _EJ_single(w):
        r, v = w[:3], w[3:]
        Lz = v[0] * r[1] - v[1] * r[0]
        E_pot = pot_fn(r[0][jnp.newaxis], r[1][jnp.newaxis], r[2][jnp.newaxis])[0]
        E_kin = 0.5 * jnp.sum(v**2)
        return E_pot + E_kin - Omega * Lz

    EJ_first = _EJ_single(wN[0])
    EJ_last = _EJ_single(wN[-1])
    delta_EJ = jnp.fabs(EJ_last / EJ_first - 1)
    valid = jnp.where(delta_EJ < 0.5, 1.0, 0.0)

    # Binning (same vectorized logic as original)
    Rzphi = jnp.array([jnp.sqrt(wN[:,0]**2 + wN[:,1]**2), wN[:,2],
                        jnp.arctan2(wN[:,1], wN[:,0])]).T

    x_pos = wN[:, :3]
    v_vel = wN[:, 3:]
    x_rot = (rotation_matrix @ x_pos.T).T
    v_rot = (rotation_matrix @ v_vel.T).T
    wN_rot = jnp.concatenate([x_rot, v_rot], axis=-1)

    XY = jnp.array([wN_rot[:, 0], wN_rot[:, 2]]).T

    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_indices = assign_regular_grid(Rzphi,
                                        grid_min=Rzphi_minmax[:, 0],
                                        grid_max=Rzphi_minmax[:, 1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)
    Rzphi_bin_counts = jax.ops.segment_sum(jnp.ones_like(Rzphi_indices),
                                           Rzphi_indices,
                                           num_segments=num_segments_Rzphi)

    XY_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    XY_indices = assign_regular_grid(XY,
                                     grid_min=XY_minmax[:, 0],
                                     grid_max=XY_minmax[:, 1],
                                     n_bins=nXY,
                                     strides=XY_strides)

    area_pixel = ((XY_minmax[0, 1] - XY_minmax[0, 0]) / nXY[0]) * \
                 ((XY_minmax[1, 1] - XY_minmax[1, 0]) / nXY[1]) * 1e6

    Vbin_indices = bin_mapping[XY_indices]

    v0_cell = v0[Vbin_indices]
    s_cell = s[Vbin_indices]

    vy = wN_rot[:, 4] * KPCGYR_TO_KMS
    w = (vy - v0_cell) / s_cell
    eps = EPSILON
    counts = jax.ops.segment_sum(jnp.ones_like(vy), Vbin_indices, num_segments=num_Vbin)
    sum_w1 = jax.ops.segment_sum(w, Vbin_indices, num_segments=num_Vbin)
    sum_w2 = jax.ops.segment_sum(w**2, Vbin_indices, num_segments=num_Vbin)
    sum_w3 = jax.ops.segment_sum(w**3, Vbin_indices, num_segments=num_Vbin)
    sum_w4 = jax.ops.segment_sum(w**4, Vbin_indices, num_segments=num_Vbin)
    norm = counts + eps
    w1 = sum_w1 / norm
    w2 = sum_w2 / norm
    w3 = sum_w3 / norm
    w4 = sum_w4 / norm

    h1 = w1
    h2 = (w2 - 1) / jnp.sqrt(2.0)
    h3 = (w3 - 3 * w1) / jnp.sqrt(6.0)
    h4 = (w4 - 6 * w2 + 3) / jnp.sqrt(24.0)
    surface_density = counts / (num_per_bin * area_pixel + eps)

    # Fix 2: jnp.where masking instead of lax.cond
    Rzphi_bin_counts = jnp.where(valid > 0.5, Rzphi_bin_counts, jnp.zeros_like(Rzphi_bin_counts))
    surface_density = jnp.where(valid > 0.5, surface_density, jnp.zeros_like(surface_density))
    h1 = jnp.where(valid > 0.5, h1, jnp.zeros_like(h1))
    h2 = jnp.where(valid > 0.5, h2, jnp.zeros_like(h2))
    h3 = jnp.where(valid > 0.5, h3, jnp.zeros_like(h3))
    h4 = jnp.where(valid > 0.5, h4, jnp.zeros_like(h4))

    return Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid


_integrate_barred_v2_vmap = jax.vmap(integrate_leapfrog_barred_v2,
                    in_axes=(
                            0, None, None, None, 0, None, None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None))


@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'n_steps', 'unroll', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_batch(w0, acc_fn, pot_fn, n_steps, dt = 0.010, t0 = 0.0, Omega = 0.0, unroll=True,
                    num_Vbin = 1028, bin_mapping = jnp.zeros(2400, dtype=jnp.int32), num_per_bin = jnp.zeros(1028, dtype=jnp.int32),
                    Rzphi_minmax=jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]), XY_minmax=jnp.array([[-10.,10.],[-2.,2.]]),
                    nRzphi=jnp.array([10,6,6]), nXY=jnp.array([40,30]), num_segments_Rzphi=360,
                    v0 = jnp.zeros(1028), s= jnp.ones(1028) * 5.0, rotation_matrix = jnp.eye(3)
                    ):
    

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid = _integrate_barred_vmap(
                        w0, acc_fn, pot_fn, n_steps, dt, t0, Omega, unroll,
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_minmax, XY_minmax,
                        nRzphi, nXY, num_segments_Rzphi,
                        v0, s, rotation_matrix)
    A_Rzphi = Rzphi_bin_counts.T / n_steps
    A_xy = surface_density.T / n_steps
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    weights = jnp.ones(A_Rzphi.shape[1]) / (valid.sum() + 0.1) #A_Rzphi.shape[1]

    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)

    Rzphi_bin_counts = A_Rzphi @ weights
    surface_density = A_xy @ weights

    h1 = A_h1 @ weights / (surface_density + EPSILON)
    h2 = A_h2 @ weights / (surface_density + EPSILON)
    h3 = A_h3 @ weights / (surface_density + EPSILON)
    h4 = A_h4 @ weights / (surface_density + EPSILON)

    return Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid.sum()

_integrate_batch_vmap = jax.vmap(integrate_batch,
                    in_axes=(
                            0, None, None, None, 0, None, None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None))


# ==========================================================================
# Adaptive step-size Bogacki-Shampine RK2(3) integrator
# ==========================================================================

def _deriv_barred(state, acc_fn, Omega):
    """RHS of the co-rotating frame ODE.

    State = [x, y, z, vx, vy, vz] where velocities are inertial
    components rotated into bar-frame axes.

    ODE:  dx/dt  = vx + Omega*y      dvx/dt = ax + Omega*vy
          dy/dt  = vy - Omega*x      dvy/dt = ay - Omega*vx
          dz/dt  = vz                dvz/dt = az
    """
    x, y_pos, z, vx, vy, vz = state[0], state[1], state[2], state[3], state[4], state[5]
    ax, ay, az = acc_fn(x, y_pos, z)
    return jnp.array([
        vx + Omega * y_pos,
        vy - Omega * x,
        vz,
        ax + Omega * vy,
        ay - Omega * vx,
        az,
    ])


def _compute_EJ(y, pot_fn, Omega):
    """Jacobi energy: E_J = 0.5*v^2 + Phi - Omega*(x*vy - y*vx)."""
    x, y_pos, z, vx, vy, vz = y[0], y[1], y[2], y[3], y[4], y[5]
    E_kin = 0.5 * (vx**2 + vy**2 + vz**2)
    E_pot = pot_fn(x[jnp.newaxis], y_pos[jnp.newaxis], z[jnp.newaxis])[0]
    Lz = x * vy - y_pos * vx
    return E_kin + E_pot - Omega * Lz


@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_barred(
    w0, acc_fn, pot_fn, N_max, T_total,
    dt_init=0.010, Omega=0.0,
    atol=1e-8, rtol=1e-6,
    dt_min=1e-5, dt_max=0.1,
    num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
    num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
    Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
    XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
    nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
    num_segments_Rzphi=360,
    v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
    rotation_matrix=jnp.eye(3),
):
    """Bogacki-Shampine RK2(3) integrator with embedded error control.

    Uses 3 acc_fn calls per step (FSAL). No pot_fn calls during integration.
    Step size controlled by local truncation error, not energy conservation.
    E_J checked only at orbit level for validity.

    Parameters
    ----------
    w0 : array (6,)
        Initial phase-space state [x, y, z, vx, vy, vz] in bar frame.
    acc_fn : callable
        Acceleration function acc_fn(x, y, z) -> (ax, ay, az) in bar frame.
    pot_fn : callable
        Potential function pot_fn(x, y, z) -> Phi (only for orbit-level E_J check).
    N_max : int
        Maximum number of scan iterations (static for jit).
    T_total : float
        Total integration time.
    dt_init : float
        Initial step size.
    Omega : float
        Bar pattern speed.
    atol : float
        Absolute error tolerance (per component).
    rtol : float
        Relative error tolerance (per component).
    dt_min, dt_max : float
        Step size bounds.

    Returns
    -------
    8-tuple: (Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, n_accepted)
    """
    # Compute initial derivative (FSAL: reused as k1 in first step)
    k1_init = _deriv_barred(w0, acc_fn, Omega)

    # Compute initial E_J for orbit-level validity check
    EJ_0 = _compute_EJ(w0, pot_fn, Omega)

    # BS23 error coefficients: err = y3 - y2
    # err = dt * (-5/72*k1 + 1/12*k2 + 1/9*k3 - 1/8*k4)
    e1, e2, e3, e4 = -5.0/72.0, 1.0/12.0, 1.0/9.0, -1.0/8.0

    safety = 0.9

    def scan_body(carry, _):
        t, y, dt, k1 = carry

        # No early stopping — all N_max iterations do useful work.
        # Orbits past T_total keep integrating; extra points improve
        # the time-averaged density/kinematics at no extra cost.

        # ── Bogacki-Shampine RK2(3) stages ──
        # k1 is carried from previous step (FSAL)
        k2 = _deriv_barred(y + 0.5 * dt * k1, acc_fn, Omega)          # stage 2
        k3 = _deriv_barred(y + 0.75 * dt * k2, acc_fn, Omega)         # stage 3
        y_new = y + dt * (2.0/9.0 * k1 + 1.0/3.0 * k2 + 4.0/9.0 * k3)  # 3rd order
        k4 = _deriv_barred(y_new, acc_fn, Omega)                       # stage 4 (FSAL)

        # ── Embedded error estimate ──
        err_vec = dt * (e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4)

        # Per-component scaling (AGAMA-style)
        scale = atol + rtol * jnp.maximum(jnp.fabs(y), jnp.fabs(y_new))
        err_scaled = err_vec / scale
        err_norm = jnp.sqrt(jnp.mean(err_scaled**2))  # RMS norm

        # Accept if err_norm <= 1
        accept = err_norm <= 1.0

        # Update state
        y_out = jnp.where(accept, y_new, y)
        t_new = jnp.where(accept, t + dt, t)
        # FSAL: if accepted, next k1 = k4; if rejected, keep old k1
        k1_new = jnp.where(accept, k4, k1)

        # Step size adjustment: dt_new = dt * safety * err^(-1/3) for 3rd order method
        scale_factor = safety * jnp.power(jnp.maximum(err_norm, 1e-10), -1.0/3.0)
        scale_factor = jnp.clip(scale_factor, 0.2, 5.0)
        dt_new = dt * scale_factor
        dt_new = jnp.clip(dt_new, dt_min, dt_max)

        valid_flag = accept.astype(jnp.float32)
        # Output dt_used for time-weighting (0 if rejected)
        dt_out = jnp.where(accept, dt, 0.0)

        return (t_new, y_out, dt_new, k1_new), (y_out, valid_flag, dt_out)

    init_carry = (0.0, w0, dt_init, k1_init)
    (t_final, y_final, dt_final, _), (y_traj, valid_mask, dt_traj) = jax.lax.scan(
        scan_body, init_carry, xs=None, length=N_max
    )

    n_accepted = jnp.sum(valid_mask)

    # Orbit-level validity: check total E_J drift (same threshold as leapfrog)
    EJ_final = _compute_EJ(y_final, pot_fn, Omega)
    delta_EJ = jnp.fabs(EJ_final / EJ_0 - 1.0)
    valid = jnp.where((delta_EJ < 0.1) & (t_final > T_total / 10), 1.0, 0.0)#

    # ── Apply 4-fold bar symmetry before binning ──
    # Each orbit point generates 4 symmetric copies (triaxial bar symmetries).
    # sign_sym shape: (4, 6); y_traj shape: (N_max, 6)
    sign_sym = jnp.array([
        [ 1,  1,  1,  1,  1,  1],   # identity
        [ 1,  1, -1,  1,  1, -1],   # z-reflection
        [-1, -1,  1, -1, -1,  1],   # xy point symmetry
        [-1, -1, -1, -1, -1, -1],   # combined
    ])
    y_traj = (y_traj[None, :, :] * sign_sym[:, None, :]).reshape(-1, 6)  # (4*N_max, 6)
    dt_traj = jnp.tile(dt_traj, 4)

    # ── Post-scan binning with dt-weighting ──
    # Rzphi binning (pre-rotation frame)
    R_vals = jnp.sqrt(y_traj[:, 0]**2 + y_traj[:, 1]**2)
    phi_vals = jnp.arctan2(y_traj[:, 1], y_traj[:, 0])
    Rzphi = jnp.stack([R_vals, y_traj[:, 2], phi_vals], axis=-1)

    # Rotation for XY projection
    x_pos = y_traj[:, :3]
    v_vel = y_traj[:, 3:]
    x_rot = (rotation_matrix @ x_pos.T).T
    v_rot = (rotation_matrix @ v_vel.T).T
    wN_rot = jnp.concatenate([x_rot, v_rot], axis=-1)
    XY = jnp.stack([wN_rot[:, 0], wN_rot[:, 2]], axis=-1)

    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_indices = assign_regular_grid(Rzphi,
                                        grid_min=Rzphi_minmax[:, 0],
                                        grid_max=Rzphi_minmax[:, 1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)

    XY_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    XY_indices = assign_regular_grid(XY,
                                     grid_min=XY_minmax[:, 0],
                                     grid_max=XY_minmax[:, 1],
                                     n_bins=nXY,
                                     strides=XY_strides)

    area_pixel = ((XY_minmax[0, 1] - XY_minmax[0, 0]) / nXY[0]) * \
                 ((XY_minmax[1, 1] - XY_minmax[1, 0]) / nXY[1]) * 1e6

    Vbin_indices = bin_mapping[XY_indices]

    # Normalize dt weights so each orbit contributes order unity.
    # sum(dt_norm) = 1, so segment sums give fractional time in each bin.
    T_integrated = jnp.sum(dt_traj) + EPSILON
    dt_norm = dt_traj / T_integrated

    # dt-normalized segment sums
    Rzphi_bin_counts = jax.ops.segment_sum(dt_norm,
                                           Rzphi_indices,
                                           num_segments=num_segments_Rzphi)

    v0_cell = v0[Vbin_indices]
    s_cell = s[Vbin_indices]

    vy = wN_rot[:, 4] * KPCGYR_TO_KMS
    w = (vy - v0_cell) / s_cell
    eps = EPSILON

    counts = jax.ops.segment_sum(dt_norm, Vbin_indices, num_segments=num_Vbin)
    sum_w1 = jax.ops.segment_sum(dt_norm * w, Vbin_indices, num_segments=num_Vbin)
    sum_w2 = jax.ops.segment_sum(dt_norm * w**2, Vbin_indices, num_segments=num_Vbin)
    sum_w3 = jax.ops.segment_sum(dt_norm * w**3, Vbin_indices, num_segments=num_Vbin)
    sum_w4 = jax.ops.segment_sum(dt_norm * w**4, Vbin_indices, num_segments=num_Vbin)

    norm = counts + eps
    w1 = sum_w1 / norm
    w2 = sum_w2 / norm
    w3 = sum_w3 / norm
    w4 = sum_w4 / norm

    h1 = w1
    h2 = (w2 - 1) / jnp.sqrt(2.0)
    h3 = (w3 - 3 * w1) / jnp.sqrt(6.0)
    h4 = (w4 - 6 * w2 + 3) / jnp.sqrt(24.0)
    surface_density = counts / (num_per_bin * area_pixel + eps)

    # Zero out bad orbits
    Rzphi_bin_counts = jnp.where(valid > 0.5, Rzphi_bin_counts, jnp.zeros_like(Rzphi_bin_counts))
    surface_density = jnp.where(valid > 0.5, surface_density, jnp.zeros_like(surface_density))
    h1 = jnp.where(valid > 0.5, h1, jnp.zeros_like(h1))
    h2 = jnp.where(valid > 0.5, h2, jnp.zeros_like(h2))
    h3 = jnp.where(valid > 0.5, h3, jnp.zeros_like(h3))
    h4 = jnp.where(valid > 0.5, h4, jnp.zeros_like(h4))

    return Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, n_accepted, T_integrated


_integrate_adaptive_vmap = jax.vmap(integrate_adaptive_barred,
                    in_axes=(
                            0, None, None, None, 0,    # w0 per-orbit, T_total per-orbit
                            0, None,       # dt_init per-orbit, Omega broadcast
                            None, None,    # atol, rtol broadcast
                            None, None,    # dt_min, dt_max
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None))


# ==========================================================================
# Chunked adaptive integrator: bins every chunk_size steps (vectorized).
# Memory: O(chunk_size * 6) per orbit instead of O(N_max * 6).
# ==========================================================================
@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'chunk_size', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_barred_chunked(
    w0, acc_fn, pot_fn, N_max, T_total,
    dt_init=0.010, Omega=0.0,
    atol=1e-8, rtol=1e-6,
    dt_min=1e-5, dt_max=0.1,
    num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
    num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
    Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
    XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
    nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
    num_segments_Rzphi=360,
    v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
    rotation_matrix=jnp.eye(3),
    chunk_size=500,
):
    """Adaptive BS2(3) integrator with chunked binning.

    Same physics as integrate_adaptive_barred, but instead of storing the full
    (N_max, 6) trajectory and binning at the end, it processes the orbit in
    chunks of `chunk_size` steps: each chunk stores a small (chunk_size, 6)
    trajectory buffer, bins it vectorially, and accumulates into running sums.

    Memory per orbit: O(chunk_size * 6 + num_Vbin + num_segments_Rzphi)
    instead of O(N_max * 6 + num_Vbin + num_segments_Rzphi).

    N_max must be divisible by chunk_size.
    """
    n_chunks = N_max // chunk_size

    # Precompute grid constants
    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    XY_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    area_pixel = ((XY_minmax[0, 1] - XY_minmax[0, 0]) / nXY[0]) * \
                 ((XY_minmax[1, 1] - XY_minmax[1, 0]) / nXY[1]) * 1e6

    # 4-fold bar symmetry signs
    sign_sym = jnp.array([
        [ 1,  1,  1,  1,  1,  1],
        [ 1,  1, -1,  1,  1, -1],
        [-1, -1,  1, -1, -1,  1],
        [-1, -1, -1, -1, -1, -1],
    ])

    # Initial derivative and E_J
    k1_init = _deriv_barred(w0, acc_fn, Omega)
    EJ_0 = _compute_EJ(w0, pot_fn, Omega)

    # BS23 error coefficients
    e1, e2, e3, e4 = -5.0/72.0, 1.0/12.0, 1.0/9.0, -1.0/8.0
    safety = 0.9

    # ── Inner scan: one RK step ──
    def rk_step(carry, _):
        t, y, dt, k1 = carry

        k2 = _deriv_barred(y + 0.5 * dt * k1, acc_fn, Omega)
        k3 = _deriv_barred(y + 0.75 * dt * k2, acc_fn, Omega)
        y_new = y + dt * (2.0/9.0 * k1 + 1.0/3.0 * k2 + 4.0/9.0 * k3)
        k4 = _deriv_barred(y_new, acc_fn, Omega)

        err_vec = dt * (e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4)
        scale = atol + rtol * jnp.maximum(jnp.fabs(y), jnp.fabs(y_new))
        err_norm = jnp.sqrt(jnp.mean((err_vec / scale)**2))
        accept = err_norm <= 1.0

        y_out = jnp.where(accept, y_new, y)
        t_new = jnp.where(accept, t + dt, t)
        k1_new = jnp.where(accept, k4, k1)

        scale_factor = safety * jnp.power(jnp.maximum(err_norm, 1e-10), -1.0/3.0)
        scale_factor = jnp.clip(scale_factor, 0.2, 5.0)
        dt_new = jnp.clip(dt * scale_factor, dt_min, dt_max)

        dt_out = jnp.where(accept, dt, 0.0)
        return (t_new, y_out, dt_new, k1_new), (y_out, dt_out)

    # ── Bin a chunk of trajectory points (vectorized) ──
    def bin_chunk(y_chunk, dt_chunk, Rzphi_acc, counts_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc):
        # Apply 4-fold symmetry: (chunk_size, 6) -> (4*chunk_size, 6)
        y_sym = (y_chunk[None, :, :] * sign_sym[:, None, :]).reshape(-1, 6)
        dt_sym = jnp.tile(dt_chunk, 4)

        # Rzphi binning (pre-rotation frame)
        R_vals = jnp.sqrt(y_sym[:, 0]**2 + y_sym[:, 1]**2)
        phi_vals = jnp.arctan2(y_sym[:, 1], y_sym[:, 0])
        Rzphi = jnp.stack([R_vals, y_sym[:, 2], phi_vals], axis=-1)

        Rzphi_indices = assign_regular_grid(Rzphi,
                                            grid_min=Rzphi_minmax[:, 0],
                                            grid_max=Rzphi_minmax[:, 1],
                                            n_bins=nRzphi,
                                            strides=Rzphi_strides)

        # XY binning (post-rotation)
        x_pos = y_sym[:, :3]
        v_vel = y_sym[:, 3:]
        x_rot = (rotation_matrix @ x_pos.T).T
        v_rot = (rotation_matrix @ v_vel.T).T
        XY = jnp.stack([x_rot[:, 0], x_rot[:, 2]], axis=-1)

        XY_indices = assign_regular_grid(XY,
                                         grid_min=XY_minmax[:, 0],
                                         grid_max=XY_minmax[:, 1],
                                         n_bins=nXY,
                                         strides=XY_strides)
        Vbin_indices = bin_mapping[XY_indices]

        # GH moment accumulation (unnormalized — we normalize at the end)
        vy = v_rot[:, 1] * KPCGYR_TO_KMS
        v0_cell = v0[Vbin_indices]
        s_cell = s[Vbin_indices]
        w = (vy - v0_cell) / s_cell

        # Accumulate with dt weighting (unnormalized)
        Rzphi_acc = Rzphi_acc + jax.ops.segment_sum(dt_sym, Rzphi_indices, num_segments=num_segments_Rzphi)
        counts_acc = counts_acc + jax.ops.segment_sum(dt_sym, Vbin_indices, num_segments=num_Vbin)
        sw1_acc = sw1_acc + jax.ops.segment_sum(dt_sym * w, Vbin_indices, num_segments=num_Vbin)
        sw2_acc = sw2_acc + jax.ops.segment_sum(dt_sym * w**2, Vbin_indices, num_segments=num_Vbin)
        sw3_acc = sw3_acc + jax.ops.segment_sum(dt_sym * w**3, Vbin_indices, num_segments=num_Vbin)
        sw4_acc = sw4_acc + jax.ops.segment_sum(dt_sym * w**4, Vbin_indices, num_segments=num_Vbin)
        T_acc = T_acc + jnp.sum(dt_sym)

        return Rzphi_acc, counts_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc

    # ── Outer scan: process one chunk per iteration ──
    def chunk_body(carry, _):
        t, y, dt, k1, Rzphi_acc, counts_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc = carry

        # Inner scan: chunk_size RK steps
        (t_new, y_new, dt_new, k1_new), (y_chunk, dt_chunk) = jax.lax.scan(
            rk_step, (t, y, dt, k1), xs=None, length=chunk_size
        )

        # Bin the chunk
        Rzphi_acc, counts_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc = \
            bin_chunk(y_chunk, dt_chunk, Rzphi_acc, counts_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc)

        return (t_new, y_new, dt_new, k1_new, Rzphi_acc, counts_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc), None

    # Initialize accumulators
    init_carry = (
        0.0, w0, dt_init, k1_init,
        jnp.zeros(num_segments_Rzphi),  # Rzphi_acc
        jnp.zeros(num_Vbin),            # counts_acc
        jnp.zeros(num_Vbin),            # sw1_acc
        jnp.zeros(num_Vbin),            # sw2_acc
        jnp.zeros(num_Vbin),            # sw3_acc
        jnp.zeros(num_Vbin),            # sw4_acc
        0.0,                            # T_acc
    )

    (t_final, y_final, _, _, Rzphi_acc, counts_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc), _ = \
        jax.lax.scan(chunk_body, init_carry, xs=None, length=n_chunks)

    # ── Orbit-level validity ──
    EJ_final = _compute_EJ(y_final, pot_fn, Omega)
    delta_EJ = jnp.fabs(EJ_final / EJ_0 - 1.0)
    valid = jnp.where((delta_EJ < 0.1) & (t_final > T_total / 10), 1.0, 0.0)

    # ── Normalize and compute GH moments ──
    T_integrated = T_acc + EPSILON
    Rzphi_bin_counts = Rzphi_acc / T_integrated

    eps = EPSILON
    norm = counts_acc / T_integrated + eps
    w1 = (sw1_acc / T_integrated) / norm
    w2 = (sw2_acc / T_integrated) / norm
    w3 = (sw3_acc / T_integrated) / norm
    w4 = (sw4_acc / T_integrated) / norm

    h1 = w1
    h2 = (w2 - 1) / jnp.sqrt(2.0)
    h3 = (w3 - 3 * w1) / jnp.sqrt(6.0)
    h4 = (w4 - 6 * w2 + 3) / jnp.sqrt(24.0)
    surface_density = (counts_acc / T_integrated) / (num_per_bin * area_pixel + eps)

    n_accepted = T_acc / (dt_init + EPSILON)  # approximate

    # Zero out bad orbits
    Rzphi_bin_counts = jnp.where(valid > 0.5, Rzphi_bin_counts, jnp.zeros_like(Rzphi_bin_counts))
    surface_density = jnp.where(valid > 0.5, surface_density, jnp.zeros_like(surface_density))
    h1 = jnp.where(valid > 0.5, h1, jnp.zeros_like(h1))
    h2 = jnp.where(valid > 0.5, h2, jnp.zeros_like(h2))
    h3 = jnp.where(valid > 0.5, h3, jnp.zeros_like(h3))
    h4 = jnp.where(valid > 0.5, h4, jnp.zeros_like(h4))

    return Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, n_accepted, T_integrated


_integrate_adaptive_chunked_vmap = jax.vmap(integrate_adaptive_barred_chunked,
                    in_axes=(
                            0, None, None, None, 0,
                            0, None,
                            None, None,
                            None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None,
                            None))


@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'chunk_size', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_batch_chunked(w0, acc_fn, pot_fn, N_max, T_total,
                             dt_init=0.010, Omega=0.0,
                             atol=1e-8, rtol=1e-6,
                             dt_min=1e-5, dt_max=0.1,
                             num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
                             num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
                             Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
                             XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
                             nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
                             num_segments_Rzphi=360,
                             v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
                             rotation_matrix=jnp.eye(3),
                             chunk_size=500):
    """Batch adaptive integration with chunked binning, analogous to integrate_adaptive_batch."""

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, n_accepted, T_integrated = _integrate_adaptive_chunked_vmap(
        w0, acc_fn, pot_fn, N_max, T_total,
        dt_init, Omega, atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_minmax, XY_minmax,
        nRzphi, nXY, num_segments_Rzphi,
        v0, s, rotation_matrix,
        chunk_size)

    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    weights = jnp.ones(A_Rzphi.shape[1]) / (valid.sum() + 0.1)
    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)

    Rzphi_bin_counts_out = A_Rzphi @ weights
    surface_density_out = A_xy @ weights
    h1_out = A_h1 @ weights / (surface_density_out + EPSILON)
    h2_out = A_h2 @ weights / (surface_density_out + EPSILON)
    h3_out = A_h3 @ weights / (surface_density_out + EPSILON)
    h4_out = A_h4 @ weights / (surface_density_out + EPSILON)

    return Rzphi_bin_counts_out, surface_density_out, h1_out, h2_out, h3_out, h4_out, valid.sum()


_integrate_adaptive_batch_chunked_vmap = jax.vmap(integrate_adaptive_batch_chunked,
                    in_axes=(
                            0, None, None, None, 0,
                            0, None,
                            None, None,
                            None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None,
                            None))


@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_batch(w0, acc_fn, pot_fn, N_max, T_total,
                             dt_init=0.010, Omega=0.0,
                             atol=1e-8, rtol=1e-6,
                             dt_min=1e-5, dt_max=0.1,
                             num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
                             num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
                             Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
                             XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
                             nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
                             num_segments_Rzphi=360,
                             v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
                             rotation_matrix=jnp.eye(3)):
    """Batch adaptive integration over multiple orbits, analogous to integrate_batch."""

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, n_accepted, T_integrated = _integrate_adaptive_vmap(
        w0, acc_fn, pot_fn, N_max, T_total,
        dt_init, Omega, atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_minmax, XY_minmax,
        nRzphi, nXY, num_segments_Rzphi,
        v0, s, rotation_matrix)

    # Per-orbit normalization already done inside integrate_adaptive_barred
    # (dt_norm = dt / sum(dt)), so bin counts are already fractional.
    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T

    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    weights = jnp.ones(A_Rzphi.shape[1]) / (valid.sum() + 0.1)

    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)

    Rzphi_bin_counts_out = A_Rzphi @ weights
    surface_density_out = A_xy @ weights

    h1_out = A_h1 @ weights / (surface_density_out + EPSILON)
    h2_out = A_h2 @ weights / (surface_density_out + EPSILON)
    h3_out = A_h3 @ weights / (surface_density_out + EPSILON)
    h4_out = A_h4 @ weights / (surface_density_out + EPSILON)

    return Rzphi_bin_counts_out, surface_density_out, h1_out, h2_out, h3_out, h4_out, valid.sum()


_integrate_adaptive_batch_vmap = jax.vmap(integrate_adaptive_batch,
                    in_axes=(
                            0, None, None, None, 0,
                            0, None,
                            None, None,
                            None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None))


@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'chunk_size', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_batch_chunked_psf(
    w0, acc_fn, pot_fn, N_max, T_total,
    dt_init=0.010, Omega=0.0,
    atol=1e-8, rtol=1e-6,
    dt_min=1e-5, dt_max=0.1,
    num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
    num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
    Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
    XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
    nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
    num_segments_Rzphi=360,
    v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
    rotation_matrix=jnp.eye(3),
    chunk_size=500,
    P_psf=jnp.eye(1),
):
    """Adaptive integration with chunked binning and PSF smearing.

    Same as integrate_adaptive_batch_chunked, but after building the orbit
    library matrices, applies the PSF transfer matrix P to the projected
    (XY) quantities using flux-weighted convolution.

    Parameters
    ----------
    P_psf : array (num_Vbin, num_Vbin)
        PSF transfer matrix. P_psf[i,j] = fraction of flux from bin j
        observed in bin i. Pass identity matrix for no smearing.
        Build with utils.compute_transfer_matrix().
    All other parameters: same as integrate_adaptive_batch_chunked.

    Returns
    -------
    A_Rzphi : array (num_segments_Rzphi, n_orbits)
        3D density orbit library (NOT smeared — intrinsic).
    A_xy : array (num_Vbin, n_orbits)
        Surface density orbit library (PSF-smeared).
    A_h1, A_h2, A_h3, A_h4 : array (num_Vbin, n_orbits)
        Flux-weighted GH moment orbit libraries (PSF-smeared).
        These are already multiplied by A_xy: model h_i = (A_hi @ w) / (A_xy @ w).
    valid_count : float
        Number of valid orbits.
    """
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, n_accepted, T_integrated = \
        _integrate_adaptive_chunked_vmap(
            w0, acc_fn, pot_fn, N_max, T_total,
            dt_init, Omega, atol, rtol, dt_min, dt_max,
            num_Vbin, bin_mapping, num_per_bin,
            Rzphi_minmax, XY_minmax,
            nRzphi, nXY, num_segments_Rzphi,
            v0, s, rotation_matrix,
            chunk_size)

    # Transpose to orbit library matrices: (n_bins, n_orbits)
    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    # Flux-weight the GH moments: A_hi becomes Sigma * h_i per orbit
    A_h1, A_h2, A_h3, A_h4 = A_h1 * A_xy, A_h2 * A_xy, A_h3 * A_xy, A_h4 * A_xy

    # Apply PSF transfer matrix to projected quantities
    # P @ A_xy smears surface density; P @ (A_hi) smears flux-weighted moments
    # Kinematics recovery: h_i_obs = (P @ A_hi) @ w / ((P @ A_xy) @ w)
    A_xy = P_psf @ A_xy
    A_h1 = P_psf @ A_h1
    A_h2 = P_psf @ A_h2
    A_h3 = P_psf @ A_h3
    A_h4 = P_psf @ A_h4

    return A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, valid.sum()


_integrate_adaptive_batch_chunked_psf_vmap = jax.vmap(
    integrate_adaptive_batch_chunked_psf,
    in_axes=(
        0, None, None, None, 0,
        0, None,
        None, None,
        None, None,
        None, None, None,
        None, None,
        None, None, None,
        None, None, None,
        None, None))