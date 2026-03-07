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



@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_leapfrog_barred(w0, acc_fn, n_steps, dt = 0.010, t0 = 0.0, Omega = 0.0, unroll=True, 
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

    return Rzphi_bin_counts, surface_density, h1, h2, h3, h4



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
