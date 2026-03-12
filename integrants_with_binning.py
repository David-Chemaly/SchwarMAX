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
    """Leapfrog (KDK) — accumulates bin statistics in the scan carry.

    Avoids storing the full (n_steps, 6) trajectory; instead accumulates
    Rzphi/XY counts and GH-moment sums incrementally.  Peak memory is
    O(n_bins) per orbit instead of O(n_steps).

    Energy conservation is checked only at t=0 and t=final (2 pot_fn
    evaluations instead of n_steps), which is mathematically equivalent
    since delta_EJ only uses E_J[0] and E_J[-1].

    num_segments_Rzphi: int
        Number of segments for Rzphi bin counting. MUST equal to nRzphi.prod()
    num_Vbin: int
        Number of Voronoi bins.

    v0 and s are arrays of reference velocity and dispersion for each
    Voronoi bin.  Length must equal num_Vbin.
    """

    # ------------------------------------------------------------------
    # Precompute loop-invariant constants outside the scan
    # ------------------------------------------------------------------
    theta    = Omega * dt
    c_th     = jnp.cos(theta)
    s_th     = jnp.sin(theta)

    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    XY_strides    = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])

    area_pixel = (
        (XY_minmax[0, 1] - XY_minmax[0, 0]) / nXY[0]
      * (XY_minmax[1, 1] - XY_minmax[1, 0]) / nXY[1]
      * 1e6
    )

    # ------------------------------------------------------------------
    # Scan: carry = (t, state, Rzphi_counts+1, counts, sw1..sw4)
    # The "+1" extra slot in Rzphi_counts absorbs out-of-bounds points
    # (segment_sum discards indices >= num_segments; we replicate that by
    #  writing OOB into slot num_segments_Rzphi and slicing it off at the end).
    # ------------------------------------------------------------------
    init_carry = (
        t0,
        w0,
        jnp.zeros(num_segments_Rzphi + 1, dtype=jnp.float32),
        jnp.zeros(num_Vbin,               dtype=jnp.float32),
        jnp.zeros(num_Vbin,               dtype=jnp.float32),
        jnp.zeros(num_Vbin,               dtype=jnp.float32),
        jnp.zeros(num_Vbin,               dtype=jnp.float32),
        jnp.zeros(num_Vbin,               dtype=jnp.float32),
    )

    def step(carry, _):
        t, y, Rzphi_counts, counts, sw1, sw2, sw3, sw4 = carry
        r, vel = _split(y)

        # --- Leapfrog KDK in rotating (bar) frame ---
        a0     = acc_fn(*r)
        v_half = vel + 0.5 * dt * a0

        # Drift + exact bar-frame rotation over dt
        x_bar = r[0] + dt * v_half[0]
        y_bar = r[1] + dt * v_half[1]
        x_new =  c_th * x_bar + s_th * y_bar
        y_new = -s_th * x_bar + c_th * y_bar
        z_new =  r[2] + dt * v_half[2]

        vx_rot =  c_th * v_half[0] + s_th * v_half[1]
        vy_rot = -s_th * v_half[0] + c_th * v_half[1]
        vz_rot =  v_half[2]

        r_new   = jnp.array([x_new, y_new, z_new])
        v_rot   = jnp.array([vx_rot, vy_rot, vz_rot])
        t_new   = t + dt

        a1    = acc_fn(*r_new)
        v_new = v_rot + 0.5 * dt * a1
        y_new_state = _merge(r_new, v_new)

        # --- Rzphi binning (bar frame, before observation rotation) ---
        R_new   = jnp.sqrt(x_new**2 + y_new**2)
        phi_new = jnp.arctan2(y_new, x_new)
        Rzphi_idx = assign_regular_grid(
            jnp.array([[R_new, z_new, phi_new]]),
            grid_min=Rzphi_minmax[:, 0],
            grid_max=Rzphi_minmax[:, 1],
            n_bins=nRzphi,
            strides=Rzphi_strides,
        )[0]
        Rzphi_counts = Rzphi_counts.at[Rzphi_idx].add(1.0)

        # --- XY / Voronoi binning (observation frame after rotation) ---
        r_obs = rotation_matrix @ r_new   # (3,)
        v_obs = rotation_matrix @ v_new   # (3,)

        XY_idx = assign_regular_grid(
            jnp.array([[r_obs[0], r_obs[2]]]),
            grid_min=XY_minmax[:, 0],
            grid_max=XY_minmax[:, 1],
            n_bins=nXY,
            strides=XY_strides,
        )[0]
        Vbin_idx = bin_mapping[XY_idx]

        vy_obs = v_obs[1] * KPCGYR_TO_KMS
        w_val  = (vy_obs - v0[Vbin_idx]) / s[Vbin_idx]
        w2_val = w_val * w_val
        w3_val = w2_val * w_val
        w4_val = w3_val * w_val

        counts = counts.at[Vbin_idx].add(1.0)
        sw1    = sw1.at[Vbin_idx].add(w_val)
        sw2    = sw2.at[Vbin_idx].add(w2_val)
        sw3    = sw3.at[Vbin_idx].add(w3_val)
        sw4    = sw4.at[Vbin_idx].add(w4_val)

        return (t_new, y_new_state, Rzphi_counts, counts, sw1, sw2, sw3, sw4), None

    final_carry, _ = jax.lax.scan(
        step, init_carry, xs=None, length=n_steps, unroll=unroll
    )
    _, y_final, Rzphi_counts, counts, sw1, sw2, sw3, sw4 = final_carry

    # Discard the OOB slot (index num_segments_Rzphi)
    Rzphi_counts = Rzphi_counts[:num_segments_Rzphi]

    # ------------------------------------------------------------------
    # Energy conservation check — only at t=0 and t=final (2 evaluations
    # instead of n_steps; mathematically equivalent since only E_J[0] and
    # E_J[-1] were used in the original).
    # ------------------------------------------------------------------
    r0_arr, vel0_arr = _split(w0)
    E_pot_0 = pot_fn(r0_arr[0:1], r0_arr[1:2], r0_arr[2:3])[0]
    E_kin_0 = 0.5 * jnp.dot(vel0_arr, vel0_arr)
    Lz_0    = vel0_arr[0] * r0_arr[1] - vel0_arr[1] * r0_arr[0]
    E_J_0   = E_pot_0 + E_kin_0 - Omega * Lz_0

    r_f, v_f = _split(y_final)
    E_pot_f = pot_fn(r_f[0:1], r_f[1:2], r_f[2:3])[0]
    E_kin_f = 0.5 * jnp.dot(v_f, v_f)
    Lz_f    = v_f[0] * r_f[1] - v_f[1] * r_f[0]
    E_J_f   = E_pot_f + E_kin_f - Omega * Lz_f

    delta_EJ = jnp.fabs(E_J_f / E_J_0 - 1)

    # ------------------------------------------------------------------
    # Final statistics (computed unconditionally; zeros returned for bad
    # orbits via lax.cond — same logic as original).
    # ------------------------------------------------------------------
    eps  = EPSILON
    norm = counts + eps
    w1   = sw1 / norm
    w2   = sw2 / norm
    w3   = sw3 / norm
    w4   = sw4 / norm

    h1 = w1
    h2 = (w2 - 1.0) / jnp.sqrt(2.0)
    h3 = (w3 - 3.0 * w1) / jnp.sqrt(6.0)
    h4 = (w4 - 6.0 * w2 + 3.0) / jnp.sqrt(24.0)

    surface_density = counts / (num_per_bin * area_pixel + eps)

    def good_orb(_):
        return Rzphi_counts.astype(jnp.float32), surface_density, h1, h2, h3, h4, 1.

    def bad_orb(_):
        return (
            jnp.zeros(num_segments_Rzphi, dtype=jnp.float32),
            jnp.zeros(num_Vbin, dtype=jnp.float32),
            jnp.zeros(num_Vbin, dtype=jnp.float32),
            jnp.zeros(num_Vbin, dtype=jnp.float32),
            jnp.zeros(num_Vbin, dtype=jnp.float32),
            jnp.zeros(num_Vbin, dtype=jnp.float32),
            0.0,
        )

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid = jax.lax.cond(
        delta_EJ < 0.5, good_orb, bad_orb, None
    )
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

@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'n_steps', 'unroll', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_batch(w0, acc_fn, pot_fn, n_steps, dt = 0.010, t0 = 0.0, Omega = 0.0, unroll=True, 
                    num_Vbin = 1028, bin_mapping = jnp.zeros(2400, dtype=jnp.int32), num_per_bin = jnp.zeros(1028, dtype=jnp.int32),
                    Rzphi_minmax=jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]), XY_minmax=jnp.array([[-10.,10.],[-2.,2.]]),
                    nRzphi=jnp.array([10,6,6]), nXY=jnp.array([40,30]), num_segments_Rzphi=360,
                    v0 = jnp.zeros(1028), s= jnp.ones(1028) * 5.0, rotation_matrix = jnp.eye(3)
                    ):
    

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid = _integrate_barred_vmap(
                        w0, acc_fn, pot_fn, n_steps, dt, t0, -Omega, unroll,
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