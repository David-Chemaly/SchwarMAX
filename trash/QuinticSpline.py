import jax
import jax.numpy as jnp

@jax.jit
def jax_compute_derivatives(x, y):
    """
    Compute first and second derivatives for a Natural Quintic Spline.
    
    Parameters
    ----------
    x : array
        Grid points (must be strictly increasing)
    y : array
        Function values at grid points
        
    Returns
    -------
    D1 : array
        First derivatives at grid points
    D2 : array
        Second derivatives at grid points
    """
    n = len(x)
    h = jnp.diff(x)
    
    # System size is 2*n (solving for y' and y'' at each knot)
    # Layout: x0_d1, x0_d2, x1_d1, x1_d2, ...
    dim = 2 * n
    A = jnp.zeros((dim, dim))
    b = jnp.zeros(dim)
    
    # --- Natural Boundary Conditions at x[0] ---
    # S'''(0) = 0
    # Eq: -36/h0^2 * d0 - 9/h0 * s0 - 24/h0^2 * d1 + 3/h0 * s1 = 60/h0^3 * (y0 - y1)
    h0 = h[0]
    row = 0
    A = A.at[row, 0].set(-36.0 / h0**2) # d0
    A = A.at[row, 1].set(-9.0 / h0)     # s0
    A = A.at[row, 2].set(-24.0 / h0**2) # d1
    A = A.at[row, 3].set(3.0 / h0)      # s1
    b = b.at[row].set(60.0 / h0**3 * (y[0] - y[1]))
    
    # S''''(0) = 0
    # Eq: 192/h0^3 * d0 + 36/h0^2 * s0 + 168/h0^3 * d1 - 24/h0^2 * s1 = -360/h0^4 * (y0 - y1)
    row = 1
    A = A.at[row, 0].set(192.0 / h0**3) # d0
    A = A.at[row, 1].set(36.0 / h0**2)  # s0
    A = A.at[row, 2].set(168.0 / h0**3) # d1
    A = A.at[row, 3].set(-24.0 / h0**2) # s1
    b = b.at[row].set(-360.0 / h0**4 * (y[0] - y[1]))

    # --- Interior Knots Continuity (C3 and C4) ---
    def interior_row(i, Ab):
        # i is the knot index, from 1 to n-2
        # Corresponds to interaction between interval i-1 and i
        A_curr, b_curr = Ab
        
        hL = h[i-1] # Left interval
        hR = h[i]   # Right interval
        
        # Indices in the flattened vector
        idx_L = 2 * (i - 1) # d_{i-1}
        idx_C = 2 * i       # d_i
        idx_R = 2 * (i + 1) # d_{i+1}
        
        # Continuity of 3rd derivative: S'_{i-1}'''(hL) = S'_i'''(0)
        # 24/hL^2 d_{i-1} + 3/hL s_{i-1} + (36/hL^2 + 36/hR^2) d_i + (9/hR - 9/hL) s_i + 24/hR^2 d_{i+1} - 3/hR s_{i+1}
        # = -60/hL^3 (yi - y_{i-1}) + 60/hR^3 (y_{i+1} - yi)
        
        r = 2 * i # Row index
        
        A_curr = A_curr.at[r, idx_L].set(24.0 / hL**2)      # d_{i-1}
        A_curr = A_curr.at[r, idx_L+1].set(3.0 / hL)        # s_{i-1}
        
        A_curr = A_curr.at[r, idx_C].set(36.0/hL**2 + 36.0/hR**2) # d_i
        A_curr = A_curr.at[r, idx_C+1].set(9.0/hR - 9.0/hL)       # s_i
        
        A_curr = A_curr.at[r, idx_R].set(24.0 / hR**2)      # d_{i+1}
        A_curr = A_curr.at[r, idx_R+1].set(-3.0 / hR)       # s_{i+1}
        
        dyL = y[i] - y[i-1]
        dyR = y[i+1] - y[i]
        rhs = -60.0/(hL**3) * dyL + 60.0/(hR**3) * dyR
        b_curr = b_curr.at[r].set(rhs)
        
        # Continuity of 4th derivative: S'_{i-1}''''(hL) = S'_i''''(0)
        # -168/hL^3 d_{i-1} - 24/hL^2 s_{i-1} - (192/hL^3 + 192/hR^3) d_i + (36/hR^2 - 36/hL^2) s_i - 168/hR^3 d_{i+1} + 24/hR^2 s_{i+1}
        # = 360/hL^4 (yi - y_{i-1}) - 360/hR^4 (y_{i+1} - yi)
        
        r = 2 * i + 1
        
        A_curr = A_curr.at[r, idx_L].set(-168.0 / hL**3)    # d_{i-1}
        A_curr = A_curr.at[r, idx_L+1].set(-24.0 / hL**2)   # s_{i-1}
        
        A_curr = A_curr.at[r, idx_C].set(-(192.0/hL**3 + 192.0/hR**3)) # d_i
        A_curr = A_curr.at[r, idx_C+1].set(36.0/hR**2 - 36.0/hL**2)    # s_i
        
        A_curr = A_curr.at[r, idx_R].set(-168.0 / hR**3)    # d_{i+1}
        A_curr = A_curr.at[r, idx_R+1].set(24.0 / hR**2)    # s_{i+1}
        
        rhs = 360.0/(hL**4) * dyL - 360.0/(hR**4) * dyR
        b_curr = b_curr.at[r].set(rhs)
        
        return A_curr, b_curr

    A, b = jax.lax.fori_loop(1, n-1, interior_row, (A, b))

    # --- Natural Boundary Conditions at x[n-1] ---
    # S'''(L) = 0
    # Eq: 24/hL^2 d_{n-2} + 3/hL s_{n-2} + 36/hL^2 d_{n-1} - 9/hL s_{n-1} = -60/hL^3 (y_{n-1} - y_{n-2})
    hL = h[n-2]
    row = 2 * n - 2
    idx_prev = 2 * (n - 2)
    idx_last = 2 * (n - 1)
    
    A = A.at[row, idx_prev].set(24.0 / hL**2)    # d_{n-2}
    A = A.at[row, idx_prev+1].set(3.0 / hL)      # s_{n-2}
    A = A.at[row, idx_last].set(36.0 / hL**2)    # d_{n-1}
    A = A.at[row, idx_last+1].set(-9.0 / hL)     # s_{n-1}
    b = b.at[row].set(-60.0 / hL**3 * (y[n-1] - y[n-2]))
    
    # S''''(L) = 0
    # Eq: -168/hL^3 d_{n-2} - 24/hL^2 s_{n-2} - 192/hL^3 d_{n-1} + 36/hL^2 s_{n-1} = 360/hL^4 (y_{n-1} - y_{n-2})
    row = 2 * n - 1
    A = A.at[row, idx_prev].set(-168.0 / hL**3)  # d_{n-2}
    A = A.at[row, idx_prev+1].set(-24.0 / hL**2) # s_{n-2}
    A = A.at[row, idx_last].set(-192.0 / hL**3)  # d_{n-1}
    A = A.at[row, idx_last+1].set(36.0 / hL**2)  # s_{n-1}
    b = b.at[row].set(360.0 / hL**4 * (y[n-1] - y[n-2]))

    # Solve system
    X = jnp.linalg.solve(A, b)
    
    # Extract first (D1) and second (D2) derivatives
    D1 = X[0::2]
    D2 = X[1::2]
    
    return D1, D2


@jax.jit
def jax_precompute_splines(grid, values):
    """
    Precompute first and second derivatives for quintic splines along each dimension.
    """
    nx, ny = values.shape

    # Compute derivatives for splines along x (for each y)
    # We need D1 and D2 for each row
    D1_x = jnp.zeros((nx, ny))
    D2_x = jnp.zeros((nx, ny))

    def compute_deriv_loop_x(i, val):
        D1_curr, D2_curr = val
        d1, d2 = jax_compute_derivatives(grid[0], values[:, i])
        D1_curr = D1_curr.at[:, i].set(d1)
        D2_curr = D2_curr.at[:, i].set(d2)
        return D1_curr, D2_curr
        
    D1_x, D2_x = jax.lax.fori_loop(0, ny, compute_deriv_loop_x, (D1_x, D2_x))

    # Compute derivatives for splines along y (for each x)
    D1_y = jnp.zeros((nx, ny))
    D2_y = jnp.zeros((nx, ny))

    def compute_deriv_loop_y(i, val):
        D1_curr, D2_curr = val
        d1, d2 = jax_compute_derivatives(grid[1], values[i, :])
        D1_curr = D1_curr.at[i, :].set(d1)
        D2_curr = D2_curr.at[i, :].set(d2)
        return D1_curr, D2_curr
        
    D1_y, D2_y = jax.lax.fori_loop(0, nx, compute_deriv_loop_y, (D1_y, D2_y))

    return D1_x, D2_x, D1_y, D2_y


@jax.jit
def quintic_spline_evaluate(xi, grid, values, D1_x, D2_x, D1_y, D2_y, fill_value=jnp.nan):
    """
    Evaluate interpolator at given points using separable quintic interpolation.
    
    Parameters
    ----------
    xi : array-like
        Points at which to interpolate. Shape (..., 2)
        Last dimension corresponds to (x, y) coordinates.
    grid : tuple
        (x_grid, y_grid) arrays
    values : array
        2D array of function values
    D1_x, D2_x : arrays
        First and second derivatives along x-axis
    D1_y, D2_y : arrays
        First and second derivatives along y-axis
        
    Returns
    -------
    result : array
        Interpolated values
    """
    
    original_shape = xi.shape[:-1]
    xi = xi.reshape(-1, 2)
    n_points = len(xi)
    
    x_pts = xi[:, 0]
    y_pts = xi[:, 1]
    
    # Check bounds
    out_of_bounds = jnp.zeros(n_points, dtype=bool)
    out_of_bounds |= (x_pts < grid[0][0]) | (x_pts > grid[0][-1])
    out_of_bounds |= (y_pts < grid[1][0]) | (y_pts > grid[1][-1])

    # Clamp to boundaries
    x_pts = jnp.clip(x_pts, grid[0][0], grid[0][-1])
    y_pts = jnp.clip(y_pts, grid[1][0], grid[1][-1])

    # Find x intervals
    i_x = jnp.searchsorted(grid[0], x_pts) - 1
    i_x = jnp.clip(i_x, 0, len(grid[0]) - 2)

    # Find y intervals
    i_y = jnp.searchsorted(grid[1], y_pts) - 1
    i_y = jnp.clip(i_y, 0, len(grid[1]) - 2)

    # Normalized coordinates
    h_x = grid[0][i_x + 1] - grid[0][i_x]
    t_x = (x_pts - grid[0][i_x]) / h_x

    h_y = grid[1][i_y + 1] - grid[1][i_y]
    t_y = (y_pts - grid[1][i_y]) / h_y
    
    # Quintic Hermite Basis Functions
    def basis_functions(t):
        # Value coeffs
        A0 = (1.0 - t)**3 * (6.0 * t**2 + 3.0 * t + 1.0)
        A1 = t**3 * (6.0 * (1.0 - t)**2 + 3.0 * (1.0 - t) + 1.0)
        # First deriv coeffs (scaled by h)
        B0 = t * (1.0 - t)**3 * (3.0 * t + 1.0)
        B1 = -t**3 * (1.0 - t) * (3.0 * (1.0 - t) + 1.0) # = -t^3(1-t)(4-3t) = -4t^3 + 7t^4 - 3t^5
        # Second deriv coeffs (scaled by h^2)
        C0 = 0.5 * t**2 * (1.0 - t)**3
        C1 = 0.5 * t**3 * (1.0 - t)**2
        return A0, A1, B0, B1, C0, C1

    Ax0, Ax1, Bx0, Bx1, Cx0, Cx1 = basis_functions(t_x)
    Ay0, Ay1, By0, By1, Cy0, Cy1 = basis_functions(t_y)

    # --- Interpolate along X at y_j and y_{j+1} ---
    # We need values at the four corners to form the Y-interval
    # Corners: (ix, iy), (ix+1, iy), (ix, iy+1), (ix+1, iy+1)
    
    # Values (Z) at corners
    Z00 = values[i_x, i_y]
    Z10 = values[i_x + 1, i_y]
    Z01 = values[i_x, i_y + 1]
    Z11 = values[i_x + 1, i_y + 1]
    
    # X-derivatives at corners
    D1x00 = D1_x[i_x, i_y]; D1x10 = D1_x[i_x + 1, i_y]
    D1x01 = D1_x[i_x, i_y + 1]; D1x11 = D1_x[i_x + 1, i_y + 1]
    
    D2x00 = D2_x[i_x, i_y]; D2x10 = D2_x[i_x + 1, i_y]
    D2x01 = D2_x[i_x, i_y + 1]; D2x11 = D2_x[i_x + 1, i_y + 1]
    
    # Quintic interp along X for bottom edge (y_j)
    val_x0 = (Ax0 * Z00 + Ax1 * Z10 + 
              Bx0 * D1x00 * h_x + Bx1 * D1x10 * h_x + 
              Cx0 * D2x00 * h_x**2 + Cx1 * D2x10 * h_x**2)

    # Quintic interp along X for top edge (y_{j+1})
    val_x1 = (Ax0 * Z01 + Ax1 * Z11 + 
              Bx0 * D1x01 * h_x + Bx1 * D1x11 * h_x + 
              Cx0 * D2x01 * h_x**2 + Cx1 * D2x11 * h_x**2)

    # --- Interpolate Y-derivatives along X ---
    # We need the 1st and 2nd derivatives w.r.t Y at the specific X location.
    # We have these values at the grid points. We linearly interpolate them 
    # to match the structural simplification of the CubicSpline script.
    
    # Y-Derivatives at corners (D1_y)
    D1y00 = D1_y[i_x, i_y]; D1y10 = D1_y[i_x + 1, i_y]
    D1y01 = D1_y[i_x, i_y + 1]; D1y11 = D1_y[i_x + 1, i_y + 1]
    
    # Linear interp along X
    D1y_x0 = (1.0 - t_x) * D1y00 + t_x * D1y10
    D1y_x1 = (1.0 - t_x) * D1y01 + t_x * D1y11
    
    # Y-Second Derivatives at corners (D2_y)
    D2y00 = D2_y[i_x, i_y]; D2y10 = D2_y[i_x + 1, i_y]
    D2y01 = D2_y[i_x, i_y + 1]; D2y11 = D2_y[i_x + 1, i_y + 1]
    
    # Linear interp along X
    D2y_x0 = (1.0 - t_x) * D2y00 + t_x * D2y10
    D2y_x1 = (1.0 - t_x) * D2y01 + t_x * D2y11
    
    # --- Final Interpolation along Y ---
    # Now we have Value, D1y, D2y at (x, y_j) and (x, y_{j+1}).
    # Use quintic formula in Y direction.
    
    result = (Ay0 * val_x0 + Ay1 * val_x1 + 
              By0 * D1y_x0 * h_y + By1 * D1y_x1 * h_y + 
              Cy0 * D2y_x0 * h_y**2 + Cy1 * D2y_x1 * h_y**2)

    result = jnp.where(out_of_bounds, fill_value, result)
    
    return result.reshape(original_shape)