"""
Pressure Correction Solver for SIMPLE Algorithm

Solves the pressure correction equation derived from continuity:
∇·(ρd∇p') = -∇·(ρv*)

where v* is the velocity from momentum solver (doesn't satisfy continuity)
and p' is the correction needed to enforce mass conservation.

Uses Line TDMA (alternating direction method) for efficient solution
of the 2D elliptic equation.
"""

from model_code import *

class PressureSolver:
    """
    Solves pressure correction equation using Line TDMA.
    
    The pressure correction equation is derived by:
    1. Substituting velocity correction into continuity equation
    2. Using d-coefficients from momentum solver
    3. Setting p' = 0 at outlet as reference
    
    Solution method:
    - Alternating radial and axial sweeps (ADI-like)
    - Tridiagonal solves in each direction
    - Iterations until convergence
    """
    def __init__(self, grid, config):
        """
        Initialize pressure solver.
        
        Args:
            grid: StaggeredGrid object
            config: SimulationConfig object
        """
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz

    def solve_pressure_correction(self, vr, vz, p, p_prime, rho, d_z, d_r):
        """
        Solve pressure correction equation with Line TDMA.
        
        Pressure correction equation (per cell):
        a_P*p'_P = a_E*p'_E + a_W*p'_W + a_N*p'_N + a_S*p'_S + b
        
        where:
        - a_nb = ρ*d*A (from momentum d-coefficients)
        - b = -mass_imbalance (continuity residual)
        
        Uses alternating sweeps in r and z directions, solving
        tridiagonal systems along each line.
        
        Args:
            vr: Radial velocity (nr+1, nz)
            vz: Axial velocity (nr, nz+1)
            p: Absolute pressure (nr, nz) - updated at end
            p_prime: Pressure correction (nr, nz) - solved for
            rho: Density (nr, nz)
            d_z: Axial d-coefficient (nr, nz+1)
            d_r: Radial d-coefficient (nr+1, nz)
        
        Returns:
            p: Updated absolute pressure (p_new = p_old + α_p*p')
        """
        nr, nz = self.nr, self.nz
        p_prime[:] = 0.0  # Initialize correction to zero

        # Iterate until p' converges
        for outer_iter in range(100):
            p_prime_old = p_prime.copy()
            
            # ========== RADIAL SWEEPS (solve in r-direction) ==========
            for j in range(1, nz - 1):
                n = nr - 2  # Number of interior points in r
                
                # Tridiagonal matrix coefficients
                a = np.zeros(n)  # Sub-diagonal (West neighbor)
                b = np.zeros(n)  # Diagonal (Center)
                c = np.zeros(n)  # Super-diagonal (East neighbor)
                d = np.zeros(n)  # RHS
                
                for idx, i in enumerate(range(1, nr - 1)):  
                    # Grid geometry
                    r = self.grid.r_centers[i]
                    dr = self.grid.dr[i]
                    dz = self.grid.dz[j]
                    
                    # Face positions
                    r_e = self.grid.r_faces[i + 1]
                    r_w = self.grid.r_faces[i]
                    
                    # Face areas (per radian, axisymmetric)
                    A_e = r_e * dz  # East face (radial)
                    A_w = r_w * dz  # West face (radial)
                    A_n = r * dr    # North face (axial)
                    A_s = r * dr    # South face (axial)

                    # Calculate face-interpolated densities
                    rho_e = 0.5 * (rho[i, j] + rho[min(i+1, nr-1), j])
                    rho_w = 0.5 * (rho[max(i-1, 0), j] + rho[i, j])
                    rho_n = 0.5 * (rho[i, j] + rho[i, min(j+1, nz-1)])
                    rho_s = 0.5 * (rho[i, max(j-1, 0)] + rho[i, j])

                    # Pressure equation coefficients: a_nb = ρ*d*A
                    a_E = rho_e * d_r[i+1, j] * A_e if i + 1 < nr else 0
                    a_W = rho_w * d_r[i, j] * A_w if i > 0 else 0
                    a_N = rho_n * d_z[i, j+1] * A_n if j + 1 < nz else 0
                    a_S = rho_s * d_z[i, j] * A_s if j > 0 else 0
                    a_P = a_E + a_W + a_N + a_S

                    # RHS: negative of mass imbalance
                    mass_imb = self._compute_mass_imbalance(i, j, vr, vz, rho)

                    # === Handle WEST boundary (i=0, the axis) ===
                    if idx == 0:
                        b[idx] = a_P - a_W  # Symmetry: p'[0] = p'[1]
                        a[idx] = 0  # No West connection
                    else:
                        b[idx] = a_P
                        a[idx] = -a_W
                    
                    # === Handle EAST boundary (i=nr-1, outer wall) ===
                    if idx == n - 1:
                        b[idx] = a_P - a_E  # Zero gradient: p'[nr-1] = p'[nr-2]
                        c[idx] = 0  # No East connection
                    else:
                        c[idx] = -a_E
                    
                    # RHS includes North/South contributions (from previous sweep)
                    d[idx] = -mass_imb + a_N * p_prime[i, j+1] + a_S * p_prime[i, j-1]
                
                # Solve tridiagonal system for this row
                solution = self._tdma(a, b, c, d)
                for idx, i in enumerate(range(1, nr - 1)):
                    p_prime[i, j] = solution[idx]
            
            # ========== AXIAL SWEEPS (solve in z-direction) ==========
            for i in range(1, nr - 1):
                n = nz - 2  # Number of interior points in z
                
                # Tridiagonal matrix coefficients
                a = np.zeros(n)  # Sub-diagonal (South neighbor)
                b = np.zeros(n)  # Diagonal (Center)
                c = np.zeros(n)  # Super-diagonal (North neighbor)
                d = np.zeros(n)  # RHS
                
                for idx, j in enumerate(range(1, nz - 1)):   
                    # Grid geometry
                    r = self.grid.r_centers[i]
                    dr = self.grid.dr[i]
                    dz = self.grid.dz[j]
                    
                    # Face positions
                    r_e = self.grid.r_faces[i + 1]
                    r_w = self.grid.r_faces[i]
                    
                    # Face areas
                    A_e = r_e * dz
                    A_w = r_w * dz
                    A_n = r * dr
                    A_s = r * dr

                    # Calculate densities
                    rho_e = 0.5 * (rho[i, j] + rho[min(i+1, nr-1), j])
                    rho_w = 0.5 * (rho[max(i-1, 0), j] + rho[i, j])
                    rho_n = 0.5 * (rho[i, j] + rho[i, min(j+1, nz-1)])
                    rho_s = 0.5 * (rho[i, max(j-1, 0)] + rho[i, j])

                    # Pressure equation coefficients
                    a_E = rho_e * d_r[i+1, j] * A_e
                    a_W = rho_w * d_r[i, j] * A_w
                    a_N = rho_n * d_z[i, j+1] * A_n
                    a_S = rho_s * d_z[i, j] * A_s
                    a_P = a_E + a_W + a_N + a_S

                    mass_imb = self._compute_mass_imbalance(i, j, vr, vz, rho)  

                    # === Handle SOUTH boundary (j=0, bottom/outlet) ===
                    if idx == 0:
                        b[idx] = a_P  # p'[i,0] = 0 (reference point)
                        a[idx] = 0  # No South connection in tridiag
                    else:
                        b[idx] = a_P
                        a[idx] = -a_S
                    
                    # === Handle NORTH boundary (j=nz-1, top/inlet) ===
                    if idx == n - 1:
                        b[idx] = a_P - a_N  # Zero gradient
                        c[idx] = 0  # No North connection
                    else:
                        c[idx] = -a_N
                    
                    # RHS includes East/West contributions (from radial sweep)
                    d[idx] = -mass_imb + a_E * p_prime[i+1, j] + a_W * p_prime[i-1, j]
                
                # Solve tridiagonal system for this column
                solution = self._tdma(a, b, c, d)
                for idx, j in enumerate(range(1, nz - 1)):
                    p_prime[i, j] = solution[idx]
            
            # ========== Apply BCs to p_prime array ==========
            # Axis (r=0): symmetry
            p_prime[0, :] = p_prime[1, :]
            
            # Outer wall (r=r_max): zero gradient
            p_prime[-1, :] = p_prime[-2, :]
            
            # Bottom (z=0): Dirichlet p'=0 (reference point for pressure)
            p_prime[:, 0] = 0.0
            
            # Top (z=z_max): zero gradient
            p_prime[:, -1] = p_prime[:, -2]
                
            # Check convergence of p' field
            change = np.max(np.abs(p_prime - p_prime_old))
            if change < 1e-6:
                break
        
        # Update absolute pressure: p_new = p_old + α_p * p'
        p += self.config.under_relaxation_p * p_prime
        
        # Prevent negative pressures (physical constraint)
        return np.maximum(p, 0.1 * self.config.pressure_outlet)
    
    def _tdma(self, a, b, c, d):
        """
        Thomas Algorithm - solves tridiagonal system Ax = d.
        
        Matrix structure:
        [ b0  c0   0   ...  0  ] [x0]   [d0]
        [ a1  b1  c1   ...  0  ] [x1]   [d1]
        [  0  a2  b2   ...  0  ] [x2] = [d2]
        [ ...              ... ] [..]   [..]
        [  0   0   0   an  bn  ] [xn]   [dn]
        
        Args:
            a: sub-diagonal (a[0] not used)
            b: main diagonal
            c: super-diagonal (c[-1] not used)
            d: right-hand side
        
        Returns:
            x: solution vector
        """
        n = len(d)
        
        # Forward elimination
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
        
        # Back substitution
        x = np.zeros(n)
        x[-1] = d_prime[-1]
        
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
        return x
    
    def _compute_mass_imbalance(self, i, j, vr, vz, rho):
        """
        Compute mass conservation residual at cell (i,j).
        
        Continuity equation (axisymmetric):
        ∂(ρu_r)/∂r + ρu_r/r + ∂(ρu_z)/∂z = 0
        
        Integrated over cell:
        m_e - m_w + m_n - m_s = 0 (should be zero if converged)
        
        Args:
            i, j: Cell indices
            vr: Radial velocity
            vz: Axial velocity
            rho: Density
        
        Returns:
            mass_imb: Mass imbalance for this cell (kg/(s·rad))
        """
        nr, nz = self.nr, self.nz
        
        r = self.grid.r_centers[i]
        r_e = self.grid.r_faces[i+1]
        r_w = self.grid.r_faces[i]
        dr, dz = self.grid.dr[i], self.grid.dz[j]
        
        # Face-interpolated densities
        rho_e = 0.5 * (rho[i, j] + rho[min(i+1, nr-1), j])
        rho_w = 0.5 * (rho[max(i-1, 0), j] + rho[i, j])
        rho_n = 0.5 * (rho[i, j] + rho[i, min(j+1, nz-1)])
        rho_s = 0.5 * (rho[i, max(j-1, 0)] + rho[i, j])

        # Calculate mass fluxes through each face (per radian)
        m_e = rho_e * vr[i+1, j] * r_e * dz
        m_w = rho_w * vr[i, j] * r_w * dz
        m_n = rho_n * vz[i, j+1] * r * dr
        m_s = rho_s * vz[i, j] * r * dr
        
        # Mass imbalance: outflow - inflow
        return m_e - m_w + m_n - m_s

    def solve(self, vr, vz, p, p_prime, rho, d_z, d_r):
        """
        Main solver interface for pressure correction.
        
        Args:
            vr: Radial velocity
            vz: Axial velocity
            p: Pressure
            p_prime: Pressure correction
            rho: Density
            d_z, d_r: d-coefficients from momentum solver
        
        Returns:
            p: Updated pressure
        """
        return self.solve_pressure_correction(vr, vz, p, p_prime, rho, d_z, d_r)