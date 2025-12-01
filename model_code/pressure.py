from model_code import *

class PressureSolver:
    def __init__(self, grid, config):
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz

    def solve_pressure_correction(self, vr, vz, p, p_prime, rho, d_z, d_r):
        nr, nz = self.nr, self.nz
        p_prime[:] = 0.0
        
        if False:
            print(f"\n  d_r range: [{d_r.min():.2e}, {d_r.max():.2e}]")
            print(f"  d_z range: [{d_z.min():.2e}, {d_z.max():.2e}]")
            print(f"  d_r nonzero: {np.count_nonzero(d_r)} / {d_r.size}")
            print(f"  d_z nonzero: {np.count_nonzero(d_z)} / {d_z.size}")

        for outer_iter in range(200):
            p_prime_old = p_prime.copy()
            
            # ========== RADIAL SWEEPS ==========
            for j in range(1, nz - 1):
                n = nr - 2
                
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
                    
                    # Face areas (per radian, for axisymmetric)
                    A_e = r_e * dz  # East face (radial)
                    A_w = r_w * dz  # West face (radial)
                    A_n = r * dr    # North face (axial)
                    A_s = r * dr    # South face (axial)

                    # Calculate densities
                    rho_e = 0.5 * (rho[i, j] + rho[min(i+1, nr-1), j])
                    rho_w = 0.5 * (rho[max(i-1, 0), j] + rho[i, j])
                    rho_n = 0.5 * (rho[i, j] + rho[i, min(j+1, nz-1)])
                    rho_s = 0.5 * (rho[i, max(j-1, 0)] + rho[i, j])

                    # Standard coefficients
                    a_E = rho_e * d_r[i+1, j] * A_e if i + 1 < nr else 0
                    a_W = rho_w * d_r[i, j] * A_w if i > 0 else 0
                    a_N = rho_n * d_z[i, j+1] * A_n if j + 1 < nz else 0
                    a_S = rho_s * d_z[i, j] * A_s if j > 0 else 0
                    a_P = a_E + a_W + a_N + a_S

                    mass_imb = self._compute_mass_imbalance_at(i, j, vr, vz, rho)

                    # === Handle WEST boundary (i=0, the axis) ===
                    if idx == 0:
                        b[idx] = a_P - a_W
                        a[idx] = 0  # No West connection
                    else:
                        b[idx] = a_P
                        a[idx] = -a_W
                    
                    # === Handle EAST boundary (i=nr-1, outer wall) ===
                    if idx == n - 1:
                        b[idx] = a_P - a_E
                        c[idx] = 0  # No East connection
                    else:
                        c[idx] = -a_E
                    
                    # RHS includes North/South contributions (from previous sweep)
                    d[idx] = -mass_imb + a_N * p_prime[i, j+1] + a_S * p_prime[i, j-1]
                
                # Solve and store
                solution = self._tdma(a, b, c, d)
                for idx, i in enumerate(range(1, nr - 1)):
                    p_prime[i, j] = solution[idx]
            
            # ========== AXIAL SWEEPS ==========
            for i in range(1, nr - 1):
                n = nz - 2
                
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

                    # Standard coefficients
                    a_E = rho_e * d_r[i+1, j] * A_e if i + 1 < nr else 0
                    a_W = rho_w * d_r[i, j] * A_w if i > 0 else 0
                    a_N = rho_n * d_z[i, j+1] * A_n if j + 1 < nz else 0
                    a_S = rho_s * d_z[i, j] * A_s if j > 0 else 0
                    a_P = a_E + a_W + a_N + a_S

                    mass_imb = self._compute_mass_imbalance_at(i, j, vr, vz, rho)  

                    # === Handle SOUTH boundary (j=0, bottom/outlet) ===
                    if idx == 0:
                        b[idx] = a_P
                        a[idx] = 0  # No South connection in tridiag
                    else:
                        b[idx] = a_P
                        a[idx] = -a_S
                    
                    # === Handle NORTH boundary (j=nz-1, top/inlet) ===
                    if idx == n - 1:
                        b[idx] = a_P - a_N
                        c[idx] = 0  # No North connection
                    else:
                        c[idx] = -a_N
                    
                    # RHS includes East/West contributions (from radial sweep)
                    d[idx] = -mass_imb + a_E * p_prime[i+1, j] + a_W * p_prime[i-1, j]
                
                # Solve and store
                solution = self._tdma(a, b, c, d)
                for idx, j in enumerate(range(1, nz - 1)):
                    p_prime[i, j] = solution[idx]
            
                # ========== Apply BCs to p_prime array ==========
                # Axis (r=0): symmetry
                p_prime[0, :] = p_prime[1, :]
                
                # Outer wall (r=r_max): zero gradient
                p_prime[-1, :] = p_prime[-2, :]
                
                # Bottom (z=0): Dirichlet p'=0
                p_prime[:, 0] = 0.0
                
                # Top (z=z_max): zero gradient
                p_prime[:, -1] = p_prime[:, -2]
                
            # Check convergence
            change = np.max(np.abs(p_prime - p_prime_old))
            if change < 1e-6:
                break
        
        # Update pressure
        p += self.config.under_relaxation_p * p_prime
        return np.maximum(p, 0.1 * self.config.pressure_outlet)
    
    def _tdma(self, a, b, c, d):
        """
        Thomas Algorithm - solves tridiagonal system
        
        a: sub-diagonal (a[0] not used)
        b: main diagonal
        c: super-diagonal (c[-1] not used)
        d: right-hand side
        
        Returns: solution vector
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
    
    def _compute_mass_imbalance_at(self, i, j, vr, vz, rho):
        """Compute continuity residual at cell (i,j)"""
        r = self.grid.r_centers[i]
        r_e = self.grid.r_faces[i+1]
        r_w = self.grid.r_faces[i]
        dr, dz = self.grid.dr[i], self.grid.dz[j]
        
        # Calculate Mass Fluxes
        m_e = rho[i, j] * vr[i+1, j] * r_e * dz
        m_w = rho[i, j] * vr[i, j] * r_w * dz
        m_n = rho[i, j] * vz[i, j+1] * r * dr
        m_s = rho[i, j] * vz[i, j] * r * dr
        
        return m_e - m_w + m_n - m_s

    def solve(self, vr, vz, p, p_prime, rho, d_z, d_r):
        return self.solve_pressure_correction(vr, vz, p, p_prime, rho, d_z, d_r)