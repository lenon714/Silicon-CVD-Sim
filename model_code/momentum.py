"""
Momentum Solver using SIMPLE-style discretization
Implements the equations from Kleijn et al. (1989) in cylindrical coordinates

Solves steady-state Navier-Stokes equations for incompressible flow
in axisymmetric cylindrical geometry with variable density.
"""

from model_code import *

class MomentumSolver:
    """
    Solves the momentum equations:
    
    Radial:  a_P v_r = Σ a_nb v_r,nb + (P_W - P_E) A_r + b
    Axial:   a_P v_z = Σ a_nb v_z,nb + (P_S - P_N) A_z + b
    
    Using:
    - Upwind differencing for convection terms
    - Central differencing for diffusion terms
    - Source terms for cylindrical coordinate effects
    - Under-relaxation for stability
    
    Velocities stored on staggered grid:
    - v_r at radial faces: between cells (i-1,j) and (i,j)
    - v_z at axial faces: between cells (i,j-1) and (i,j)
    """
    
    def __init__(self, grid, config):
        """
        Initialize momentum solver.
        
        Args:
            grid: StaggeredGrid object containing mesh information
            config: SimulationConfig object with solver parameters
        """
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz
        
        # Store d coefficients for pressure correction equation
        # d = A/a_P, where a_P is diagonal coefficient from momentum equation
        self.d_r = np.zeros((self.nr + 1, self.nz))
        self.d_z = np.zeros((self.nr, self.nz + 1))
        
    def solve_radial_momentum(self, vr, vz, p, rho, mu):
        """
        Solve radial momentum equation in cylindrical coordinates.
        
        Equation (steady, axisymmetric):
        ∂(ρu_r)/∂t + 1/r * ∂(rρu_ru_r)/∂r + ∂(ρu_zu_r)/∂z = 
            -∂p/∂r + 1/r * ∂/∂r[rμ(2∂u_r/∂r)] + ∂/∂z[μ(∂u_r/∂z)] + 2μu_r/r²
        
        Discretized on control volumes around v_r faces.
        
        Args:
            vr: Radial velocity array (nr+1, nz) on radial faces
            vz: Axial velocity array (nr, nz+1) on axial faces
            p: Pressure array (nr, nz) at cell centers
            rho: Density array (nr, nz) at cell centers
            mu: Dynamic viscosity array (nr, nz) at cell centers
        
        Returns:
            vr_new: Updated radial velocity with under-relaxation
        """
        nr, nz = self.nr, self.nz
        vr_new = vr.copy()
        alpha = self.config.under_relaxation_v  # Under-relaxation factor
        
        # Loop over interior radial faces (not at axis i=0 or outer wall i=nr)
        for i in range(1, nr):
            for j in range(1, nz - 1):
                
                r = self.grid.r_faces[i]  # Radius of this face
                if r < 1e-10:
                    continue  # Skip axis singularity
                
                # === GRID GEOMETRY ===
                # v_r[i,j] sits between pressure cells (i-1,j) and (i,j)
                dr_w = self.grid.dr[i-1] if i > 0 else self.grid.dr[0]
                dr_e = self.grid.dr[i] if i < nr - 1 else self.grid.dr[-1]
                dr_p = 0.5 * (dr_w + dr_e)  # CV size in radial direction
                
                dz_s = self.grid.dz[j-1] if j > 0 else self.grid.dz[0]
                dz_n = self.grid.dz[j] if j < nz - 1 else self.grid.dz[-1]
                dz_p = 0.5 * (dz_s + dz_n)  # CV size in axial direction
                
                # Face areas for momentum CV (per-radian formulation)
                r_e = r + 0.5 * dr_e
                r_w = r - 0.5 * dr_w
                A_e = r_e * dz_p
                A_w = r_w * dz_p
                A_n = r * dr_p
                A_s = r * dr_p
                
                # Control volume (per radian in axisymmetric)
                dV = r * dr_p * dz_p
                
                # === DENSITY AT CV CENTER ===
                # Average of adjacent pressure cells
                i_w = max(i - 1, 0)
                i_e = min(i, nr - 1)
                rho_p = 0.5 * (rho[i_w, j] + rho[i_e, j])
                
                # === MASS FLUXES AT CV FACES ===
                # East face (between v_r[i] and v_r[i+1])
                vr_e = 0.5 * (vr[i, j] + vr[min(i+1, nr), j]) if i < nr else vr[i, j]
                m_dot_e = rho_p * vr_e * A_e
                
                # West face (between v_r[i-1] and v_r[i])
                vr_w = 0.5 * (vr[max(i-1, 0), j] + vr[i, j]) if i > 0 else vr[i, j]
                m_dot_w = rho_p * vr_w * A_w
                
                # North face (interpolate v_z from adjacent cells)
                vz_n = 0.5 * (vz[i_w, j+1] + vz[i_e, j+1])
                m_dot_n = rho_p * vz_n * A_n
                
                # South face
                vz_s = 0.5 * (vz[i_w, j] + vz[i_e, j])
                m_dot_s = rho_p * vz_s * A_s
                
                # === VISCOSITIES ===
                mu_e = mu[i_e, j]
                mu_w = mu[i_w, j]
                mu_p = 0.5 * (mu_w + mu_e)

                # North: interpolate between adjacent cells
                j_n = min(j + 1, nz - 1)
                mu_n = 0.5 * (mu[i_w, j_n] + mu[i_e, j_n])
                mu_n = 0.5 * (mu_p + mu_n)

                # South
                j_s = max(j - 1, 0)
                mu_s = 0.5 * (mu[i_w, j_s] + mu[i_e, j_s])
                mu_s = 0.5 * (mu_p + mu_s)

                # === DIFFUSION COEFFICIENTS ===
                # D = μ*A/Δ for diffusion terms
                # Factor of 2 for radial direction (cylindrical coordinates)
                D_e = 2 * mu_e * A_e / dr_e
                D_w = 2 * mu_w * A_w / dr_w
                D_n = mu_n * A_n / dz_n
                D_s = mu_s * A_s / dz_s
                
                # === NEIGHBOR COEFFICIENTS ===
                # Upwind scheme: max(0, inflow) for stability
                a_E = D_e + max(-m_dot_e, 0)
                a_W = D_w + max(m_dot_w, 0)
                a_N = D_n + max(-m_dot_n, 0)
                a_S = D_s + max(m_dot_s, 0)
                
                # Center coefficient (diagonal term)
                a_P0 = a_E + a_W + a_N + a_S
                # Note: mass flux imbalance term commented out (can cause instability)
                # + (m_dot_e - m_dot_w + m_dot_n - m_dot_s) 
                
                # Ensure positive diagonal for stability
                a_P0 = max(a_P0, 1e-10)
                
                # === NEIGHBOR VELOCITIES ===
                vr_E = vr[min(i+1, nr), j]
                vr_W = vr[max(i-1, 0), j]
                vr_N = vr[i, min(j+1, nz-1)]
                vr_S = vr[i, max(j-1, 0)]
                
                # === PRESSURE GRADIENT ===
                # Pressure at west (i-1,j) and east (i,j) of this face
                P_W = p[i_w, j]
                P_E = p[i_e, j]
                A_pressure = r * dz_p  # Area for pressure force
                
                # === SOURCE TERM ===
                # Cylindrical coordinate source: 2μ*v_r/r²
                b = 2 * mu[i, j] * dV / (r * r)  # Coefficient of v_r in source
                
                # === SOLVE FOR v_r ===
                # Under-relaxation: a_P = a_P0/α
                a_P = (a_P0 + b) / alpha
                
                # Add deferred correction for under-relaxation
                source_old = ((1 - alpha) / alpha) * a_P0 * vr[i, j]

                # Right-hand side
                numerator = (a_E * vr_E + a_W * vr_W + a_N * vr_N + a_S * vr_S 
                            + (P_W - P_E) * A_pressure + source_old)
                
                vr_new[i, j] = numerator / a_P

                # Store d coefficient for pressure correction
                # d = A/a_P (relates velocity correction to pressure correction)
                self.d_r[i, j] = A_pressure / a_P
        
        return vr_new
    
    def solve_axial_momentum(self, vr, vz, p, rho, mu):
        """
        Solve axial momentum equation in cylindrical coordinates.
        
        Equation (steady, axisymmetric):
        ∂(ρu_z)/∂t + 1/r * ∂(rρu_ru_z)/∂r + ∂(ρu_zu_z)/∂z = 
            -∂p/∂z + 1/r * ∂/∂r[rμ(∂u_z/∂r)] + ∂/∂z[μ(2∂u_z/∂z)] + ρg
        
        Discretized on control volumes around v_z faces.
        
        Args:
            vr: Radial velocity array (nr+1, nz)
            vz: Axial velocity array (nr, nz+1) on axial faces
            p: Pressure array (nr, nz)
            rho: Density array (nr, nz)
            mu: Dynamic viscosity array (nr, nz)
        
        Returns:
            vz_new: Updated axial velocity with under-relaxation
        """
        nr, nz = self.nr, self.nz
        g = self.config.gravity
        vz_new = vz.copy()
        alpha = self.config.under_relaxation_v
        
        # Loop over interior axial faces
        for i in range(1, nr - 1):
            for j in range(1, nz):  # Axial faces (not at bottom j=0)
                
                r = self.grid.r_centers[i]
                
                # === GRID GEOMETRY ===
                dr_w = self.grid.dr[i-1] if i > 0 else self.grid.dr[0]
                dr_e = self.grid.dr[i] if i < nr - 1 else self.grid.dr[-1]
                dr_p = 0.5 * (dr_w + dr_e)
                
                dz_s = self.grid.dz[j-1] if j > 0 else self.grid.dz[0]
                dz_n = self.grid.dz[j] if j < nz else self.grid.dz[-1]
                dz_p = 0.5 * (dz_s + dz_n)
                
                # Face areas (per-radian)
                r_e = r + 0.5 * dr_e
                r_w = r - 0.5 * dr_w
                A_e = r_e * dz_p
                A_w = r_w * dz_p
                A_n = r * dr_p
                A_s = r * dr_p
                
                # Cell volume (per radian)
                dV = r * dr_p * dz_p
                
                # === DENSITY ===
                # Average of adjacent pressure cells (south and north)
                j_s = max(j - 1, 0)
                j_n = min(j, nz - 1)
                rho_p = 0.5 * (rho[i, j_s] + rho[i, j_n])
                
                # === MASS FLUXES ===
                # East face (interpolate v_r)
                vr_e = 0.5 * (vr[i+1, j_s] + vr[i+1, j_n]) if j_n < nz else vr[i+1, j_s]
                m_dot_e = rho_p * vr_e * A_e
                
                # West face
                vr_w = 0.5 * (vr[i, j_s] + vr[i, j_n]) if j_n < nz else vr[i, j_s]
                m_dot_w = rho_p * vr_w * A_w
                
                # North face (between v_z[j] and v_z[j+1])
                vz_n = 0.5 * (vz[i, j] + vz[i, min(j+1, nz)]) if j < nz else vz[i, j]
                m_dot_n = rho_p * vz_n * A_n
                
                # South face (between v_z[j-1] and v_z[j])
                vz_s = 0.5 * (vz[i, max(j-1, 0)] + vz[i, j]) if j > 0 else vz[i, j]
                m_dot_s = rho_p * vz_s * A_s
                
                # === VISCOSITY ===
                mu_n = mu[i, j_n]
                mu_s = mu[i, j_s]
                mu_p = 0.5 * (mu_s + mu_n)
            
                # East face viscosity
                i_e = min(i + 1, nr - 1)
                mu_e = 0.5 * (mu[i_e, j_s] + mu[i_e, j_n])
                mu_e = 0.5 * (mu_p + mu_e)
                
                # West face viscosity
                i_w = max(i - 1, 0)
                mu_w = 0.5 * (mu[i_w, j_s] + mu[i_w, j_n])
                mu_w = 0.5 * (mu_p + mu_w)               

                # === DIFFUSION COEFFICIENTS ===
                # Factor of 2 for axial direction (cylindrical coordinates)
                D_e = mu_e * A_e / dr_e
                D_w = mu_w * A_w / dr_w
                D_n = 2 * mu_n * A_n / dz_n
                D_s = 2 * mu_s * A_s / dz_s
                
                # === NEIGHBOR COEFFICIENTS ===
                a_E = D_e + max(-m_dot_e, 0)
                a_W = D_w + max(m_dot_w, 0)
                a_N = D_n + max(-m_dot_n, 0)
                a_S = D_s + max(m_dot_s, 0)
                
                # Center coefficient
                a_P0 = a_E + a_W + a_N + a_S 
                # + (m_dot_e - m_dot_w + m_dot_n - m_dot_s)
                
                a_P0 = max(a_P0, 1e-10)
                
                # === NEIGHBOR VELOCITIES ===
                vz_E = vz[min(i+1, nr-1), j]
                vz_W = vz[max(i-1, 0), j]
                vz_N = vz[i, min(j+1, nz)]
                vz_S = vz[i, max(j-1, 0)]
                
                # === PRESSURE GRADIENT ===
                P_S = p[i, j_s]
                P_N = p[i, j_n]
                A_pressure = r * dr_p
                
                # === SOURCE TERM ===
                # Gravity acts in -z direction (if z points up)
                b = -rho_p * g * dV
                
                # === SOLVE FOR v_z ===
                # Under-relaxation
                a_P = a_P0 / alpha
                source_old = ((1 - alpha) / alpha) * a_P0 * vz[i, j]
                
                numerator = (a_E * vz_E + a_W * vz_W + a_N * vz_N + a_S * vz_S
                            + (P_S - P_N) * A_pressure + b + source_old)
                
                vz_new[i, j] = numerator / a_P
                
                # Store d coefficient for pressure correction
                self.d_z[i, j] = A_pressure / a_P
        
        return vz_new
    
    def solve(self, vr, vz, p, rho, mu):
        """
        Solve both momentum equations and return updated velocities.
        
        Solves radial momentum first, then uses updated vr in axial momentum.
        This sequential approach helps stability.
        
        Args:
            vr: Radial velocity (nr+1, nz)
            vz: Axial velocity (nr, nz+1)
            p: Pressure (nr, nz)
            rho: Density (nr, nz)
            mu: Dynamic viscosity (nr, nz)
        
        Returns:
            vr_new, vz_new: Updated velocity fields
        """
        vr_new = self.solve_radial_momentum(vr, vz, p, rho, mu)
        vz_new = self.solve_axial_momentum(vr_new, vz, p, rho, mu)
        
        return vr_new, vz_new
    
    def get_d_coefficients(self):
        """
        Return the d coefficients for pressure correction equation.
        
        These relate velocity corrections to pressure correction gradients:
        Δv = d * ∇p'
        
        Returns:
            d_r, d_z: d coefficient arrays for radial and axial directions
        """
        return self.d_r, self.d_z