"""
Multicomponent Species Diffusion Solver

Implements species transport with:
- Ordinary (Fickian) diffusion from concentration gradients
- Thermal diffusion (Soret effect) from temperature gradients
- Convective transport by bulk flow

Uses Stefan-Maxwell formulation for multicomponent diffusion with
temperature-dependent binary diffusion coefficients and thermal diffusion ratios.
"""

from model_code import *

@dataclass
class BinaryDiffusionCoeffs:
    """
    Coefficients for binary diffusion coefficient correlation.
    
    Temperature and pressure dependence: D_ij = (c0 + c1*T + c2*T²) / P
    
    Attributes:
        species_pair: Tuple of species names, e.g., ("N2", "SiH4")
        c0, c1, c2: Polynomial coefficients for T-dependence
    """
    species_pair: tuple  # e.g., ("N2", "SiH4")
    c0: float
    c1: float
    c2: float
    
    def __call__(self, T, P):
        """
        Calculate binary diffusion coefficient [m²/s].
        
        Args:
            T: Temperature (K)
            P: Pressure (Pa)
        
        Returns:
            D_ij: Binary diffusion coefficient (m²/s)
        """
        return (self.c0 + self.c1 * T + self.c2 * T**2) / P

@dataclass 
class ThermalDiffusionCoeffs:
    """
    Coefficients for thermal diffusion ratio (Soret effect).
    
    Form: k_ij = c0*x1*x2*(1 + c1*x1 + c2*x1² + c3*x1³)*(1 + c4*exp(c5*T))
    
    The thermal diffusion ratio k_ij represents the coupling between
    temperature gradients and species fluxes.
    
    Attributes:
        species_pair: Tuple where first species is "species 1" in formula
        c0-c5: Correlation coefficients from experimental data
    """
    species_pair: tuple  # e.g., ("N2", "SiH4") where N2 is species 1
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    
    def __call__(self, x1, x2, T):
        """
        Calculate thermal diffusion ratio (dimensionless).
        
        Args:
            x1: Mole fraction of species 1
            x2: Mole fraction of species 2
            T: Temperature (K)
        
        Returns:
            k_ij: Thermal diffusion ratio (dimensionless)
        """
        composition_term = 1 + self.c1 * x1 + self.c2 * x1**2 + self.c3 * x1**3
        temperature_term = 1 + self.c4 * np.exp(self.c5 * T)
        return self.c0 * x1 * x2 * composition_term * temperature_term

class DiffusionSolver():
    """
    Solves multicomponent species transport with thermal diffusion.
    
    Implements convection-diffusion equations for each species:
    ∂(ρω_i)/∂t + ∇·(ρvω_i) = ∇·(ρD'_i∇ω_i) + ∇·(ρD^T_i∇T/T)
    
    where:
    - D'_i is effective multicomponent diffusion coefficient
    - D^T_i is thermal diffusion coefficient
    """
    def __init__(self, grid, config):
        """
        Initialize diffusion solver with binary diffusion and thermal diffusion data.
        
        Args:
            grid: StaggeredGrid object
            config: SimulationConfig object
        """
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz
        self.R = 8.314  # J/(mol·K) - Universal gas constant
        self.masses = np.array(self.config.masses)
        
        # Binary diffusion coefficient correlations for each species pair
        # Coefficients from Kleijn et al. (1989)
        D_sih4_n2 = BinaryDiffusionCoeffs(
            species_pair=('sih4', 'n2'), c0=-9.64e-1, c1=6.25e-3, c2=8.50e-6
        )
        D_sih4_h2 = BinaryDiffusionCoeffs(
            species_pair=('sih4', 'h2'), c0=-2.90e0, c1=2.06e-2, c2=2.81e-5
        )
        D_n2_h2 = BinaryDiffusionCoeffs(
            species_pair=('n2', 'h2'), c0=-3.20e0, c1=2.44e-2, c2=3.37e-5
        )

        # Store in 3x3 matrix: D_binary[i][j] gives D_ij
        # Diagonal elements are None (D_ii not defined)
        self.D_binary = [
            [None, D_sih4_n2, D_n2_h2],     # i=0: N2
            [D_sih4_n2, None, D_sih4_h2],   # i=1: SiH4
            [D_n2_h2, D_sih4_h2, None]      # i=2: H2
        ]

        # Thermal diffusion ratio correlations
        # Note: k_ij = -k_ji (antisymmetric)
        k_n2_sih4 = ThermalDiffusionCoeffs(
            species_pair=('n2', 'sih4'), c0=-0.0515, c1=0.186, c2=0.0294, c3=0.00443, c4=-1.69, c5=-0.00494
        )
        k_h2_n2 = ThermalDiffusionCoeffs(
            species_pair=('h2', 'n2'), c0=-0.271, c1=0.727, c2=-0.357, c3=0.987, c4=-1.61, c5=-0.00915
        )
        k_h2_sih4 = ThermalDiffusionCoeffs(
            species_pair=('h2', 'sih4'), c0=-0.274, c1=1.01, c2=-1.04, c3=1.90, c4=-1.70, c5=-0.00635
        )
        
        self.k_thermal = [
            [None, k_n2_sih4, k_h2_n2],
            [k_n2_sih4, None, k_h2_sih4],
            [k_h2_n2, k_h2_sih4, None]
        ]

    def get_binary_diffusion(self, i, j, T, P):
        """
        Get binary diffusion coefficient D_ij at given conditions.
        
        Args:
            i: Species index (0=N2, 1=SiH4, 2=H2)
            j: Species index
            T: Temperature (K)
            P: Pressure (Pa)
        
        Returns:
            D_ij: Binary diffusion coefficient (m²/s)
        """
        if i == j:
            return 0.0
        coeff = self.D_binary[i][j]
        return coeff(T, P)
    
    def get_thermal_diffusion_ratio(self, i, j, x, T):
        """
        Get thermal diffusion ratio k_ij at given composition and temperature.
        
        Note: k_ji = -k_ij (antisymmetric property)
        
        Args:
            i: Species index
            j: Species index
            x: Mole fraction array [x_N2, x_SiH4, x_H2]
            T: Temperature (K)
        
        Returns:
            k_ij: Thermal diffusion ratio (dimensionless)
        """
        if i == j:
            return 0.0
        
        # Determine which is "species 1" based on stored coefficients
        if i < j:
            coeff = self.k_thermal[i][j]
            return coeff(x[i], x[j], T)
        else:
            # Use antisymmetry: k_ji = -k_ij
            coeff = self.k_thermal[j][i]
            return -coeff(x[j], x[i], T)
        
    def compute_effective_diffusion(self, i, T, P, omega, m_mix):
        """
        Compute effective multicomponent diffusion coefficient D'_i.
        
        Stefan-Maxwell formulation for multicomponent mixtures:
        D'_i = (1 - M_mix*ω_i/M_i) / [Σ_{j≠i} (M_mix*ω_j)/(M_j*D_ij)]
        
        This accounts for interactions between all species, not just
        binary diffusion between i and each j.
        
        Args:
            i: Species index
            T: Temperature (K)
            P: Pressure (Pa)
            omega: Mass fraction array [ω_N2, ω_SiH4, ω_H2]
            m_mix: Mean molecular mass (kg/mol)
        
        Returns:
            D'_i: Effective diffusion coefficient (m²/s)
        """
        mi = self.masses[i]
        
        # First term: (1 - M_mix*ω_i/M_i)
        term1 = 1.0 - m_mix * omega[i] / mi
        
        # Sum term: Σ_{j≠i} (M_mix*ω_j)/(M_j*D_ij)
        sum_term = 0.0
        for j in range(3):
            if j != i and omega[j] > 1e-10:
                mj = self.masses[j]
                Dij = self.get_binary_diffusion(i, j, T, P)
                if Dij > 1e-30:
                    sum_term += m_mix * omega[j] / (mj * Dij)
        
        if sum_term < 1e-30:
            return 0.0
        
        return term1 / sum_term
    
    def compute_thermal_diffusion_coeff(self, i, T, P, x, rho):
        """
        Compute multicomponent thermal diffusion coefficient D^T_i.
        
        Represents species flux due to temperature gradients (Soret effect):
        j_i^thermal = -ρ*D^T_i * ∇T/T
        
        Formula: D^T_i = Σ_{j≠i} (c²/ρ) * M_i * M_j * D_ij * k_ij
        where c = P/(RT) is total molar concentration
        
        Args:
            i: Species index
            T: Temperature (K)
            P: Pressure (Pa)
            x: Mole fraction array [x_N2, x_SiH4, x_H2]
            rho: Density (kg/m³)
        
        Returns:
            D^T_i: Thermal diffusion coefficient (kg/(m·s))
        """
        c = P / (self.R * T)  # mol/m³ - total molar concentration
        mi = self.masses[i]
        
        DiT = 0.0
        for j in range(3):
            if j != i:
                mj = self.masses[j]
                Dij = self.get_binary_diffusion(i, j, T, P)
                kij = self.get_thermal_diffusion_ratio(i, j, x, T)
                DiT += (c**2 / rho) * mi * mj * Dij * kij
        
        return DiT
    
    def mass_to_mole_fraction(self, omega):
        """
        Convert mass fractions to mole fractions.
        
        Formula: x_i = (ω_i/M_i) / Σ(ω_j/M_j)
        
        Args:
            omega: Mass fraction array or list [ω_N2, ω_SiH4, ω_H2]
        
        Returns:
            x: Mole fraction array [x_N2, x_SiH4, x_H2]
        """
        sum_term = sum(omega[k] / self.masses[k] for k in range(3))
        if sum_term < 1e-30:
            return [1.0, 0.0, 0.0]  # Default to pure N2
        return [omega[k] / self.masses[k] / sum_term for k in range(3)]
    
    def mole_to_mass_fraction(self, x):
        """
        Convert mole fractions to mass fractions.
        
        Formula: ω_i = x_i * M_i / Σ(x_j * M_j)
        
        Args:
            x: Mole fraction array
        
        Returns:
            omega: Mass fraction array
        """
        # Mean molecular mass: M_mix = Σ(x_j * M_j)
        m_mix = np.sum(x * np.array(self.masses), axis=-1, keepdims=True)
        # Mass fraction: ω_i = x_i * M_i / M_mix
        omega = x * self.masses / m_mix
        return omega

    def mean_mole_mass(self, x):
        """
        Calculate mixture molecular mass from mole fractions.
        
        Formula: M_mix = Σ(x_k * M_k)
        
        Args:
            x: Mole fraction array [x_N2, x_SiH4, x_H2]
        
        Returns:
            M_mix: Mean molecular mass (kg/mol)
        """
        return sum(x[k] * self.masses[k] for k in range(3))
    
    def solve_diffusion(self, species_idx, vr, vz, rho, T, omega, P):
        """
        Solve convection-diffusion equation for a single species.
        
        Discretizes and solves:
        ∂(ρω_i)/∂t + ∇·(ρvω_i) = ∇·(ρD'_i∇ω_i) + ∇·(ρD^T_i∇T/T)
        
        Uses:
        - Upwind differencing for convection
        - Central differencing for diffusion
        - Gauss-Seidel iteration with under-relaxation
        
        Args:
            species_idx: Index of species to solve (1=SiH4, 2=H2)
            vr: Radial velocity (nr+1, nz)
            vz: Axial velocity (nr, nz+1)
            rho: Density (nr, nz)
            T: Temperature (nr, nz)
            omega: Mass fractions (nr, nz, 3)
            P: Pressure (nr, nz)
        
        Returns:
            omega: Updated mass fractions with species_idx solved
        """
        nr, nz = self.nr, self.nz
        alpha_omega = 0.7  # Under-relaxation factor for stability
        
        # Gauss-Seidel iterations
        for outer_iter in range(20):
            omega_old = omega[:, :, species_idx].copy()
            
            # Sweep through interior cells
            for i in range(1, nr - 1):
                for j in range(1, nz - 1):
                    r = self.grid.r_centers[i]
                    if r < 1e-10:
                        continue  # Skip axis singularity
                    
                    # Grid spacings around cell P
                    dr_w = self.grid.dr[i-1] if i > 0 else self.grid.dr[0]
                    dr_e = self.grid.dr[i] if i < nr - 1 else self.grid.dr[-1]
                    dr_p = 0.5 * (dr_w + dr_e)
                    
                    dz_s = self.grid.dz[j-1] if j > 0 else self.grid.dz[0]
                    dz_n = self.grid.dz[j] if j < nz - 1 else self.grid.dz[-1]
                    dz_p = 0.5 * (dz_s + dz_n)
                    
                    # Face positions and areas (per radian)
                    r_e = r + 0.5 * dr_e
                    r_w = r - 0.5 * dr_w
                    A_e = r_e * dz_p
                    A_w = r_w * dz_p
                    A_n = r * dr_p
                    A_s = r * dr_p
                    
                    # Local properties at cell center
                    T_P = float(T[i, j])
                    P_P = float(P[i, j])
                    rho_P = float(rho[i, j])
                    omega_local = [float(omega[i, j, k]) for k in range(3)]
                    x_local = self.mass_to_mole_fraction(omega_local)
                    m_mix = self.mean_mole_mass(x_local)
                    
                    # Effective diffusion coefficient for species i
                    D_eff = self.compute_effective_diffusion(
                        species_idx, T_P, P_P, omega_local, m_mix
                    )
                    
                    # Thermal diffusion coefficient
                    D_T = self.compute_thermal_diffusion_coeff(
                        species_idx, T_P, P_P, x_local, rho_P
                    )
                    
                    # Mass fluxes at faces (convection)
                    m_dot_e = rho[i, j] * vr[i+1, j] * A_e
                    m_dot_w = rho[i, j] * vr[i, j] * A_w
                    m_dot_n = rho[i, j] * vz[i, j+1] * A_n
                    m_dot_s = rho[i, j] * vz[i, j] * A_s
                    
                    # Diffusion coefficients at faces
                    D_e = rho_P * D_eff * A_e / dr_e
                    D_w = rho_P * D_eff * A_w / dr_w
                    D_n = rho_P * D_eff * A_n / dz_n
                    D_s = rho_P * D_eff * A_s / dz_s
                    
                    # Neighbor coefficients (upwind for convection + central for diffusion)
                    a_E = D_e + max(-m_dot_e, 0)  # max(0, inflow)
                    a_W = D_w + max(m_dot_w, 0)
                    a_N = D_n + max(-m_dot_n, 0)
                    a_S = D_s + max(m_dot_s, 0)
                    a_P = a_E + a_W + a_N + a_S
                    
                    if a_P < 1e-30:
                        continue  # Skip degenerate cells
                    
                    # Neighbor values
                    omega_E = omega[i+1, j, species_idx]
                    omega_W = omega[i-1, j, species_idx]
                    omega_N = omega[i, j+1, species_idx]
                    omega_S = omega[i, j-1, species_idx]
                    
                    # Thermodiffusion source term: ∇·(ρD^T∇T/T)
                    T_E, T_W = T[i+1, j], T[i-1, j]
                    T_N, T_S = T[i, j+1], T[i, j-1]
                    T_e = 0.5 * (T_P + T_E)
                    T_w = 0.5 * (T_P + T_W)
                    T_n = 0.5 * (T_P + T_N)
                    T_s = 0.5 * (T_P + T_S)
                    
                    S_T = 0.0
                    if abs(D_T) > 1e-30:
                        # Radial thermodiffusion flux
                        S_T += D_T / T_e * (T_E - T_P) / dr_e * A_e
                        S_T -= D_T / T_w * (T_P - T_W) / dr_w * A_w
                        # Axial thermodiffusion flux
                        S_T += D_T / T_n * (T_N - T_P) / dz_n * A_n
                        S_T -= D_T / T_s * (T_P - T_S) / dz_s * A_s
                    
                    # Solve for new omega at cell P
                    omega_new = (a_E * omega_E + a_W * omega_W + 
                                 a_N * omega_N + a_S * omega_S + S_T) / a_P
                    
                    # Apply under-relaxation
                    omega[i, j, species_idx] = (alpha_omega * omega_new + 
                                               (1 - alpha_omega) * omega[i, j, species_idx])
            
            # Apply boundary conditions after each sweep
            omega = self.apply_boundary_conditions(omega, species_idx)
            
            # Check convergence of inner iterations
            change = np.max(np.abs(omega[:, :, species_idx] - omega_old))
            if change < 1e-6:
                break
        
        return omega
    
    def apply_boundary_conditions(self, omega, species_idx):
        """
        Apply boundary conditions for species transport.
        
        Conditions:
        - Inlet: Fixed composition (Dirichlet)
        - Walls: Zero flux (Neumann, zero gradient)
        - Axis: Symmetry (zero gradient)
        - Wafer: Will be overwritten by chemistry solver
        
        Args:
            omega: Mass fraction array (nr, nz, 3)
            species_idx: Species index being solved
        
        Returns:
            omega: Updated mass fractions with BCs applied
        """
        nr, nz = self.nr, self.nz
        inlet_mass_frac = self.mole_to_mass_fraction(self.config.inlet_composition)
        
        # Inlet (top, inside pipe): specified composition
        R_pipe = self.config.pipe_radius
        for i in range(nr):
            r = self.grid.r_centers[i]
            if r < R_pipe:
                omega[i, -1, species_idx] = inlet_mass_frac[species_idx]
        
        # Top wall (outside pipe): zero flux (use interior value)
        for i in range(nr):
            r = self.grid.r_centers[i]
            if r >= R_pipe:
                omega[i, -1, species_idx] = omega[i, -2, species_idx]
        
        # Symmetry axis (r=0): zero gradient
        omega[0, :, species_idx] = omega[1, :, species_idx]
        
        # Outer wall (r=r_max): zero flux
        omega[-1, :, species_idx] = omega[-2, :, species_idx]
        
        # Bottom - wafer and outlet: zero gradient for now
        # (Chemistry solver will overwrite wafer region with reaction flux BC)
        omega[:, 0, species_idx] = omega[:, 1, species_idx]
        
        return omega
    
    def solve(self, vr, vz, rho, T, omega, P):
        """
        Solve species transport equations for all species.
        
        Solves for SiH4 and H2 explicitly, then computes N2 from
        mass fraction constraint (Σω_i = 1).
        
        Args:
            vr: Radial velocity (nr+1, nz)
            vz: Axial velocity (nr, nz+1)
            rho: Density (nr, nz)
            T: Temperature (nr, nz)
            omega: Mass fractions (nr, nz, 3) - [N2, SiH4, H2]
            P: Pressure field (nr, nz)
        
        Returns:
            omega: Updated mass fractions
        """
        # Solve for SiH4 and H2 (minor species)
        for species_idx in [1, 2]:
            omega = self.solve_diffusion(species_idx, vr, vz, rho, T, omega, P)
        
        # Enforce constraint: ω_N2 = 1 - ω_SiH4 - ω_H2
        omega[:, :, 0] = 1.0 - omega[:, :, 1] - omega[:, :, 2]
        
        # Clamp to valid range [0,1]
        omega = np.clip(omega, 0.0, 1.0)
        
        # Renormalize to ensure sum = 1 exactly
        omega_sum = np.sum(omega, axis=2, keepdims=True)
        omega = omega / (omega_sum + 1e-30)
        
        return omega