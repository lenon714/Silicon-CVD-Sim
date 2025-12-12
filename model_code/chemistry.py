"""
Surface Chemistry for LPCVD Silicon Deposition
Based on Kleijn et al. (1989)

Reaction: SiH4 -> Si(s) + 2H2
"""
import numpy as np

class ChemistrySolver:
    """
    Reactive boundary condition handler for wafer surface.
    
    Applies species flux BCs based on Langmuir-Hinshelwood surface reaction rate.
    The reaction consumes SiH4 and produces H2, creating a net mass flux (Stefan flow)
    normal to the wafer surface that affects the fluid dynamics.
    """
    def __init__(self, grid, config):
        """
        Initialize chemistry solver with kinetic parameters.
        
        Args:
            grid: StaggeredGrid object containing mesh information
            config: SimulationConfig object with reactor parameters
        """
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz

        # Molecular masses (kg/mol)
        self.M_SIH4 = 0.03212  # Silane
        self.M_H2 = 0.00202    # Hydrogen
        self.M_SI = 0.02809    # Silicon

        # Langmuir-Hinshelwood adsorption constants from Kleijn et al.
        self.K_H = 0.19    # Pa^(-1/2) - H2 adsorption equilibrium constant
        self.K_S = 0.70    # Pa^(-1) - SiH4 adsorption equilibrium constant


    def reaction_rate_constant(self, T):
        """
        Calculate temperature-dependent reaction rate constant.
        
        Arrhenius form: k(T) = k0 * exp(-Ea/(RT))
        From Kleijn et al.: k = 1.6e4 * exp(-18500/T)
        
        Args:
            T: Temperature (K), scalar or array
        
        Returns:
            k: Reaction rate constant (mol/(m²·s·Pa))
        """
        return 1.6e4 * np.exp(-18500.0 / T)


    def surface_reaction_rate(self, T, P_total, x_SiH4, x_H2):
        """
        Calculate silicon deposition rate using Langmuir-Hinshelwood kinetics.
        
        Reaction: SiH4(g) -> Si(s) + 2H2(g)
        
        Rate equation (Kleijn et al., Eq. 30):
        R = k(T) * P_SiH4 / [1 + K_H*sqrt(P_H2) + K_S*P_SiH4]
        
        The denominator accounts for competitive adsorption of H2 and SiH4
        on the silicon surface sites.
        
        Args:
            T: Surface temperature (K)
            P_total: Total pressure (Pa)
            x_SiH4: SiH4 mole fraction (dimensionless)
            x_H2: H2 mole fraction (dimensionless)
        
        Returns:
            R: Silicon deposition rate (mol/(m²·s))
        """
        # Calculate partial pressures from mole fractions
        P_SiH4 = x_SiH4 * P_total
        P_H2 = x_H2 * P_total
        
        # Temperature-dependent rate constant
        k = self.reaction_rate_constant(T)
        
        # Langmuir-Hinshelwood denominator (competitive adsorption)
        denom = 1.0 + self.K_H * np.sqrt(P_H2 + 1e-30) + self.K_S * P_SiH4
        
        # Surface reaction rate
        R = k * P_SiH4 / denom
        
        return R


    def surface_mass_fluxes(self, T, P_total, x_SiH4, x_H2):
        """
        Calculate species mass fluxes at wafer surface from reaction stoichiometry.
        
        Reaction: SiH4 -> Si(s) + 2H2
        Stoichiometry: 1 mol SiH4 consumed, 2 mol H2 produced per mol Si deposited
        
        Args:
            T: Surface temperature (K)
            P_total: Total pressure (Pa)
            x_SiH4: SiH4 mole fraction
            x_H2: H2 mole fraction
        
        Returns:
            j_SiH4: SiH4 mass flux (kg/(m²·s)), negative = consumption
            j_H2: H2 mass flux (kg/(m²·s)), positive = production
            R: Molar reaction rate (mol/(m²·s))
        """
        # Get molar reaction rate
        R = self.surface_reaction_rate(T, P_total, x_SiH4, x_H2)
        
        # Convert to mass fluxes using molecular masses and stoichiometry
        j_SiH4 = -self.M_SIH4 * R      # Consumed (negative flux into surface)
        j_H2 = 2.0 * self.M_H2 * R      # Produced (2 moles H2 per mole SiH4)
        
        return j_SiH4, j_H2, R


    def stefan_velocity(self, T, P_total, x_SiH4, x_H2, rho):
        """
        Calculate Stefan velocity due to net mass flux from surface reaction.
        
        The reaction consumes 1 mol SiH4 but produces 2 mol H2, creating a
        net outward mass flux. This induces a velocity normal to the surface
        that must be accounted for in the momentum boundary condition.
        
        Stefan velocity: v_s = -j_net / ρ
        where j_net = j_SiH4 + j_H2 (net mass flux from gas phase perspective)
        
        Args:
            T: Surface temperature (K)
            P_total: Total pressure (Pa)
            x_SiH4: SiH4 mole fraction
            x_H2: H2 mole fraction
            rho: Gas density at surface (kg/m³)
        
        Returns:
            v_stefan: Stefan velocity (m/s), positive = away from surface
        """
        # Get mass fluxes from reaction
        j_SiH4, j_H2, R = self.surface_mass_fluxes(T, P_total, x_SiH4, x_H2)
        
        # Net gas-phase mass flux (negative = net consumption, positive = net production)
        j_net = j_SiH4 + j_H2
        
        # Stefan velocity: negative j_net means flow toward surface
        v_stefan = -j_net / (rho + 1e-30)
        
        return v_stefan


    def deposition_rate(self, T, P_total, x_SiH4, x_H2):
        """
        Calculate silicon film growth rate in practical units.
        
        Converts molar reaction rate to:
        - Mass deposition rate
        - Thickness growth rate (nm/min)
        
        Args:
            T: Surface temperature (K)
            P_total: Total pressure (Pa)
            x_SiH4: SiH4 mole fraction
            x_H2: H2 mole fraction
        
        Returns:
            G: Molar deposition rate (mol/(m²·s))
            G_mass: Mass deposition rate (kg/(m²·s))
            G_thickness: Thickness growth rate (nm/min)
        """
        # Get molar reaction rate
        R = self.surface_reaction_rate(T, P_total, x_SiH4, x_H2)
        
        # Molar deposition rate (1:1 stoichiometry for Si)
        G = R  # mol/(m²·s)
        
        # Mass deposition rate
        G_mass = self.M_SI * R  # kg/(m²·s)
        
        # Thickness rate: convert mass flux to thickness growth
        # G_thickness = (mass flux) / (solid density) * unit conversions
        rho_Si = 2329.0  # kg/m³ (solid silicon density at room temp)
        G_thickness = (G_mass / rho_Si) * 1e9 * 60  # Convert m/s -> nm/min
        
        return G, G_mass, G_thickness
    
    def apply_species_bc(self, omega, x, T, P, rho, D_eff):
        """
        Apply species boundary conditions at wafer surface based on reaction fluxes.
        
        Implements flux boundary condition: ρD ∂ω/∂n = j_species
        Discretized as: ω_surface = ω_interior + j*dz/(ρ*D)
        
        Args:
            omega: Mass fraction array (nr, nz, 3) - [N2, SiH4, H2]
            x: Mole fraction array (nr, nz, 3) - [N2, SiH4, H2]
            T: Temperature array (nr, nz)
            P: Pressure array (nr, nz)
            rho: Density array (nr, nz)
            D_eff: Effective diffusion coefficients (nr, nz, 3)
        
        Returns:
            omega: Updated mass fractions with reaction BCs applied
            deposition_rates: Array of deposition rates across wafer (nr,)
        """
        R_wafer = self.config.wafer_radius
        dz = self.grid.dz[0]  # Grid spacing at bottom boundary
        
        deposition_rates = np.zeros(self.nr)
        
        for i in range(self.nr):
            r = self.grid.r_centers[i]
            
            if r < R_wafer:
                # This cell is on the wafer surface
                T_surf = T[i, 0]
                P_surf = P[i, 0]
                rho_surf = rho[i, 0]
                
                # Use first interior cell for mole fractions (ghost cell approach)
                x_SiH4 = x[i, 1, 1]  # Index: cell i, z=1, species=SiH4
                x_H2 = x[i, 1, 2]    # Index: cell i, z=1, species=H2
                
                # Calculate surface reaction rate and mass fluxes
                j_SiH4, j_H2, R = self.surface_mass_fluxes(T_surf, P_surf, x_SiH4, x_H2)
                deposition_rates[i] = R
                
                # Apply flux BC for SiH4: ω_surface = ω_interior + j*dz/(ρ*D)
                D_SiH4 = D_eff[i, 0, 1]
                if D_SiH4 > 1e-30:
                    omega[i, 0, 1] = omega[i, 1, 1] + j_SiH4 * dz / (rho_surf * D_SiH4)
                
                # Apply flux BC for H2
                D_H2 = D_eff[i, 0, 2]
                if D_H2 > 1e-30:
                    omega[i, 0, 2] = omega[i, 1, 2] + j_H2 * dz / (rho_surf * D_H2)
                
                # N2 from mass fraction constraint (sum = 1)
                omega[i, 0, 0] = 1.0 - omega[i, 0, 1] - omega[i, 0, 2]
        
        # Clamp to valid range [0,1]
        omega[:, 0, :] = np.clip(omega[:, 0, :], 0.0, 1.0)
        
        return omega, deposition_rates
    
    def get_stefan_velocity(self, x, T, P, rho):
        """
        Calculate Stefan velocity array across wafer surface for momentum BC.
        
        Args:
            x: Mole fraction array (nr, nz, 3)
            T: Temperature array (nr, nz)
            P: Pressure array (nr, nz)
            rho: Density array (nr, nz)
        
        Returns:
            v_stefan: Stefan velocity array (nr,) at wafer surface
        """
        R_wafer = self.config.wafer_radius
        v_stefan = np.zeros(self.nr)
        
        for i in range(self.nr):
            r = self.grid.r_centers[i]
            
            if r < R_wafer:
                # Extract surface conditions
                T_surf = T[i, 0]
                P_surf = P[i, 0]
                rho_surf = rho[i, 0]
                x_SiH4 = x[i, 0, 1]
                x_H2 = x[i, 0, 2]
                
                # Calculate Stefan velocity
                v_stefan[i] = self.stefan_velocity(T_surf, P_surf, x_SiH4, x_H2, rho_surf)
        
        return v_stefan
    
    def solve(self, omega, x, T, P, rho, D_eff):
        """
        Main solver interface: apply chemistry boundary conditions.
        
        Args:
            omega: Mass fraction array (nr, nz, 3)
            x: Mole fraction array (nr, nz, 3)
            T: Temperature array (nr, nz)
            P: Pressure array (nr, nz)
            rho: Density array (nr, nz)
            D_eff: Effective diffusion coefficients (nr, nz, 3)
        
        Returns:
            omega: Updated mass fractions with reaction BCs applied
        """
        return self.apply_species_bc(omega, x, T, P, rho, D_eff)