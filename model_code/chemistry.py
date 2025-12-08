"""
Surface Chemistry for LPCVD Silicon Deposition
Based on Kleijn et al. (1989) Equation [30]

Reaction: SiH4 -> Si(s) + 2H2
"""
import numpy as np

class ChemistrySolver:
    """
    Reactive boundary condition handler for wafer surface.
    
    Applies species flux BCs based on surface reaction rate.
    """
    def __init__(self, grid, config):
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz

        self.M_SIH4 = 0.03212
        self.M_H2 = 0.00202
        self.M_SI = 0.02809 

        # Reaction constants from the paper
        self.K_H = 0.19    # Pa^(-1/2) - H2 adsorption constant
        self.K_S = 0.70    # Pa^(-1) - SiH4 adsorption constant


    def reaction_rate_constant(self, T):
        return 1.6e4 * np.exp(-18500.0 / T)


    def surface_reaction_rate(self, T, P_total, x_SiH4, x_H2):
        # Partial pressures
        P_SiH4 = x_SiH4 * P_total
        P_H2 = x_H2 * P_total
        
        # Rate constant
        k = self.reaction_rate_constant(T)
        
        # Denominator (adsorption terms)
        denom = 1.0 + self.K_H * np.sqrt(P_H2 + 1e-30) + self.K_S * P_SiH4
        
        # Reaction rate
        R = k * P_SiH4 / denom
        
        return R


    def surface_mass_fluxes(self, T, P_total, x_SiH4, x_H2):
        R = self.surface_reaction_rate(T, P_total, x_SiH4, x_H2)
        
        # Mass fluxes from stoichiometry
        j_SiH4 = -self.M_SIH4 * R  # Consumed (negative flux into surface)
        j_H2 = 2.0 * self.M_H2 * R  # Produced (2 moles H2 per mole SiH4)
        
        return j_SiH4, j_H2, R


    def stefan_velocity(self, T, P_total, x_SiH4, x_H2, rho):
        j_SiH4, j_H2, R = self.surface_mass_fluxes(T, P_total, x_SiH4, x_H2)
        
        j_net = j_SiH4 + j_H2  # Net gas-phase mass flux
        
        # Stefan velocity: v = -j_net / rho (negative j_net means flow toward surface)
        v_stefan = -j_net / (rho + 1e-30)
        
        return v_stefan


    def deposition_rate(self, T, P_total, x_SiH4, x_H2):
        R = self.surface_reaction_rate(T, P_total, x_SiH4, x_H2)
        
        G = R  # mol/(m^2·s) - 1:1 stoichiometry for Si
        G_mass = self.M_SI * R  # kg/(m^2·s)
        
        # Thickness rate: G_mass / rho_Si, converted to nm/min
        rho_Si = 2329.0  # kg/m^3 (solid silicon density)
        G_thickness = (G_mass / rho_Si) * 1e9 * 60  # nm/min
        
        return G, G_mass, G_thickness
    
    def apply_species_bc(self, omega, x, T, P, rho, D_eff):
        R_wafer = self.config.wafer_radius
        dz = self.grid.dz[0]  # Grid spacing at bottom
        
        deposition_rates = np.zeros(self.nr)
        
        for i in range(self.nr):
            r = self.grid.r_centers[i]
            
            if r < R_wafer:
                # This cell is on the wafer
                T_surf = T[i, 0]
                P_surf = P[i, 0]
                rho_surf = rho[i, 0]
                
                # Surface mole fractions (use first interior cell as approximation)
                x_SiH4 = x[i, 1, 1]  # Index: cell i, z=1, species=SiH4
                x_H2 = x[i, 1, 2]    # Index: cell i, z=1, species=H2
                
                # Calculate surface reaction rate and fluxes
                j_SiH4, j_H2, R = self.surface_mass_fluxes(T_surf, P_surf, x_SiH4, x_H2)
                deposition_rates[i] = R
                
                # Apply flux BC for SiH4
                D_SiH4 = D_eff[i, 0, 1]
                if D_SiH4 > 1e-30:
                    # ω[i,0] = ω[i,1] + j * dz / (rho * D')
                    omega[i, 0, 1] = omega[i, 1, 1] + j_SiH4 * dz / (rho_surf * D_SiH4)
                
                # Apply flux BC for H2
                D_H2 = D_eff[i, 0, 2]
                if D_H2 > 1e-30:
                    omega[i, 0, 2] = omega[i, 1, 2] + j_H2 * dz / (rho_surf * D_H2)
                
                # N2 from constraint
                omega[i, 0, 0] = 1.0 - omega[i, 0, 1] - omega[i, 0, 2]
        
        # Clamp to valid range
        omega[:, 0, :] = np.clip(omega[:, 0, :], 0.0, 1.0)
        
        return omega, deposition_rates
    
    def get_stefan_velocity(self, x, T, P, rho):
        R_wafer = self.config.wafer_radius
        v_stefan = np.zeros(self.nr)
        
        for i in range(self.nr):
            r = self.grid.r_centers[i]
            
            if r < R_wafer:
                T_surf = T[i, 0]
                P_surf = P[i, 0]
                rho_surf = rho[i, 0]
                x_SiH4 = x[i, 0, 1]
                x_H2 = x[i, 0, 2]
                
                v_stefan[i] = self.stefan_velocity(T_surf, P_surf, x_SiH4, x_H2, rho_surf)
        
        return v_stefan
    
    def solve(self, omega, x, T, P, rho, D_eff):
        return self.apply_species_bc(omega, x, T, P, rho, D_eff)