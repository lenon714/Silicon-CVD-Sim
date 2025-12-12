"""
Mixture Property Calculations

Computes temperature-dependent properties of gas mixtures:
- Density from ideal gas law
- Viscosity using Wilke's mixing rule
- Thermal conductivity using weighted average
- Specific heat using mass-weighted average
- Multicomponent diffusion coefficients
"""

from model_code import *

@dataclass
class SpeciesProperties:
    """
    Temperature-dependent pure species properties.
    
    All properties follow polynomial form: property = c0 + c1*T + c2*T²
    Coefficients from Kleijn et al. (1989).
    
    Attributes:
        name: Species name
        mass: Molecular mass (kg/mol)
        mu_c0, mu_c1, mu_c2: Viscosity coefficients (Pa·s)
        lambda_c0, lambda_c1, lambda_c2: Thermal conductivity coefficients (W/(m·K))
        cp_c0, cp_c1, cp_c2: Specific heat coefficients (J/(kg·K))
    """
    name: str
    mass: float  # kg/mol
    # Viscosity coefficients (Pa·s)
    mu_c0: float
    mu_c1: float
    mu_c2: float
    # Thermal conductivity coefficients (W/(m·K))
    lambda_c0: float
    lambda_c1: float
    lambda_c2: float
    # Specific heat coefficients (J/(kg·K))
    cp_c0: float
    cp_c1: float
    cp_c2: float
    
    def viscosity(self, T):
        """
        Calculate dynamic viscosity at temperature T.
        
        Args:
            T: Temperature (K), scalar or array
        
        Returns:
            μ: Dynamic viscosity (Pa·s)
        """
        return self.mu_c0 + self.mu_c1 * T + self.mu_c2 * T**2
    
    def thermal_conductivity(self, T):
        """
        Calculate thermal conductivity at temperature T.
        
        Args:
            T: Temperature (K), scalar or array
        
        Returns:
            λ: Thermal conductivity (W/(m·K))
        """
        return self.lambda_c0 + self.lambda_c1 * T + self.lambda_c2 * T**2
    
    def specific_heat(self, T):
        """
        Calculate specific heat at temperature T.
        
        Args:
            T: Temperature (K), scalar or array
        
        Returns:
            cp: Specific heat at constant pressure (J/(kg·K))
        """
        return self.cp_c0 + self.cp_c1 * T + self.cp_c2 * T**2

# Pure species property data from Kleijn et al. (1989)
Nitrogen = SpeciesProperties(
    name="N2",
    mass=0.02801,  # kg/mol
    mu_c0=5.73e-6, mu_c1=4.37e-8, mu_c2=-9.28e-12,
    lambda_c0=8.54e-3, lambda_c1=6.23e-5, lambda_c2=-4.34e-9,
    cp_c0=9.83e2, cp_c1=1.58e-1, cp_c2=1.69e-5
)

SiliconHydride = SpeciesProperties(
    name="SiH4",
    mass=0.03212,  # kg/mol
    mu_c0=1.47e-6, mu_c1=3.66e-8, mu_c2=-6.81e-12,
    lambda_c0=-2.12e-2, lambda_c1=1.45e-4, lambda_c2=-1.31e-8,
    cp_c0=4.74e2, cp_c1=3.26e0, cp_c2=-1.08e-3
)

Hydrogen = SpeciesProperties(
    name="H2",
    mass=0.00202,  # kg/mol
    mu_c0=3.11e-6, mu_c1=2.06e-8, mu_c2=-3.54e-12,
    lambda_c0=1.05e-1, lambda_c1=3.21e-4, lambda_c2=-2.50e-9,
    cp_c0=1.47e4, cp_c1=-1.01e0, cp_c2=1.29e-3
)

class MixturePropertySolver():
    """
    Calculates mixture properties from pure species properties and composition.
    
    Uses standard mixing rules appropriate for gas mixtures:
    - Ideal gas law for density
    - Wilke's rule for viscosity
    - Weighted average for thermal conductivity
    - Mass-weighted average for specific heat
    """
    def __init__(self, grid, config):
        """
        Initialize mixture property solver.
        
        Args:
            grid: StaggeredGrid object
            config: SimulationConfig object
        """
        self.R = 8.314  # J/(mol·K) - Universal gas constant
        self.species = [Nitrogen, SiliconHydride, Hydrogen]
        self.n_species = len(self.species)        
        self.grid = grid
        self.config = config

    def mean_mole_mass(self, x):
        """
        Calculate mean molecular mass of mixture.
        
        Formula: M_mix = Σ(x_i * M_i)
        
        Args:
            x: Mole fraction array [..., 3]
        
        Returns:
            M_mix: Mean molecular mass (kg/mol), same shape as x without last dim
        """
        masses = np.array([s.mass for s in self.species])
        return np.sum(x * masses)
    
    def density(self, P, T, x):
        """
        Calculate mixture density from ideal gas law.
        
        Ideal gas law: ρ = P*M_mix/(R*T)
        where M_mix is the mean molecular mass
        
        Args:
            P: Pressure (Pa), scalar or array
            T: Temperature (K), scalar or array
            x: Mole fractions, array [..., 3]
        
        Returns:
            ρ: Density (kg/m³)
        """
        m = self.mean_mole_mass(x)
        return P * m / (self.R * T)

    def mole_to_mass_fraction(self, x):
        """
        Convert mole fractions to mass fractions.
        
        Formula: ω_i = x_i * M_i / M_mix
        
        Args:
            x: Mole fraction array
        
        Returns:
            omega: Mass fraction array with same shape
        """
        masses = np.array([s.mass for s in self.species])
        m_mix = self.mean_mole_mass(x)
        # Handle broadcasting for field arrays
        if x.ndim > 1:
            return x * masses / m_mix[..., np.newaxis]
        return x * masses / m_mix
    
    def mixture_specific_heat(self, T, x):
        """
        Calculate mixture specific heat using mass-weighted average.
        
        Formula: cp_mix = Σ(ω_i * cp_i)
        
        This is exact for ideal gas mixtures.
        
        Args:
            T: Temperature (K)
            x: Mole fractions
        
        Returns:
            cp_mix: Mixture specific heat (J/(kg·K))
        """
        omega = self.mole_to_mass_fraction(x)
        cp_mix = np.zeros_like(T)
        for i, species in enumerate(self.species):
            cp_i = species.specific_heat(T)
            if omega.ndim > 1:
                cp_mix += omega[..., i] * cp_i
            else:
                cp_mix += omega[i] * cp_i
        return cp_mix
    
    def _wilke_phi(self, i, j, T):
        """
        Calculate Wilke's interaction parameter φ_ij.
        
        Used in Wilke's mixing rule for viscosity:
        φ_ij = (1/√8) * (1 + M_i/M_j)^(-1/2) * [1 + (μ_i/μ_j)^(1/2) * (M_j/M_i)^(1/4)]²
        
        Args:
            i: Species index i
            j: Species index j
            T: Temperature (K)
        
        Returns:
            φ_ij: Wilke interaction parameter (dimensionless)
        """
        mi, mj = self.species[i].mass, self.species[j].mass
        mu_i = self.species[i].viscosity(T)
        mu_j = self.species[j].viscosity(T)
        
        term1 = 1.0 / np.sqrt(8.0) * (1.0 + mi/mj)**(-0.5)
        term2 = (1.0 + np.sqrt(mu_i/mu_j) * (mj/mi)**0.25)**2
        return term1 * term2
    
    def mixture_viscosity(self, T, x):
        """
        Calculate mixture viscosity using Wilke's mixing rule.
        
        Wilke's rule (semi-empirical, accurate for gas mixtures):
        μ_mix = Σ_i [x_i * μ_i / Σ_j(x_j * φ_ij)]
        
        Accounts for molecular size and interaction effects.
        
        Args:
            T: Temperature (K)
            x: Mole fractions
        
        Returns:
            μ_mix: Mixture dynamic viscosity (Pa·s)
        """
        mu_mix = np.zeros_like(T)
        
        for i, species_i in enumerate(self.species):
            mu_i = species_i.viscosity(T)
            # Shortcut for pure species
            if x[i] > 0.999:
                return species_i.viscosity(T)
            
            # Compute denominator: Σ_j(x_j * φ_ij)
            denom = np.zeros_like(T)
            for j in range(self.n_species):
                phi_ij = self._wilke_phi(i, j, T)
                if x.ndim > 1:
                    denom += x[..., j] * phi_ij
                else:
                    denom += x[j] * phi_ij
            
            # Add contribution from species i
            if x.ndim > 1:
                mu_mix += x[..., i] * mu_i / (denom + 1e-30)
            else:
                mu_mix += x[i] * mu_i / (denom + 1e-30)
        
        return mu_mix

    def mixture_thermal_conductivity(self, T, x, alpha=0.5):
        """
        Calculate mixture thermal conductivity using weighted average.
        
        Uses combination of linear and harmonic averages:
        λ_mix = α * Σ(λ_i * x_i) + (1-α) * [Σ(x_i/λ_i)]^(-1)
        
        where α=0.5 provides reasonable accuracy for gas mixtures.
        
        Args:
            T: Temperature (K)
            x: Mole fractions
            alpha: Weight factor for linear vs harmonic average (default 0.5)
        
        Returns:
            λ_mix: Mixture thermal conductivity (W/(m·K))
        """
        # Linear average term
        linear = np.zeros_like(T)
        # Harmonic average term
        harmonic_inv = np.zeros_like(T)
        
        for i, species in enumerate(self.species):
            lambda_i = species.thermal_conductivity(T)
            if x.ndim > 1:
                linear += x[..., i] * lambda_i
                harmonic_inv += x[..., i] / (lambda_i + 1e-30)
            else:
                linear += x[i] * lambda_i
                harmonic_inv += x[i] / (lambda_i + 1e-30)
        
        harmonic = 1.0 / (harmonic_inv + 1e-30)
        return alpha * linear + (1 - alpha) * harmonic
    
    def solve(self, nr, nz, p, T, x, rho, mu, tc, cp, omega):
        """
        Update all mixture properties for entire field.
        
        Calculates:
        - Density from ideal gas law
        - Viscosity from Wilke's rule
        - Thermal conductivity from weighted average
        - Specific heat from mass-weighted average
        - Effective diffusion coefficients for each species
        
        Uses under-relaxation on density for numerical stability
        given extreme property variations in LPCVD.
        
        Args:
            nr, nz: Grid dimensions
            p: Pressure field (nr, nz)
            T: Temperature field (nr, nz)
            x: Mole fractions (nr, nz, 3)
            rho: Density field (nr, nz) - input for under-relaxation
            mu: Viscosity field (nr, nz) - overwritten
            tc: Thermal conductivity field (nr, nz) - overwritten
            cp: Specific heat field (nr, nz) - overwritten
            omega: Mass fractions (nr, nz, 3)
        
        Returns:
            rho, mu, tc, cp, D_eff: Updated property fields
        """
        # Under-relaxation for density to prevent oscillations
        alpha_rho = 0.7
        D_eff = np.zeros((nr, nz, 3))
        diffusion_solver = DiffusionSolver(self.grid, self.config)

        for i in range(nr):
            for j in range(nz):
                # Calculate new properties
                rho_new = self.density(p[i, j], T[i, j], x[i, j, :])
                mu_new = self.mixture_viscosity(T[i, j], x[i, j, :])
                tc_new = self.mixture_thermal_conductivity(T[i, j], x[i, j, :])
                cp_new = self.mixture_specific_heat(T[i, j], x[i, j, :])
                
                # Apply under-relaxation to density only
                rho[i, j] = alpha_rho * rho_new + (1 - alpha_rho) * rho[i, j]
                # Other properties updated directly (more stable)
                mu[i, j] = mu_new
                tc[i, j] = tc_new
                cp[i, j] = cp_new

                # Calculate effective diffusion coefficient for each species
                omega_local = [float(omega[i, j, k]) for k in range(3)]
                m_mix = self.mean_mole_mass(x[i, j, :])

                for k in range(3):
                    D_eff[i, j, k] = diffusion_solver.compute_effective_diffusion(
                        k, T[i, j], p[i, j], omega_local, m_mix
                    )

        return rho, mu, tc, cp, D_eff