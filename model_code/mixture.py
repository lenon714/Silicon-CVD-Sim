from model_code import *

@dataclass
class SpeciesProperties:
    """Temperature-dependent properties: property = c0 + c1*T + c2*T^2"""
    name: str
    mass: float  # kg/mol
    # Viscosity coefficients (Pa·s)
    mu_c0: float
    mu_c1: float
    mu_c2: float
    # Thermal conductivity coefficients (W/m·K)
    lambda_c0: float
    lambda_c1: float
    lambda_c2: float
    # Specific heat coefficients (J/kg·K)
    cp_c0: float
    cp_c1: float
    cp_c2: float
    
    def viscosity(self, T):
        return self.mu_c0 + self.mu_c1 * T + self.mu_c2 * T**2
    
    def thermal_conductivity(self, T):
        return self.lambda_c0 + self.lambda_c1 * T + self.lambda_c2 * T**2
    
    def specific_heat(self, T):
        return self.cp_c0 + self.cp_c1 * T + self.cp_c2 * T**2

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
    def __init__(self, grid, config):
        self.R = 8.314
        self.species = [Nitrogen, SiliconHydride, Hydrogen]
        self.n_species = len(self.species)        
        self.grid = grid
        self.config = config

    def mean_mole_mass(self, x):
        masses = np.array([s.mass for s in self.species])
        return np.sum(x * masses)
    
    def density(self, P, T, x):
        m = self.mean_mole_mass(x)
        return P * m / (self.R * T)

    def mole_to_mass_fraction(self, x):
        """
        omega_i = x_i * m_i / m_mix
        """
        masses = np.array([s.mass for s in self.species])
        m_mix = self.mean_mole_mass(x)
        # Handle broadcasting for field arrays
        if x.ndim > 1:
            return x * masses / m_mix[..., np.newaxis]
        return x * masses / m_mix
    
    def mixture_specific_heat(self, T, x):
        """
        cp = sum(omega_i * cp_i)
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
        phi_ij = (1/sqrt(8)) * (1 + m_i/m_j)^(-1/2) * 
                 [1 + (mu_i/mu_j)^(1/2) * (m_j/m_i)^(1/4)]^2
        """
        mi, mj = self.species[i].mass, self.species[j].mass
        mu_i = self.species[i].viscosity(T)
        mu_j = self.species[j].viscosity(T)
        
        term1 = 1.0 / np.sqrt(8.0) * (1.0 + mi/mj)**(-0.5)
        term2 = (1.0 + np.sqrt(mu_i/mu_j) * (mj/mi)**0.25)**2
        return term1 * term2
    
    def mixture_viscosity(self, T, x):
        """
        mu = sum(x_i * mu_i / sum(x_j * phi_ij))
        """
        mu_mix = np.zeros_like(T)
        
        for i, species_i in enumerate(self.species):
            mu_i = species_i.viscosity(T)
            if x[i] > 0.999:
                return species_i.viscosity(T)
            
            # Compute denominator: sum_j(x_j * phi_ij)
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
        lambda = alpha * sum(lambda_i * x_i) + 
                 (1-alpha) * (sum(x_i/lambda_i))^(-1)
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
        # Under-relaxation for stability
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
                
                # Apply with under-relaxation
                rho[i, j] = alpha_rho * rho_new + (1 - alpha_rho) * rho[i, j]
                mu[i, j] = mu_new
                tc[i, j] = tc_new
                cp[i, j] = cp_new

                omega_local = [float(omega[i, j, k]) for k in range(3)]
                m_mix = self.mean_mole_mass(x[i, j, :])

                for k in range(3):
                    D_eff[i, j, k] = diffusion_solver.compute_effective_diffusion(
                        k, T[i, j], p[i, j], omega_local, m_mix
                    )

        return rho, mu, tc, cp, D_eff