"""Properties and configuration for LPCVD Model"""

from model_code import *

@dataclass
class FluidProperties:
    """Fluid properties"""
    density: float = 0.15  # kg/m³ (N2 at 1 torr, 300K)
    viscosity: float = 1.5e-5  # Pa·s
    bulk_viscosity: float = 0.0
    
@dataclass
class SimulationConfig:
    """Configuration for the LPCVD reactor simulation"""
    # Grid dimensions
    nr: int = 20 
    nz: int = 30
    
    # Physical dimensions
    r_max: float = 0.21
    z_max: float = 0.36
    
    # Reactor geometry
    pipe_radius: float = 0.025
    pipe_height: float = 0.025
    wafer_radius: float = 0.125
    
    # Process conditions
    inlet_velocity: float = 0.5  # m/s
    pressure_outlet: float = 133.0  # Pa
    
    # Solver parameter
    max_iterations: int = 2000
    convergence_criteria: float = 1e-4
    under_relaxation_p: float = 0.1
    under_relaxation_v: float = 0.3
    
    # Gravity
    gravity: float = 9.81

class StaggeredGrid:
    """Staggered grid in cylindrical coordinates"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.nr, self.nz = config.nr, config.nz
        
        # Use UNIFORM grid for stability during debugging
        self.r_faces = np.linspace(0, config.r_max, config.nr + 1)
        self.z_faces = np.linspace(0, config.z_max, config.nz + 1)
        
        # Cell centers
        self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        self.z_centers = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])
        
        # Grid spacing (constant for uniform grid)
        self.dr = np.diff(self.r_faces)
        self.dz = np.diff(self.z_faces)
        
        # Check for zero spacing
        assert np.all(self.dr > 0), "Zero radial spacing detected!"
        assert np.all(self.dz > 0), "Zero axial spacing detected!"
        
        print(f"Grid created: dr_min={self.dr.min():.6f}, dz_min={self.dz.min():.6f}")
        