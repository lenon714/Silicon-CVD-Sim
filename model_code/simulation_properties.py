"""
Properties and configuration for LPCVD Model

Defines simulation configuration and grid generation for the
axisymmetric cylindrical reactor geometry.
"""

from model_code import *

@dataclass
class SimulationConfig:
    """
    Configuration for the LPCVD reactor simulation.
    
    Contains all parameters needed to define:
    - Computational grid
    - Reactor geometry
    - Process conditions
    - Solver parameters
    - Species properties
    """
    # ========== GRID DIMENSIONS ==========
    nr: int = 38  # Number of cells in radial direction
    nz: int = 53  # Number of cells in axial direction
    
    # ========== PHYSICAL DIMENSIONS ==========
    r_max: float = 0.21   # Maximum radius (m) - reactor chamber radius
    z_max: float = 0.36   # Maximum height (m) - reactor chamber height
    
    # ========== REACTOR GEOMETRY ==========
    # Inlet configuration
    pipe_radius: float = 0.025      # Inlet pipe radius (m)
    pipe_height: float = 0.025      # Extension height of inlet pipe (m)
    z_pipe_bottom: float = 0.02     # Height where pipe ends (m)
    
    # Wafer configuration
    wafer_radius: float = 0.125     # Wafer radius (m) - typically 250mm diameter
    
    # ========== PROCESS CONDITIONS ==========
    inlet_velocity: float = 0.5     # Inlet velocity (m/s)
    pressure_outlet: float = 133.0  # Outlet pressure (Pa) ~ 1 torr for LPCVD
    
    # ========== SOLVER PARAMETERS ==========
    max_iterations: int = 2000              # Maximum SIMPLE iterations
    convergence_criteria: float = 1e-4     # Mass residual convergence threshold
    under_relaxation_p: float = 0.3        # Pressure under-relaxation factor
    under_relaxation_v: float = 0.5       # Velocity under-relaxation factor
    
    # ========== GRAVITY ==========
    gravity: float = 0  # Gravitational acceleration (m/s²) - set to 0 for horizontal reactor
    
    # ========== WALL TEMPERATURES ==========
    T_inlet: float = 290   # Inlet gas temperature (K) - room temperature
    T_wall: float = 290    # Cooled wall temperature (K)
    T_wafer: float = 1000  # Wafer/susceptor temperature (K) - typical LPCVD temperature
    
    # ========== INITIAL CONCENTRATIONS ==========
    # Initial field composition (pure N2 for startup)
    n2: float = 1      # N2 mole fraction
    sih4: float = 0    # SiH4 mole fraction
    h2: float = 0      # H2 mole fraction
    init_composition: tuple[float] = (1, 0, 0)
    
    # Inlet composition (N2, SiH4, H2)
    inlet_composition: tuple[float] = (0.9, 0.08, 0.02)  # 8% SiH4, 2% H2 in N2 carrier
    
    # ========== MOLAR MASSES ==========
    # Molecular masses (kg/mol) for [N2, SiH4, H2]
    masses: tuple[float] = (0.02801, 0.03212, 0.00202)

class StaggeredGrid:
    """
    Staggered grid in cylindrical coordinates (r, z).
    
    Uses Marker-and-Cell (MAC) staggered grid arrangement:
    - Scalar quantities (p, T, ρ, etc.) stored at cell centers
    - Radial velocity v_r stored at radial faces (i+1/2, j)
    - Axial velocity v_z stored at axial faces (i, j+1/2)
    
    This arrangement:
    - Avoids pressure checkerboarding
    - Naturally satisfies continuity at faces
    - Simplifies pressure-velocity coupling
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize staggered grid for cylindrical reactor.
        
        Creates non-uniform mesh with clustering near boundaries where
        gradients are steep (inlet, axis, wafer surface).
        
        Args:
            config: SimulationConfig object with grid parameters
        """
        self.config = config
        self.nr, self.nz = config.nr, config.nz
        
        # ========== GRID FACES ==========
        # Face locations define cell boundaries
        # Currently uniform, but can be modified for clustering
        self.r_faces = np.linspace(0, config.r_max, config.nr + 1)
        self.z_faces = np.linspace(0, config.z_max, config.nz + 1)
        
        # ========== CELL CENTERS ==========
        # Scalar quantities stored here (average of adjacent faces)
        self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        self.z_centers = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])
        
        # ========== GRID SPACING ==========
        # Distance between faces (cell size)
        self.dr = np.diff(self.r_faces)  # Array of length nr
        self.dz = np.diff(self.z_faces)  # Array of length nz
        
        # ========== VALIDATION ==========
        # Check for zero spacing
        assert np.all(self.dr > 0), "Zero radial spacing detected!"
        assert np.all(self.dz > 0), "Zero axial spacing detected!"
        
        print(f"Grid created: dr_min={self.dr.min():.6f}, dz_min={self.dz.min():.6f}")