"""Boundary Conditions from Kleijn Paper"""

from model_code import *

class VelocityBoundaryConditions:
    """
    Velocity boundary conditions for LPCVD reactor.
    
    Domain layout (axisymmetric, r-z coordinates):
    
                              INLET PIPE
                         ┌──────────────────┐
                         │ v_r = 0          │
                         │ v_z = V_in       │
                         └────────┬─────────┘
        ┌────────────────────────┴────────────────────────┐ z = z_max
        │  TOP WALL              │           TOP WALL     │
        │  v_r = v_z = 0         │           v_r = v_z = 0│
        │                        │                        │
        │                                                 │
    A   │                                                 │ OUTER WALL
    X   │ v_r = 0                                         │ v_r = v_z = 0
    I   │ ∂v_z/∂r = 0                                     │
    S   │                                                 │
        │                                                 │
    r=0 │                                                 │ r = r_max
        │                                                 │
        │                                                 │
        ├─────────────────────────┬───────────────────────┤ z = 0
        │       WAFER             │     BOTTOM WALL       │
        │  v_r = 0                │     v_r = v_z = 0     │
        │  v_z = M/ρ (Stefan)     │                       │
        └─────────────────────────┴───────────────────────┘
                                  │
                           ┌──────┴──────┐
                           │   OUTLET    │
                           │ v_r = 0     │
                           │∂(ρv_z)/∂z=0 │
                           └─────────────┘
    """
    
    def __init__(self, grid, config):
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz
        
    def apply(self, vr: np.ndarray, vz: np.ndarray, 
              rho: np.ndarray = None, 
              M_stefan: np.ndarray = None):
        
        self._apply_inlet(vr, vz)
        self._apply_top_wall(vr, vz)
        self._apply_symmetry_axis(vr, vz)
        self._apply_outer_wall(vr, vz)
        self._apply_wafer_surface(vr, vz, rho, M_stefan)
        self._apply_bottom_wall(vr, vz)
        self._apply_outlet(vr, vz)
        
        return vr, vz
    
    def _apply_inlet(self, vr: np.ndarray, vz: np.ndarray):
        R_pipe = self.config.pipe_radius
        V_in = self.config.inlet_velocity
        
        for i in range(self.nr + 1):
            r_face = self.grid.r_faces[i]
            if r_face < R_pipe:
                vr[i, -1] = 0.0
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center < R_pipe:
                vz[i, -1] = -V_in 
    
    def _apply_top_wall(self, vr: np.ndarray, vz: np.ndarray):
        R_pipe = self.config.pipe_radius
        
        for i in range(self.nr + 1):
            r_face = self.grid.r_faces[i]
            if r_face >= R_pipe:
                vr[i, -1] = 0.0
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_pipe:
                vz[i, -1] = 0.0
    
    def _apply_symmetry_axis(self, vr: np.ndarray, vz: np.ndarray):
        vr[0, :] = 0.0
        vz[0, :] = vz[1, :]
    
    def _apply_outer_wall(self, vr: np.ndarray, vz: np.ndarray):
        vr[-1, :] = 0.0
        vz[-1, :] = 0.0
    
    def _apply_wafer_surface(self, vr: np.ndarray, vz: np.ndarray,
                              rho: np.ndarray = None,
                              M_stefan: np.ndarray = None):
        R_wafer = self.config.wafer_radius
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            
            if r_center < R_wafer:
                vr[i, 0] = 0.0
                if i + 1 <= self.nr:
                    vr[i + 1, 0] = 0.0
                
                if M_stefan is not None and rho is not None:
                    vz[i, 0] = M_stefan[i] / (rho[i, 0] + 1e-30)
                else:
                    vz[i, 0] = 0.0
    
    def _apply_bottom_wall(self, vr: np.ndarray, vz: np.ndarray):
        R_wafer = self.config.wafer_radius
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            
            if r_center >= R_wafer:
                vr[i, 0] = 0.0
                if i + 1 <= self.nr:
                    vr[i + 1, 0] = 0.0
                vz[i, 0] = 0.0
    
    def _apply_outlet(self, vr: np.ndarray, vz: np.ndarray):
        R_wafer = self.config.wafer_radius
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                vz[i, 0] = vz[i, 1]


class TemperatureBoundaryConditions:
    """
    Temperature boundary conditions for LPCVD reactor.
    
    From Kleijn et al. (1989), Figure 2:
    
                              INLET PIPE
                         ┌──────────────────┐
                         │ T = T_in         │
                         └────────┬─────────┘
        ┌────────────────────────┴────────────────────────┐ z = z_max
        │  TOP WALL              │           TOP WALL     │
        │  T = T_w               │           T = T_w      │
        │                        │                        │
        │                                                 │
    A   │                                                 │ OUTER WALL
    X   │ ∂T/∂r = 0                                       │ T = T_w
    I   │                                                 │
    S   │                                                 │
        │                                                 │
    r=0 │                                                 │ r = r_max
        │                                                 │
        │                                                 │
        ├─────────────────────────┬───────────────────────┤ z = 0
        │       WAFER             │     BOTTOM WALL       │
        │  T = T_h (heated)       │     T = T_w           │
        └─────────────────────────┴───────────────────────┘
                                  │
                           ┌──────┴──────┐
                           │   OUTLET    │
                           │ ∂T/∂z = 0   │
                           └─────────────┘
    
    Typical values from the paper:
        T_h (wafer/susceptor) = 900-1000 K
        T_w (cooled walls) = 290 K
        T_in (inlet) = 290 K
    """
    
    def __init__(self, grid, config):
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz
        
    def apply(self, T: np.ndarray):
        """
        Apply all temperature boundary conditions.
        
        Args:
            T: Temperature array, shape (nr, nz), stored at cell centers
        
        Returns:
            T: Modified temperature array
        """
        self._apply_inlet(T)
        self._apply_top_wall(T)
        self._apply_symmetry_axis(T)
        self._apply_outer_wall(T)
        self._apply_wafer_surface(T)
        self._apply_bottom_wall(T)
        self._apply_outlet(T)
        
        return T
    
    def _apply_inlet(self, T: np.ndarray):
        """
        INLET: Top boundary inside pipe (z = z_max, r < R_pipe)
        
        Condition: T = T_in
        """
        R_pipe = self.config.pipe_radius
        T_in = self.config.T_inlet
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center < R_pipe:
                T[i, -1] = T_in
    
    def _apply_top_wall(self, T: np.ndarray):
        """
        TOP WALL: Top boundary outside pipe (z = z_max, r >= R_pipe)
        
        Condition: T = T_w (cooled wall)
        """
        R_pipe = self.config.pipe_radius
        T_w = self.config.T_wall
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_pipe:
                T[i, -1] = T_w
    
    def _apply_symmetry_axis(self, T: np.ndarray):
        """
        SYMMETRY AXIS: Centerline (r = 0)
        
        Condition: ∂T/∂r = 0 (zero gradient)
        """
        T[0, :] = T[1, :]
    
    def _apply_outer_wall(self, T: np.ndarray):
        """
        OUTER WALL: Outer radial boundary (r = r_max)
        
        Condition: T = T_w (cooled wall)
        """
        T_w = self.config.T_wall
        T[-1, :] = T_w
    
    def _apply_wafer_surface(self, T: np.ndarray):
        """
        WAFER SURFACE: Bottom boundary where wafer is located (z = 0, r < R_wafer)
        
        Condition: T = T_h (heated susceptor)
        """
        R_wafer = self.config.wafer_radius
        T_h = self.config.T_wafer
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center < R_wafer:
                T[i, 0] = T_h
    
    def _apply_bottom_wall(self, T: np.ndarray):
        """
        BOTTOM WALL: Bottom boundary outside wafer (z = 0, r >= R_wafer)
        
        Condition: T = T_w (cooled wall)
        """
        R_wafer = self.config.wafer_radius
        T_w = self.config.T_wall
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                T[i, 0] = T_w
    
    def _apply_outlet(self, T: np.ndarray):
        """
        OUTLET: Outflow boundary (if separate from bottom wall)
        
        Condition: ∂T/∂z = 0 (zero gradient for outflow)
        
        Note: In the Kleijn paper geometry, the outlet is at the 
        bottom-outer region. If you want pure outflow BC there:
        """
        # Currently handled by bottom_wall setting T = T_w
        # If you want zero gradient outflow instead, uncomment:
        # R_wafer = self.config.wafer_radius
        # for i in range(self.nr):
        #     r_center = self.grid.r_centers[i]
        #     if r_center >= R_wafer:
        #         T[i, 0] = T[i, 1]
        pass


class PressureBoundaryConditions:
    """
    Pressure boundary conditions for LPCVD reactor.
    
    From the paper, pressure is specified at the outlet only.
    All other boundaries use zero-gradient (Neumann) conditions
    for the pressure correction equation.
    
                              INLET PIPE
                         ┌──────────────────┐
                         │ ∂p/∂z = 0        │
                         └────────┬─────────┘
        ┌────────────────────────┴────────────────────────┐ z = z_max
        │  TOP WALL              │           TOP WALL     │
        │  ∂p/∂z = 0             │           ∂p/∂z = 0    │
        │                        │                        │
        │                                                 │
    A   │                                                 │ OUTER WALL
    X   │ ∂p/∂r = 0                                       │ ∂p/∂r = 0
    I   │                                                 │
    S   │                                                 │
        │                                                 │
    r=0 │                                                 │ r = r_max
        │                                                 │
        │                                                 │
        ├─────────────────────────┬───────────────────────┤ z = 0
        │       WAFER             │     OUTLET            │
        │  ∂p/∂z = 0              │     p = P_out         │
        └─────────────────────────┴───────────────────────┘
    
    Note: These are applied to the pressure correction p', not p directly.
    The actual pressure field p is updated via p = p + α_p * p'
    """
    
    def __init__(self, grid, config):
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz
        
    def apply(self, p: np.ndarray):
        """
        Apply pressure boundary conditions.
        
        Note: For SIMPLE algorithm, these are typically applied to p_prime.
        The pressure field itself only needs the outlet reference pressure.
        
        Args:
            p: Pressure array, shape (nr, nz), stored at cell centers
        
        Returns:
            p: Modified pressure array
        """
        self._apply_symmetry_axis(p)
        self._apply_outer_wall(p)
        self._apply_top(p)
        self._apply_bottom(p)
        
        return p
    
    def _apply_symmetry_axis(self, p: np.ndarray):
        """
        SYMMETRY AXIS: r = 0
        
        Condition: ∂p/∂r = 0
        """
        p[0, :] = p[1, :]
    
    def _apply_outer_wall(self, p: np.ndarray):
        """
        OUTER WALL: r = r_max
        
        Condition: ∂p/∂r = 0
        """
        p[-1, :] = p[-2, :]
    
    def _apply_top(self, p: np.ndarray):
        """
        TOP: z = z_max (inlet and top wall)
        
        Condition: ∂p/∂z = 0
        """
        p[:, -1] = p[:, -2]
    
    def _apply_bottom(self, p: np.ndarray):
        """
        BOTTOM: z = 0
        
        For the outlet region, pressure is specified (Dirichlet).
        For the wafer region, zero gradient.
        
        In SIMPLE, we typically set p' = 0 at the outlet (reference point).
        """
        R_wafer = self.config.wafer_radius
        P_out = self.config.pressure_outlet
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                # Outlet: fixed pressure
                p[i, 0] = P_out
            else:
                # Wafer: zero gradient
                p[i, 0] = p[i, 1]
    
    def apply_to_correction(self, p_prime: np.ndarray):
        """
        Apply boundary conditions specifically for pressure correction p'.
        
        For p', we set p' = 0 at the outlet (reference) and zero gradient elsewhere.
        This is what your pressure solver already does internally.
        """
        # Symmetry axis
        p_prime[0, :] = p_prime[1, :]
        
        # Outer wall
        p_prime[-1, :] = p_prime[-2, :]
        
        # Top
        p_prime[:, -1] = p_prime[:, -2]
        
        # Bottom - outlet is reference (p' = 0)
        R_wafer = self.config.wafer_radius
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                p_prime[i, 0] = 0.0
            else:
                p_prime[i, 0] = p_prime[i, 1]
        
        return p_prime