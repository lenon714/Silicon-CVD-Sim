"""Boundary Conditions from Kleijn Paper"""

from model_code import *

class VelocityBoundaryConditions:
    """
    Velocity boundary conditions for LPCVD reactor.
    Implements no-slip conditions on walls, inlet velocity profile,
    symmetry at axis, and Stefan velocity at reacting wafer surface.
    """
    
    def __init__(self, grid, config):
        """
        Initialize velocity boundary condition handler.
        
        Args:
            grid: StaggeredGrid object containing mesh information
            config: SimulationConfig object with reactor geometry and process parameters
        """
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz
        
    def apply(self, vr: np.ndarray, vz: np.ndarray, 
              rho: np.ndarray = None, 
              M_stefan: np.ndarray = None):
        """
        Apply all velocity boundary conditions to the velocity fields.
        
        Args:
            vr: Radial velocity array (nr+1, nz) on radial faces
            vz: Axial velocity array (nr, nz+1) on axial faces
            rho: Density array (nr, nz) at cell centers (optional, for Stefan velocity)
            M_stefan: Stefan mass flux array (nr,) at wafer surface (optional)
        
        Returns:
            vr, vz: Updated velocity arrays with boundary conditions applied
        """
        self._apply_inlet(vr, vz)
        self._apply_top_wall(vr, vz)
        self._apply_symmetry_axis(vr, vz)
        self._apply_outer_wall(vr, vz)
        self._apply_wafer_surface(vr, vz, rho, M_stefan)
        self._apply_bottom_wall(vr, vz)
        self._apply_outlet(vr, vz)
        
        return vr, vz
    
    def _apply_inlet(self, vr: np.ndarray, vz: np.ndarray):
        """
        Apply inlet boundary conditions at top of reactor inside inlet pipe.
        
        Location: z = z_max, r < R_pipe
        Conditions:
            - vr = 0 (no radial flow at inlet)
            - vz = -V_in (uniform axial inflow, negative because flow is downward)
        
        Args:
            vr: Radial velocity array
            vz: Axial velocity array
        """
        R_pipe = self.config.pipe_radius
        V_in = self.config.inlet_velocity
        
        # Set radial velocity to zero at inlet faces
        for i in range(self.nr + 1):
            r_face = self.grid.r_faces[i]
            if r_face < R_pipe:
                vr[i, -1] = 0.0
        
        # Set axial velocity to inlet velocity (negative = inflow)
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center < R_pipe:
                vz[i, -1] = -V_in 
    
    def _apply_top_wall(self, vr: np.ndarray, vz: np.ndarray):
        """
        Apply no-slip wall conditions at top of reactor outside inlet pipe.
        
        Location: z = z_max, r >= R_pipe
        Conditions:
            - vr = 0 (no radial velocity at wall)
            - vz = 0 (no flow through solid wall)
        
        Args:
            vr: Radial velocity array
            vz: Axial velocity array
        """
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
        """
        Apply symmetry boundary conditions at centerline.
        
        Location: r = 0 (axis of cylindrical symmetry)
        Conditions:
            - vr = 0 (no radial velocity at axis by symmetry)
            - ∂vz/∂r = 0 (zero gradient, implemented by copying adjacent cell)
        
        Args:
            vr: Radial velocity array
            vz: Axial velocity array
        """
        vr[0, :] = 0.0
        vz[0, :] = vz[1, :]
    
    def _apply_outer_wall(self, vr: np.ndarray, vz: np.ndarray):
        """
        Apply no-slip conditions at outer radial wall.
        
        Location: r = r_max (outer cylindrical wall)
        Conditions:
            - vr = 0 (no radial velocity at wall)
            - vz = 0 (no axial velocity at wall)
        
        Args:
            vr: Radial velocity array
            vz: Axial velocity array
        """
        vr[-1, :] = 0.0
        vz[-1, :] = 0.0
    
    def _apply_wafer_surface(self, vr: np.ndarray, vz: np.ndarray,
                              rho: np.ndarray = None,
                              M_stefan: np.ndarray = None):
        """
        Apply boundary conditions at heated wafer surface with surface reaction.
        
        Location: z = 0, r < R_wafer (heated susceptor)
        Conditions:
            - vr = 0 (no radial velocity at surface)
            - vz = M_stefan/ρ (Stefan velocity due to surface reaction)
              If M_stefan not provided, vz = 0
        
        The Stefan velocity accounts for net mass flux from surface reaction:
        SiH4 → Si(s) + 2H2, which creates a flow normal to the surface.
        
        Args:
            vr: Radial velocity array
            vz: Axial velocity array
            rho: Density array (needed for Stefan velocity calculation)
            M_stefan: Stefan mass flux array (kg/(m²·s)) at wafer surface
        """
        R_wafer = self.config.wafer_radius
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            
            if r_center < R_wafer:
                # No radial velocity at wafer surface
                vr[i, 0] = 0.0
                if i + 1 <= self.nr:
                    vr[i + 1, 0] = 0.0
                
                # Axial velocity: Stefan velocity or no-penetration
                if M_stefan is not None and rho is not None:
                    vz[i, 0] = M_stefan[i] / (rho[i, 0] + 1e-30)
                else:
                    vz[i, 0] = 0.0
    
    def _apply_bottom_wall(self, vr: np.ndarray, vz: np.ndarray):
        """
        Apply conditions at bottom wall outside wafer region.
        
        Location: z = 0, r >= R_wafer (cooled wall at bottom)
        Conditions:
            - vr = 0 (no radial velocity at wall)
            - vz: outflow condition (zero gradient if flow is exiting, zero if recirculating)
        
        Args:
            vr: Radial velocity array
            vz: Axial velocity array
        """
        R_wafer = self.config.wafer_radius
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                # Check interior velocity to determine outflow behavior
                vz_interior = vz[i, 1]
                if vz_interior > 0:
                    # Flow is toward wall, set to zero (no penetration)
                    vz[i, 0] = 0.0
                else:
                    # Flow is away from wall, allow outflow (zero gradient)
                    vz[i, 0] = vz_interior
                
                # No radial velocity at wall
                vr[i, 0] = 0.0
                if i + 1 <= self.nr:
                    vr[i + 1, 0] = 0.0
    
    def _apply_outlet(self, vr: np.ndarray, vz: np.ndarray):
        """
        Apply outlet boundary condition (zero gradient for outflow).
        
        Location: z = 0, r >= R_wafer (outlet annular region)
        Condition: ∂vz/∂z = 0 (zero gradient outflow)
        
        Note: This overlaps with _apply_bottom_wall but uses simpler zero gradient.
        
        Args:
            vr: Radial velocity array
            vz: Axial velocity array
        """
        R_wafer = self.config.wafer_radius
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                vz[i, 0] = vz[i, 1]

class TemperatureBoundaryConditions:
    """
    Temperature boundary conditions for LPCVD reactor.
    
    Typical values from the paper:
        T_h (wafer/susceptor) = 900-1000 K
        T_w (cooled walls) = 290 K
        T_in (inlet) = 290 K
    
    Implements Dirichlet (fixed temperature) conditions on heated/cooled surfaces
    and Neumann (zero gradient) conditions at symmetry boundaries.
    """
    
    def __init__(self, grid, config):
        """
        Initialize temperature boundary condition handler.
        
        Args:
            grid: StaggeredGrid object containing mesh information
            config: SimulationConfig object with temperature specifications
        """
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
            T: Modified temperature array with boundary conditions applied
        """
        self._apply_inlet(T)
        self._apply_top_wall(T)
        self._apply_symmetry_axis(T)
        self._apply_outer_wall(T)
        self._apply_wafer_surface(T)
        # self._apply_bottom_wall(T)
        self._apply_outlet(T)
        
        return T
    
    def _apply_inlet(self, T: np.ndarray):
        """
        Apply inlet temperature condition.
        
        Location: z = z_max, r < R_pipe (inlet region)
        Condition: T = T_in (fixed inlet temperature)
        
        Args:
            T: Temperature array at cell centers
        """
        R_pipe = self.config.pipe_radius
        T_in = self.config.T_inlet
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center < R_pipe:
                T[i, -1] = T_in
    
    def _apply_top_wall(self, T: np.ndarray):
        """
        Apply cooled wall temperature at top of reactor.
        
        Location: z = z_max, r >= R_pipe (top wall outside inlet)
        Condition: T = T_w (cooled wall temperature)
        
        Args:
            T: Temperature array at cell centers
        """
        R_pipe = self.config.pipe_radius
        T_w = self.config.T_wall
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_pipe:
                T[i, -1] = T_w
    
    def _apply_symmetry_axis(self, T: np.ndarray):
        """
        Apply symmetry condition at centerline.
        
        Location: r = 0 (axis of cylindrical symmetry)
        Condition: ∂T/∂r = 0 (zero gradient by symmetry)
        Implementation: Copy value from adjacent cell
        
        Args:
            T: Temperature array at cell centers
        """
        T[0, :] = T[1, :]
    
    def _apply_outer_wall(self, T: np.ndarray):
        """
        Apply cooled wall temperature at outer boundary.
        
        Location: r = r_max (outer cylindrical wall)
        Condition: T = T_w (cooled wall temperature)
        
        Args:
            T: Temperature array at cell centers
        """
        T_w = self.config.T_wall
        T[-1, :] = T_w
    
    def _apply_wafer_surface(self, T: np.ndarray):
        """
        Apply heated wafer temperature condition.
        
        Location: z = 0, r < R_wafer (susceptor/wafer surface)
        Condition: T = T_h (heated susceptor temperature)
        
        This is the hot surface where silicon deposition occurs.
        
        Args:
            T: Temperature array at cell centers
        """
        R_wafer = self.config.wafer_radius
        T_h = self.config.T_wafer
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center < R_wafer:
                T[i, 0] = T_h
    
    def _apply_bottom_wall(self, T: np.ndarray):
        """
        Apply cooled wall temperature at bottom outside wafer.
        
        Location: z = 0, r >= R_wafer (bottom wall outside wafer)
        Condition: T = T_w (cooled wall temperature)
        
        Args:
            T: Temperature array at cell centers
        """
        R_wafer = self.config.wafer_radius
        T_w = self.config.T_wall
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                T[i, 0] = T_w
    
    def _apply_outlet(self, T: np.ndarray):
        """
        Apply outlet boundary condition for temperature.
        
        Location: Bottom boundary at outlet region
        Condition: ∂T/∂z = 0 (zero gradient for outflow)
        
        Args:
            T: Temperature array at cell centers
        """
        R_wafer = self.config.wafer_radius
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                T[i, 0] = T[i, 1]
        pass

class PressureBoundaryConditions:
    """
    Pressure boundary conditions for LPCVD reactor.
    
    From the paper, pressure is specified at the outlet only.
    All other boundaries use zero-gradient (Neumann) conditions
    for the pressure correction equation.
    
    Note: These are applied to the pressure correction p', not p directly.
    The actual pressure field p is updated via p = p + α_p * p'
    """
    
    def __init__(self, grid, config):
        """
        Initialize pressure boundary condition handler.
        
        Args:
            grid: StaggeredGrid object containing mesh information
            config: SimulationConfig object with outlet pressure specification
        """
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz
        
    def apply(self, p: np.ndarray):
        """
        Apply pressure boundary conditions to absolute pressure field.
        
        Args:
            p: Pressure array, shape (nr, nz), stored at cell centers
        
        Returns:
            p: Modified pressure array with boundary conditions applied
        """
        self._apply_symmetry_axis(p)
        self._apply_outer_wall(p)
        self._apply_top(p)
        self._apply_bottom(p)
        
        return p
    
    def _apply_symmetry_axis(self, p: np.ndarray):
        """
        Apply symmetry condition at centerline for pressure.
        
        Location: r = 0 (axis of cylindrical symmetry)
        Condition: ∂p/∂r = 0 (zero gradient by symmetry)
        
        Args:
            p: Pressure array at cell centers
        """
        p[0, :] = p[1, :]
    
    def _apply_outer_wall(self, p: np.ndarray):
        """
        Apply zero gradient condition at outer wall.
        
        Location: r = r_max (outer cylindrical wall)
        Condition: ∂p/∂r = 0 (zero gradient, no-penetration wall)
        
        Args:
            p: Pressure array at cell centers
        """
        p[-1, :] = p[-2, :]
    
    def _apply_top(self, p: np.ndarray):
        """
        Apply zero gradient condition at top boundary.
        
        Location: z = z_max (inlet and top wall)
        Condition: ∂p/∂z = 0 (zero gradient)
        
        Args:
            p: Pressure array at cell centers
        """
        p[:, -1] = p[:, -2]
    
    def _apply_bottom(self, p: np.ndarray):
        """
        Apply mixed boundary conditions at bottom.
        
        Location: z = 0 (bottom boundary)
        Conditions:
            - r >= R_wafer (outlet): p = P_out (fixed pressure, Dirichlet)
            - r < R_wafer (wafer): ∂p/∂z = 0 (zero gradient, Neumann)
        
        The outlet serves as the pressure reference point for the domain.
        
        Args:
            p: Pressure array at cell centers
        """
        R_wafer = self.config.wafer_radius
        P_out = self.config.pressure_outlet
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                # Outlet: fixed pressure (Dirichlet)
                p[i, 0] = P_out
            else:
                # Wafer: zero gradient (Neumann)
                p[i, 0] = p[i, 1]
    
    def apply_to_correction(self, p_prime: np.ndarray):
        """
        Apply boundary conditions specifically for pressure correction p'.
        
        For the pressure correction equation in SIMPLE algorithm:
        - p' = 0 at outlet (reference point, Dirichlet)
        - ∂p'/∂n = 0 at all other boundaries (Neumann)
        
        This ensures the correction field has a unique solution and
        properly couples with the velocity correction.
        
        Args:
            p_prime: Pressure correction array at cell centers
        
        Returns:
            p_prime: Modified pressure correction array
        """
        # Symmetry axis: zero gradient
        p_prime[0, :] = p_prime[1, :]
        
        # Outer wall: zero gradient
        p_prime[-1, :] = p_prime[-2, :]
        
        # Top: zero gradient
        p_prime[:, -1] = p_prime[:, -2]
        
        # Bottom: outlet is reference (p' = 0), wafer is zero gradient
        R_wafer = self.config.wafer_radius
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                # Outlet: p' = 0 (reference point)
                p_prime[i, 0] = 0.0
            else:
                # Wafer: zero gradient
                p_prime[i, 0] = p_prime[i, 1]
        
        return p_prime