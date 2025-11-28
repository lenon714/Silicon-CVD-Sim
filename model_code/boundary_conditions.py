"""Boundary Conditions from Klein Paper Extracted by Claude"""

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
        """
        Args:
            grid: StaggeredGrid object with r_centers, r_faces, z_centers, z_faces
            config: SimulationConfig with pipe_radius, wafer_radius, inlet_velocity, etc.
        """
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz
        
    def apply(self, vr: np.ndarray, vz: np.ndarray, 
              rho: np.ndarray = None, 
              M_stefan: np.ndarray = None):
        """
        Apply all velocity boundary conditions.
        
        Args:
            vr: Radial velocity array, shape (nr+1, nz)
                - Stored at radial FACES, axial cell centers
            vz: Axial velocity array, shape (nr, nz+1)
                - Stored at radial cell centers, axial FACES
            rho: Density array, shape (nr, nz) - needed for Stefan velocity
            M_stefan: Net mass flux at wafer surface [kg/m²/s], shape (nr,)
                      Positive = mass leaving surface (gas production)
                      From Eq. [27]: M = Σ_i Σ_k γ_ik R_k m_i
        
        Returns:
            vr, vz: Modified velocity arrays
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
        INLET: Top boundary inside pipe (z = z_max, r < R_pipe)
        
        Conditions:
            v_r = 0       (no radial velocity at inlet)
            v_z = V_in    (specified inlet velocity, positive = downward)
        """
        R_pipe = self.config.pipe_radius
        V_in = self.config.inlet_velocity
        
        # v_r at inlet (top row of vr array)
        # vr is at radial faces, so check r_faces
        for i in range(self.nr + 1):
            r_face = self.grid.r_faces[i]
            if r_face < R_pipe:
                vr[i, -1] = 0.0
        
        # v_z at inlet (top face of vz array, index nz)
        # vz is at radial centers, so check r_centers
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center < R_pipe:
                vz[i, -1] = V_in  # Positive = flow in +z direction
                                  # If z increases upward and flow is down,
                                  # this should be negative. Check your convention!
    
    def _apply_top_wall(self, vr: np.ndarray, vz: np.ndarray):
        """
        TOP WALL: Top boundary outside pipe (z = z_max, r ≥ R_pipe)
        
        Conditions:
            v_r = 0    (no-slip)
            v_z = 0    (no-slip)
        """
        R_pipe = self.config.pipe_radius
        
        # v_r at top wall
        for i in range(self.nr + 1):
            r_face = self.grid.r_faces[i]
            if r_face >= R_pipe:
                vr[i, -1] = 0.0
        
        # v_z at top wall
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_pipe:
                vz[i, -1] = 0.0
    
    def _apply_symmetry_axis(self, vr: np.ndarray, vz: np.ndarray):
        """
        SYMMETRY AXIS: Centerline (r = 0)
        
        Conditions:
            v_r = 0           (no flow through axis)
            ∂v_z/∂r = 0       (zero gradient, symmetry)
        """
        # v_r = 0 at axis (first radial face, i=0)
        vr[0, :] = 0.0
        
        # ∂v_z/∂r = 0 at axis
        # vz[0, :] should equal vz[1, :] (zero gradient)
        vz[0, :] = vz[1, :]
    
    def _apply_outer_wall(self, vr: np.ndarray, vz: np.ndarray):
        """
        OUTER WALL: Outer radial boundary (r = r_max)
        
        Conditions:
            v_r = 0    (no-slip, no flow through wall)
            v_z = 0    (no-slip)
        """
        # v_r = 0 at outer wall (last radial face, i=nr)
        vr[-1, :] = 0.0
        
        # v_z = 0 at outer wall (last radial cell, i=nr-1)
        vz[-1, :] = 0.0
    
    def _apply_wafer_surface(self, vr: np.ndarray, vz: np.ndarray,
                              rho: np.ndarray = None,
                              M_stefan: np.ndarray = None):
        """
        WAFER SURFACE: Bottom boundary where wafer is located (z = 0, r < R_wafer)
        
        Conditions:
            v_r = 0                    (no radial velocity at surface)
            v_z = M/ρ                  (Stefan velocity, Eq. [28])
        
        The Stefan velocity accounts for the net mass flux due to:
        - Consumption of reactants (e.g., SiH4)
        - Production of products (e.g., H2)
        - Deposition of solid (Si)
        
        For silicon deposition from silane:
            SiH4(g) → Si(s) + 2H2(g)
        
        Net gaseous mass change: 2*M_H2 - M_SiH4 = 2(2.016) - 32.12 = -28.09 g/mol
        So there's net mass consumption → v_z points INTO the surface (negative if z+ is up)
        """
        R_wafer = self.config.wafer_radius
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            
            if r_center < R_wafer:
                # Inside wafer region
                
                # v_r = 0 at wafer surface
                # Set both faces of this cell at z=0
                vr[i, 0] = 0.0
                if i + 1 <= self.nr:
                    vr[i + 1, 0] = 0.0
                
                # v_z = M/ρ (Stefan velocity)
                if M_stefan is not None and rho is not None:
                    # M_stefan[i] = net mass flux [kg/m²/s]
                    # Positive M = mass leaving surface (net gas production)
                    # This gives positive vz (flow away from surface)
                    vz[i, 0] = M_stefan[i] / (rho[i, 0] + 1e-30)
                else:
                    # No reaction or simplified case: no Stefan flow
                    vz[i, 0] = 0.0
    
    def _apply_bottom_wall(self, vr: np.ndarray, vz: np.ndarray):
        """
        BOTTOM WALL: Bottom boundary outside wafer (z = 0, r ≥ R_wafer)
        
        Conditions:
            v_r = 0    (no-slip)
            v_z = 0    (no-slip)
        """
        R_wafer = self.config.wafer_radius
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            
            if r_center >= R_wafer:
                # Outside wafer - solid wall
                vr[i, 0] = 0.0
                if i + 1 <= self.nr:
                    vr[i + 1, 0] = 0.0
                vz[i, 0] = 0.0
    
    def _apply_outlet(self, vr: np.ndarray, vz: np.ndarray):
        """
        OUTLET: Outflow boundary
        
        The paper shows the outlet at the bottom-outer corner.
        
        Conditions:
            v_r = 0              (no radial outflow)
            ∂(ρv_z)/∂z = 0       (zero gradient for mass flux)
        
        For constant density, this simplifies to:
            ∂v_z/∂z = 0
        
        Note: In your current setup, the outlet is handled by the
        wafer/bottom wall conditions. If you want a separate outlet
        region (e.g., at z=0 for certain r values), implement here.
        """
        # The outlet condition ∂v_z/∂z = 0 is typically applied by
        # setting the boundary value equal to the interior value.
        # This is often already handled by the wafer surface BC
        # (where vz[i,0] is either Stefan velocity or comes from interior).
        
        # If you have a specific outlet region, you can add it here:
        # For example, if outlet is at z=0 for r > wafer_radius:
        R_wafer = self.config.wafer_radius
        
        for i in range(self.nr):
            r_center = self.grid.r_centers[i]
            if r_center >= R_wafer:
                # Zero gradient outflow
                vz[i, 0] = vz[i, 1]


def apply_velocity_bc_simple(vr, vz, grid, config, rho=None, M_stefan=None):
    """
    Simple function version (no class) for easy integration.
    
    Call this in your solver's apply_boundary_conditions method.
    """
    nr, nz = grid.nr, grid.nz
    R_pipe = config.pipe_radius
    R_wafer = config.wafer_radius
    V_in = config.inlet_velocity
    
    # ==================== INLET ====================
    # z = z_max, r < R_pipe
    # v_r = 0, v_z = V_in
    for i in range(nr + 1):
        if grid.r_faces[i] < R_pipe:
            vr[i, -1] = 0.0
    for i in range(nr):
        if grid.r_centers[i] < R_pipe:
            vz[i, -1] = V_in
    
    # ==================== TOP WALL ====================
    # z = z_max, r >= R_pipe
    # v_r = v_z = 0
    for i in range(nr + 1):
        if grid.r_faces[i] >= R_pipe:
            vr[i, -1] = 0.0
    for i in range(nr):
        if grid.r_centers[i] >= R_pipe:
            vz[i, -1] = 0.0
    
    # ==================== SYMMETRY AXIS ====================
    # r = 0
    # v_r = 0, ∂v_z/∂r = 0
    vr[0, :] = 0.0
    vz[0, :] = vz[1, :]
    
    # ==================== OUTER WALL ====================
    # r = r_max
    # v_r = v_z = 0
    vr[-1, :] = 0.0
    vz[-1, :] = 0.0
    
    # ==================== WAFER SURFACE ====================
    # z = 0, r < R_wafer
    # v_r = 0, v_z = M/ρ
    for i in range(nr):
        if grid.r_centers[i] < R_wafer:
            vr[i, 0] = 0.0
            vr[min(i+1, nr), 0] = 0.0
            
            if M_stefan is not None and rho is not None:
                vz[i, 0] = M_stefan[i] / (rho[i, 0] + 1e-30)
            else:
                vz[i, 0] = 0.0
    
    # ==================== BOTTOM WALL ====================
    # z = 0, r >= R_wafer
    # v_r = v_z = 0
    for i in range(nr):
        if grid.r_centers[i] >= R_wafer:
            vr[i, 0] = 0.0
            vr[min(i+1, nr), 0] = 0.0
            vz[i, 0] = 0.0
    
    return vr, vz