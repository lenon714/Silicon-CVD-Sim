"""Navier-Stokes Solver for LPCVD Model"""

from model_code import *

class NavierStokesSolver:
    """Navier-Stokes Solver in Cylindrical Coordinates"""
    
    def __init__(self, config: SimulationConfig, fluid: FluidProperties):
        self.config = config
        self.fluid = fluid
        self.grid = StaggeredGrid(config)
        self._initialize_fields()
        
    def _initialize_fields(self):
        """Initialize with physically reasonable values"""
        nr, nz = self.config.nr, self.config.nz
        
        # Velocities
        self.vr = np.zeros((nr + 1, nz))
        self.vz = np.zeros((nr, nz + 1))
        
        # Initialize vz with linear profile from inlet to outlet
        for i in range(nr):
            self.vz[i, :] = np.linspace(self.config.inlet_velocity, 0.1, nz + 1)
        
        # Pressure
        self.p = np.ones((nr, nz)) * self.config.pressure_outlet
        for j in range(nz):
            # Higher pressure at bottom
            self.p[:, j] = self.config.pressure_outlet * (1 + 0.1 * (nz - j) / nz)
        
        # Corrections
        self.p_prime = np.zeros((nr, nz))
        self.vr_star = np.zeros((nr + 1, nz))
        self.vz_star = np.zeros((nr, nz + 1))
        
        # Density and temperature / constant for now**
        self.rho = np.ones((nr, nz)) * self.fluid.density
        self.T = np.ones((nr, nz)) * 300.0
        
        print("Fields initialized successfully")
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions"""
        nr, nz = self.config.nr, self.config.nz
        
        # === INLET ===
        inlet_mask_vr = self.grid.r_faces < self.config.pipe_radius
        inlet_mask_vz = self.grid.r_centers < self.config.pipe_radius
        
        self.vr[inlet_mask_vr, -1] = 0.0
        self.vz[inlet_mask_vz, -1] = self.config.inlet_velocity
        
        # === SYMMETRY AXIS ===
        self.vr[0, :] = 0.0
        
        # === WALLS (no-slip) ===
        # Top wall (outside inlet)
        top_wall_vr = self.grid.r_faces >= self.config.pipe_radius
        top_wall_vz = self.grid.r_centers >= self.config.pipe_radius
        self.vr[top_wall_vr, -1] = 0.0
        self.vz[top_wall_vz, -1] = 0.0
        
        # Bottom (wafer + wall)
        self.vr[:, 0] = 0.0
        self.vz[:, 0] = 0.0
        
        # Outer wall
        self.vr[-1, :] = 0.0
        self.vz[-1, :] = 0.0
        
    def solve_momentum_explicit(self):
        """
        Simplified explicit momentum solver
        More stable than implicit for debugging
        """
        nr, nz = self.config.nr, self.config.nz
        mu = self.fluid.viscosity
        alpha_v = self.config.under_relaxation_v
        
        # Store old velocities
        vr_old = self.vr.copy()
        vz_old = self.vz.copy()
        
        # === RADIAL MOMENTUM ===
        for i in range(1, nr):
            for j in range(1, nz - 1):
                r = self.grid.r_faces[i]
                if r < 1e-10:  # Skip near axis
                    continue
                
                # Pressure gradient
                i_left = max(i-1, 0)
                i_right = min(i, nr-1)
                dp_dr = (self.p[i_right, j] - self.p[i_left, j]) / self.grid.dr[i-1]
                
                # Diffusion (Laplacian approximation)
                dvr_dr2 = 0.0
                if i > 1 and i < nr:
                    dvr_dr2 = (vr_old[i+1, j] - 2*vr_old[i, j] + vr_old[i-1, j]) / (self.grid.dr[i-1]**2)
                
                dvr_dz2 = (vr_old[i, j+1] - 2*vr_old[i, j] + vr_old[i, j-1]) / (self.grid.dz[j]**2)
                
                # Viscosity term
                viscous = mu * (dvr_dr2 + dvr_dz2)
                
                # Update
                dt_eff = alpha_v * min(self.grid.dr[i-1]**2, self.grid.dz[j]**2) / (4 * mu / self.rho[i_left, j])
                
                self.vr[i, j] = vr_old[i, j] + dt_eff * (viscous / self.rho[i_left, j] - dp_dr / self.rho[i_left, j])
        
        # === AXIAL MOMENTUM ===
        for i in range(1, nr - 1):
            for j in range(1, nz):
                # Pressure gradient
                dp_dz = (self.p[i, j] - self.p[i, j-1]) / self.grid.dz[j-1]
                
                # Simple diffusion
                dvz_dr2 = (vz_old[i+1, j] - 2*vz_old[i, j] + vz_old[i-1, j]) / (self.grid.dr[i]**2)
                
                dvz_dz2 = 0.0
                if j > 1 and j < nz:
                    dvz_dz2 = (vz_old[i, j+1] - 2*vz_old[i, j] + vz_old[i, j-1]) / (self.grid.dz[j-1]**2)
                
                # Viscous term
                viscous = mu * (dvz_dr2 + dvz_dz2)
                
                # Gravity
                gravity_force = -self.rho[i, j-1] * self.config.gravity
                
                # Update
                dt_eff = alpha_v * min(self.grid.dr[i]**2, self.grid.dz[j-1]**2) / (4 * mu / self.rho[i, j-1])
                
                self.vz[i, j] = vz_old[i, j] + dt_eff * (
                    viscous / self.rho[i, j-1] - dp_dz / self.rho[i, j-1] + gravity_force / self.rho[i, j-1]
                )
        
        # Clip velocities to prevent instability
        self.vr = np.clip(self.vr, -10*self.config.inlet_velocity, 10*self.config.inlet_velocity)
        self.vz = np.clip(self.vz, -10*self.config.inlet_velocity, 10*self.config.inlet_velocity)
        
    def solve_pressure_correction(self):
        """
        Pressure correction using Jacobi iteration
        """
        nr, nz = self.config.nr, self.config.nz
        alpha_p = self.config.under_relaxation_p
        
        # Reset pressure correction
        self.p_prime[:] = 0.0
        
        # Inner iterations for pressure
        for inner_iter in range(10):
            p_prime_old = self.p_prime.copy()
            
            for i in range(1, nr - 1):
                for j in range(1, nz - 1):
                    r = self.grid.r_centers[i]
                    if r < 1e-10:
                        continue
                    
                    # Compute mass imbalance (continuity residual)
                    # Radial flux: (1/r) * d(r*rho*vr)/dr
                    r_plus = self.grid.r_faces[i+1]
                    r_minus = self.grid.r_faces[i]
                    
                    flux_r_plus = r_plus * self.rho[i, j] * self.vr[i+1, j]
                    flux_r_minus = r_minus * self.rho[i, j] * self.vr[i, j]
                    div_r = (flux_r_plus - flux_r_minus) / (r * self.grid.dr[i])
                    
                    # Axial flux: d(rho*vz)/dz
                    flux_z_plus = self.rho[i, j] * self.vz[i, j+1]
                    flux_z_minus = self.rho[i, j] * self.vz[i, j]
                    div_z = (flux_z_plus - flux_z_minus) / self.grid.dz[j]
                    
                    mass_imbalance = div_r + div_z
                    
                    # Pressure correction (Jacobi-style)
                    coeff = 2.0 / (self.grid.dr[i]**2) + 2.0 / (self.grid.dz[j]**2)
                    
                    self.p_prime[i, j] = -alpha_p * mass_imbalance / (coeff + 1e-10)
            
            # Check inner convergence
            change = np.max(np.abs(self.p_prime - p_prime_old))
            if change < 1e-6:
                break
        
        # Update pressure
        self.p += self.p_prime
        
        # Ensure positive pressure
        self.p = np.maximum(self.p, 0.1 * self.config.pressure_outlet)
        
    def correct_velocities(self):
        """Correct velocities based on pressure correction"""
        nr, nz = self.config.nr, self.config.nz
        
        # Correct radial velocities
        for i in range(1, nr):
            for j in range(1, nz - 1):
                i_left = max(i-1, 0)
                i_right = min(i, nr-1)
                dp_prime_dr = (self.p_prime[i_right, j] - self.p_prime[i_left, j]) / self.grid.dr[i-1]
                
                # Simple correction
                self.vr[i, j] -= 0.1 * dp_prime_dr / (self.rho[i_left, j] + 1e-10)
        
        # Correct axial velocities
        for i in range(1, nr - 1):
            for j in range(1, nz):
                dp_prime_dz = (self.p_prime[i, j] - self.p_prime[i, j-1]) / self.grid.dz[j-1]
                
                self.vz[i, j] -= 0.1 * dp_prime_dz / (self.rho[i, j-1] + 1e-10)
        
    def compute_residuals(self) -> Tuple[float, float, float]:
        """Compute residuals"""
        nr, nz = self.config.nr, self.config.nz
        
        mass_res = 0.0
        count = 0
        
        for i in range(1, nr - 1):
            for j in range(1, nz - 1):
                r = self.grid.r_centers[i]
                if r < 1e-10:
                    continue
                
                # Divergence
                r_plus = self.grid.r_faces[i+1]
                r_minus = self.grid.r_faces[i]
                flux_r = (r_plus * self.vr[i+1, j] - r_minus * self.vr[i, j]) / (r * self.grid.dr[i])
                flux_z = (self.vz[i, j+1] - self.vz[i, j]) / self.grid.dz[j]
                
                div = flux_r + flux_z
                
                if not np.isnan(div) and not np.isinf(div):
                    mass_res += abs(div)
                    count += 1
        
        mass_res = mass_res / max(count, 1)
        
        # Check for NaNs in velocities
        if np.any(np.isnan(self.vr)) or np.any(np.isnan(self.vz)):
            print("WARNING: NaN detected in velocities!")
            return np.nan, np.nan, np.nan
        
        vr_res = np.mean(np.abs(self.vr))
        vz_res = np.mean(np.abs(self.vz))
        
        return mass_res, vr_res, vz_res
    
    def solve(self, verbose: bool = True) -> bool:
        """Main solution loop with stability checks"""
        
        print("\n" + "="*70)
        print("Starting iteration")
        print("="*70)
        
        for iteration in range(self.config.max_iterations):
            # 1. Apply boundary conditions
            self.apply_boundary_conditions()
            
            # 2. Solve momentum (explicit for stability)
            self.solve_momentum_explicit()
            
            # 3. Apply BCs again (important!)
            self.apply_boundary_conditions()
            
            # 4. Solve pressure correction
            self.solve_pressure_correction()
            
            # 5. Correct velocities
            self.correct_velocities()
            
            # 6. Apply BCs one more time
            self.apply_boundary_conditions()
            
            # 7. Check for NaNs
            if np.any(np.isnan(self.vr)) or np.any(np.isnan(self.vz)) or np.any(np.isnan(self.p)):
                print(f"\nERROR: NaN detected at iteration {iteration}")
                print(f"  vr range: [{np.nanmin(self.vr):.6f}, {np.nanmax(self.vr):.6f}]")
                print(f"  vz range: [{np.nanmin(self.vz):.6f}, {np.nanmax(self.vz):.6f}]")
                print(f"  p range:  [{np.nanmin(self.p):.6f}, {np.nanmax(self.p):.6f}]")
                return False
            
            # 8. Check convergence
            if iteration % 10 == 0:
                mass_res, vr_res, vz_res = self.compute_residuals()
                
                if verbose and iteration % 100 == 0:
                    print(f"Iteration {iteration:4d}: mass_res={mass_res:.6e}, "
                          f"vr_max={np.max(np.abs(self.vr)):.6f}, "
                          f"vz_max={np.max(np.abs(self.vz)):.6f}")
                
                if not np.isnan(mass_res) and mass_res < self.config.convergence_criteria:
                    print(f"\n✓ Converged after {iteration} iterations!")
                    print(f"  Final mass residual: {mass_res:.6e}")
                    return True
        
        print(f"\n✗ Did not converge after {self.config.max_iterations} iterations")
        return False
    
    def visualize(self):
        """Visualize the converged solution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Interpolate velocities to cell centers for plotting
        vr_center = 0.5 * (self.vr[:-1, :] + self.vr[1:, :])
        vz_center = 0.5 * (self.vz[:, :-1] + self.vz[:, 1:])
        v_mag = np.sqrt(vr_center**2 + vz_center**2)
        
        R, Z = np.meshgrid(self.grid.r_centers, self.grid.z_centers, indexing='ij')
        
        # Velocity magnitude
        im0 = axes[0, 0].contourf(R, Z, v_mag, levels=20, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude (m/s)')
        axes[0, 0].set_xlabel('Radius (m)')
        axes[0, 0].set_ylabel('Height (m)')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Streamlines
        axes[0, 1].streamplot(R.T, Z.T, vr_center.T, vz_center.T, density=2, color=v_mag.T, cmap='viridis')
        axes[0, 1].set_title('Streamlines')
        axes[0, 1].set_xlabel('Radius (m)')
        axes[0, 1].set_ylabel('Height (m)')
        
        # Pressure
        im2 = axes[1, 0].contourf(R, Z, self.p, levels=20, cmap='RdBu_r')
        axes[1, 0].set_title('Pressure (Pa)')
        axes[1, 0].set_xlabel('Radius (m)')
        axes[1, 0].set_ylabel('Height (m)')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Velocity profiles
        center_idx = 0
        axes[1, 1].plot(self.grid.z_centers, vz_center[center_idx, :], 'b-', label='Vz at centerline')
        axes[1, 1].set_title('Axial Velocity at Centerline')
        axes[1, 1].set_xlabel('Height z (m)')
        axes[1, 1].set_ylabel('Velocity (m/s)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('converged_solution.png', dpi=150, bbox_inches='tight')
        plt.show()