"""Solver for LPCVD Model"""

from model_code import *

class CVDSolver:
    """Navier-Stokes Solver in Cylindrical Coordinates"""
    
    def __init__(self, config: SimulationConfig, fluid: FluidProperties):
        self.config = config
        self.fluid = fluid
        self.grid = StaggeredGrid(config)
        self._initialize_fields()
        self.mass_res = []
        self.iteration = []
        
    def _initialize_fields(self):
        """Initialize with physically reasonable values"""
        nr, nz = self.config.nr, self.config.nz
        
        # Velocities
        self.vr = np.zeros((nr + 1, nz))
        self.vz = np.zeros((nr, nz + 1))
        
        # # Initialize vz with linear profile from inlet to outlet
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
        
        # Density and temperature
        self.T = np.ones((nr, nz)) * self.config.T_inlet
        for i in range(nr):
            r = self.grid.r_centers[i]
            for j in range(nz):
                z_frac = self.grid.z_centers[j] / self.config.z_max
                # Hot near bottom (wafer), cold near top (inlet)
                if r < self.config.wafer_radius:
                    self.T[i, j] = self.config.T_wafer - (self.config.T_wafer - self.config.T_inlet) * z_frac
                else:
                    self.T[i, j] = self.config.T_wall

        M_N2 = 0.02802  # kg/mol
        R = 8.314  # J/(mol·K)
        self.rho = self.p * M_N2 / (R * self.T)
        
        self.mu = np.ones((nr, nz)) * self.fluid.viscosity
        self.tc = np.ones((nr, nz)) * self.fluid.tc
        self.cp = np.ones((nr, nz)) * 1040

        self.d_r = np.zeros((nr + 1, nz))
        self.d_z = np.zeros((nr, nz + 1))
        
        print("Fields initialized successfully")
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions"""
        velocity_bc = VelocityBoundaryConditions(self.grid, self.config)
        pressure_bc = PressureBoundaryConditions(self.grid, self.config)
        temperature_bc = TemperatureBoundaryConditions(self.grid, self.config)
        self.vr, self.vz = velocity_bc.apply(self.vr, self.vz, self.rho)
        # self.p = pressure_bc.apply(self.p)
        # self.T = temperature_bc.apply(self.T)
        
    def solve_momentum(self):
        """Momentum Solver"""
        momentum_solver = MomentumSolver(self.grid, self.fluid, self.config)
        self.vr, self.vz = momentum_solver.solve(
            self.vr, self.vz, self.p, self.rho, self.mu
        )
        self.d_r, self.d_z = momentum_solver.get_d_coefficients()
        
    def solve_pressure_correction(self):
        """Line TDMA with proper boundary conditions"""
        pressure_solver = PressureSolver(self.grid, self.config)
        self.p = pressure_solver.solve(
            self.vr, self.vz, self.p, self.p_prime, self.rho, self.d_z, self.d_r
        )
        
    def correct_velocities(self):
        """Correct velocities based on pressure correction"""
        nr, nz = self.config.nr, self.config.nz
        
        # Correct radial velocities
        for i in range(1, nr):
            for j in range(1, nz - 1):
                i_w = max(i-1, 0)
                i_e = min(i, nr-1)

                self.vr[i, j] += self.d_r[i,j] * (self.p_prime[i_w, j] - self.p_prime[i_e, j])
        
        # Correct axial velocities
        for i in range(1, nr - 1):
            for j in range(1, nz):          

                self.vz[i, j] += self.d_z[i,j] * (self.p_prime[i, j-1] - self.p_prime[i, j] )
        
    def solve_temperature(self):
        temperature_solver = TemperatureSolver(self.grid, self.config)
        self.T = temperature_solver.solve(
            self.vr, self.vz, self.rho, self.tc, self.T
        )

    def solve_properties(self):
        """Update fluid properties with under-relaxation for stability"""
        nr, nz = self.config.nr, self.config.nz
        
        # Nitrogen property coefficients
        a0, a1, a2 = 5.73e-6, 4.37e-8, -9.28e-12
        b0, b1, b2 = 8.54e-3, 6.23e-5, -4.34e-9
        c0, c1, c2 = 9.83e2, 1.58e-1, 1.69e-5
        M_N2 = 0.02802 # kg/mol
        R = 8.314
        
        alpha_rho = 0.2
        
        for i in range(nr):
            for j in range(nz):
                T_ij = self.T[i, j]
                
                # Update viscosity
                self.mu[i, j] = a0 + a1 * T_ij + a2 * T_ij**2
                # Update Thermal Conductivity
                self.tc[i, j] = b0 + b1 * T_ij + b2 * T_ij**2
                # Update Heat Capacity
                self.cp[i, j] = c0 + c1 * T_ij + c2 * T_ij**2
                
                # Under relaxation
                rho_new = self.p[i, j] * M_N2 / (R * T_ij)
                self.rho[i, j] = alpha_rho * rho_new + (1 - alpha_rho) * self.rho[i, j]

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
        
        # Initial Solve
        self.solve_properties()

        for iteration in range(self.config.max_iterations):
            # 1. Apply boundary conditions
            self.apply_boundary_conditions()
            
            # 2. Solve momentum
            self.solve_momentum()
            
            # 3. Apply BCs again
            self.apply_boundary_conditions()
            
            # 4. Solve pressure correction
            self.solve_pressure_correction()
            
            # 5. Correct velocities
            self.correct_velocities()
            
            # 6. Apply BCs one more time
            self.apply_boundary_conditions()

            # 7. Solve temperature
            self.solve_temperature()

            # 8. Solve remaining fluid properties
            # self.solve_properties()

            # self.apply_boundary_conditions()

            # Check for NaNs
            if np.any(np.isnan(self.vr)) or np.any(np.isnan(self.vz)) or np.any(np.isnan(self.p)):
                print(f"\nERROR: NaN detected at iteration {iteration}")
                print(f"  vr range: [{np.nanmin(self.vr):.6f}, {np.nanmax(self.vr):.6f}]")
                print(f"  vz range: [{np.nanmin(self.vz):.6f}, {np.nanmax(self.vz):.6f}]")
                print(f"  p range:  [{np.nanmin(self.p):.6f}, {np.nanmax(self.p):.6f}]")
                return False

            # 8. Check convergence
            if iteration % 5 == 0:
                mass_res, vr_res, vz_res = self.compute_residuals()
                
                if verbose and iteration % 50 == 0:
                    print(f"Iteration {iteration:4d}: mass_res={mass_res:.6e}, "
                          f"vr_max={np.max(np.abs(self.vr)):.6f}, "
                          f"vz_max={np.max(np.abs(self.vz)):.6f}")
                    self.mass_res.append(mass_res)
                    self.iteration.append(iteration)
                
                if not np.isnan(mass_res) and mass_res < self.config.convergence_criteria:
                    print(f"\n✓ Converged after {iteration} iterations!")
                    print(f"  Final mass residual: {mass_res:.6e}")
                    self.mass_res.append(mass_res)
                    self.iteration.append(iteration)
                    return True                
        
        print(f"\n✗ Did not converge after {self.config.max_iterations} iterations")
        return False
    
    def visualize(self):
        """Visualize the converged solution"""
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        
        # Interpolate velocities to cell centers for plotting
        vr_center = 0.5 * (self.vr[:-1, :] + self.vr[1:, :])
        vz_center = 0.5 * (self.vz[:, :-1] + self.vz[:, 1:])
        v_mag = np.sqrt(vr_center**2 + vz_center**2)
        
        R, Z = np.meshgrid(self.grid.r_centers, self.grid.z_centers, indexing='ij')
        
        # Velocity magnitude
        im0 = axes[0, 0].contourf(R, Z, v_mag, levels=30, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude (m/s)')
        axes[0, 0].set_xlabel('Radius (m)')
        axes[0, 0].set_ylabel('Height (m)')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Streamlines
        axes[0, 1].streamplot(R.T, Z.T, vr_center.T, vz_center.T, density=3, color=v_mag.T, cmap='viridis')
        axes[0, 1].set_title('Streamlines')
        axes[0, 1].set_xlabel('Radius (m)')
        axes[0, 1].set_ylabel('Height (m)')
        
        # Pressure
        im2 = axes[1, 0].contourf(R, Z, self.p, levels=30, cmap='plasma')
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

        im4 = axes[0, 2].contourf(R, Z, self.T, levels=20, cmap='autumn')
        axes[0, 2].set_title('Temperature Profile')
        plt.colorbar(im4, ax=axes[0, 2])

        im5 = axes[1, 2].contourf(R, Z, self.rho, levels=20, cmap='autumn')
        axes[1, 2].set_title('Density')
        plt.colorbar(im5, ax=axes[1, 2])

        axes[2, 2].plot(self.iteration, self.mass_res)
        axes[2, 2].set_title('Mass Residual at each iteration')
        axes[2, 2].set_xlabel('Iterations')
        axes[2, 2].set_ylabel('Mass Residual')

        plt.tight_layout()
        plt.savefig('Navier-Stokes_Solutions\converged_solution_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+ '.png', dpi=150, bbox_inches='tight')
        plt.show()