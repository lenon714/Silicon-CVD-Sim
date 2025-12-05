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
        
        # Visualize initial temperature
        # R, Z = np.meshgrid(self.grid.r_centers, self.grid.z_centers, indexing='ij')
        # plt.contourf(R, Z, self.T, levels=20, cmap='autumn')
        # plt.show()
        
        self.rho = np.ones((nr, nz)) * self.fluid.density
        
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
                T = self.T[i, j]
                
                # Update viscosity
                self.mu[i, j] = a0 + a1 * T + a2 * T**2
                # Update Thermal Conductivity
                self.tc[i, j] = b0 + b1 * T + b2 * T**2
                # Update Heat Capacity
                self.cp[i, j] = c0 + c1 * T + c2 * T**2
                
                # Under relaxation
                rho_new = self.p[i, j] * M_N2 / (R * T)
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
                
                r_e = self.grid.r_faces[i+1]
                r_w = self.grid.r_faces[i]
                dr = self.grid.dr[i]
                dz = self.grid.dz[j]
                
                # Face-interpolated densities
                rho_e = 0.5 * (self.rho[i, j] + self.rho[min(i+1, nr-1), j])
                rho_w = 0.5 * (self.rho[max(i-1, 0), j] + self.rho[i, j])
                rho_n = 0.5 * (self.rho[i, j] + self.rho[i, min(j+1, nz-1)])
                rho_s = 0.5 * (self.rho[i, max(j-1, 0)] + self.rho[i, j])
                
                # Mass fluxes
                m_e = rho_e * self.vr[i+1, j] * r_e * dz
                m_w = rho_w * self.vr[i, j] * r_w * dz
                m_n = rho_n * self.vz[i, j+1] * r * dr
                m_s = rho_s * self.vz[i, j] * r * dr
                
                # Mass imbalance for this cell
                mass_imb = m_e - m_w + m_n - m_s
                
                if not np.isnan(mass_imb) and not np.isinf(mass_imb):
                    mass_res += abs(mass_imb)
                    count += 1
        
        # Normalize by inlet mass flux for meaningful comparison
        # Calculate inlet mass flux
        inlet_mass_flux = 0.0
        for i in range(nr):
            r = self.grid.r_centers[i]
            if r < self.config.pipe_radius:
                dr = self.grid.dr[i]
                inlet_mass_flux += self.rho[i, -1] * abs(self.vz[i, -1]) * r * dr
        
        # Normalize residual
        if inlet_mass_flux > 1e-20:
            mass_res = mass_res / (count * inlet_mass_flux + 1e-20)
        else:
            mass_res = mass_res / max(count, 1)
        
        vr_res = np.mean(np.abs(self.vr))
        vz_res = np.mean(np.abs(self.vz))
        
        return mass_res, vr_res, vz_res
    
    def solve(self, verbose: bool = True) -> bool:
        """Main solution loop with staged coupling"""
        
        print("\n" + "="*70)
        print("Starting iteration with staged coupling")
        print("="*70)
        
        # Staged coupling thresholds
        ISOTHERMAL_ITERS = self.config.max_iterations // 25
        TEMP_ONLY_ITERS = self.config.max_iterations // 10  
        FULL_COUPLING_ITERS = self.config.max_iterations // 6
        
        best_mass_res = 1e10
        
        for iteration in range(self.config.max_iterations):
            
            # 1. Apply boundary conditions
            self.apply_boundary_conditions()
            
            # 2. Solve momentum
            self.solve_momentum()
            
            # 3. Apply BCs
            self.apply_boundary_conditions()
            
            # 4. Solve pressure correction
            self.solve_pressure_correction()
            
            # 5. Correct velocities
            self.correct_velocities()
            
            # 6. Apply BCs
            self.apply_boundary_conditions()
            
            # 7. Solve temperature (after isothermal phase)
            if iteration >= ISOTHERMAL_ITERS:
                self.solve_temperature()
            
            # 8. Update properties (after temperature stabilizes)
            if iteration >= FULL_COUPLING_ITERS:
                self.solve_properties()
            elif iteration >= TEMP_ONLY_ITERS and iteration % 20 == 0:
                # Gradual property introduction
                self.solve_properties()
            
            # Apply BCs after property update
            self.apply_boundary_conditions()
            
            # Check for NaNs
            if np.any(np.isnan(self.vr)) or np.any(np.isnan(self.vz)) or np.any(np.isnan(self.p)):
                print(f"\nERROR: NaN detected at iteration {iteration}")
                return False
            
            # Check convergence
            if iteration % 10 == 0:
                mass_res, vr_res, vz_res = self.compute_residuals()
                
                if mass_res < best_mass_res:
                    best_mass_res = mass_res
                
                if verbose and iteration % 50 == 0:
                    stage = "isothermal" if iteration < ISOTHERMAL_ITERS else \
                            "temp only" if iteration < TEMP_ONLY_ITERS else \
                            "gradual" if iteration < FULL_COUPLING_ITERS else "full"
                    print(f"Iter {iteration:4d} [{stage:10s}]: mass_res={mass_res:.4e}, "
                        f"rho=[{self.rho.min():.5f},{self.rho.max():.5f}], "
                        f"T=[{self.T.min():.0f},{self.T.max():.0f}]")
                    self.mass_res.append(mass_res)
                    self.iteration.append(iteration)
                
                if not np.isnan(mass_res) and mass_res < self.config.convergence_criteria:
                    print(f"\n✓ Converged after {iteration} iterations!")
                    print(f"  Final mass residual: {mass_res:.6e}")
                    return True
        
        print(f"\n✗ Did not converge after {self.config.max_iterations} iterations")
        print(f"  Best mass residual achieved: {best_mass_res:.6e}")
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

        # Temperature profiles
        im4 = axes[0, 2].contourf(R, Z, self.T, levels=20, cmap='autumn')
        axes[0, 2].set_title('Temperature Profile')
        plt.colorbar(im4, ax=axes[0, 2])

        # Density profiles
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