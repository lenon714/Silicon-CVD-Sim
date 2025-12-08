"""Solver for LPCVD Model"""

from model_code import *

class CVDSolver:
    """Navier-Stokes Solver in Cylindrical Coordinates"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
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
            self.vz[i, :] = np.linspace(0.0, -self.config.inlet_velocity, nz + 1)
        
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

        self.x = np.zeros((nr, nz, 3))
        self.x[:, :, 0] = self.config.n2
        self.x[:, :, 1] = self.config.sih4
        self.x[:, :, 2] = self.config.h2

        self.rho = np.zeros((nr, nz))
        self.mu = np.ones((nr, nz))
        self.tc = np.ones((nr, nz))
        self.cp = np.ones((nr, nz))        
        
        self.omega = self.mole_to_mass_fraction(self.x)
        self.D_eff = np.zeros((nr, nz, 3))
        self.v_stefan = np.zeros(nr)
        self.deposition_rates = np.zeros(nr)

        self.solve_properties()

        self.d_r = np.zeros((nr + 1, nz))
        self.d_z = np.zeros((nr, nz + 1))
        
        print("Fields initialized successfully")
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions"""
        velocity_bc = VelocityBoundaryConditions(self.grid, self.config)
        pressure_bc = PressureBoundaryConditions(self.grid, self.config)
        temperature_bc = TemperatureBoundaryConditions(self.grid, self.config)
        self.vr, self.vz = velocity_bc.apply(self.vr, self.vz, self.rho, M_stefan=self.v_stefan*self.rho[:,0])
        self.p = pressure_bc.apply(self.p)
        self.T = temperature_bc.apply(self.T)
        
    def mass_to_mole_fraction(self, omega):
        """Convert mass fractions to mole fractions"""
        masses = np.array(self.config.masses)
        # ωi/mi for each species
        molar = omega / masses
        # Sum along species axis
        molar_sum = np.sum(molar, axis=-1, keepdims=True)
        # Mole fraction
        x = molar / (molar_sum + 1e-30)
        return x
    
    def mole_to_mass_fraction(self, x):
        """Convert mole fractions to mass fractions"""    
        # Mean molecular mass: m_mix = Σ(xj * mj)
        m_mix = np.sum(x * np.array(self.config.masses), axis=-1, keepdims=True)
        # Mass fraction: ωi = xi * mi / m_mix
        omega = x * self.config.masses / m_mix
        return omega

    def solve_momentum(self):
        """Momentum Solver"""
        momentum_solver = MomentumSolver(self.grid, self.config)
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
        mixture_solver = MixturePropertySolver(self.grid, self.config)
        self.rho, self.mu, self.tc, self.cp, self.D_eff = mixture_solver.solve(
            nr, nz, self.p, self.T, self.x, self.rho, self.mu, self.tc, self.cp, self.omega
        )
        
    def solve_diffusion(self):
        diffusion_solver = DiffusionSolver(self.grid, self.config)
        self.omega = diffusion_solver.solve(
            self.vr, self.vz, self.rho, self.T, self.omega, self.p
        )
        self.x = self.mass_to_mole_fraction(self.omega)
    
    def solve_chemistry(self):
        chemistry_solver = ChemistrySolver(self.grid, self.config)
        self.omega = self.mole_to_mass_fraction(self.x)
        self.omega, self.deposition_rates = chemistry_solver.solve(
            self.omega, self.x, self.T, self.p, self.rho, self.D_eff
        )
        self.x = self.mass_to_mole_fraction(self.omega)

        # Update Stefan velocity for next iteration
        self.v_stefan = chemistry_solver.get_stefan_velocity(
            self.x, self.T, self.p, self.rho
        )

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
        """Main solution loop"""

        print("\n" + "="*70)
        print("Starting iteration")
        print("="*70)
        
        # Staged coupling thresholds
        temp_iteration = 1
        
        best_mass_res = 1e10

        for iteration in range(self.config.max_iterations):
            
            # 1. Apply boundary conditions
            self.apply_boundary_conditions()
            
            # 2. Solve momentum
            self.solve_momentum()
            
            # Apply BCs
            self.apply_boundary_conditions()
            
            # 3. Solve pressure correction
            self.solve_pressure_correction()
            
            # 4. Correct velocities
            self.correct_velocities()
            
            # Apply BCs
            self.apply_boundary_conditions()
            
            # 5. Solve temperature
            self.solve_temperature()

            # 6. Update properties
            self.solve_properties()
            
            # Apply BCs
            self.apply_boundary_conditions()

            # 7. Solve Diffusion
            self.solve_diffusion()

            # 8. Solve Chemistry
            self.solve_chemistry()

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
                    stage = 'Full' if iteration % temp_iteration == 0 else 'Isothermal'
                    print(f"Iter {iteration:4d} [{stage:10s}]: mass_res={mass_res:.4e}, "
                        f"rho=[{self.rho.min():.5f},{self.rho.max():.5f}], "
                        f"T=[{self.T.min():.0f},{self.T.max():.0f}]")
                    self.mass_res.append(mass_res)
                    self.iteration.append(iteration)
                    
                    # diag.run_diagnostics(self)
                
                if not np.isnan(mass_res) and mass_res < self.config.convergence_criteria and iteration > 300:
                    print(f"\n✓ Converged after {iteration} iterations!")
                    print(f"  Final mass residual: {mass_res:.6e}")
                    self.report_deposition()
                    return True
                
        
        print(f"\n✗ Did not converge after {self.config.max_iterations} iterations")
        print(f"  Best mass residual achieved: {best_mass_res:.6e}")
        return False
    
    def report_deposition(self, simulation_time=60.0):
        """
        Report silicon deposition on the wafer.
        
        Args:
            simulation_time: Time in seconds to project growth (default 60s = 1 min)
        """
        print("\n" + "="*70)
        print("SILICON DEPOSITION REPORT")
        print("="*70)
        
        chemistry_solver = ChemistrySolver(self.grid, self.config)
        R_wafer = self.config.wafer_radius
        
        # Collect deposition data across wafer
        radii = []
        dep_rates_mol = []      # mol/(m²·s)
        dep_rates_nm_min = []   # nm/min
        
        for i in range(self.config.nr):
            r = self.grid.r_centers[i]
            if r < R_wafer:
                T_surf = self.T[i, 0]
                P_surf = self.p[i, 0]
                x_SiH4 = self.x[i, 0, 1]
                x_H2 = self.x[i, 0, 2]
                
                G_mol, G_mass, G_nm_min = chemistry_solver.deposition_rate(
                    T_surf, P_surf, x_SiH4, x_H2
                )
                
                radii.append(r * 1000)  # Convert to mm
                dep_rates_mol.append(G_mol)
                dep_rates_nm_min.append(G_nm_min)
        
        radii = np.array(radii)
        dep_rates_nm_min = np.array(dep_rates_nm_min)
        dep_rates_mol = np.array(dep_rates_mol)
        
        # Statistics
        avg_rate = np.mean(dep_rates_nm_min)
        min_rate = np.min(dep_rates_nm_min)
        max_rate = np.max(dep_rates_nm_min)
        
        # Non-uniformity (standard semiconductor metric)
        non_uniformity = (max_rate - min_rate) / (2 * avg_rate) * 100
        
        # Total thickness after simulation_time
        thickness_nm = avg_rate * (simulation_time / 60.0)  # nm
        thickness_um = thickness_nm / 1000  # μm
        
        # Total silicon mass deposited
        # Integrate over wafer area: ∫ G_mass * 2πr dr
        total_mass = 0.0
        M_Si = 0.02809  # kg/mol
        rho_Si = 2329.0  # kg/m³
        
        for i in range(len(radii) - 1):
            r1 = radii[i] / 1000  # Back to meters
            r2 = radii[i+1] / 1000
            r_avg = 0.5 * (r1 + r2)
            dr = r2 - r1
            G_avg = 0.5 * (dep_rates_mol[i] + dep_rates_mol[i+1])
            
            # Mass rate for this annulus: G * M_Si * 2πr * dr
            total_mass += G_avg * M_Si * 2 * np.pi * r_avg * dr
        
        total_mass_time = total_mass * simulation_time  # kg deposited
        
        print(f"\nProcess Conditions:")
        print(f"  Wafer temperature:  {self.config.T_wafer} K")
        print(f"  Total pressure:     {self.config.pressure_outlet} Pa ({self.config.pressure_outlet/133.3:.2f} torr)")
        print(f"  Inlet SiH4:         {self.config.inlet_composition[1]*100:.1f}%")
        
        print(f"\nDeposition Rate:")
        print(f"  Average:  {avg_rate:.2f} nm/min")
        print(f"  Minimum:  {min_rate:.2f} nm/min (at r = {radii[np.argmin(dep_rates_nm_min)]:.1f} mm)")
        print(f"  Maximum:  {max_rate:.2f} nm/min (at r = {radii[np.argmax(dep_rates_nm_min)]:.1f} mm)")
        
        print(f"\nUniformity:")
        print(f"  Non-uniformity:  {non_uniformity:.2f}%")
        print(f"  (max-min)/(2*avg) metric")
        
        print(f"\nProjected Growth (t = {simulation_time:.0f} s):")
        print(f"  Average thickness:  {thickness_nm:.1f} nm  ({thickness_um:.3f} μm)")
        print(f"  Total Si deposited: {total_mass_time*1e6:.4f} mg")
        
        print(f"\nRadial Profile:")
        print(f"  {'Radius (mm)':<12} {'Rate (nm/min)':<15} {'SiH4 (%)':<10} {'H2 (%)':<10}")
        print(f"  {'-'*47}")
        
        # Print every few points
        step = max(1, len(radii) // 10)
        for idx in range(0, len(radii), step):
            i = idx
            for j in range(self.config.nr):
                if abs(self.grid.r_centers[j]*1000 - radii[idx]) < 0.1:
                    i = j
                    break
            x_sih4_pct = self.x[i, 0, 1] * 100
            x_h2_pct = self.x[i, 0, 2] * 100
            print(f"  {radii[idx]:<12.1f} {dep_rates_nm_min[idx]:<15.2f} {x_sih4_pct:<10.2f} {x_h2_pct:<10.2f}")
        
        print("="*70)
        
        return {
            'radii_mm': radii,
            'deposition_rate_nm_min': dep_rates_nm_min,
            'average_rate': avg_rate,
            'non_uniformity': non_uniformity,
            'thickness_nm': thickness_nm
        }

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
        
        # Pressure
        im1 = axes[0, 1].contourf(R, Z, self.p, levels=30, cmap='plasma')
        axes[0, 1].set_title('Pressure (Pa)')
        axes[0, 1].set_xlabel('Radius (m)')
        axes[0, 1].set_ylabel('Height (m)')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Streamlines
        axes[0, 2].streamplot(R.T, Z.T, vr_center.T, vz_center.T, density=3, color=v_mag.T, cmap='viridis')
        axes[0, 2].set_title('Streamlines')
        axes[0, 2].set_xlabel('Radius (m)')
        axes[0, 2].set_ylabel('Height (m)')
        
        
        # # Velocity profiles
        # center_idx = 0
        # axes[1, 1].plot(self.grid.z_centers, vz_center[center_idx, :], 'b-', label='Vz at centerline')
        # axes[1, 1].set_title('Axial Velocity at Centerline')
        # axes[1, 1].set_xlabel('Height z (m)')
        # axes[1, 1].set_ylabel('Velocity (m/s)')
        # axes[1, 1].grid(True)
        # axes[1, 1].legend()

        # Temperature profiles
        im2 = axes[1, 0].contourf(R, Z, self.T, levels=20, cmap='autumn')
        axes[1, 0].set_title('Temperature Profile')
        plt.colorbar(im2, ax=axes[1, 0])

        # Density profiles
        im3 = axes[1, 1].contourf(R, Z, self.rho, levels=20, cmap='autumn')
        axes[1, 1].set_title('Density')
        plt.colorbar(im3, ax=axes[1, 1])

        # Mass residual
        axes[1, 2].plot(self.iteration, self.mass_res)
        axes[1, 2].set_title('Mass Residual at each iteration')
        axes[1, 2].set_xlabel('Iterations')
        axes[1, 2].set_ylabel('Mass Residual')

        # SiH4 mole fraction
        im4 = axes[2, 0].contourf(R, Z, self.x[:, :, 1], levels=20, cmap='Blues')
        axes[2, 0].set_title('SiH₄ Mole Fraction')
        axes[2, 0].set_xlabel('Radius (m)')
        axes[2, 0].set_ylabel('Height (m)')
        plt.colorbar(im4, ax=axes[2, 0])

        # H2 mole fraction
        im5 = axes[2, 1].contourf(R, Z, self.x[:, :, 2], levels=20, cmap='Greens')
        axes[2, 1].set_title('H₂ Mole Fraction')
        axes[2, 1].set_xlabel('Radius (m)')
        axes[2, 1].set_ylabel('Height (m)')
        plt.colorbar(im5, ax=axes[2, 1])

        # N2 mole fraction
        im6 = axes[2, 2].contourf(R, Z, self.x[:, :, 0], levels=20, cmap='Reds')
        axes[2, 2].set_title('N₂ Mole Fraction')
        axes[2, 2].set_xlabel('Radius (m)')
        axes[2, 2].set_ylabel('Height (m)')
        plt.colorbar(im6, ax=axes[2, 2])

        plt.tight_layout()
        plt.savefig('Navier-Stokes_Solutions\converged_solution_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+ '.png', dpi=150, bbox_inches='tight')
        plt.show()