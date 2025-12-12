"""Solver for LPCVD Model - Main simulation orchestrator"""

from model_code import *

class CVDSolver:
    """
    Main Navier-Stokes solver for LPCVD reactor in cylindrical coordinates.
    
    Implements SIMPLE algorithm for pressure-velocity coupling with:
    - Momentum equations for velocity field
    - Pressure correction for mass conservation
    - Energy equation for temperature field
    - Species transport with multicomponent diffusion
    - Surface chemistry for deposition kinetics
    
    Solves coupled multiphysics problem with staged integration approach
    for numerical stability under extreme property variations.
    """
    
    def __init__(self, config: SimulationConfig, use_prev_run: bool = False):
        """
        Initialize CVD solver with configuration and grid.
        
        Args:
            config: SimulationConfig object with all simulation parameters
        """            
        self.config = config
        self.grid = StaggeredGrid(config)

        if not use_prev_run:
            print("="*70)
            print("LPCVD REACTOR SIMULATION")
            print("="*70)
            print(f"\nConfiguration:")
            print(f"  Grid: {config.nr} × {config.nz}")
            print(f"  Inlet velocity: {config.inlet_velocity} m/s")
            print(f"  Pressure: {config.pressure_outlet} Pa ({config.pressure_outlet/133.322:.1f} torr)")
            print(f"  Under-relaxation: α_p={config.under_relaxation_p}, α_v={config.under_relaxation_v}")
        
            self._initialize_fields()
            self.mass_res = []
        self.iteration = []

        if use_prev_run:
            nr, nz = self.config.nr, self.config.nz
            # === PRESSURE CORRECTION FIELDS ===
            # Used in SIMPLE algorithm for pressure-velocity coupling
            self.p_prime = np.zeros((nr, nz))       # Pressure correction
            self.vr_star = np.zeros((nr + 1, nz))   # Intermediate radial velocity
            self.vz_star = np.zeros((nr, nz + 1))   # Intermediate axial velocity
        
    def _initialize_fields(self):
        """
        Initialize all field variables with physically reasonable values.
        
        Initializes:
        - Velocity fields (vr, vz) with approximate inlet profile
        - Pressure field with slight gradient from inlet to outlet
        - Temperature field with gradient from hot wafer to cold walls
        - Species concentrations (initially pure N2, then inlet composition)
        - Fluid properties (density, viscosity, thermal conductivity, cp)
        - Diffusion coefficients
        
        Good initialization critical for convergence of coupled problem.
        """
        nr, nz = self.config.nr, self.config.nz
        
        # === VELOCITY FIELDS ===
        # Stored on staggered grid: vr at radial faces, vz at axial faces
        self.vr = np.zeros((nr + 1, nz))
        self.vz = np.zeros((nr, nz + 1))
        
        # Initialize vz with linear profile from inlet to outlet
        for i in range(nr):
            self.vz[i, :] = np.linspace(0.0, -self.config.inlet_velocity, nz + 1)
        
        # === PRESSURE FIELD ===
        # Initialize with small gradient from bottom (outlet) to top
        self.p = np.ones((nr, nz)) * self.config.pressure_outlet
        for j in range(nz):
            # Higher pressure at bottom to drive flow toward outlet
            self.p[:, j] = self.config.pressure_outlet * (1 + 0.1 * (nz - j) / nz)
        
        # === PRESSURE CORRECTION FIELDS ===
        # Used in SIMPLE algorithm for pressure-velocity coupling
        self.p_prime = np.zeros((nr, nz))       # Pressure correction
        self.vr_star = np.zeros((nr + 1, nz))   # Intermediate radial velocity
        self.vz_star = np.zeros((nr, nz + 1))   # Intermediate axial velocity
        
        # === TEMPERATURE FIELD ===
        # Initialize with gradient from hot wafer to cold inlet/walls
        self.T = np.ones((nr, nz)) * self.config.T_inlet
        for i in range(nr):
            r = self.grid.r_centers[i]
            for j in range(nz):
                z_frac = self.grid.z_centers[j] / self.config.z_max
                # Hot near bottom (wafer), cold near top (inlet)
                if r < self.config.wafer_radius:
                    # Linear temperature gradient over wafer
                    self.T[i, j] = self.config.T_wafer - (self.config.T_wafer - self.config.T_inlet) * z_frac
                else:
                    # Outside wafer: cold wall temperature
                    self.T[i, j] = self.config.T_wall

        # === SPECIES MOLE FRACTIONS ===
        # Array shape (nr, nz, 3) for [N2, SiH4, H2]
        self.x = np.zeros((nr, nz, 3))
        self.x[:, :, 0] = self.config.n2
        self.x[:, :, 1] = self.config.sih4
        self.x[:, :, 2] = self.config.h2

        # === FLUID PROPERTIES ===
        # Initialize arrays for mixture properties
        self.rho = np.zeros((nr, nz))     # Density (kg/m³)
        self.mu = np.ones((nr, nz))       # Dynamic viscosity (Pa·s)
        self.tc = np.ones((nr, nz))       # Thermal conductivity (W/(m·K))
        self.cp = np.ones((nr, nz))       # Specific heat (J/(kg·K))
        
        # === SPECIES MASS FRACTIONS ===
        self.omega = self.mole_to_mass_fraction(self.x)
        
        # === DIFFUSION COEFFICIENTS ===
        # Effective multicomponent diffusion coefficients
        self.D_eff = np.zeros((nr, nz, 3))
        
        # === SURFACE CHEMISTRY VARIABLES ===
        self.v_stefan = np.zeros(nr)           # Stefan velocity at wafer (m/s)
        self.deposition_rates = np.zeros(nr)   # Deposition rate distribution (mol/(m²·s))

        # Calculate initial fluid properties from ideal gas law and correlations
        self.solve_properties()

        # === d-COEFFICIENTS FOR PRESSURE EQUATION ===
        # Stored from momentum solver for pressure correction
        self.d_r = np.zeros((nr + 1, nz))  # Radial d-coefficients
        self.d_z = np.zeros((nr, nz + 1))  # Axial d-coefficients
        
        print("Fields initialized successfully")
        
    def apply_boundary_conditions(self):
        """
        Apply all boundary conditions to velocity, pressure, and temperature fields.
        
        Called multiple times per iteration to enforce BCs after each sub-solver.
        Ensures consistency between field updates.
        """
        velocity_bc = VelocityBoundaryConditions(self.grid, self.config)
        pressure_bc = PressureBoundaryConditions(self.grid, self.config)
        temperature_bc = TemperatureBoundaryConditions(self.grid, self.config)
        
        # Apply velocity BCs with Stefan velocity from surface reaction
        self.vr, self.vz = velocity_bc.apply(self.vr, self.vz, self.rho, M_stefan=self.v_stefan*self.rho[:,0])
        
        # Apply pressure and temperature BCs
        self.p = pressure_bc.apply(self.p)
        self.T = temperature_bc.apply(self.T)
        
    def mass_to_mole_fraction(self, omega):
        """
        Convert species mass fractions to mole fractions.
        
        Formula: x_i = (ω_i/M_i) / Σ(ω_j/M_j)
        
        Args:
            omega: Mass fraction array (nr, nz, 3) or (3,)
        
        Returns:
            x: Mole fraction array with same shape as omega
        """
        masses = np.array(self.config.masses)
        # ω_i/M_i for each species
        molar = omega / masses
        # Sum along species axis
        molar_sum = np.sum(molar, axis=-1, keepdims=True)
        # Mole fraction
        x = molar / (molar_sum + 1e-30)
        return x
    
    def mole_to_mass_fraction(self, x):
        """
        Convert species mole fractions to mass fractions.
        
        Formula: ω_i = x_i * M_i / Σ(x_j * M_j)
        
        Args:
            x: Mole fraction array (nr, nz, 3) or (3,)
        
        Returns:
            omega: Mass fraction array with same shape as x
        """
        # Mean molecular mass: M_mix = Σ(x_j * M_j)
        m_mix = np.sum(x * np.array(self.config.masses), axis=-1, keepdims=True)
        # Mass fraction: ω_i = x_i * M_i / M_mix
        omega = x * self.config.masses / m_mix
        return omega

    def solve_momentum(self):
        """
        Solve momentum equations for velocity field.
        
        Uses MomentumSolver to compute vr and vz from current pressure,
        density, and viscosity fields. Returns d-coefficients needed
        for pressure correction equation.
        """
        momentum_solver = MomentumSolver(self.grid, self.config)
        self.vr, self.vz = momentum_solver.solve(
            self.vr, self.vz, self.p, self.rho, self.mu
        )
        # Store d-coefficients for pressure correction
        self.d_r, self.d_z = momentum_solver.get_d_coefficients()
        
    def solve_pressure_correction(self):
        """
        Solve pressure correction equation and update pressure field.
        
        Uses Line TDMA (alternating direction) method to solve the
        pressure correction equation derived from continuity constraint.
        Updates pressure: p = p + α_p * p'
        """
        pressure_solver = PressureSolver(self.grid, self.config)
        self.p = pressure_solver.solve(
            self.vr, self.vz, self.p, self.p_prime, self.rho, self.d_z, self.d_r
        )
        
    def correct_velocities(self):
        """
        Correct velocities based on pressure correction field.
        
        SIMPLE velocity correction: v = v* + d*(∇p')
        where v* is the intermediate velocity from momentum solver
        and d is the coefficient from momentum equation diagonal.
        """
        nr, nz = self.config.nr, self.config.nz
        
        # Correct radial velocities: v_r = v_r* + d_r*(p'_W - p'_E)
        for i in range(1, nr):
            for j in range(1, nz - 1):
                i_w = max(i-1, 0)   # West cell index
                i_e = min(i, nr-1)  # East cell index

                self.vr[i, j] += self.d_r[i,j] * (self.p_prime[i_w, j] - self.p_prime[i_e, j])
        
        # Correct axial velocities: v_z = v_z* + d_z*(p'_S - p'_N)
        for i in range(1, nr - 1):
            for j in range(1, nz):
                self.vz[i, j] += self.d_z[i,j] * (self.p_prime[i, j-1] - self.p_prime[i, j])
        
    def solve_temperature(self):
        """
        Solve energy equation for temperature field.
        
        Solves convection-diffusion equation for temperature using
        current velocity, density, and thermal conductivity fields.
        """
        temperature_solver = TemperatureSolver(self.grid, self.config)
        self.T = temperature_solver.solve(
            self.vr, self.vz, self.rho, self.tc, self.T
        )

    def solve_properties(self):
        """
        Update fluid properties from current state.
        
        Calculates mixture properties using:
        - Ideal gas law for density: ρ = PM/(RT)
        - Wilke's mixing rule for viscosity
        - Weighted average for thermal conductivity
        - Mass-weighted average for specific heat
        - Multicomponent diffusion coefficients
        
        Uses under-relaxation for density to prevent oscillations.
        """
        nr, nz = self.config.nr, self.config.nz
        mixture_solver = MixturePropertySolver(self.grid, self.config)
        self.rho, self.mu, self.tc, self.cp, self.D_eff = mixture_solver.solve(
            nr, nz, self.p, self.T, self.x, self.rho, self.mu, self.tc, self.cp, self.omega
        )
        
    def solve_diffusion(self):
        """
        Solve species transport equations with multicomponent diffusion.
        
        Solves convection-diffusion equations for each species including:
        - Ordinary diffusion (concentration gradients)
        - Thermal diffusion (Soret effect)
        - Convective transport
        
        Updates mass fractions omega and converts to mole fractions x.
        """
        diffusion_solver = DiffusionSolver(self.grid, self.config)
        self.omega = diffusion_solver.solve(
            self.vr, self.vz, self.rho, self.T, self.omega, self.p
        )
        self.x = self.mass_to_mole_fraction(self.omega)
    
    def solve_chemistry(self):
        """
        Apply surface chemistry boundary conditions at wafer.
        
        Calculates:
        - Species flux BCs from Langmuir-Hinshelwood kinetics
        - Stefan velocity from net mass flux
        - Deposition rate distribution
        
        Updates species concentrations at wafer surface and Stefan velocity
        for next momentum solve.
        """
        chemistry_solver = ChemistrySolver(self.grid, self.config)
        self.omega = self.mole_to_mass_fraction(self.x)
        self.omega, self.deposition_rates = chemistry_solver.solve(
            self.omega, self.x, self.T, self.p, self.rho, self.D_eff
        )
        self.x = self.mass_to_mole_fraction(self.omega)

        # Update Stefan velocity for next iteration's velocity BCs
        self.v_stefan = chemistry_solver.get_stefan_velocity(
            self.x, self.T, self.p, self.rho
        )

    def compute_residuals(self):
        """
        Compute residuals for convergence monitoring.
        
        Calculates:
        - Mass residual: cell-by-cell continuity equation imbalance
        - Velocity residuals: mean magnitudes of vr and vz
        
        Returns:
            mass_res: Normalized mass conservation residual (dimensionless)
            vr_res: Mean radial velocity magnitude (m/s)
            vz_res: Mean axial velocity magnitude (m/s)
        """
        nr, nz = self.config.nr, self.config.nz

        mass_res = 0.0
        count = 0
        
        # Loop over interior cells only
        for i in range(1, nr - 1):
            for j in range(1, nz - 1):
                r = self.grid.r_centers[i]
                if r < 1e-10:
                    continue  # Skip axis singularity
                
                # Grid information for this cell
                r_e = self.grid.r_faces[i+1]  # East face radius
                r_w = self.grid.r_faces[i]    # West face radius
                dr = self.grid.dr[i]
                dz = self.grid.dz[j]
                
                # Face-interpolated densities (arithmetic average)
                rho_e = 0.5 * (self.rho[i, j] + self.rho[min(i+1, nr-1), j])
                rho_w = 0.5 * (self.rho[max(i-1, 0), j] + self.rho[i, j])
                rho_n = 0.5 * (self.rho[i, j] + self.rho[i, min(j+1, nz-1)])
                rho_s = 0.5 * (self.rho[i, max(j-1, 0)] + self.rho[i, j])
                
                # Mass fluxes through each face (per radian in axisymmetric)
                # Positive = outflow from cell
                m_e = rho_e * self.vr[i+1, j] * r_e * dz
                m_w = rho_w * self.vr[i, j] * r_w * dz
                m_n = rho_n * self.vz[i, j+1] * r * dr
                m_s = rho_s * self.vz[i, j] * r * dr
                
                # Mass imbalance for this cell (should be zero if converged)
                # Continuity: m_e - m_w + m_n - m_s = 0
                mass_imb = m_e - m_w + m_n - m_s
                
                if not np.isnan(mass_imb) and not np.isinf(mass_imb):
                    mass_res += abs(mass_imb)
                    count += 1
        
        # Calculate inlet mass flux for normalization
        inlet_mass_flux = 0.0
        for i in range(nr):
            r = self.grid.r_centers[i]
            if r < self.config.pipe_radius:
                dr = self.grid.dr[i]
                # Mass flux through inlet (per radian)
                inlet_mass_flux += self.rho[i, -1] * abs(self.vz[i, -1]) * r * dr
        
        # Normalize residual by inlet flux and number of cells
        if inlet_mass_flux > 1e-20:
            mass_res = mass_res / (count * inlet_mass_flux + 1e-20)
        else:
            mass_res = mass_res / max(count, 1)
        
        # Velocity residuals (simple magnitude averages)
        vr_res = np.mean(np.abs(self.vr))
        vz_res = np.mean(np.abs(self.vz))
        
        return mass_res, vr_res, vz_res
        
    def solve(self, verbose: bool = True):
        """
        Main solution loop implementing SIMPLE algorithm
        
        Iterative sequence per iteration:
        1. Apply boundary conditions
        2. Solve momentum equations → v*
        3. Solve pressure correction → p'
        4. Correct velocities → v
        5. Solve temperature equation → T
        6. Update fluid properties → ρ, μ, k, cp, D
        7. Solve species transport → ω, x
        8. Apply surface chemistry → reaction BCs, Stefan velocity
        9. Check convergence
        
        Args:
            verbose: If True, print iteration progress and diagnostics
        
        Returns:
            bool: True if converged, False if max iterations reached
        """

        print("\n" + "="*70)
        print("Starting iteration")
        print("="*70)
        
        # Staged coupling threshold (currently not actively used)
        temp_iteration = 1
        
        best_mass_res = 1e10

        for iteration in range(self.config.max_iterations):
            
            # 1. Apply boundary conditions
            self.apply_boundary_conditions()
            
            # 2. Solve momentum equations for v*
            self.solve_momentum()
            
            # Apply BCs to v*
            self.apply_boundary_conditions()
            
            # 3. Solve pressure correction equation
            self.solve_pressure_correction()
            
            # 4. Correct velocities based on pressure correction
            self.correct_velocities()
            
            # Apply BCs to corrected velocities
            self.apply_boundary_conditions()
            
            # 5. Solve energy equation for temperature
            self.solve_temperature()

            # 6. Update mixture properties (ρ, μ, k, cp, D)
            self.solve_properties()
            
            # Apply BCs after property update
            self.apply_boundary_conditions()

            # 7. Solve species transport equations
            self.solve_diffusion()

            # 8. Apply surface chemistry boundary conditions
            self.solve_chemistry()

            # Check for NaNs (indicates divergence)
            if np.any(np.isnan(self.vr)) or np.any(np.isnan(self.vz)) or np.any(np.isnan(self.p)):
                print(f"\nERROR: NaN detected at iteration {iteration}")
                return False
            
            # Check convergence every 10 iterations
            if iteration % 10 == 0:
                mass_res, vr_res, vz_res = self.compute_residuals()
                
                # Track best residual achieved
                if mass_res < best_mass_res:
                    best_mass_res = mass_res
                
                # Print progress every 50 iterations
                if verbose and iteration % 50 == 0:
                    stage = 'Full' if iteration % temp_iteration == 0 else 'Isothermal'
                    print(f"Iter {iteration:4d} [{stage:10s}]: mass_res={mass_res:.4e}, "
                        f"rho=[{self.rho.min():.5f},{self.rho.max():.5f}], "
                        f"T=[{self.T.min():.0f},{self.T.max():.0f}]")
                    self.mass_res.append(mass_res)
                    self.iteration.append(iteration)
                    
                    # Optional: run diagnostics (currently commented out)
                    # diag.run_diagnostics(self)
                
                # Check convergence criterion
                if not np.isnan(mass_res) and mass_res < self.config.convergence_criteria:
                    print(f"\n✓ Converged after {iteration} iterations!")
                    print(f"  Final mass residual: {mass_res:.6e}")
                    self.report_deposition()
                    converged=True
                    break
                
        if converged and verbose:
            print("\n" + "="*70)
            print("SUCCESS - Visualizing results")
            print("="*70)
            self.visualize()
        elif not converged:
            # Maximum iterations reached without convergence
            print(f"\n✗ Did not converge after {self.config.max_iterations} iterations")
            print(f"  Best mass residual achieved: {best_mass_res:.6e}")
            print("\n" + "="*70)
            print("FAILED - Check parameters and try again")
            print("="*70)
            return False
        self.save_run()
        return True
    
    def report_deposition(self, simulation_time=60.0):
        """
        Report silicon deposition statistics on the wafer.
        
        Calculates and displays:
        - Deposition rate distribution (nm/min)
        - Average, min, max rates
        - Non-uniformity metric (semiconductor standard)
        - Projected thickness after simulation_time
        - Total silicon mass deposited
        - Radial profile of deposition and species concentrations
        
        Args:
            simulation_time: Time in seconds for thickness projection (default 60s)
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
                
                radii.append(r * 1000)  # Convert to mm for display
                dep_rates_mol.append(G_mol)
                dep_rates_nm_min.append(G_nm_min)
        
        radii = np.array(radii)
        dep_rates_nm_min = np.array(dep_rates_nm_min)
        dep_rates_mol = np.array(dep_rates_mol)
        
        # Calculate statistics
        avg_rate = np.mean(dep_rates_nm_min)
        min_rate = np.min(dep_rates_nm_min)
        max_rate = np.max(dep_rates_nm_min)
        
        # Non-uniformity: standard semiconductor metric
        # NU = (max - min) / (2 * avg) * 100%
        non_uniformity = (max_rate - min_rate) / (2 * avg_rate) * 100
        
        # Projected thickness after simulation_time
        thickness_nm = avg_rate * (simulation_time / 60.0)  # nm
        thickness_um = thickness_nm / 1000  # μm
        
        # Calculate total silicon mass deposited by integrating over wafer area
        # Total mass = ∫∫ G_mass * r dr dθ = 2π ∫ G_mass * r dr
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
        
        total_mass_time = total_mass * simulation_time  # kg deposited in simulation_time
        
        # Display results
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
        
        # Print representative radial points
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

    def visualize(self, filename=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        """
        Create comprehensive visualization of converged solution.
        
        Generates 3x3 subplot figure showing:
        - Velocity magnitude contours
        - Pressure field
        - Streamlines with velocity coloring
        - Temperature distribution
        - Density field
        - Mass residual convergence history
        - Species mole fractions (SiH4, H2, N2)
        
        Each plot includes reactor geometry annotations (inlet, wafer, outlet)
        for clear identification of boundary regions.
        """
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        
        # Interpolate staggered velocities to cell centers for plotting
        vr_center = 0.5 * (self.vr[:-1, :] + self.vr[1:, :])
        vz_center = 0.5 * (self.vz[:, :-1] + self.vz[:, 1:])
        v_mag = np.sqrt(vr_center**2 + vz_center**2)
        
        # Create meshgrid for contour plotting
        R, Z = np.meshgrid(self.grid.r_centers, self.grid.z_centers, indexing='ij')
        
        # Geometry parameters for annotations
        r_wafer = self.config.wafer_radius
        r_pipe = self.config.pipe_radius
        r_wall = self.config.r_max
        z_max = self.config.z_max
        
        def add_geometry_annotations(ax, show_legend=False):
            """Helper function to add reactor geometry markers to plots"""
            # Wafer surface (bottom, r < r_wafer)
            ax.axhline(y=0, xmin=0, xmax=r_wafer/r_wall, color='orange', 
                    linewidth=2.5, linestyle='-', label='Wafer')
            
            # Inlet pipe (top, r < r_pipe)
            ax.axhline(y=z_max, xmin=0, xmax=r_pipe/r_wall, color='cyan', 
                    linewidth=2.5, linestyle='-', label='Inlet')
            
            # Outlet (bottom, r > r_pipe) - annular region
            ax.axhline(y=z_max, xmin=r_pipe/r_wall, xmax=1.0, color='magenta', 
                    linewidth=2.5, linestyle='--', label='Outlet')
            
            if show_legend:
                ax.legend(loc='center left', fontsize=8, framealpha=0.9)
        
        # Plot 1: Velocity magnitude
        im0 = axes[0, 0].contourf(R, Z, v_mag, levels=30, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude (m/s)')
        axes[0, 0].set_xlabel('Radius (m)')
        axes[0, 0].set_ylabel('Height (m)')
        add_geometry_annotations(axes[0, 0], show_legend=False)
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Plot 2: Pressure field
        im1 = axes[0, 1].contourf(R, Z, self.p, levels=30, cmap='plasma')
        axes[0, 1].set_title('Pressure (Pa)')
        axes[0, 1].set_xlabel('Radius (m)')
        axes[0, 1].set_ylabel('Height (m)')
        add_geometry_annotations(axes[0, 1])
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Plot 3: Streamlines
        axes[0, 2].streamplot(R.T, Z.T, vr_center.T, vz_center.T, density=3, 
                              color=v_mag.T, cmap='viridis')
        axes[0, 2].set_title('Streamlines')
        axes[0, 2].set_xlabel('Radius (m)')
        axes[0, 2].set_ylabel('Height (m)')
        axes[0, 2].set_xlim(0, r_wall)
        axes[0, 2].set_ylim(0, z_max)
        add_geometry_annotations(axes[0, 2])
        
        # Plot 4: Temperature
        im2 = axes[1, 0].contourf(R, Z, self.T, levels=20, cmap='hot')
        axes[1, 0].set_title('Temperature (K)')
        axes[1, 0].set_xlabel('Radius (m)')
        axes[1, 0].set_ylabel('Height (m)')
        add_geometry_annotations(axes[1, 0])
        plt.colorbar(im2, ax=axes[1, 0])

        # Plot 5: Density
        im3 = axes[1, 1].contourf(R, Z, self.rho, levels=20, cmap='coolwarm')
        axes[1, 1].set_title('Density (kg/m³)')
        axes[1, 1].set_xlabel('Radius (m)')
        axes[1, 1].set_ylabel('Height (m)')
        add_geometry_annotations(axes[1, 1])
        plt.colorbar(im3, ax=axes[1, 1])

        # Plot 6: Convergence history
        axes[1, 2].semilogy(self.iteration, self.mass_res, 'b-', linewidth=1.5)
        axes[1, 2].set_title('Mass Residual Convergence')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Mass Residual')
        axes[1, 2].grid(True, alpha=0.3)

        # Plot 7: SiH4 mole fraction
        im4 = axes[2, 0].contourf(R, Z, self.x[:, :, 1], levels=20, cmap='Blues')
        axes[2, 0].set_title('SiH₄ Mole Fraction')
        axes[2, 0].set_xlabel('Radius (m)')
        axes[2, 0].set_ylabel('Height (m)')
        add_geometry_annotations(axes[2, 0])
        plt.colorbar(im4, ax=axes[2, 0])

        # Plot 8: H2 mole fraction
        im5 = axes[2, 1].contourf(R, Z, self.x[:, :, 2], levels=20, cmap='Greens')
        axes[2, 1].set_title('H₂ Mole Fraction')
        axes[2, 1].set_xlabel('Radius (m)')
        axes[2, 1].set_ylabel('Height (m)')
        add_geometry_annotations(axes[2, 1])
        plt.colorbar(im5, ax=axes[2, 1])

        # Plot 9: N2 mole fraction
        im6 = axes[2, 2].contourf(R, Z, self.x[:, :, 0], levels=20, cmap='Oranges')
        axes[2, 2].set_title('N₂ Mole Fraction')
        axes[2, 2].set_xlabel('Radius (m)')
        axes[2, 2].set_ylabel('Height (m)')
        add_geometry_annotations(axes[2, 2])
        plt.colorbar(im6, ax=axes[2, 2])

        plt.tight_layout()
        plt.savefig('Navier-Stokes_Solutions/converged_solution_' + filename + '.png', 
                    dpi=150, bbox_inches='tight')
        plt.show()

    def save_run(self, run_name=None, output_dir='Saved Runs'):
        """
        Save complete simulation state to CSV files for later analysis.
        
        Creates a directory with:
        - config.json: All simulation parameters
        - fields.csv: All field variables (vr, vz, p, T, rho, etc.)
        - convergence.csv: Convergence history
        - deposition.csv: Deposition rate profile
        - metadata.json: Run information (timestamp, convergence status)
        
        Args:
            run_name: Optional name for this run (default: timestamp)
            output_dir: Directory to save runs (created if doesn't exist)
        
        Returns:
            save_path: Path to saved run directory
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate run name if not provided
        if run_name is None:
            run_name = datetime.now().strftime("/run_%Y%m%d_%H%M%S")
        
        # Create subdirectory for this run
        save_path = os.path.join(output_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\nSaving run to: {save_path}")
        
        # ========== SAVE CONFIGURATION ==========
        config_dict = {
            'nr': self.config.nr,
            'nz': self.config.nz,
            'r_max': self.config.r_max,
            'z_max': self.config.z_max,
            'pipe_radius': self.config.pipe_radius,
            'wafer_radius': self.config.wafer_radius,
            'inlet_velocity': self.config.inlet_velocity,
            'pressure_outlet': self.config.pressure_outlet,
            'T_inlet': self.config.T_inlet,
            'T_wall': self.config.T_wall,
            'T_wafer': self.config.T_wafer,
            'inlet_composition': list(self.config.inlet_composition),
            'masses': list(self.config.masses),
            'max_iterations': self.config.max_iterations,
            'convergence_criteria': self.config.convergence_criteria,
            'under_relaxation_p': self.config.under_relaxation_p,
            'under_relaxation_v': self.config.under_relaxation_v,
            'gravity': self.config.gravity,
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # ========== SAVE GRID ==========
        grid_dict = {
            'r_centers': self.grid.r_centers.tolist(),
            'z_centers': self.grid.z_centers.tolist(),
            'r_faces': self.grid.r_faces.tolist(),
            'z_faces': self.grid.z_faces.tolist(),
            'dr': self.grid.dr.tolist(),
            'dz': self.grid.dz.tolist(),
        }
        
        with open(os.path.join(save_path, 'grid.json'), 'w') as f:
            json.dump(grid_dict, f, indent=2)
        
        # ========== SAVE FIELD VARIABLES ==========
        # Create a DataFrame with all field data
        # Flatten 2D/3D arrays with multi-index (i, j) or (i, j, species)
        nr, nz = self.config.nr, self.config.nz
        
        fields_data = []
        
        for i in range(nr):
            for j in range(nz):
                row = {
                    'i': i,
                    'j': j,
                    'r': self.grid.r_centers[i],
                    'z': self.grid.z_centers[j],
                    # Scalar fields at cell centers
                    'p': self.p[i, j],
                    'T': self.T[i, j],
                    'rho': self.rho[i, j],
                    'mu': self.mu[i, j],
                    'tc': self.tc[i, j],
                    'cp': self.cp[i, j],
                    # Species mole fractions
                    'x_N2': self.x[i, j, 0],
                    'x_SiH4': self.x[i, j, 1],
                    'x_H2': self.x[i, j, 2],
                    # Species mass fractions
                    'omega_N2': self.omega[i, j, 0],
                    'omega_SiH4': self.omega[i, j, 1],
                    'omega_H2': self.omega[i, j, 2],
                    # Diffusion coefficients
                    'D_eff_N2': self.D_eff[i, j, 0],
                    'D_eff_SiH4': self.D_eff[i, j, 1],
                    'D_eff_H2': self.D_eff[i, j, 2],
                }
                
                # Velocities (interpolated to cell centers for convenience)
                if i < nr and j < nz:
                    row['vz'] = self.vz[i, j]
                if i < nr and j < nz:
                    row['vr'] = 0.5 * (self.vr[i, j] + self.vr[i+1, j]) if i+1 <= nr else self.vr[i, j]
                
                fields_data.append(row)
        
        fields_df = pd.DataFrame(fields_data)
        fields_df.to_csv(os.path.join(save_path, 'fields.csv'), index=False)
        print(f"  ✓ Saved field variables ({len(fields_df)} cells)")
        
        # ========== SAVE CONVERGENCE HISTORY ==========
        if len(self.mass_res) > 0:
            convergence_df = pd.DataFrame({
                'iteration': self.iteration,
                'mass_residual': self.mass_res
            })
            convergence_df.to_csv(os.path.join(save_path, 'convergence.csv'), index=False)
            print(f"  ✓ Saved convergence history ({len(convergence_df)} points)")
        
        # ========== SAVE DEPOSITION PROFILE ==========
        chemistry_solver = ChemistrySolver(self.grid, self.config)
        R_wafer = self.config.wafer_radius
        
        deposition_data = []
        for i in range(nr):
            r = self.grid.r_centers[i]
            if r < R_wafer:
                T_surf = self.T[i, 0]
                P_surf = self.p[i, 0]
                x_SiH4 = self.x[i, 0, 1]
                x_H2 = self.x[i, 0, 2]
                
                G_mol, G_mass, G_nm_min = chemistry_solver.deposition_rate(
                    T_surf, P_surf, x_SiH4, x_H2
                )
                
                deposition_data.append({
                    'i': i,
                    'r_mm': r * 1000,
                    'T_surf': T_surf,
                    'P_surf': P_surf,
                    'x_SiH4': x_SiH4,
                    'x_H2': x_H2,
                    'deposition_rate_mol': G_mol,
                    'deposition_rate_mass': G_mass,
                    'deposition_rate_nm_min': G_nm_min,
                    'v_stefan': self.v_stefan[i]
                })
        
        if deposition_data:
            deposition_df = pd.DataFrame(deposition_data)
            deposition_df.to_csv(os.path.join(save_path, 'deposition.csv'), index=False)
            print(f"  ✓ Saved deposition profile ({len(deposition_df)} points)")
        
        # ========== SAVE METADATA ==========
        # metadata = {
        #     'run_name': run_name,
        #     'timestamp': datetime.now().isoformat(),
        #     'converged': len(self.mass_res) > 0 and self.mass_res[-1] < self.config.convergence_criteria,
        #     'final_mass_residual': float(self.mass_res[-1]) if len(self.mass_res) > 0 else None,
        #     'total_iterations': len(self.iteration),
        #     'grid_size': f"{nr} x {nz}",
        # }
        
        # with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        #     json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved metadata")
        print(f"\n✓ Run saved successfully to: {save_path}\n")
        
        return save_path

    def update_config(self, config):
        self.config = config
        print("="*70)
        print("LPCVD REACTOR SIMULATION")
        print("="*70)
        print(f"\nUpdated Configuration:")
        print(f"  Grid: {config.nr} × {config.nz}")
        print(f"  Inlet velocity: {config.inlet_velocity} m/s")
        print(f"  Pressure: {config.pressure_outlet} Pa ({config.pressure_outlet/133.322:.1f} torr)")
        print(f"  Under-relaxation: α_p={config.under_relaxation_p}, α_v={config.under_relaxation_v}")

def load_run(run_path, visualize_after_load=False):
    """
    Load a previously saved simulation run and optionally visualize it.
    
    Creates a new CVDSolver instance with the saved configuration and
    loads all field variables to restore the exact simulation state.
    
    Args:
        run_path: Path to saved run directory
        visualize_after_load: If True, generate visualization plot
    
    Returns:
        solver: CVDSolver instance with loaded state
    """
    print(f"\nLoading run from: {run_path}")
    
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    # ========== LOAD CONFIGURATION ==========
    config_path = os.path.join(run_path, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create SimulationConfig from loaded parameters
    config = SimulationConfig(
        nr=config_dict['nr'],
        nz=config_dict['nz'],
        r_max=config_dict['r_max'],
        z_max=config_dict['z_max'],
        pipe_radius=config_dict['pipe_radius'],
        wafer_radius=config_dict['wafer_radius'],
        inlet_velocity=config_dict['inlet_velocity'],
        pressure_outlet=config_dict['pressure_outlet'],
        T_inlet=config_dict['T_inlet'],
        T_wall=config_dict['T_wall'],
        T_wafer=config_dict['T_wafer'],
        inlet_composition=tuple(config_dict['inlet_composition']),
        masses=tuple(config_dict['masses']),
        max_iterations=config_dict['max_iterations'],
        convergence_criteria=config_dict['convergence_criteria'],
        under_relaxation_p=config_dict['under_relaxation_p'],
        under_relaxation_v=config_dict['under_relaxation_v'],
        gravity=config_dict['gravity'],
    )
    
    print(f"  ✓ Loaded configuration")
    
    # ========== CREATE SOLVER ==========
    solver = CVDSolver(config, use_prev_run=True)
    
    # ========== LOAD FIELD VARIABLES ==========
    fields_path = os.path.join(run_path, 'fields.csv')
    if not os.path.exists(fields_path):
        raise FileNotFoundError(f"Fields file not found: {fields_path}")
    
    fields_df = pd.read_csv(fields_path)
    
    nr, nz = config.nr, config.nz
    
    # Initialize arrays
    solver.p = np.zeros((nr, nz))
    solver.T = np.zeros((nr, nz))
    solver.rho = np.zeros((nr, nz))
    solver.mu = np.zeros((nr, nz))
    solver.tc = np.zeros((nr, nz))
    solver.cp = np.zeros((nr, nz))
    solver.x = np.zeros((nr, nz, 3))
    solver.omega = np.zeros((nr, nz, 3))
    solver.D_eff = np.zeros((nr, nz, 3))
    solver.vr = np.zeros((nr + 1, nz))
    solver.vz = np.zeros((nr, nz + 1))
    
    # Load data into arrays
    for _, row in fields_df.iterrows():
        i, j = int(row['i']), int(row['j'])
        
        # Scalar fields
        solver.p[i, j] = row['p']
        solver.T[i, j] = row['T']
        solver.rho[i, j] = row['rho']
        solver.mu[i, j] = row['mu']
        solver.tc[i, j] = row['tc']
        solver.cp[i, j] = row['cp']
        
        # Species mole fractions
        solver.x[i, j, 0] = row['x_N2']
        solver.x[i, j, 1] = row['x_SiH4']
        solver.x[i, j, 2] = row['x_H2']
        
        # Species mass fractions
        solver.omega[i, j, 0] = row['omega_N2']
        solver.omega[i, j, 1] = row['omega_SiH4']
        solver.omega[i, j, 2] = row['omega_H2']
        
        # Diffusion coefficients
        solver.D_eff[i, j, 0] = row['D_eff_N2']
        solver.D_eff[i, j, 1] = row['D_eff_SiH4']
        solver.D_eff[i, j, 2] = row['D_eff_H2']
        
        # Velocities (note: these are approximations from cell-center values)
        if 'vz' in row and not np.isnan(row['vz']):
            solver.vz[i, j] = row['vz']
        if 'vr' in row and not np.isnan(row['vr']):
            solver.vr[i, j] = row['vr']
    
    print(f"  ✓ Loaded field variables ({len(fields_df)} cells)")
    
    # ========== LOAD CONVERGENCE HISTORY ==========
    convergence_path = os.path.join(run_path, 'convergence.csv')
    if os.path.exists(convergence_path):
        convergence_df = pd.read_csv(convergence_path)
        solver.iteration = convergence_df['iteration'].tolist()
        solver.mass_res = convergence_df['mass_residual'].tolist()
        print(f"  ✓ Loaded convergence history ({len(convergence_df)} points)")
    
    # ========== LOAD DEPOSITION PROFILE ==========
    deposition_path = os.path.join(run_path, 'deposition.csv')
    if os.path.exists(deposition_path):
        deposition_df = pd.read_csv(deposition_path)
        
        # Reconstruct v_stefan array
        solver.v_stefan = np.zeros(nr)
        for _, row in deposition_df.iterrows():
            i = int(row['i'])
            solver.v_stefan[i] = row['v_stefan']
        
        print(f"  ✓ Loaded deposition profile ({len(deposition_df)} points)")
    
    # ========== LOAD METADATA ==========
    # metadata_path = os.path.join(run_path, 'metadata.json')
    # if os.path.exists(metadata_path):
    #     with open(metadata_path, 'r') as f:
    #         metadata = json.load(f)
        
    #     print(f"\n  Run Information:")
    #     print(f"    Name: {metadata.get('run_name', 'N/A')}")
    #     print(f"    Timestamp: {metadata.get('timestamp', 'N/A')}")
    #     print(f"    Converged: {metadata.get('converged', 'N/A')}")
    #     print(f"    Final residual: {metadata.get('final_mass_residual', 'N/A'):.4e}")
    #     print(f"    Total iterations: {metadata.get('total_iterations', 'N/A')}")
    
    print(f"\n✓ Run loaded successfully\n")
    
    # ========== VISUALIZE ==========
    if visualize_after_load:
        print("Generating visualization...")
        solver.visualize()
    
    return solver
