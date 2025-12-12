"""
Sensitivity Analysis Framework for LPCVD Model

Provides tools to:
1. Vary single parameters and observe effects
2. Run parameter sweeps (1D and 2D)
3. Calculate sensitivity metrics
4. Generate comparison plots
5. Export results for further analysis
"""

from model_code import *

@dataclass
class SensitivityResult:
    """
    Store results from a single simulation run in sensitivity analysis.
    
    Attributes:
        parameter_name: Name of parameter being varied
        parameter_value: Value of parameter for this run
        converged: Whether simulation converged
        final_residual: Final mass residual
        avg_deposition_rate: Average deposition rate (nm/min)
        min_deposition_rate: Minimum deposition rate (nm/min)
        max_deposition_rate: Maximum deposition rate (nm/min)
        non_uniformity: Non-uniformity percentage
        center_deposition: Deposition rate at wafer center (nm/min)
        edge_deposition: Deposition rate at wafer edge (nm/min)
        avg_temperature: Average temperature on wafer (K)
        wafer_temp_variation: Temperature variation across wafer (K)
        runtime: Simulation runtime (seconds)
    """
    parameter_name: str
    parameter_value: float
    converged: bool
    final_residual: float
    avg_deposition_rate: float
    min_deposition_rate: float
    max_deposition_rate: float
    non_uniformity: float
    center_deposition: float
    edge_deposition: float
    avg_temperature: float
    wafer_temp_variation: float
    runtime: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary for export."""
        return asdict(self)

class SensitivityAnalyzer:
    """
    Perform sensitivity analysis on LPCVD model parameters.
    
    Can vary:
    - Process conditions (temperature, pressure, composition)
    - Geometry (wafer radius, inlet dimensions)
    - Flow conditions (inlet velocity)
    - Solver parameters (under-relaxation factors)
    """
    
    def __init__(self, baseline_config: SimulationConfig):
        """
        Initialize sensitivity analyzer with baseline configuration.
        
        Args:
            baseline_config: Baseline SimulationConfig to perturb
        """
        self.baseline_config = baseline_config
        self.results: List[SensitivityResult] = []
        
    def run_single_case(self, param_name: str, param_value: float, 
                       verbose: bool = False) -> SensitivityResult:
        """
        Run single simulation with modified parameter.
        
        Args:
            param_name: Name of parameter to modify (e.g., 'T_wafer', 'inlet_velocity')
            param_value: New value for parameter
            verbose: Print solver progress
        
        Returns:
            result: SensitivityResult with metrics
        """
        import time
        
        # Create modified config
        config_dict = asdict(self.baseline_config)
        
        # Special handling for inlet_composition (tuple)
        if param_name == 'inlet_sih4_fraction':
            # Modify SiH4 fraction, keep H2 constant, adjust N2 to balance
            h2_frac = config_dict['inlet_composition'][2]
            sih4_frac = param_value
            n2_frac = 1.0 - sih4_frac - h2_frac
            
            # Validate composition
            if n2_frac < 0 or sih4_frac < 0 or h2_frac < 0:
                raise ValueError(f"Invalid composition: N2={n2_frac:.3f}, SiH4={sih4_frac:.3f}, H2={h2_frac:.3f}")
            if abs((n2_frac + sih4_frac + h2_frac) - 1.0) > 1e-6:
                raise ValueError(f"Composition does not sum to 1.0: sum={n2_frac + sih4_frac + h2_frac:.6f}")
            
            config_dict['inlet_composition'] = (n2_frac, sih4_frac, h2_frac)
            
        elif param_name == 'inlet_h2_fraction':
            # Modify H2 fraction, keep SiH4 constant, adjust N2 to balance
            sih4_frac = config_dict['inlet_composition'][1]
            h2_frac = param_value
            n2_frac = 1.0 - sih4_frac - h2_frac
            
            # Validate composition
            if n2_frac < 0 or sih4_frac < 0 or h2_frac < 0:
                raise ValueError(f"Invalid composition: N2={n2_frac:.3f}, SiH4={sih4_frac:.3f}, H2={h2_frac:.3f}")
            if abs((n2_frac + sih4_frac + h2_frac) - 1.0) > 1e-6:
                raise ValueError(f"Composition does not sum to 1.0: sum={n2_frac + sih4_frac + h2_frac:.6f}")
            
            config_dict['inlet_composition'] = (n2_frac, sih4_frac, h2_frac)
            
        else:
            # Direct parameter modification
            if param_name not in config_dict:
                raise ValueError(f"Parameter '{param_name}' not found in config")
            config_dict[param_name] = param_value
        
        # Create new config
        config = SimulationConfig(**config_dict)
        
        # VERIFICATION: Double-check composition sums to 1.0
        inlet_comp = config.inlet_composition
        comp_sum = sum(inlet_comp)
        if abs(comp_sum - 1.0) > 1e-6:
            raise ValueError(f"Configuration inlet_composition does not sum to 1.0: {inlet_comp}, sum={comp_sum:.6f}")
        
        # Print composition for verification
        if param_name in ['inlet_sih4_fraction', 'inlet_h2_fraction']:
            print(f"  Composition: N2={inlet_comp[0]:.4f}, SiH4={inlet_comp[1]:.4f}, H2={inlet_comp[2]:.4f}, Sum={comp_sum:.6f}")
        
        # Run simulation
        print(f"\nRunning: {param_name} = {param_value}")
        use_prev_run = True
        run_path = 'Saved Runs/run_20251210_155432'
        if use_prev_run:
            solver = load_run(run_path)
            solver.update_config(config)
        else:
            solver = CVDSolver(config)
        
        start_time = time.time()
        converged = solver.solve(verbose=verbose)
        runtime = time.time() - start_time
        
        # Extract metrics
        metrics = self.extract_metrics(converged, solver)
        
        # Create result
        result = SensitivityResult(
            parameter_name=param_name,
            parameter_value=param_value,
            runtime=runtime,
            **metrics
        )
        
        self.results.append(result)
        
        print(f"  Converged: {result.converged}")
        print(f"  Avg deposition: {result.avg_deposition_rate:.2f} nm/min")
        print(f"  Non-uniformity: {result.non_uniformity:.2f}%")
        print(f"  Runtime: {runtime:.1f} s")
        
        return result
    
    def parameter_sweep_1d(self, param_name: str, param_values: np.ndarray,
                          verbose: bool = False, save_results: bool = True,
                          output_dir: str = 'sensitivity_results') -> pd.DataFrame:
        """
        Perform 1D parameter sweep.
        
        Args:
            param_name: Parameter to vary
            param_values: Array of parameter values to test
            verbose: Print detailed solver output
            save_results: Save results to CSV
            output_dir: Directory for output files
        
        Returns:
            df: DataFrame with all results
        """
        print(f"\n{'='*70}")
        print(f"1D PARAMETER SWEEP: {param_name}")
        print(f"{'='*70}")
        print(f"Testing {len(param_values)} values from {param_values.min():.3e} to {param_values.max():.3e}")
        
        # Clear previous results
        self.results = []
        
        # Run each case
        for value in param_values:
            try:
                self.run_single_case(param_name, value, verbose=verbose)
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                # Create failed result
                self.results.append(SensitivityResult(
                    parameter_name=param_name,
                    parameter_value=value,
                    converged=False,
                    final_residual=np.inf,
                    avg_deposition_rate=0.0,
                    min_deposition_rate=0.0,
                    max_deposition_rate=0.0,
                    non_uniformity=0.0,
                    center_deposition=0.0,
                    edge_deposition=0.0,
                    avg_temperature=0.0,
                    wafer_temp_variation=0.0,
                ))
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sweep_1d_{param_name}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"\n✓ Results saved to: {filepath}")
        
        return df
    
    def plot_sensitivity_1d(self, df: pd.DataFrame, param_name: str,
                           metrics: List[str] = None, save_fig: bool = True):
        """
        Plot results from 1D parameter sweep.
        
        Args:
            df: DataFrame from parameter_sweep_1d
            param_name: Name of parameter that was varied
            metrics: List of metrics to plot (default: key metrics)
            save_fig: Save figure to file
        """
        if metrics is None:
            metrics = ['avg_deposition_rate', 'non_uniformity', 
                      'center_deposition', 'edge_deposition']
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            # Filter converged results
            df_conv = df[df['converged'] == True]
            
            if len(df_conv) == 0:
                ax.text(0.5, 0.5, 'No converged results', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot
            ax.plot(df_conv['parameter_value'], df_conv[metric], 'o-', linewidth=2, markersize=8)
            ax.set_xlabel(param_name, fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Mark baseline if available
            baseline_val = getattr(self.baseline_config, param_name, None)
            if baseline_val is not None and param_name in df_conv['parameter_name'].values:
                baseline_metric = df_conv[df_conv['parameter_value'] == baseline_val][metric]
                if len(baseline_metric) > 0:
                    ax.axvline(baseline_val, color='r', linestyle='--', alpha=0.5, label='Baseline')
                    ax.legend()
        
        plt.tight_layout()
        
        if save_fig:
            os.makedirs('sensitivity_results', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plot_1d_{param_name}_{timestamp}.png"
            plt.savefig(os.path.join('sensitivity_results', filename), dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: sensitivity_results/{filename}")
        
        plt.show()

    def extract_metrics(self, converged, solver: 'CVDSolver') -> Dict:
        """
        Extract key performance metrics from solved simulation.
        
        Args:
            solver: Converged CVDSolver instance
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        chemistry_solver = ChemistrySolver(solver.grid, solver.config)
        R_wafer = solver.config.wafer_radius
        
        # Collect deposition data across wafer
        dep_rates = []
        temperatures = []
        radii = []
        
        for i in range(solver.config.nr):
            r = solver.grid.r_centers[i]
            if r < R_wafer:
                T_surf = solver.T[i, 0]
                P_surf = solver.p[i, 0]
                x_SiH4 = solver.x[i, 0, 1]
                x_H2 = solver.x[i, 0, 2]
                
                _, _, G_nm_min = chemistry_solver.deposition_rate(
                    T_surf, P_surf, x_SiH4, x_H2
                )
                
                dep_rates.append(G_nm_min)
                temperatures.append(T_surf)
                radii.append(r * 1000)  # mm
        
        dep_rates = np.array(dep_rates)
        temperatures = np.array(temperatures)
        
        # Calculate metrics
        avg_rate = np.mean(dep_rates)
        min_rate = np.min(dep_rates)
        max_rate = np.max(dep_rates)
        non_uniformity = (max_rate - min_rate) / (2 * avg_rate) * 100
        
        # Center and edge deposition
        center_dep = dep_rates[0]  # At r=0
        edge_dep = dep_rates[-1]   # At r=R_wafer
        
        # Temperature statistics
        avg_temp = np.mean(temperatures)
        temp_variation = np.max(temperatures) - np.min(temperatures)
        
        return {
            'converged': converged,
            'final_residual': solver.mass_res[-1] if len(solver.mass_res) > 0 else np.inf,
            'avg_deposition_rate': avg_rate,
            'min_deposition_rate': min_rate,
            'max_deposition_rate': max_rate,
            'non_uniformity': non_uniformity,
            'center_deposition': center_dep,
            'edge_deposition': edge_dep,
            'avg_temperature': avg_temp,
            'wafer_temp_variation': temp_variation,
        }

class ComprehensiveSensitivitySweep:
    """
    Run comprehensive sensitivity analysis on all key parameters.
    
    Performs 1D sweeps on all important parameters and generates
    a complete sensitivity report with rankings and visualizations.
    """
    
    def __init__(self, baseline_config: SimulationConfig):
        """
        Initialize comprehensive sweep.
        
        Args:
            baseline_config: Baseline configuration
        """
        self.baseline_config = baseline_config
        self.analyzer = SensitivityAnalyzer(baseline_config)
        self.all_results = {}
        
    def define_parameter_ranges(self, sweep_range: str = 'moderate') -> Dict[str, np.ndarray]:
        """
        Define ranges for each parameter based on sweep intensity.
        
        Args:
            sweep_range: 'narrow' (±10%, 5 pts), 'moderate' (±20%, 7 pts), 
                        'wide' (±40%, 9 pts), or 'extreme' (±50%, 11 pts)
        
        Returns:
            param_ranges: Dictionary of parameter names to value arrays
        """
        ranges_config = {
            'narrow': (0.10, 5),    # ±10%, 5 points
            'moderate': (0.20, 7),  # ±20%, 7 points  
            'wide': (0.40, 9),      # ±40%, 9 points
            'extreme': (0.50, 11),  # ±50%, 11 points
        }
        
        if sweep_range not in ranges_config:
            raise ValueError(f"sweep_range must be one of {list(ranges_config.keys())}")
        
        delta_frac, n_points = ranges_config[sweep_range]
        
        param_ranges = {}
        
        # Temperature parameters
        T_wafer_base = self.baseline_config.T_wafer
        param_ranges['T_wafer'] = np.linspace(
            T_wafer_base * (1 - delta_frac),
            T_wafer_base * (1 + delta_frac),
            n_points
        )
        
        T_inlet_base = self.baseline_config.T_inlet
        param_ranges['T_inlet'] = np.linspace(
            max(T_inlet_base * (1 - delta_frac), 280),  # Don't go below 280K
            T_inlet_base * (1 + delta_frac),
            n_points
        )
        
        T_wall_base = self.baseline_config.T_wall
        param_ranges['T_wall'] = np.linspace(
            max(T_wall_base * (1 - delta_frac), 280),
            T_wall_base * (1 + delta_frac),
            n_points
        )
        
        # Pressure
        P_base = self.baseline_config.pressure_outlet
        param_ranges['pressure_outlet'] = np.linspace(
            max(P_base * (1 - delta_frac), 50),  # Don't go below 50 Pa
            P_base * (1 + delta_frac),
            n_points
        )
        
        # Inlet velocity
        v_base = self.baseline_config.inlet_velocity
        param_ranges['inlet_velocity'] = np.linspace(
            max(v_base * (1 - delta_frac), 0.1),  # Don't go below 0.1 m/s
            v_base * (1 + delta_frac),
            n_points
        )
        
        # Composition - SiH4 fraction
        sih4_base = self.baseline_config.inlet_composition[1]
        param_ranges['inlet_sih4_fraction'] = np.linspace(
            max(sih4_base * (1 - delta_frac), 0.02),  # Don't go below 2%
            min(sih4_base * (1 + delta_frac), 0.20),  # Don't exceed 20%
            n_points
        )
        
        # Composition - H2 fraction
        h2_base = self.baseline_config.inlet_composition[2]
        param_ranges['inlet_h2_fraction'] = np.linspace(
            max(h2_base * (1 - delta_frac), 0.0),
            min(h2_base * (1 + delta_frac), 0.10),  # Don't exceed 10%
            n_points
        )
        
        # Geometry - wafer radius (if you want to vary)
        r_wafer_base = self.baseline_config.wafer_radius
        param_ranges['wafer_radius'] = np.linspace(
            r_wafer_base * (1 - 0.2),  # ±20% for geometry
            r_wafer_base * (1 + 0.2),
            5  # Fewer points for geometry
        )
        
        # Geometry - pipe radius
        r_pipe_base = self.baseline_config.pipe_radius
        param_ranges['pipe_radius'] = np.linspace(
            r_pipe_base * (1 - 0.3),  # ±30% for geometry
            r_pipe_base * (1 + 0.3),
            5
        )
        
        return param_ranges
    
    def run_all_sweeps(self, sweep_range: str = 'moderate', 
                       verbose: bool = False,
                       skip_parameters: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Run 1D sweeps on all parameters.
        
        Args:
            sweep_range: Intensity of sweep ('narrow', 'moderate', 'wide', 'extreme')
            verbose: Print detailed solver output
            skip_parameters: List of parameter names to skip
        
        Returns:
            all_results: Dictionary mapping parameter names to result DataFrames
        """
        if skip_parameters is None:
            skip_parameters = []
        
        # Get parameter ranges
        param_ranges = self.define_parameter_ranges(sweep_range)
        
        # Filter out skipped parameters
        param_ranges = {k: v for k, v in param_ranges.items() if k not in skip_parameters}
        
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE SENSITIVITY ANALYSIS")
        print(f"{'='*70}")
        print(f"Sweep range: {sweep_range}")
        print(f"Parameters to sweep: {len(param_ranges)}")
        print(f"Total simulations: {sum(len(v) for v in param_ranges.values())}")
        print(f"{'='*70}\n")
        
        # Run each parameter sweep
        total_params = len(param_ranges)
        for idx, (param_name, param_values) in enumerate(param_ranges.items(), 1):
            print(f"\n[{idx}/{total_params}] Sweeping {param_name}...")
            print(f"  Range: {param_values.min():.4e} to {param_values.max():.4e}")
            print(f"  Points: {len(param_values)}")
            
            try:
                df = self.analyzer.parameter_sweep_1d(
                    param_name, 
                    param_values, 
                    verbose=verbose,
                    save_results=True
                )
                self.all_results[param_name] = df
                
                # Print quick summary
                df_conv = df[df['converged'] == True]
                if len(df_conv) > 0:
                    print(f"  ✓ Converged: {len(df_conv)}/{len(df)}")
                    print(f"  Deposition rate range: {df_conv['avg_deposition_rate'].min():.2f} - {df_conv['avg_deposition_rate'].max():.2f} nm/min")
                    print(f"  Best uniformity: {df_conv['non_uniformity'].min():.2f}%")
                else:
                    print(f"  ✗ No converged results")
                    
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return self.all_results
    
    def generate_sensitivity_report(self, output_dir: str = 'sensitivity_results'):
        """
        Generate comprehensive sensitivity report with rankings and plots.
        
        Creates:
        - Summary CSV with sensitivity metrics for each parameter
        - Individual plots for each parameter sweep
        - Combined comparison plot
        - Sensitivity ranking (which parameters matter most)
        
        Args:
            output_dir: Directory for output files
        """
        if not self.all_results:
            print("No results to report. Run run_all_sweeps() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*70}")
        print(f"GENERATING SENSITIVITY REPORT")
        print(f"{'='*70}\n")
        
        # ========== 1. CALCULATE SENSITIVITY METRICS ==========
        sensitivity_metrics = []
        
        for param_name, df in self.all_results.items():
            df_conv = df[df['converged'] == True]
            
            if len(df_conv) == 0:
                continue
            
            # Calculate variation metrics
            dep_rate_variation = (df_conv['avg_deposition_rate'].max() - 
                                 df_conv['avg_deposition_rate'].min())
            dep_rate_pct_change = (dep_rate_variation / 
                                  df_conv['avg_deposition_rate'].mean() * 100)
            
            uniformity_variation = (df_conv['non_uniformity'].max() - 
                                   df_conv['non_uniformity'].min())
            
            # Normalized sensitivity (how much output changes per unit input change)
            param_variation = (df_conv['parameter_value'].max() - 
                             df_conv['parameter_value'].min())
            param_mean = df_conv['parameter_value'].mean()
            param_pct_change = param_variation / param_mean * 100
            
            # Sensitivity = (% change in output) / (% change in input)
            deposition_sensitivity = dep_rate_pct_change / param_pct_change
            uniformity_sensitivity = uniformity_variation / param_pct_change
            
            sensitivity_metrics.append({
                'parameter': param_name,
                'converged_cases': len(df_conv),
                'total_cases': len(df),
                'dep_rate_min': df_conv['avg_deposition_rate'].min(),
                'dep_rate_max': df_conv['avg_deposition_rate'].max(),
                'dep_rate_variation': dep_rate_variation,
                'dep_rate_pct_change': dep_rate_pct_change,
                'uniformity_min': df_conv['non_uniformity'].min(),
                'uniformity_max': df_conv['non_uniformity'].max(),
                'uniformity_variation': uniformity_variation,
                'deposition_sensitivity': abs(deposition_sensitivity),
                'uniformity_sensitivity': abs(uniformity_sensitivity),
            })
        
        sensitivity_df = pd.DataFrame(sensitivity_metrics)
        
        # Sort by deposition sensitivity
        sensitivity_df = sensitivity_df.sort_values('deposition_sensitivity', ascending=False)
        
        # Save summary
        summary_file = os.path.join(output_dir, f'sensitivity_summary_{timestamp}.csv')
        sensitivity_df.to_csv(summary_file, index=False)
        print(f"✓ Saved sensitivity summary: {summary_file}\n")
        
        # ========== 2. PRINT RANKINGS ==========
        print("SENSITIVITY RANKINGS (Most to Least Influential):\n")
        print("Deposition Rate Sensitivity:")
        print("-" * 70)
        for idx, row in sensitivity_df.iterrows():
            print(f"  {row['parameter']:25s} : {row['deposition_sensitivity']:8.3f} "
                  f"(Δ = {row['dep_rate_pct_change']:6.2f}%)")
        
        print("\nUniformity Sensitivity:")
        print("-" * 70)
        sensitivity_df_uniform = sensitivity_df.sort_values('uniformity_sensitivity', ascending=False)
        for idx, row in sensitivity_df_uniform.iterrows():
            print(f"  {row['parameter']:25s} : {row['uniformity_sensitivity']:8.3f} "
                  f"(Δ = {row['uniformity_variation']:6.2f}%)")
        
        # ========== 3. GENERATE INDIVIDUAL PLOTS ==========
        print(f"\nGenerating individual parameter plots...")
        for param_name, df in self.all_results.items():
            try:
                self.analyzer.plot_sensitivity_1d(df, param_name, save_fig=True)
                print(f"  ✓ {param_name}")
            except Exception as e:
                print(f"  ✗ {param_name}: {str(e)}")
        
        # ========== 4. GENERATE COMPARISON PLOT ==========
        print(f"\nGenerating comparison plots...")
        self._plot_all_parameters_comparison(output_dir, timestamp)
        
        print(f"\n{'='*70}")
        print(f"REPORT COMPLETE")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")
        print(f"Summary file: sensitivity_summary_{timestamp}.csv")
    
    def _plot_all_parameters_comparison(self, output_dir: str, timestamp: str):
        """
        Create comparison plot showing all parameters normalized.
        
        Args:
            output_dir: Output directory
            timestamp: Timestamp string for filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['avg_deposition_rate', 'non_uniformity', 
                  'center_deposition', 'edge_deposition']
        metric_labels = ['Avg Deposition Rate (nm/min)', 'Non-uniformity (%)',
                        'Center Deposition (nm/min)', 'Edge Deposition (nm/min)']
        
        for ax, metric, label in zip(axes, metrics, metric_labels):
            for param_name, df in self.all_results.items():
                df_conv = df[df['converged'] == True]
                
                if len(df_conv) == 0:
                    continue
                
                # Normalize parameter values to 0-1 range
                param_vals = df_conv['parameter_value'].values
                param_norm = (param_vals - param_vals.min()) / (param_vals.max() - param_vals.min() + 1e-10)
                
                # Plot
                ax.plot(param_norm, df_conv[metric].values, 
                       'o-', label=param_name.replace('_', ' '), alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Normalized Parameter Value (0=min, 1=max)', fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'comparison_all_parameters_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot: {filename}")
        plt.close()
    
    def find_optimal_conditions(self, objective: str = 'maximize_deposition',
                               constraint: Dict[str, float] = None) -> pd.DataFrame:
        """
        Find optimal parameter combinations based on objective.
        
        Args:
            objective: 'maximize_deposition', 'minimize_uniformity', or 'balanced'
            constraint: Dict of constraints, e.g., {'non_uniformity': 5.0} (max 5%)
        
        Returns:
            optimal_df: DataFrame with top parameter combinations
        """
        if not self.all_results:
            print("No results available. Run run_all_sweeps() first.")
            return pd.DataFrame()
        
        # Combine all results
        all_data = []
        for param_name, df in self.all_results.items():
            df_conv = df[df['converged'] == True].copy()
            all_data.append(df_conv)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Apply constraints
        if constraint:
            for metric, limit in constraint.items():
                if metric in combined_df.columns:
                    if 'max' in metric.lower() or 'uniformity' in metric.lower():
                        combined_df = combined_df[combined_df[metric] <= limit]
                    else:
                        combined_df = combined_df[combined_df[metric] >= limit]
        
        # Sort by objective
        if objective == 'maximize_deposition':
            sorted_df = combined_df.sort_values('avg_deposition_rate', ascending=False)
        elif objective == 'minimize_uniformity':
            sorted_df = combined_df.sort_values('non_uniformity', ascending=True)
        elif objective == 'balanced':
            # Create composite score: high deposition, low uniformity
            combined_df['score'] = (combined_df['avg_deposition_rate'] / 
                                   combined_df['avg_deposition_rate'].max() - 
                                   combined_df['non_uniformity'] / 100)
            sorted_df = combined_df.sort_values('score', ascending=False)
        
        return sorted_df.head(10)

def run_comprehensive_sweep(sweep_range: str = 'moderate', 
                           skip_geometry: bool = True,
                           verbose: bool = False):
    """
    Convenience function to run complete sensitivity analysis.
    
    Args:
        sweep_range: 'narrow', 'moderate', 'wide', or 'extreme'
        skip_geometry: If True, skip wafer_radius and pipe_radius sweeps
        verbose: Print detailed solver output
    
    Returns:
        sweep: ComprehensiveSensitivitySweep instance with results
    """
    # Create baseline configuration
    config = SimulationConfig(
        nr=38, nz=53,
        T_wafer=1000,
        T_inlet=290,
        T_wall=290,
        inlet_velocity=0.5,
        pressure_outlet=133.0,
        inlet_composition=(0.9, 0.08, 0.02),
    )
    
    # Create sweep instance
    sweep = ComprehensiveSensitivitySweep(config)
    
    # Define parameters to skip
    skip_params = []
    if skip_geometry:
        skip_params.append('wafer_radius')
        skip_params.append('pipe_radius')
    
    # Run all sweeps
    results = sweep.run_all_sweeps(
        sweep_range=sweep_range,
        verbose=verbose,
        skip_parameters=skip_params
    )
    
    # Generate report
    sweep.generate_sensitivity_report()
    
    # Find optimal conditions
    print("\n" + "="*70)
    print("OPTIMAL CONDITIONS")
    print("="*70)
    
    print("\nMaximize Deposition (with uniformity < 10%):")
    optimal = sweep.find_optimal_conditions(
        objective='maximize_deposition',
        constraint={'non_uniformity': 10.0}
    )
    print(optimal[['parameter_name', 'parameter_value', 'avg_deposition_rate', 'non_uniformity']].head())
    
    print("\nMinimize Non-uniformity:")
    optimal = sweep.find_optimal_conditions(objective='minimize_uniformity')
    print(optimal[['parameter_name', 'parameter_value', 'avg_deposition_rate', 'non_uniformity']].head())
    
    return sweep