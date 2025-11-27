"""Main script to run LPCVD Model"""

from model_code import *

def main():
    # Configuration with stable parameters
    config = SimulationConfig(
        nr=20,
        nz=30,
        inlet_velocity=0.5,
        under_relaxation_p=0.5,
        under_relaxation_v=0.3,
        max_iterations=5000
    )
    
    fluid = FluidProperties(
        density=0.15,
        viscosity=1.5e-5
    )
    
    print("="*70)
    print("LPCVD REACTOR SIMULATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Grid: {config.nr} × {config.nz}")
    print(f"  Inlet velocity: {config.inlet_velocity} m/s")
    print(f"  Pressure: {config.pressure_outlet} Pa ({config.pressure_outlet/133.322:.1f} torr)")
    print(f"  Under-relaxation: α_p={config.under_relaxation_p}, α_v={config.under_relaxation_v}")
    
    # Create and run solver
    solver = NavierStokesSolver(config, fluid)
    
    converged = solver.solve(verbose=True)
    
    if converged:
        print("\n" + "="*70)
        print("SUCCESS - Visualizing results")
        print("="*70)
        solver.visualize()
    else:
        print("\n" + "="*70)
        print("FAILED - Check parameters and try again")
        print("="*70)

if __name__ == "__main__":
    main()