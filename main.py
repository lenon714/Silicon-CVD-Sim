"""Main script to run LPCVD Model"""

from model_code import *

def main():
    config = SimulationConfig(
        nr=38, nz=53,
        T_wafer=1000,
        T_inlet=290,
        T_wall=290,
        inlet_velocity=0.6,
        pressure_outlet=133.0,
        inlet_composition=(0.9, 0.08, 0.02),
    )

    CVDSolver(config).solve(verbose=True)
    # solver = load_run('Saved Runs/run_20251210_155432')
    # solver.update_config(config)
    # solver.solve()
    # run_comprehensive_sweep(sweep_range='moderate')

if __name__ == "__main__":
    main()