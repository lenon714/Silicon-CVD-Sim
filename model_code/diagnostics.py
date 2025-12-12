"""
Diagnostic Tools for SIMPLE Algorithm Debugging

Provides functions to check algorithm consistency, identify convergence issues,
and verify proper implementation of momentum and pressure equations.
"""

from model_code import *

def run_diagnostics(solver):
    """
    Run comprehensive diagnostics on SIMPLE algorithm implementation.
    
    Checks:
    1. Global mass conservation (inlet vs outlet)
    2. d-coefficient magnitudes and zeros
    3. Pressure correction field structure
    4. Velocity field structure and boundary conditions
    5. Cell-by-cell mass balance distribution
    6. Momentum equation discretization
    7. Pressure field structure
    8. Issues and recommendations
    
    Args:
        solver: CVDSolver instance with current solution state
    
    Returns:
        mass_imb: Array of mass imbalances per cell (nr, nz)
    """
    
    print("\n" + "="*70)
    print("SIMPLE ALGORITHM DIAGNOSTICS")
    print("="*70)
    
    nr, nz = solver.config.nr, solver.config.nz
    grid = solver.grid
    
    # ========================================
    # 1. CHECK GLOBAL MASS CONSERVATION
    # ========================================
    print("\n1. GLOBAL MASS BALANCE")
    print("-" * 40)
    
    # Calculate inlet mass flux (top boundary, inside pipe)
    # Integral: ∫ ρ·v·r dr over inlet area
    inlet_flux = 0.0
    for i in range(nr):
        r = grid.r_centers[i]
        if r < solver.config.pipe_radius:
            dr = grid.dr[i]
            # vz at top face (j = nz), negative value = inflow
            inlet_flux += solver.rho[i, -1] * (-solver.vz[i, -1]) * r * dr
    
    # Calculate outlet mass flux (bottom boundary, outside wafer)
    # Integral: ∫ ρ·v·r dr over outlet annular region
    outlet_flux = 0.0
    for i in range(nr):
        r = grid.r_centers[i]
        if r >= solver.config.wafer_radius:
            dr = grid.dr[i]
            # vz at bottom face (j = 0), negative value = outflow
            outlet_flux += solver.rho[i, 0] * (-solver.vz[i, 0]) * r * dr
    
    print(f"  Inlet mass flux:  {inlet_flux:.6e} kg/(s·rad)")
    print(f"  Outlet mass flux: {outlet_flux:.6e} kg/(s·rad)")
    print(f"  Imbalance:        {abs(inlet_flux - outlet_flux):.6e}")
    print(f"  Relative error:   {abs(inlet_flux - outlet_flux) / (abs(inlet_flux) + 1e-20):.2%}")
    
    # ========================================
    # 2. CHECK d-COEFFICIENT MAGNITUDES
    # ========================================
    print("\n2. d-COEFFICIENT CHECK")
    print("-" * 40)
    
    d_r, d_z = solver.d_r, solver.d_z
    
    # Check interior points only (avoid boundaries)
    d_r_int = d_r[1:-1, 1:-1]
    d_z_int = d_z[1:-1, 1:-1]
    
    print(f"  d_r range (interior): [{d_r_int.min():.4e}, {d_r_int.max():.4e}]")
    print(f"  d_z range (interior): [{d_z_int.min():.4e}, {d_z_int.max():.4e}]")
    print(f"  d_r zeros: {np.sum(d_r_int == 0)} / {d_r_int.size}")
    print(f"  d_z zeros: {np.sum(d_z_int == 0)} / {d_z_int.size}")
    
    # Check for very small d values (would make pressure correction weak)
    threshold = 1e-12
    print(f"  d_r < {threshold}: {np.sum(d_r_int < threshold)}")
    print(f"  d_z < {threshold}: {np.sum(d_z_int < threshold)}")
    
    # ========================================
    # 3. CHECK PRESSURE CORRECTION FIELD
    # ========================================
    print("\n3. PRESSURE CORRECTION CHECK")
    print("-" * 40)
    
    p_prime = solver.p_prime
    print(f"  p' range: [{p_prime.min():.4e}, {p_prime.max():.4e}]")
    print(f"  p' mean:  {p_prime.mean():.4e}")
    print(f"  p' at outlet (should be ~0): {p_prime[:, 0].mean():.4e}")
    
    # ========================================
    # 4. CHECK VELOCITY FIELD STRUCTURE
    # ========================================
    print("\n4. VELOCITY FIELD CHECK")
    print("-" * 40)
    
    vr, vz = solver.vr, solver.vz
    
    print(f"  vr range: [{vr.min():.4e}, {vr.max():.4e}]")
    print(f"  vz range: [{vz.min():.4e}, {vz.max():.4e}]")
    
    # Check inlet velocity matches specification
    inlet_vz = []
    for i in range(nr):
        r = grid.r_centers[i]
        if r < solver.config.pipe_radius:
            inlet_vz.append(solver.vz[i, -1])
    print(f"  Inlet vz (should be ~-{solver.config.inlet_velocity}): {np.mean(inlet_vz):.4f}")
    
    # Check axis symmetry (vr should be zero at r=0)
    print(f"  vr at axis (should be 0): {vr[0, :].max():.4e}")
    print(f"  vz gradient at axis: {np.mean(np.abs(vz[1,:] - vz[0,:])):.4e}")
    
    # ========================================
    # 5. CHECK CELL-BY-CELL MASS BALANCE
    # ========================================
    print("\n5. CELL MASS BALANCE DISTRIBUTION")
    print("-" * 40)
    
    # Calculate mass imbalance for each interior cell
    mass_imb = np.zeros((nr, nz))
    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            r = grid.r_centers[i]
            if r < 1e-10:
                continue
            
            r_e = grid.r_faces[i + 1]
            r_w = grid.r_faces[i]
            dr = grid.dr[i]
            dz = grid.dz[j]
            
            rho = solver.rho[i, j]
            
            # Mass fluxes through cell faces
            m_e = rho * vr[i + 1, j] * r_e * dz
            m_w = rho * vr[i, j] * r_w * dz
            m_n = rho * vz[i, j + 1] * r * dr
            m_s = rho * vz[i, j] * r * dr
            
            # Continuity residual
            mass_imb[i, j] = m_e - m_w + m_n - m_s
    
    mass_imb_int = mass_imb[1:-1, 1:-1]
    print(f"  Max |mass imbalance|: {np.abs(mass_imb_int).max():.4e}")
    print(f"  Mean |mass imbalance|: {np.abs(mass_imb_int).mean():.4e}")
    
    # Find worst cells for targeted debugging
    worst_idx = np.unravel_index(np.argmax(np.abs(mass_imb)), mass_imb.shape)
    print(f"  Worst cell: ({worst_idx[0]}, {worst_idx[1]})")
    print(f"  Worst imbalance: {mass_imb[worst_idx]:.4e}")
    
    # ========================================
    # 6. CHECK MOMENTUM EQUATION BALANCE
    # ========================================
    print("\n6. VERIFY MOMENTUM DISCRETIZATION (sample cell)")
    print("-" * 40)
    
    # Pick a cell away from boundaries for verification
    i_test, j_test = nr // 2, nz // 2
    
    print(f"  Test cell: ({i_test}, {j_test})")
    print(f"  r = {grid.r_centers[i_test]:.4f} m")
    print(f"  z = {grid.z_centers[j_test]:.4f} m")
    print(f"  vz at this location: {vz[i_test, j_test]:.4e}")
    print(f"  d_z at this location: {d_z[i_test, j_test]:.4e}")
    
    # ========================================
    # 7. CHECK PRESSURE FIELD
    # ========================================
    print("\n7. PRESSURE FIELD CHECK")
    print("-" * 40)
    
    p = solver.p
    print(f"  p range: [{p.min():.2f}, {p.max():.2f}] Pa")
    print(f"  p at outlet: {p[:, 0].mean():.2f} Pa (target: {solver.config.pressure_outlet})")
    print(f"  Pressure drop: {p[:, -1].mean() - p[:, 0].mean():.4f} Pa")
    
    # ========================================
    # 8. RECOMMENDATIONS
    # ========================================
    print("\n8. RECOMMENDATIONS")
    print("-" * 40)
    
    issues = []
    
    if np.sum(d_r_int == 0) > 0 or np.sum(d_z_int == 0) > 0:
        issues.append("- Zero d-coefficients found: check momentum solver a_P calculation")
    
    if abs(inlet_flux - outlet_flux) / (abs(inlet_flux) + 1e-20) > 0.1:
        issues.append("- Large global mass imbalance: check boundary conditions")
    
    if np.abs(vr[0, :]).max() > 1e-10:
        issues.append("- Non-zero vr at axis: symmetry BC not applied correctly")
    
    if d_r_int.max() < 1e-10 or d_z_int.max() < 1e-10:
        issues.append("- Very small d-coefficients: pressure correction will be weak")
    
    if abs(np.mean(inlet_vz) + solver.config.inlet_velocity) > 0.1 * solver.config.inlet_velocity:
        issues.append("- Inlet velocity incorrect: check inlet BC application")
    
    if len(issues) == 0:
        print("  No obvious issues detected. Problem may be subtle.")
    else:
        for issue in issues:
            print(issue)
    
    return mass_imb


def check_velocity_correction(solver):
    """
    Verify velocity correction is applied correctly in SIMPLE algorithm.
    
    Stores pre-correction velocities and manually computes what the
    correction should be based on pressure correction gradients and
    d-coefficients. Useful for debugging velocity update step.
    
    Args:
        solver: CVDSolver instance
    """
    print("\n" + "="*70)
    print("VELOCITY CORRECTION VERIFICATION")
    print("="*70)
    
    nr, nz = solver.config.nr, solver.config.nz
    
    # Store pre-correction velocities for comparison
    vr_before = solver.vr.copy()
    vz_before = solver.vz.copy()
    
    # Manually compute expected corrections at sample locations
    print("\nSample velocity corrections:")
    
    for i in [nr//4, nr//2, 3*nr//4]:
        for j in [nz//4, nz//2]:
            if i > 0 and i < nr and j > 0 and j < nz:
                # For vz: correction = d_z * (p'_south - p'_north)
                dp = solver.p_prime[i, j-1] - solver.p_prime[i, j]
                d = solver.d_z[i, j]
                expected_dvz = d * dp
                
                print(f"  Cell ({i},{j}): d_z={d:.4e}, dp={dp:.4e}, expected Δvz={expected_dvz:.4e}")


def check_pressure_equation_coefficients(solver):
    """
    Verify pressure equation coefficient computation for debugging.
    
    Extracts and displays coefficients (a_E, a_W, a_N, a_S, a_P) for
    a sample interior cell. Helps identify issues with pressure equation
    matrix construction.
    
    Args:
        solver: CVDSolver instance
    """
    print("\n" + "="*70)
    print("PRESSURE EQUATION COEFFICIENT CHECK")
    print("="*70)
    
    nr, nz = solver.config.nr, solver.config.nz
    grid = solver.grid
    
    # Sample cell at domain center
    i, j = nr // 2, nz // 2
    
    r = grid.r_centers[i]
    dr = grid.dr[i]
    dz = grid.dz[j]
    r_e = grid.r_faces[i + 1]
    r_w = grid.r_faces[i]
    
    # Face areas for axisymmetric formulation
    A_e = r_e * dz
    A_w = r_w * dz
    A_n = r * dr
    A_s = r * dr
    
    rho = solver.rho[i, j]
    
    # Compute pressure equation coefficients
    # a_nb = ρ * d * A (from SIMPLE formulation)
    a_E = rho * solver.d_r[i+1, j] * A_e
    a_W = rho * solver.d_r[i, j] * A_w
    a_N = rho * solver.d_z[i, j+1] * A_n
    a_S = rho * solver.d_z[i, j] * A_s
    a_P = a_E + a_W + a_N + a_S
    
    print(f"\nSample cell ({i}, {j}):")
    print(f"  Geometry: r={r:.4f}, dr={dr:.4f}, dz={dz:.4f}")
    print(f"  Areas: A_e={A_e:.6f}, A_w={A_w:.6f}, A_n={A_n:.6f}, A_s={A_s:.6f}")
    print(f"  d-coeffs: d_r[i+1]={solver.d_r[i+1,j]:.4e}, d_r[i]={solver.d_r[i,j]:.4e}")
    print(f"  d-coeffs: d_z[j+1]={solver.d_z[i,j+1]:.4e}, d_z[j]={solver.d_z[i,j]:.4e}")
    print(f"  Coefficients: a_E={a_E:.4e}, a_W={a_W:.4e}, a_N={a_N:.4e}, a_S={a_S:.4e}")
    print(f"  Sum a_P = {a_P:.4e}")
    
    # Check for common issues
    if a_P < 1e-15:
        print("  WARNING: a_P is essentially zero!")
    if a_E < 0 or a_W < 0 or a_N < 0 or a_S < 0:
            print("  WARNING: Negative coefficients detected!")