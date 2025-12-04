from model_code import *

class TemperatureSolver():
    def __init__(self, grid, config):
        self.grid = grid
        self.config = config
        self.nr = grid.nr
        self.nz = grid.nz

    def solve_temperature(self, vr, vz, rho, tc, T):
        nr, nz = self.nr, self.nz
        for outer_iter in range(20):
            T_old = T.copy() 
            for i in range(1, nr - 1):
                for j in range(1, nz - 1):

                    r = self.grid.r_centers[i]
                    if r < 1e-10:
                        continue
                    
                    # Grid spacings at this location
                    # v_r[i,j] sits between pressure cells (i-1,j) and (i,j)
                    dr_w = self.grid.dr[i-1] if i > 0 else self.grid.dr[0]
                    dr_e = self.grid.dr[i] if i < nr - 1 else self.grid.dr[-1]
                    dr_p = 0.5 * (dr_w + dr_e)
                    
                    dz_s = self.grid.dz[j-1] if j > 0 else self.grid.dz[0]
                    dz_n = self.grid.dz[j] if j < nz - 1 else self.grid.dz[-1]
                    dz_p = 0.5 * (dz_s + dz_n)
                    
                    # Face areas
                    r_e = r + 0.5 * dr_e
                    r_w = r - 0.5 * dr_w
                    A_e = r_e * dz_p
                    A_w = r_w * dz_p
                    A_n = r * dr_p
                    A_s = r * dr_p
                    
                    # === MASS FLUXES at CV faces ===
                    m_dot_e = rho[i, j] * vr[i+1, j] * A_e
                    m_dot_w = rho[i, j] * vr[i, j] * A_w
                    m_dot_n = rho[i, j] * vz[i, j+1] * A_n
                    m_dot_s = rho[i, j] * vz[i, j] * A_s

                    # === THERMAL CONDUCTIVITIES ===
                    tc_e = 0.5 * (tc[i, j] + tc[i+1, j])
                    tc_w = 0.5 * (tc[i-1, j] + tc[i, j])
                    tc_n = 0.5 * (tc[i, j] + tc[i, j+1])
                    tc_s = 0.5 * (tc[i, j-1] + tc[i, j])

                    # === DIFFUSION COEFFICIENTS ===
                    D_e = tc_e * A_e / dr_e
                    D_w = tc_w * A_w / dr_w
                    D_n = tc_n * A_n / dz_n
                    D_s = tc_s * A_s / dz_s
                    
                    # === NEIGHBOR COEFFICIENTS ===
                    a_E = D_e + max(-m_dot_e, 0)
                    a_W = D_w + max(m_dot_w, 0)
                    a_N = D_n + max(-m_dot_n, 0)
                    a_S = D_s + max(m_dot_s, 0)
                    a_P = a_E + a_W + a_N + a_S

                    T_new = (a_E * T[i+1, j] + a_W * T[i-1, j] +
                             a_N * T[i, j+1] + a_S * T[i, j-1]) / a_P
                    alpha_T = 0.7

                    T[i, j] = alpha_T * T_new + (1 - alpha_T) * T[i, j]

            T = TemperatureBoundaryConditions(self.grid, self.config).apply(T)

            change = np.max(np.abs(T - T_old))
            if change < 1e-6:
                break

        return T
    
    def solve(self, vr, vz, rho, tc, T):
        return self.solve_temperature(vr, vz, rho, tc, T)