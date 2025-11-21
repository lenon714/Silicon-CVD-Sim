"""CHANGE NAME LATER"""
"""Stores existance of the simulation"""

"""
divergence = (uT - uB + uR - uL) / cell_size;

rhs = density * divergence / dt;
h2 = cell_size * cell_size;
pressure = (pB + pT + pL + pR - h2 * rhs) / num_neighbors
"""

import matplotlib.pyplot as plt

class Simulation:
    def __init__(self):
        self.time = 0.0
        self.dt = 0.01
        self.steps = 1000
        
        # Constants
        self.density = 1.0
        self.cell_size = 1.0
        
        # Physical properties of each cell
        
        # Velocity components at each edge midpoint
        # Horizontal means velocity moves in x direction
        # Vertical means velocity moves in y direction 
        self.horz_velocity = []
        self.vert_velocity = []
        
        # Other properties at each cell center
        # 2D arrays
        self.divergence = []
        self.pressure = []

def solve_divergence():
    pass

def solve_velocity():
    pass

def main():
    pass
    
if __name__ == "__main__":
    main()