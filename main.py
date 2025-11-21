"""CHANGE NAME LATER"""
"""Stores existance of the simulation"""


"""
TO-DO
Start :)
Set up velocity solve - try before Tuesday
"""

import matplotlib.pyplot as plt

class Simulation:
    def __init__(self):
        self.time = 0.0
        self.dt = 0.01
        self.steps = 1000
        
        ## Constants
        self.density = 1.0
        self.cell_size = 1.0

        self.divergence = []
        self.pressure = []
        self.viscosity = []
        self.concentration = []
        self.temperature = []

    def solve_divergence(self):
        pass

    def solve_velocity(self):
        pass

def main():
    pass
    
if __name__ == "__main__":
    main()