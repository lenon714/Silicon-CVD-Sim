"""All imports for LPCVP Model"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
from datetime import datetime

from model_code.simulation_properties import SimulationConfig, FluidProperties, StaggeredGrid
from model_code.boundary_conditions import VelocityBoundaryConditions, TemperatureBoundaryConditions, PressureBoundaryConditions
from model_code.momentum import MomentumSolver
from model_code.pressure import PressureSolver
from model_code.temperature import TemperatureSolver
from model_code.cvd_solver import CVDSolver