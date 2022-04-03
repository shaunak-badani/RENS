import numpy as np
import pandas as pd
import math
from .units import Units
from .system import System
from .config import Config

class LangevinThermostat:

    def __init__(self, dt):
        self.friction = 0.05
        self.dt = dt
        self.damping = np.exp(-self.friction * dt)

    def step(self, x, v, force, m):
        kBT = Units.kB * Config.T()
        if not Units.arbitrary:
            kBT *= Units.kJ_mol_TO_J / (Units.AMU_TO_KG) * (Units.M_S_TO_A_PS)**2
        kBT_m = kBT / m
        sigma = np.sqrt(kBT_m * (1 - np.exp(-2 * self.friction * self.dt)))
        R = np.random.normal(size = v.shape)
        F = force(x)

        v = v + (self.dt / 2) * (F / m)
        x = x + (self.dt / 2) * v

        v = v * self.damping
        v = v + sigma * R

        x = x + (self.dt / 2) * v
        F = force(x)
        v = v + (self.dt / 2) * (F / m)
        return x, v
        