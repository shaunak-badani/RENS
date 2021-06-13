from .system import System
import numpy as np

class VelocityVerletIntegrator:
    def __init__(self):
        self.dt = 1e-3

       

    def velocity_verlet_step(self, x, v, m):
        F = System.F(x)
        v_new = v + (self.dt / 2) * (F / m)
        x_new = x + (self.dt) * v_new
        F = System.F(x_new)
        v_new = v_new + (self.dt / 2) * (F / m)
        return x_new, v_new



