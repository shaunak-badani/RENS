from src.system import System
from src.config import Config
from src.file_operations import FileOperations
from src.integrator import VelocityVerletIntegrator
from src.analysis import Analysis
from src.nose_hoover import NoseHoover

import numpy as np
import os

class Ensemble:
    
    def __init__(self, cfg):
        self.sys = System(cfg)
        self.file_io = FileOperations(cfg)
        self.stepper = VelocityVerletIntegrator()
        self.ensemble_type = cfg.run_type
        self.num_steps = cfg.num_steps

        if cfg.run_type == 'nvt':
            self.nht = NoseHoover(stepper.dt, cfg)
    
    def run_simulation(self):
        for step_no in range(self.num_steps):
            if self.ensemble_type == 'nve':
                x, v = self.stepper.velocity_verlet_step(self.sys.x, self.sys.v, self.sys.m)
            elif self.ensemble_type == 'nvt':
                v = self.nht.step(self.sys.m, self.sys.v)
                x, v = self.stepper.velocity_verlet_step(self.sys.x, v, self.sys.m)
                v = self.nht.step(self.sys.m, v)
            else:
                print("Use nve or nvt as run_type!")
                break
            pe = self.sys.U(x)
            self.sys.set_x(x)
            self.sys.set_v(v)
            ke = self.sys.K(v)
            temp = self.sys.instantaneous_T(v)
            self.file_io.write(x, v, ke, pe, temp, step_no)
        del(self.file_io)