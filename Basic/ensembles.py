from src.system import System
from src.config import Config
from src.file_operations import FileOperations
from src.integrator import VelocityVerletIntegrator
from src.andersen import Andersen
from src.analysis import Analysis
from src.nose_hoover import NoseHoover

import numpy as np
import os


def run_nve(cfg):
    sys = System(cfg)
    file_io = FileOperations(cfg)
    stepper = VelocityVerletIntegrator()

    for step_no in range(cfg.num_steps):
        x, v = stepper.velocity_verlet_step(sys.x, sys.v, sys.m)
        pe = sys.U(x)
        sys.set_x(x)
        sys.set_v(v)
        ke = sys.K(v)
        temp = sys.instantaneous_T(v)
        file_io.write(x, v, ke, pe, temp, step_no)
    del(file_io)

def run_nvt(cfg):
    sys = System(cfg)
    file_io = FileOperations(cfg)
    stepper = VelocityVerletIntegrator()
    nht = NoseHoover(stepper.dt, cfg)

    for step_no in range(cfg.num_steps):
        v = nht.step(sys.m, sys.v)
        x, v = stepper.velocity_verlet_step(sys.x, v, sys.m)
        v = nht.step(sys.m, v)
        pe = sys.U(x)
        sys.set_x(x)
        sys.set_v(v)
        ke = sys.K(v)
        temp = sys.instantaneous_T(v)
        file_io.write(x, v, ke, pe, temp, step_no)
    del(file_io)
        
