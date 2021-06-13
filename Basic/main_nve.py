from src.system import System
from src.config import Config
from src.file_operations import FileOperations
from src.integrator import VelocityVerletIntegrator
from src.andersen import Andersen
from src.analysis import Analysis

import numpy as np

num_particles = 10
num_steps = int(1e5)
reduced_temperature = 2.0
run_name = 'main_nve'


if __name__ == "__main__":
    cfg = Config(num_particles, num_steps, reduced_temperature)
    sys = System(cfg)
    file_io = FileOperations(cfg, run_name)
    stepper = VelocityVerletIntegrator()
    and_therm = Andersen(cfg, sys.m)
    for step_no in range(num_steps):
        x, v = stepper.velocity_verlet_step(sys.x, sys.v, sys.m)
        pe = sys.U(x)
        sys.set_x(x)
        sys.set_v(v)
        ke = sys.K(v)
        temp = sys.instantaneous_T(v)
        file_io.write(x, v, ke, pe, temp, step_no)
    del(file_io)
    print("Done. Analyzing .... \n")
    an = Analysis(cfg, run_name)
    an.plot_energy()
    an.plot_temperature()
    print("Analysis Done. \n")
        
