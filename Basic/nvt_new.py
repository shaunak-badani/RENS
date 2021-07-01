from src.system import System
from src.config import Config
from src.file_operations_nose_hoover import FileOperations
from src.integrator import VelocityVerletIntegrator
from src.andersen import Andersen
from src.analysis import Analysis
from src.nose_hoover import NoseHoover

import numpy as np
import os

num_particles = 10
num_steps = int(1e3)
reduced_temperature = 2.0
run_name = os.path.splitext(__file__)[0]


if __name__ == "__main__":
    cfg = Config(num_particles, num_steps, reduced_temperature)
    sys = System(cfg)
    file_io = FileOperations(cfg, run_name)
    stepper = VelocityVerletIntegrator()
    nht = NoseHoover(stepper.dt, cfg)
    for step_no in range(num_steps):
        v = nht.step(sys.m, sys.v)
        x, v = stepper.velocity_verlet_step(sys.x, v, sys.m)
        v = nht.step(sys.m, v)
        pe = sys.U(x)
        sys.set_x(x)
        sys.set_v(v)
        ke = sys.K(v)
        temp = sys.instantaneous_T(v)
        h_prime = nht.h_prime(x, v, sys.m)
        file_io.write(x, v, ke, pe, h_prime, temp, step_no)
    del(file_io)
    print("Done. Analyzing .... \n")
    an = Analysis(cfg, run_name)
    # an.plot_energy()
    an.plot_temperature(cfg.reduced_temperature)
    an.plot_hprime()
    # print("Analysis Done. \n")
        
