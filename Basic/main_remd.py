from src.system import System
from src.config import Config
from src.file_operations_remd import FileOperationsREMD
from src.integrator import VelocityVerletIntegrator
from src.andersen import Andersen
from src.analysis import Analysis
from src.nose_hoover import NoseHoover
from src.remd import REMD

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


num_particles = 10
num_steps = int(1e3)
exchange_period = int(50)
temperatures = [0.05, 0.3, 2.0, 4.0]
num_replicas = len(temperatures)
reduced_temperature = temperatures[rank]
run_name = 'main_remd'


if __name__ == "__main__":
    cfg = Config(num_particles, num_steps, reduced_temperature)
    sys = System(cfg)
    file_io = FileOperationsREMD(cfg, run_name)
    stepper = VelocityVerletIntegrator()
    nht = NoseHoover(stepper.dt, cfg)
    remd = REMD(cfg, num_replicas, exchange_period)
    for step_no in range(num_steps):
        x, v = stepper.velocity_verlet_step(sys.x, sys.v, sys.m)
        v = nht.step(sys.m, sys.v)
        remd.step(step_no, sys, cfg, file_io)
        v = remd.update_environment(cfg, nht, sys)
        pe = sys.U(x)
        sys.set_x(x)
        sys.set_v(v)
        ke = sys.K(v)
        temp = sys.instantaneous_T(v)
        file_io.write(x, v, ke, pe, temp, step_no)
    del(file_io)
    print("Done. Analyzing .... \n")
    # an = Analysis(cfg, run_name)
    # an.plot_energy()
    # an.plot_temperature()
    # print("Analysis Done. \n")
        
