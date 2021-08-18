from src.system import System
from src.config import Config
from src.file_operations import FileOperations
from src.file_operations import FileOperationsREMD
from src.integrator import VelocityVerletIntegrator
from src.integrator import REMDIntegrator
from src.free_particle import FreeParticleSystem
from src.leonnard_jones import LJ
from src.analysis import Analysis
from src.nose_hoover import NoseHoover
from src.minimizer import Minimizer

import numpy as np
import os

class Ensemble:
    
    def __init__(self):
        self.sys = System()
        if Config.system == 'free_particle':
            self.sys = FreeParticleSystem()
        elif Config.system == 'LJ':
            self.sys = LJ()
        if Config.run_type == 'remd':
            self.file_io = FileOperationsREMD()
        else:
            self.file_io = FileOperations()
        
        self.stepper = VelocityVerletIntegrator()
        self.ensemble_type = Config.run_type
        self.num_steps = Config.num_steps

        self.nht = NoseHoover(self.stepper.dt)

        if Config.run_type == 'remd':
            self.remd_integrator = REMDIntegrator()

        if Config.run_type == 'minimize':
            self.minimizer = Minimizer(1e-2)
    
    def run_simulation(self):
        for step_no in range(self.num_steps):
            if self.ensemble_type == 'nve':
                x, v = self.stepper.step(self.sys, step_no)
            elif self.ensemble_type == 'nvt':
                KE = self.sys.K(self.sys.v)
                v = self.nht.step(KE, self.sys.v)
                x, v = self.stepper.step(self.sys, step_no, v = v)
                KE = self.sys.K(v)
                v = self.nht.step(KE, v)
            elif self.ensemble_type == 'minimize':
                x = self.minimizer.step(self.sys)
                v = self.sys.v
            elif self.ensemble_type == 'remd':
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                exchange = self.remd_integrator.step(self.sys.U(self.sys.x), step_no, self.file_io)
                KE = self.sys.K(self.sys.v)
                v = self.nht.step(KE, self.sys.v)
                x, v = self.stepper.step(self.sys, step_no, v = v)
                KE = self.sys.K(v)
                v = self.nht.step(KE, v)
            else:
                print("Use nve or nvt as run_type!")
                break
            pe = self.sys.U(x)
            self.sys.set_x(x)
            self.sys.set_v(v)
            ke = self.sys.K(v)
            temp = self.sys.instantaneous_T(v)
            self.file_io.write_vectors(x, v, step_no)
            self.file_io.write_scalars(ke, pe, temp, step_no)
            # if self.ensemble_type == 'nvt':
            surr_energy = self.nht.surr_energy()
            self.file_io.write_surrounding_energy(surr_energy, step_no)
        self.file_io.write_rst(x, v, self.sys.m)
        del(self.file_io)