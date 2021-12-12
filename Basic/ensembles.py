from src.system import System
from src.config import Config
from src.file_operations import FileOperations
from src.file_operations import FileOperationsREMD
from src.file_operations import FileOperationsRENS
from src.integrator import VelocityVerletIntegrator
from src.integrator import REMDIntegrator
from src.integrator import RENSIntegrator
from src.free_particle import FreeParticleSystem
from src.leonnard_jones import LJ
from src.analysis import Analysis
from src.nose_hoover import NoseHoover
from src.minimizer import Minimizer

import numpy as np
import pandas as pd
import os

class Ensemble:
    
    def __init__(self):
        self.starting_step = 0
        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            self.starting_step = int(df['step'].to_numpy()[0]) + 1

        first_time = (self.starting_step == 0)
        if Config.run_type == 'remd':
            self.file_io = FileOperationsREMD(first_time = first_time)
        elif Config.run_type == 'rens':
            self.file_io = FileOperationsRENS()
        else:
            self.file_io = FileOperations(first_time = first_time)
        
        self.sys = System()
        if Config.system == 'free_particle':
            self.sys = FreeParticleSystem()
        elif Config.system == 'LJ':
            self.sys = LJ()
        
        self.stepper = VelocityVerletIntegrator()
        self.ensemble_type = Config.run_type
        self.num_steps = Config.num_steps

        self.nht = NoseHoover(self.stepper.dt)

        if Config.run_type == 'remd':
            self.remd_integrator = REMDIntegrator()
        
        if Config.run_type == 'rens':
            self.rens_integrator = RENSIntegrator(self.stepper.dt)


        if Config.run_type == 'minimize':
            self.minimizer = Minimizer(1e-2)
    
    def run_simulation(self):
        for step_no in range(self.starting_step, self.num_steps):
            if self.ensemble_type == 'nve':
                x, v = self.stepper.step(self.sys, step_no)
            elif self.ensemble_type == 'nvt':
                v = self.nht.step(self.sys.m, self.sys.v)
                x, v = self.stepper.step(self.sys, step_no, v = v)
                v = self.nht.step(self.sys.m, v)
            elif self.ensemble_type == 'minimize':
                x = self.minimizer.step(self.sys)
                v = self.sys.v
            elif self.ensemble_type == 'remd':
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()

                exchange = self.remd_integrator.step(self.sys.U(self.sys.x), step_no, self.file_io)
                v = self.nht.step(self.sys.m, self.sys.v)
                x, v = self.stepper.step(self.sys, step_no, v = v)
                v = self.nht.step(self.sys.m, v)
            elif self.ensemble_type == 'rens':
                # print(self.rens_integrator.mode)
                if self.rens_integrator.mode == 0:
                    v = self.nht.step(self.sys.m, self.sys.v)
                    x, v = self.stepper.step(self.sys, step_no, v = v)
                    v = self.nht.step(self.sys.m, v)
                    self.rens_integrator.attempt(self.sys, x, v)
                else:
                    x, v = self.rens_integrator.step(self.sys, step_no, self.file_io)
            else:
                print("Use nve or nvt as run_type!")
                break
            pe = self.sys.U(x)
            self.sys.set_x(x)
            self.sys.set_v(v)
            ke = self.sys.K(v)
            temp = self.sys.instantaneous_T(v)

            if Config.run_type == 'rens':
                self.file_io.write_vectors(x, v, step_no, self.rens_integrator.mode)
                
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
            else:
                self.file_io.write_vectors(x, v, step_no)
            self.file_io.write_scalars(ke, pe, temp, step_no)
            if self.ensemble_type == 'nvt' or self.ensemble_type == 'remd':
                univ_energy = self.nht.universe_energy(ke, pe)
                self.file_io.write_hprime(univ_energy, step_no)
        self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.nht.xi, self.nht.vxi, self.num_steps - 1)
        del(self.file_io)