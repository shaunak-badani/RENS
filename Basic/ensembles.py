from src.system import System
from src.harmonic_oscillator import HarmonicOscillator
from src.config import Config
from src.file_operations import FileOperations
from src.file_operations import FileOperationsREMD
from src.file_operations import FileOperationsRENS
from src.integrator import VelocityVerletIntegrator
from src.integrator import REMDIntegrator
from src.integrator import RENSIntegrator
from src.free_particle import FreeParticleSystem
from src.muller import MullerSystem
from src.leonnard_jones import LJ
from src.nose_hoover import NoseHoover
from src.minimizer import Minimizer

import numpy as np
import pandas as pd
import os

class Ensemble:
    
    def __init__(self):
        starting_step = 0
        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            starting_step = int(df['step'].to_numpy()[0]) + 1

        first_time = (starting_step == 0)
        if Config.run_type == 'remd':
            self.file_io = FileOperationsREMD(first_time = first_time)
        elif Config.run_type == 'rens':
            self.file_io = FileOperationsRENS()
        else:
            self.file_io = FileOperations(first_time = first_time)
        
        sys = System()
        if Config.system == 'free_particle':
            sys = FreeParticleSystem()
        elif Config.system == 'LJ':
            sys = LJ()
        elif Config.system == 'Harmonic':
            sys = HarmonicOscillator()
        elif Config.system == 'Muller':
            sys = MullerSystem()

        nsteps = Config.num_steps

        if Config.run_type == 'nve':
            self.ensemble = NVE_Ensemble(sys, nsteps, starting_step, first_time)

        if Config.run_type == 'nvt':
            self.ensemble = NVT_Ensemble(sys, nsteps, starting_step, first_time)

        if Config.run_type == 'minimize':
            self.ensemble = MinimizerEnsemble(sys, nsteps, starting_step, first_time)

        if Config.run_type == 'remd':
            self.ensemble = REMD_Ensemble(sys, nsteps, starting_step, first_time)

        if Config.run_type == 'rens':
            self.ensemble = RENS_Ensemble(sys, nsteps, starting_step, first_time)
        
    def run_simulation(self):
        self.ensemble.run_simulation()

class NVE_Ensemble:

    def __init__(self, sys, num_steps, starting_step, first_time):
        self.sys = sys

        self.stepper = VelocityVerletIntegrator()
        self.num_steps = num_steps
        self.starting_step = starting_step
        self.file_io = FileOperations(first_time = first_time)

    def run_simulation(self):
        for step_no in range(self.starting_step, self.num_steps):
            x, v = self.stepper.step(self.sys, step_no)

            pe = self.sys.U(x)
            self.sys.set_x(x)
            self.sys.set_v(v)
            ke = self.sys.K(v)
            temp = self.sys.instantaneous_T(v)
            self.file_io.write_vectors(x, v, step_no)
            self.file_io.write_scalars(ke, pe, temp, step_no)

        self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1)
        del(self.file_io)

class MinimizerEnsemble(NVE_Ensemble):

    def __init__(self, *args):
        super().__init__(*args)
        self.minimizer = Minimizer(1e-2)

    def run_simulation(self):
        for step_no in range(self.starting_step, self.num_steps):
            x = self.minimizer.step(self.sys)
            v = self.sys.v
            pe = self.sys.U(x)
            self.sys.set_x(x)
            self.sys.set_v(v)
            ke = self.sys.K(v)
            temp = self.sys.instantaneous_T(v)
            self.file_io.write_vectors(x, v, step_no)
            self.file_io.write_scalars(ke, pe, temp, step_no)

        self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1)
        del(self.file_io)
        

class NVT_Ensemble(NVE_Ensemble):

    def __init__(self, *args):
        super().__init__(*args)
        
        self.nht = NoseHoover(self.stepper.dt)

    def run_simulation(self):
        for step_no in range(self.starting_step, self.num_steps):
            v = self.nht.step(self.sys.m, self.sys.v)
            x, v = self.stepper.step(self.sys, step_no, v = v)
            v = self.nht.step(self.sys.m, v)

            pe = self.sys.U(x)
            self.sys.set_x(x)
            self.sys.set_v(v)
            ke = self.sys.K(v)
            temp = self.sys.instantaneous_T(v)
            self.file_io.write_vectors(x, v, step_no)
            self.file_io.write_scalars(ke, pe, temp, step_no)

            univ_energy = self.nht.universe_energy(ke, pe)
            self.file_io.write_hprime(univ_energy, step_no)

        self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1, xi = self.nht.xi, vxi = self.nht.vxi)
        del(self.file_io)

class REMD_Ensemble(NVT_Ensemble):

    def __init__(self, sys, num_steps, starting_step, first_time):
        self.sys = sys
        self.file_io = FileOperationsREMD(first_time = first_time)
        self.num_steps = num_steps
        self.starting_step = starting_step
        self.stepper = VelocityVerletIntegrator()
        self.nht = NoseHoover(self.stepper.dt)
        self.remd_integrator = REMDIntegrator()

    def run_simulation(self):
        for step_no in range(self.starting_step, self.num_steps):
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            v = self.sys.v
            if step_no % self.remd_integrator.exchange_period == 0:
                exchange, factor = self.remd_integrator.step(self.sys.U(self.sys.x), step_no, self.file_io)
                v = self.sys.v * factor
            v = self.nht.step(self.sys.m, v)
            x, v = self.stepper.step(self.sys, step_no, v = v)
            v = self.nht.step(self.sys.m, v)

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

            univ_energy = self.nht.universe_energy(ke, pe)
            self.file_io.write_hprime(univ_energy, step_no)

        self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1, xi = self.nht.xi, vxi = self.nht.vxi)
        del(self.file_io)

class RENS_Ensemble(REMD_Ensemble):

    def __init__(self, sys, num_steps, starting_step, first_time):
        self.sys = sys
        self.file_io = FileOperationsRENS(first_time = first_time)
        self.num_steps = num_steps
        self.starting_step = starting_step
        self.stepper = VelocityVerletIntegrator()
        self.nht = NoseHoover(self.stepper.dt)
        self.rens_integrator = RENSIntegrator(self.stepper.dt)

    def run_simulation(self):
        for step_no in range(self.starting_step, self.num_steps):
            if self.rens_integrator.mode == 0:
                v = self.nht.step(self.sys.m, self.sys.v)
                x, v = self.stepper.step(self.sys, step_no, v = v)
                v = self.nht.step(self.sys.m, v)
                self.rens_integrator.attempt(self.sys, x, v)
            else:
                x, v = self.rens_integrator.step(self.sys, step_no, self.file_io)

            pe = self.sys.U(x)
            self.sys.set_x(x)
            self.sys.set_v(v)
            ke = self.sys.K(v)
            temp = self.sys.instantaneous_T(v)
            self.file_io.write_vectors(x, v, step_no, self.rens_integrator.mode)
            self.file_io.write_scalars(ke, pe, temp, step_no)

            univ_energy = self.nht.universe_energy(ke, pe)
            self.file_io.write_hprime(univ_energy, step_no)

        self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1, xi = self.nht.xi, vxi = self.nht.vxi)
        # del(self.file_io)
        