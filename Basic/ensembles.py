from src.system import System
from src.test_sys import TestSystem

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
from src.muller_mod import MullerMod
from src.leps import LEPS_I
from src.leps import LEPS_II

from src.leonnard_jones import LJ
from src.nose_hoover import NoseHoover
from src.langevin import LangevinThermostat
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

        if Config.system == '1D_Leach':
            sys = System()
        if Config.system == 'test':
            sys = TestSystem()
        if Config.system == 'FreeParticle':
            sys = FreeParticleSystem()
        elif Config.system == 'LJ':
            sys = LJ()
        elif Config.system == 'Harmonic':
            sys = HarmonicOscillator()
        elif Config.system == 'Muller':
            sys = MullerSystem()
        elif Config.system == 'MullerMod':
            sys = MullerMod()
        elif Config.system == 'LEPS_I':
            sys = LEPS_I()
        elif Config.system == 'LEPS_II':
            sys = LEPS_II()


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
            if Config.system == 'LJ':
                L = self.sys.L
                x = L * (x <= 0) - L * (x >= L) + x
            self.sys.set_x(x)
            self.sys.set_v(v)
            if step_no % self.file_io.output_period != 0:
                continue

            pe = self.sys.U(x)
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

            if Config.system == 'LJ':
                L = self.sys.L
                x = L * (x <= 0) - L * (x >= L) + x
            self.sys.set_x(x)
            if step_no % self.file_io.output_period != 0:
                continue
            
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
        
        if Config.thermostat == 'nh':
            self.thermostat = NoseHoover(self.stepper.dt, d = self.sys.v.shape[1])
        elif Config.thermostat == 'langevin':
            self.thermostat = LangevinThermostat(self.stepper.dt)

    def run_simulation(self):
        for step_no in range(self.starting_step, self.num_steps):
            t = step_no * self.stepper.dt
            if Config.thermostat == 'nh':
                v = self.thermostat.step(self.sys)
                x, v = self.stepper.step(self.sys, step_no, v = v)
                v = self.thermostat.step(self.sys, v = v)
            else:
                x, v = self.thermostat.step(self.sys.x, self.sys.v, self.sys.F, self.sys.m)
            
            if Config.system == 'LJ':
                L = self.sys.L
                x = L * (x <= 0) - L * (x >= L) + x

            self.sys.set_x(x)
            self.sys.set_v(v)
            if step_no % self.file_io.output_period != 0:
                continue

            pe = self.sys.U(x)
            ke = self.sys.K(v)
            
            temp = self.sys.instantaneous_T(v)
            self.file_io.write_vectors(x, v, t)
            self.file_io.write_scalars(ke, pe, temp, t)

            if Config.thermostat == 'nh':
                univ_energy = self.thermostat.universe_energy(ke, pe)
                self.file_io.write_hprime(univ_energy, t)

        if Config.thermostat == 'nh':
            self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1, xi = self.thermostat.xi, vxi = self.thermostat.vxi)
        else:
            self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1)
        del(self.file_io)

class REMD_Ensemble(NVT_Ensemble):

    def __init__(self, sys, num_steps, starting_step, first_time):
        self.sys = sys
        self.file_io = FileOperationsREMD(first_time = first_time)
        self.num_steps = num_steps
        self.starting_step = starting_step
        self.stepper = VelocityVerletIntegrator()
        self.remd_integrator = REMDIntegrator()

        if Config.thermostat == 'nh':
            self.nht = NoseHoover(self.stepper.dt, d = self.sys.v.shape[1])
        else:
            self.thermostat = LangevinThermostat(self.stepper.dt)


    def run_simulation(self):
        for step_no in range(self.starting_step, self.num_steps):

            t = step_no * self.stepper.dt
            v = self.sys.v
            x = self.sys.x
            if step_no != 0 and step_no % self.remd_integrator.exchange_period == 0:
                y_x, y_v = self.remd_integrator.step(x, v, self.sys.U(self.sys.x), step_no, self.file_io)
                x, v = y_x, y_v
            else: 
                if Config.thermostat == 'nh':
                    v = self.nht.step(self.sys.m, v)
                    x, v = self.stepper.step(self.sys, step_no, v = v)
                    v = self.nht.step(self.sys.m, v)
                else:
                    x, v = self.thermostat.step(self.sys.x, self.sys.v, self.sys.F, self.sys.m)
            
            if Config.system == 'LJ':
                L = self.sys.L
                x = L * (x <= 0) - L * (x >= L) + x

            self.sys.set_x(x)
            self.sys.set_v(v)
            if step_no % self.file_io.output_period != 0:
                continue

            pe = self.sys.U(x)
            ke = self.sys.K(v)
            temp = self.sys.instantaneous_T(v)

            self.file_io.write_vectors(x, v, t)
            self.file_io.write_scalars(ke, pe, temp, step_no * self.stepper.dt)
            
            if Config.thermostat == 'nh':
                univ_energy = self.nht.universe_energy(ke, pe)
                self.file_io.write_hprime(univ_energy, step_no)

        if Config.thermostat == 'nh':
            self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1, xi = self.nht.xi, vxi = self.nht.vxi)
        else:
            self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1)
        self.file_io.wrap_up()


class RENS_Ensemble(REMD_Ensemble):

    def __init__(self, sys, num_steps, starting_step, first_time):
        self.sys = sys
        self.file_io = FileOperationsRENS(first_time = first_time)
        self.num_steps = num_steps
        self.starting_step = starting_step
        self.stepper = VelocityVerletIntegrator()
        self.rens_integrator = RENSIntegrator(self.stepper.dt)

        if Config.thermostat == 'nh':
            self.nht = NoseHoover(self.stepper.dt)
        else:
            self.thermostat = LangevinThermostat(self.stepper.dt)
            


    def run_simulation(self):
        for step_no in range(self.starting_step, self.num_steps):
            t = step_no * self.stepper.dt
            if self.rens_integrator.mode == 0:
                if Config.thermostat == 'nh':
                    v = self.nht.step(self.sys)
                    x, v = self.stepper.step(self.sys, step_no, v = v)
                    v = self.nht.step(self.sys)
                else:
                    x, v = self.thermostat.step(self.sys.x, self.sys.v, self.sys.F, self.sys.m)
                self.rens_integrator.attempt(self.sys, x, v)
            else:
                x, v = self.rens_integrator.step(self.sys, t, self.file_io)

            if Config.system == 'LJ':
                L = self.sys.L
                x = L * (x <= 0) - L * (x >= L) + x

            self.sys.set_x(x)
            self.sys.set_v(v)
            if step_no % self.file_io.output_period != 0:
                continue 

            pe = self.sys.U(x)
            ke = self.sys.K(v)
            temp = self.sys.instantaneous_T(v)
            self.file_io.write_vectors(x, v, t, self.rens_integrator.mode)
            self.file_io.write_scalars(ke, pe, temp, t)

            if Config.thermostat == 'nh':
                univ_energy = self.nht.universe_energy(ke, pe)
                self.file_io.write_hprime(univ_energy, t)

        if Config.thermostat == 'nh':
            self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1, xi = self.nht.xi, vxi = self.nht.vxi)
        else:
            self.file_io.write_rst(self.sys.x, self.sys.v, self.sys.m, self.num_steps - 1)

        # del(self.file_io)
        
            
