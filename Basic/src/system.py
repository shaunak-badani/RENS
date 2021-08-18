from .units import Units

import numpy as np
import random
import math
import pandas as pd
from .config import Config

class System:
   
    BOLTZMANN = 1.380649e-23 
    AVOGADRO = 6.02214076e23
    KILO = 1e3
    RGAS = BOLTZMANN*AVOGADRO
    BOLTZ = (RGAS/KILO)  

    def __init_velocities(self):
        N = Config.num_particles
        T = Config.T()
        v = np.random.random(size = (N, 1)) - 0.5
        sumv2 = np.sum(self.m * v**2)
        fs = np.sqrt((N * Units.kB * T) / sumv2)
        self.v = v * fs 
        

    def __init__(self):
        N = Config.num_particles
        self.x = np.random.normal(0, 1.0, size = (N, 1))
        self.m = np.full(N, 1) # kg / mol
        self.__init_velocities()


        if Config.rst:
            df = pd.read_csv(cfg.rst, sep = ' ')
            self.x = df['x'].to_numpy().reshape(-1, 1)
            N = self.x.shape[0]
            self.v = df['v'].to_numpy().reshape(-1, 1)
            self.m = df['m'].to_numpy().reshape(-1, 1)
            self.N = N
            cfg.num_particles = N
        
    def pot_energy(self, x):
        if x < -1.25:
            return (4 * (np.pi**2)) * (x + 1.25)**2
        
        if x >= -1.25 and x <= -0.25:
            return 2 * (1 + np.sin(2 * np.pi * x))
            
        if x >= -0.25 and x <= 0.75:
            return 3 * (1 + np.sin(2 * np.pi * x))
                    
        if x >= 0.75 and x <= 1.75:
            return 4 * (1 + np.sin(2 * np.pi * x))
                    
        # if x >= 1.75:
        return 64 * ((x - 1.75) ** 2)

    def U(self, x):
        reduced_x = x.flatten()
        pot = 0
        for i in reduced_x:
            pot += self.pot_energy(i)
        return pot

    def K(self, v):
        KE = 0.5 * np.sum(self.m * v**2)
        # KE_in_KJmol = KE / 1e4
        # return KE_in_KJmol
        return KE

    
    @staticmethod
    def __force(x):
        if x <= -1.25:
            return (-8 * (np.pi)**2) * (x + 1.25)
        
        if x > -1.25 and x <= -0.25:
            return -4 * np.pi * np.cos(2 * np.pi * x)
            
        if x >= -0.25 and x <= 0.75:
            return -6 * np.pi * np.cos(2 * np.pi * x)
                    
        if x >= 0.75 and x <= 1.75:
            return -8 * np.pi * np.cos(2 * np.pi * x)
            
        # if x >= 1.75:
        return -(128) * (x - 1.75)
                    
    @staticmethod
    def F(x):
        forces = []
        reduced_x = x.flatten()
        
        for particle in reduced_x:
            f = System.__force(particle)
            forces.append(f)
        F_x = np.array(forces).reshape(x.shape)
        return F_x

    def instantaneous_T(self, v):
        N = Config.num_particles
        sumv2 = 2 * self.K(v)
        t = sumv2 / (N * Units.kB)
        # r_t = (Units.kB / Units.epsilon) * t
        return t

    # Setters
    def set_x(self, x):
        self.x = x

    def set_v(self, v):
        self.v = v

    def __del__(self):
        pass
