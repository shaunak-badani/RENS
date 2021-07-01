from .units import Units

import numpy as np
import random
import math

class System:
   
    BOLTZMANN = 1.380649e-23 
    AVOGADRO = 6.02214076e23
    KILO = 1e3
    RGAS = BOLTZMANN*AVOGADRO
    BOLTZ = (RGAS/KILO)  

    def __init_velocities(self, cfg):
        N = cfg.num_particles
        # T = cfg.temperature
        T = cfg.reduced_temperature
        self.T = T
        self.N = N
        v = np.random.random(size = (N, 1)) - 0.5
        sumv2 = np.sum(self.m * v**2)
        fs = np.sqrt((N * Units.kB * T) / sumv2)
        self.v = v * fs

    def __init__(self, cfg):
        N = cfg.num_particles
        self.x = np.random.normal(0, 0.5, size = (N, 1))
        self.m = np.ones((N, 1))
        self.__init_velocities(cfg)

    def __pot_energy(x):
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

    @staticmethod
    def U(x):
        reduced_x = x.squeeze()
        pot = 0
        for i in reduced_x:
            pot += System.__pot_energy(i)
        return pot

    
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
        reduced_x = x.squeeze()
        for particle in reduced_x:
            f = System.__force(particle)
            forces.append(f)
        F_x = np.array(forces).reshape(x.shape)
        return F_x

    def K(self, v):
        ke = 0.5 * self.m * (self.v ** 2)
        return ke.sum()

    def instantaneous_T(self, v):
        N = self.N
        sumv2 = np.sum(self.m * v**2)
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
