from .units import Units

import numpy as np
import random
import math
import os
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

        # Sampling from maxwell boltzmann distribution
        # beta = 1 / (Config.T() * Units.kB)
        # sigma = 1 / np.sqrt(self.m * beta)
        # v = np.random.normal(size = (N, 1), scale = sigma)
        # fs = 1
        self.v = v * fs 

    def __init_positions(self):
        N = Config.num_particles
        beta = 1 / (Units.kB * Config.T())
        sigma_p = 1 / np.sqrt(beta)
        linsp = np.linspace(-2, 2.25, 10000)
        u = np.array([self.U(i) for i in linsp])
        prob = np.exp(- beta * u)
        prob /= prob.sum()

        x = np.random.choice(linsp, size = (N, 1), p = prob)
        self.x = x
        

    def __init__(self, file_io = None):
        N = Config.num_particles
        self.m = np.full((N, 1), 1)
        self.__init_positions()
        self.__init_velocities()


        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            self.x = df['x'].dropna().to_numpy().reshape(-1, 1)
            N = self.x.shape[0]
            self.v = df['v'].dropna().to_numpy().reshape(-1, 1)
            self.m = df['m'].dropna().to_numpy().reshape(-1, 1)
            self.N = N
            Config.num_particles = N
        
        
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
        return 8 * (np.pi**2) * ((x - 1.75) ** 2)

    def U(self, x):
        u = np.zeros_like(x)
        u[x < -1.25] = 4 * np.pi**2 * (x[x < -1.25] + 1.25)**2
        
        ind = np.logical_and(x >= -1.25, x < -0.25)
        u[ind] = 2 * (1 + np.sin(2*np.pi*x[ind]))
        
        ind = np.logical_and(x >= -0.25, x < 0.75)
        u[ind] = 3 * (1 + np.sin(2*np.pi*x[ind]))
        
        ind = np.logical_and(x >= 0.75, x < 1.75)
        u[ind] = 4 * (1 + np.sin(2*np.pi*x[ind]))
        
        ind = (x >= 1.75)
        u[ind] = 8 * (np.pi**2) * (x[ind] - 1.75)**2
        
        return u.sum()

    def K(self, v):
        KE = 0.5 * np.sum(self.m * v**2)
        # print("KE : ", KE)
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
            
        if x >= 1.75:
            return (-16 * (np.pi)**2) * (x - 1.75)
                    
    @staticmethod
    def F(x):
        f = np.zeros_like(x)
        f[x < -1.25] = -8 * np.pi**2 * (x[x < -1.25] + 1.25)
        
        ind = np.logical_and(x >= -1.25, x < -0.25)
        f[ind] = -4 * np.pi * np.cos(2 * np.pi * x[ind])
        
        ind = np.logical_and(x >= -0.25, x < 0.75)
        f[ind] = -6 * np.pi * np.cos(2 * np.pi * x[ind])
        
        ind = np.logical_and(x >= 0.75, x < 1.75)
        f[ind] = -8 * np.pi * np.cos(2 * np.pi * x[ind])
        
        ind = (x >= 1.75)
        f[ind] = -16 * np.pi**2 * (x[ind] - 1.75)
        
        return f

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
