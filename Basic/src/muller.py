from .units import Units

import numpy as np
import random
import math
import os
import pandas as pd
from .config import Config

from .system import System

class MullerSystem(System):

    def __init_velocities(self):
        N = Config.num_particles
        T = Config.T()
        v = np.random.random(size = (N, 2)) - 0.5
        sumv2 = np.sum(self.m * v**2)
        fs = np.sqrt((N * Units.kB * T) / sumv2)
        self.v = v * fs 
        

    def __init__(self, file_io = None):
        N = Config.num_particles

        # Initializing in one of the basins
        # To see if it samples the other
        a = np.random.uniform(-0.8, -0.3, size = (N, 1))
        b = np.random.uniform(1.0, 1.9, size = (N, 1))
        self.x = np.hstack((a, b))
        self.m = np.full((N, 1), 1) # kg / mol
        self.__init_velocities()

        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            self.x = df['x'].dropna().to_numpy().reshape(-1, 2)
            N = self.x.shape[0]
            self.v = df['v'].dropna().to_numpy().reshape(-1, 2)
            self.m = df['m'].dropna().to_numpy().reshape(-1, 2)
            self.N = N
            Config.num_particles = N

        self.A = np.array([-200, -100, -170, 15])
        self.mean_x = np.array([1, 0, -0.5, -1])
        self.mean_y = np.array([0, 0.5, 1.5, 1])
        self.a = np.array([-1, -1, -6.5, 0.7])
        self.b = np.array([0, 0, 11, 0.6])
        self.c = np.array([-10, -10, -6.5, 0.7])

        # Constants for vectorized function
        self.VA = self.A.reshape(-1, 1)
        self.Vmean_x = self.mean_x.reshape(-1, 1)
        self.Vmean_y = self.mean_y.reshape(-1, 1)
        self.Va = self.a.reshape(-1, 1)
        self.Vb = self.b.reshape(-1, 1)
        self.Vc = self.c.reshape(-1, 1)

        
    def pot_energy(self, x, y):
        exponent = self.a * (x - self.mean_x)**2
        exponent += self.b * (x - self.mean_x) * (y - self.mean_y)
        exponent += self.c * (y - self.mean_y)**2
        V_vector = self.A * np.exp(exponent)
        return V_vector.sum()

    def U(self, q):
        """
        Args : 
        q : Vector of size (N, 2) containing N particles on 2D space
        
        Returns : 
        u : scalar, potential energy of vector
        """

        x = q[:, 0]
        y = q[:, 1]
        gaussians = self.Va * (x - self.Vmean_x)**2
        gaussians += self.Vb * (x - self.Vmean_x) * (y - self.Vmean_y)
        gaussians += self.Vc * (y - self.Vmean_y)**2
        pot = (self.VA * np.exp(gaussians))
        return pot.sum()
    
    def __force(self, x, y):
        exponent = self.a * (x - self.mean_x)**2 
        exponent += self.b * (x - self.mean_x) * (y - self.mean_y)  
        exponent += self.c * (y - self.mean_y)**2
        V_vector = self.A * np.exp(exponent)
        F_x = -V_vector * (2 * self.a * (x - mean_x) + self.b * (y - mean_y))
        F_y = -V_vector * (2 * self.c * (y - mean_y) + self.b * (x - mean_x))
        F_x = F_x.sum()
        F_y = F_y.sum()
        return np.array([F_x, F_y])
        
                    
    def F(self, q):
        """
        Args : 
        q : Vector of size (N, 2) containing N particles on 2D space
        
        Returns : 
        F : Vector of size (N, 2) containing forces on N particles
        """
        

        x = q[:, 0]
        y = q[:, 1]
        gaussians = self.Va * (x - self.Vmean_x)**2
        gaussians += self.Vb * (x - self.Vmean_x) * (y - self.Vmean_y)
        gaussians += self.Vc * (y - self.Vmean_y)**2
        V_vector = self.VA * np.exp(gaussians)
        
        F_x = - V_vector * (2 * self.Va * (x - self.Vmean_x) + self.Vb * (y - self.Vmean_y))
        F_y = - V_vector * (2 * self.Vc * (y - self.Vmean_y) + self.Vb * (x - self.Vmean_x))
        F_x = F_x.sum(axis = 0).reshape(-1, 1)
        F_y = F_y.sum(axis = 0).reshape(-1, 1)
        return np.hstack((F_x, F_y))
