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
        self.x = np.random.normal(0, 1.0, size = (N, 2))
        self.m = np.full((N, 2), 1) # kg / mol
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
    def F(q):
        """
        Args : 
        q : Vector of size (N, 2) containing N particles on 2D space
        
        Returns : 
        F : Vector of size (N, 2) containing forces on N particles
        """
        
        A = np.array([-200, -100, -170, 15]).reshape(-1, 1)
        mean_x = np.array([1, 0, -0.5, -1]).reshape(-1, 1)
        mean_y = np.array([0, 0.5, 1.5, 1]).reshape(-1, 1)
        a = np.array([-1, -1, -6.5, 0.7]).reshape(-1, 1)
        b = np.array([0, 0, 11, 0.6]).reshape(-1, 1)
        c = np.array([-10, -10, -6.5, 0.7]).reshape(-1, 1)
        

        x = q[:, 0]
        y = q[:, 1]
        gaussians = a * (x - mean_x)**2
        gaussians += b * (x - mean_x) * (y - mean_y)
        gaussians += c * (y - mean_y)**2
        V_vector = A * np.exp(gaussians)
        
        F_x = - V_vector * (2 * a * (x - mean_x) + b * (y - mean_y))
        F_y = - V_vector * (2 * c * (y - mean_y) + b * (x - mean_x))
        F_x = F_x.sum(axis = 0).reshape(-1, 1)
        F_y = F_y.sum(axis = 0).reshape(-1, 1)
        return np.hstack((F_x, F_y))
