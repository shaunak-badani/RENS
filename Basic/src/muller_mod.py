from .units import Units

import numpy as np
import random
import math
import os
import pandas as pd
from .config import Config

from .muller import MullerSystem

class MullerMod(MullerSystem):

    def __init__(self, *args):
        super().__init__(self, *args)
        self.sigma = 0.8
        self.height = 200
        self.center = (-0.25, 0.65)

        
    def pot_energy(self, x, y):
        exponent = self.a * (x - self.mean_x)**2
        exponent += self.b * (x - self.mean_x) * (y - self.mean_y)
        exponent += self.c * (y - self.mean_y)**2
        V_vector = self.A * np.exp(exponent)

        exp = (x - self.center[0])**2 + (y - self.center[1])**2
        exp /= 2 * self.sigma**2
        extra_term = self.height * np.exp(-exp)

        return V_vector.sum()

    def __force(self, x, y):
        exponent = self.a * (x - self.mean_x)**2 
        exponent += self.b * (x - self.mean_x) * (y - self.mean_y)  
        exponent += self.c * (y - self.mean_y)**2
        V_vector = self.A * np.exp(exponent)
        F_x = -V_vector * (2 * self.a * (x - mean_x) + self.b * (y - mean_y))
        F_y = -V_vector * (2 * self.c * (y - mean_y) + self.b * (x - mean_x))

        exp = (x - self.center[0])**2 + (y - self.center[1])**2
        exp /= 2 * self.sigma**2

        F_x = F_x.sum() + self.height * np.exp(-exp) * (x - self.center[0]) / sigma**2
        F_y = F_y.sum() + self.height * np.exp(-exp) * (y - self.center[1]) / sigma**2
        return np.array([F_x, F_y])

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


        exp = (x - self.center[0])**2 + (y - self.center[1])**2
        exp /= 2 * self.sigma**2
        extra_term = self.height * np.exp(-exp)

        return pot.sum()
        
                    
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
        
        
        exp = (x - self.center[0])**2 + (y - self.center[1])**2
        exp /= 2 * self.sigma**2
        
        F_x = F_x.sum(axis = 0)
        F_y = F_y.sum(axis = 0)
        F_x += self.height * np.exp(-exp) * (x - self.center[0]) / self.sigma**2
        F_y += self.height * np.exp(-exp) * (y - self.center[1]) / self.sigma**2
        
        F_x = F_x.reshape(-1, 1)
        F_y = F_y.reshape(-1, 1)
        
        return np.hstack((F_x, F_y))
    
