from .units import Units

import numpy as np
import random
import math
from .system import System
from .config import Config

class FreeParticleSystem(System):

    def __init_velocities(self):
        N = Config.num_particles
        T = Config.temperature
        self.T = Config.T()
        self.N = N
        self.v = np.random.random(size = (N, 1))
        if N > 1:
            self.v = self.v - self.v.mean(axis = 0)

    def __init__(self, *args):
        super().__init__(*args)
        N = Config.num_particles
        self.x = np.zeros((N, 1), dtype="float")
        self.m = np.ones((N, 1), dtype="float")
        self.__init_velocities()

   
    @staticmethod
    def U(x):
        return 69

                    
    @staticmethod
    def F(x):
        return np.zeros_like(x)
