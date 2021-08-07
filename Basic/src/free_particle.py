from .units import Units

import numpy as np
import random
import math
from .system import System

class FreeParticleSystem(System):

    def __init_velocities(self, cfg):
        N = cfg.num_particles
        T = cfg.temperature
        self.T = T
        self.N = N
        self.v = np.random.random(size = (N, 1))
        print(self.v.mean(axis = 0).shape)
        self.v = self.v - self.v.mean(axis = 0)
        
        

    def __init__(self, cfg):
        super().__init__(cfg)
        N = cfg.num_particles
        self.x = np.zeros((N, 1), dtype="float")
        self.m = np.ones((N, 1), dtype="float")
        self.__init_velocities(cfg)

   
    @staticmethod
    def U(x):
        return 69

                    
    @staticmethod
    def F(x):
        return 0
