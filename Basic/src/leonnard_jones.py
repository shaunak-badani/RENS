from .units import Units
from .config import Config

import numpy as np
import pandas as pd
import random
import math
from .system import System

class LJ(System):

    def __init__(self):
        Units.kB = 1.380649 * 1e-23
        Units.epsilon = 120 * Units.kB
        Config.dt = 1e-2 # picosecond

        self.N = 108
        self.sigma = 3.4
        self.L = 10.229 * self.sigma
        self.x = np.random.uniform(0, self.L, size = (self.N, 3))
        self.m = 39.95 * 1.6747 * 1e-27 * np.ones(shape = (self.N, 1)) # kg
        self.__init_velocities()

        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            self.x = df['x'].dropna().to_numpy().reshape(-1, 3)
            N = self.x.shape[0]
            self.v = df['v'].dropna().to_numpy().reshape(-1, 3)
            self.m = df['m'].dropna().to_numpy().reshape(-1, 1)
            self.N = N
            Config.num_particles = N

    def __init_velocities(self):
        T = Config.T()
        v = np.random.random(size = (self.N, 3)) - 0.5
        sumv2 = 2 * self.K(v)
        fs = np.sqrt((self.N * 3 * Units.kB * T) / sumv2)
        self.v = v * fs  # Angstrom / picosecond

    def U(self, x):
        sigma = self.sigma

        a = x[:]
        b = x[:]
        pairwise_dist = a[:, None, :] - b[None, :, :]
        r_ij = pairwise_dist
        
        # PBC
        # L = self.L
        # r_ij = L / 2 * (pairwise_dist <= -L / 2) - L / 2 * (pairwise_dist >= L / 2) + pairwise_dist


        x = np.linalg.norm(r_ij, axis=-1)

        dist_np = x[np.triu(x) != 0]
        u = np.sum(4 * Units.epsilon * ((sigma / dist_np)**12 - (sigma / dist_np)**6))
        return u  # Joules
        
    def F(self, x):
        N = self.N
        sigma = self.sigma

        r_ij = x[:, None, :] - x[None, :, :]
        q = r_ij != [0, 0, 0]
        r_ij = r_ij[q].reshape(N, N - 1, 3)

        # PBC
        # L = self.L
        # r_ij = L / 2 * (r_ij <= -L / 2) - L / 2 * (r_ij >= L / 2) + r_ij

        dist = np.linalg.norm(r_ij, axis = -1)
        val_temp = 24 * Units.epsilon / (dist**2) * (2 * (sigma / dist)**12 - (sigma / dist)**6)
        val_temp = val_temp[:,:, np.newaxis]
        F_i = np.sum(val_temp * (r_ij), axis = 1)
        return F_i * 1e-4 # kg A / (ps)^2 

    def instantaneous_T(self, v):
        N = self.N
        sumv2 = 2 * self.K(v)
        t = sumv2 / (3 * N * Units.kB)
        return t

    def K(self, v):
        """
        v -> A / ps
        m -> kg
        A / ps * 1e2 -> m / s
        """
        KE = 0.5 * np.sum(self.m * (v * 1e2)**2)
        return KE