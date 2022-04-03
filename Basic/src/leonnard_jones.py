from .units import Units
from .config import Config

import numpy as np
import pandas as pd
import random
import math
from .system import System

class LJ(System):

    def __init__(self):
        Units.arbitrary = False
        Units.kB = Units.BOLTZ # kJ / mol K
        Units.epsilon = 120 * Units.kB # kJ / mol 
        Config.dt = 1e-2 # picosecond

        self.N = Config.num_particles
        self.sigma = 3.4 # angstrom
        self.L = 10.229 * self.sigma / 2
        self.x = np.random.uniform(0, self.L, size = (self.N, 3))
        self.m = 39.95 * np.ones(shape = (self.N, 1)) # amu

        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            self.x = df['x'].dropna().to_numpy().reshape(-1, 3)
            N = self.x.shape[0]

            self.m = df['m'].dropna().to_numpy().reshape(-1, 1)
            self.N = N
            Config.num_particles = N

        self.__init_velocities()
        

    def __init_velocities(self):
        T = Config.T()
        kBT = Units.kB * T * Units.kJ_mol_TO_J
        m = self.m * Units.AMU_TO_KG
        scale = np.sqrt(kBT  / m) * Units.M_S_TO_A_PS
        v = np.random.normal(size = (self.N, 3), scale = scale)
        self.v = v

    def U(self, x):
        sigma = self.sigma

        a = x[:]
        b = x[:]
        pairwise_dist = a[:, None, :] - b[None, :, :]
        r_ij = pairwise_dist
        
        # PBC
        L = self.L
        r_ij = L * (pairwise_dist <= -L / 2) - L * (pairwise_dist >= L / 2) + pairwise_dist


        x = np.linalg.norm(r_ij, axis=-1)

        dist_np = x[np.triu(x) != 0]
        u = np.sum(4 * Units.epsilon * ((sigma / dist_np)**12 - (sigma / dist_np)**6))
        return u  # kJ / mol
        
    def F(self, x):
        N = self.N
        sigma = self.sigma

        r_ij = x[:, None, :] - x[None, :, :]
        q = r_ij != [0, 0, 0]
        r_ij = r_ij[q].reshape(N, N - 1, 3)

        # PBC
        L = self.L
        r_ij = L * (r_ij <= -L / 2) - L * (r_ij >= L / 2) + r_ij

        dist = np.linalg.norm(r_ij, axis = -1)
        epsilon = Units.epsilon * Units.kJ_mol_TO_J * (Units.M_S_TO_A_PS)**2 / (Units.AMU_TO_KG)
        val_temp = 24 * epsilon / (dist**2) * (2 * (sigma / dist)**12 - (sigma / dist)**6)
        val_temp = val_temp[:,:, np.newaxis]
        F_i = np.sum(val_temp * (r_ij), axis = 1)
        return F_i # amu * Angstrom / ps**2

    def instantaneous_T(self, v):
        N = self.N
        sumv2 = 2 * self.K(v)
        t = sumv2 / (3 * N * Units.kB)
        return t

    def K(self, v):
        """
        v -> A / ps
        m -> amu
        A / ps * 1e2 -> m / s
        """
        v_2 = (self.m * Units.AMU_TO_KG) * (v * Units.A_PS_TO_M_S)**2 # kg m^2 / s^2
        KE = 0.5 * np.sum(v_2) * Units.NA / Units.KILO # kJ / mol
        return KE