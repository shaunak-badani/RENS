import numpy as np
import math
from .units import Units
from .system import System
from .config import Config

class NoseHoover():

    def __init__(self, dt, freq = 50, M = 2, n_c = 1):
        self.n_c = n_c
        self.M = M
        self.vxi = np.zeros(M)
        self.xi = np.zeros(M)
        self.dt = dt
        self.num_particles = Config.num_particles
        self.Q = np.full(M, Units.kB * Config.T() / freq**2)
        self.Q[0] *= Config.num_particles


    def surr_energy(self):
        total_surrounding_energy = (self.num_particles * self.xi[0] + self.xi[1:].sum()) * Units.kB * Config.T() 
        total_surrounding_energy += 0.5 * np.sum(self.Q * self.vxi**2)
        return total_surrounding_energy

    def step(self, KE, v):
        
        N_f = self.num_particles
        T = Config.T()
        M = self.M
        n_c = self.n_c
        delta = (self.dt / n_c)
        SCALE = 1.0
        
        
        for i in range(n_c):
            KE2 = 2 * KE
            
            G_M = (self.Q[M - 2] * self.vxi[M - 2]**2 - Units.kB * T) / self.Q[M - 1]
            self.vxi[M - 1] += (delta / 4) * G_M

            for j in range(M - 2, 0, -1):
                self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])
                G_j = (self.Q[j - 1] * self.vxi[j-1]**2 - Units.kB * T) / self.Q[j]
                self.vxi[j] += G_j * delta / 4
                self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])

            self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1]) 
            G_1 = (KE2 - N_f * Units.kB * T) / self.Q[0]
            self.vxi[0] += (delta / 4) * G_1 
            self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1])

            # UPDATE xi and v_new
            self.xi += (delta/2) * self.vxi
            SCALE *= np.exp(-delta / 2 * self.vxi[0])
            

            # REVERSE
            self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1]) 
            G_1 = (KE2 * SCALE * SCALE - N_f * Units.kB * T) / self.Q[0]
            self.vxi[0] += (delta / 4) * G_1 
            self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1])

            for j in range(1, M - 1):
                self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])
                G_j = (self.Q[j - 1] * self.vxi[j-1]**2 - Units.kB * T) / self.Q[j]
                self.vxi[j] += G_j * delta / 4
                self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])

            G_M = (self.Q[M - 2] * self.vxi[M - 2]**2 - Units.kB * T) / self.Q[M - 1]
            self.vxi[M - 1] += (delta / 4) * G_M
        
        v_new = v*SCALE
        return v_new
        
