import numpy as np
import pandas as pd
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
        kBT = Units.kB * Config.T()

        self.Q = np.full(M, kBT / freq**2)
        self.Q[0] *= Config.num_particles
        self.w = np.array([0.2967324292201065,  0.2967324292201065, -0.1869297168804260, 0.2967324292201065, 0.2967324292201065])

        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            if hasattr(df, 'xi'):
                self.xi = df['xi'].dropna().to_numpy()
            if hasattr(df, 'vxi'):
                self.vxi = df['vxi'].dropna().to_numpy()
            self.M = self.xi.shape[0]

        ## NVT Harmonic parameters
        # self.w = np.array([1])
        # self.M = 4
        # self.Q = np.ones(self.M)
        # self.xi = np.zeros(self.M)
        # self.vxi = np.array([1, -1, 1, -1], dtype = 'float')
        ###



    def universe_energy(self, KE, PE):
        total_universe_energy = (self.num_particles * self.xi[0] + self.xi[1:].sum()) * Units.kB * Config.T() 
        total_universe_energy += 0.5 * np.sum(self.Q * self.vxi**2)
        total_universe_energy += KE + PE
        return total_universe_energy

    def step(self, sys, v = None):
        if v is None:
            v = sys.v
        _, d = v.shape
        N_f = d * self.num_particles
        T = Config.T()
        M = self.M
        n_c = self.n_c
        n_ys = self.w.shape[0]
        SCALE = 1.0
        
        
        KE2 = 2 * sys.K(v)      
        
        for i in range(n_c):
            for w_j in self.w:
                delta = (w_j * self.dt / n_c)

                
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
                SCALE_FACTOR = np.exp(-delta / 2 * self.vxi[0])
                SCALE *= SCALE_FACTOR
                KE2 *= SCALE_FACTOR * SCALE_FACTOR

                # REVERSE
                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1]) 
                G_1 = (KE2 - N_f * Units.kB * T) / self.Q[0]
                self.vxi[0] += (delta / 4) * G_1 
                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1])

                for j in range(1, M - 1):
                    self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])
                    G_j = (self.Q[j - 1] * self.vxi[j-1]**2 - Units.kB * T) / self.Q[j]
                    self.vxi[j] += G_j * delta / 4
                    self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])

                G_M = (self.Q[M - 2] * self.vxi[M - 2]**2 - Units.kB * T) / self.Q[M - 1]
                self.vxi[M - 1] += (delta / 4) * G_M
        
        v_new = sys.v * SCALE
        return v_new
        
