import numpy as np
import math
from .units import Units
from .system import System

class NoseHoover():

    def __init__(self, dt, cfg, M = 4, n_c = 8):
        self.n_c = n_c
        self.M = M
        self.vxi = np.zeros(M)
        self.xi = np.zeros(M)
        self.dt = dt
        self.T = cfg.temperature
        self.num_particles = cfg.num_particles
        self.Q = np.full(M, 10.0)


    def surr_energy(self):
        total_surrounding_energy = (self.num_particles * self.xi[0] + self.xi[1:].sum()) * Units.kB * self.T 
        total_surrounding_energy += 0.5 * np.sum(self.Q * self.vxi**2)
        return total_surrounding_energy
    
    # def step(self, m, v):
    #     KE = 0.5 * np.sum(m * v**2)
    #     T = self.T
    #     d = v.shape[1]
    #     N_f = self.num_particles * d
    #     M = self.M
    #     SCALE = 1.0
    #     for i in range(self.n_c):
            
    #         delta_ts = self.dt / self.n_c
    #         dt_2 = 0.5 * delta_ts
    #         dt_4 = 0.5 * dt_2
    #         dt_8 = 0.5 * dt_4
            
    #         G_2 = (self.Q[0] * self.vxi[0] * self.vxi[0] - Units.kB * T) / self.Q[1]
    #         # print(self.Q[0], self.vxi[0], T)
    #         # print(G_2)
    #         self.vxi[1] += G_2 * (dt_4)
    #         # print(self.vxi[1])
    #         self.vxi[0] = self.vxi[0] * math.exp(-self.vxi[1] * delta_ts / 8)
    #         # print(self.vxi[0])
    #         G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
    #         self.vxi[0] = self.vxi[0] + (dt_4) * G_1
    #         self.vxi[0] = self.vxi[0] * math.exp(-self.vxi[1] * dt_8)

    #         SCALE = SCALE * (math.exp(-1 * dt_2 * self.vxi[0]))
    #         KE = KE * math.exp(-1 * delta_ts * self.vxi[0])
            
    #         self.xi[0] = self.xi[0] + (dt_2) * self.vxi[0]
    #         self.xi[1] = self.xi[1] + (dt_2) * self.vxi[1]

    #         G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
    #         # print(G_1)
            
    #         self.vxi[0] = self.vxi[0] * math.exp(-self.vxi[1] * dt_8)
    #         self.vxi[0] = self.vxi[0] + (dt_4) * G_1
    #         self.vxi[0] = self.vxi[0] * math.exp(-self.vxi[1] * dt_8)

    #         G_2 = (self.Q[0] * self.vxi[0] * self.vxi[0] - Units.kB * T) / self.Q[1]
    #         self.vxi[1] = self.vxi[1] + (G_2 * dt_4)
    #         # print(self.vxi[1])
    #         # print("-----")


    #     v_new = v * SCALE
    #     return v_new

    def step(self, m, v):
        
        N_f = self.num_particles
        T = self.T
        M = self.M
        n_c = self.n_c
        delta = (self.dt / n_c)
        KE = 0.5 * np.sum(m * v**2)
        SCALE = 1.0
        
        
        for i in range(n_c):
            
            if M > 1:
                G_M = (self.Q[M - 2] * self.vxi[M - 2]**2 - Units.kB * T) / self.Q[M - 1]
                self.vxi[M - 1] += (delta / 4) * G_M

            for j in range(M - 2, 0, -1):
                self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])
                G_j = (self.Q[j - 1] * self.vxi[j-1]**2 - Units.kB * T) / self.Q[j]
                self.vxi[j] += G_j * delta / 4
                self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])

            if self.M > 1:
                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1]) 
            G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
            self.vxi[0] += (delta / 4) * G_1 
            if self.M > 1:
                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1])

            # UPDATE xi and v_new
            self.xi += (delta/2) * self.vxi
            SCALE *= np.exp(-delta / 2 * self.vxi[0])
            KE *= np.exp(-delta * self.vxi[0])

            # REVERSE
            if self.M > 1:
                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1]) 
            G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
            self.vxi[0] += (delta / 4) * G_1 
            if self.M > 1:
                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1])

            for j in range(1, M - 1):
                self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])
                G_j = (self.Q[j - 1] * self.vxi[j-1]**2 - Units.kB * T) / self.Q[j]
                self.vxi[j] += G_j * delta / 4
                self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])

            if M > 1:
                G_M = (self.Q[M - 2] * self.vxi[M - 2]**2 - Units.kB * T) / self.Q[M - 1]
                self.vxi[M - 1] += (delta / 4) * G_M
        
        v_new = v*SCALE
        return v_new
        
