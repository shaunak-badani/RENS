import numpy as np
import math
from .units import Units
from .system import System

class NoseHoover():

    def __init__(self, dt, cfg, M = 2, freq = 1, n_c = 1, w = [1]):
        self.n_c = n_c
        self.w = w
        self.M = M
        self.vxi = np.zeros(M)
        self.xi = np.zeros(M)
        self.dt = dt
        self.T = cfg.temperature
        self.num_particles = cfg.num_particles
        self.Q = np.full(M, 1.0)


    def surr_energy(self):
        total_surrounding_energy = (self.num_particles * self.xi[0] + self.xi[1:].sum()) * Units.kB * self.T 
        total_surrounding_energy += 0.5 * np.sum(self.Q * self.vxi**2)
        return total_surrounding_energy

    def step_one_thermostat(self, m, v):
        KE = 0.5 * np.sum(m * v**2)
        T = self.T
        N_f = self.num_particles
        M = self.M
        SCALE = 1.0
        w = self.w
        for i in range(self.n_c):
            
            delta_ts = self.dt / self.n_c
            
            
            G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
            self.vxi[0] = self.vxi[0] + (delta_ts / 4) * G_1

            SCALE = SCALE * (np.exp(-1 * delta_ts * self.vxi[0] / 2))
            KE = KE * np.exp(-1 * delta_ts * self.vxi[0])
            
            self.xi[0] = self.xi[0] + (delta_ts / 2) * self.vxi[0]
            G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
            
            self.vxi[0] = self.vxi[0] + (delta_ts / 4) * G_1


        
        v_new = v * SCALE
        return v_new

    
    
    def step_someone_elses_code(self, m, v):
        KE = 0.5 * np.sum(m * v**2)
        T = self.T
        d = v.shape[1]
        N_f = self.num_particles * d
        M = self.M
        SCALE = 1.0
        w = self.w
        for i in range(self.n_c):
            
            delta_ts = self.dt / self.n_c
            dt_2 = 0.5 * delta_ts
            dt_4 = 0.5 * dt_2
            dt_8 = 0.5 * dt_4
            
            G_2 = self.Q[0] * self.vxi[0] * self.vxi[0] - Units.kB * T
            # print(self.Q[0], self.vxi[0], T)
            # print(G_2)
            self.vxi[1] += G_2 * (dt_4)
            # print(self.vxi[1])
            self.vxi[0] = self.vxi[0] * math.exp(-self.vxi[1] * delta_ts / 8)
            # print(self.vxi[0])
            G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
            self.vxi[0] = self.vxi[0] + (dt_4) * G_1
            self.vxi[0] = self.vxi[0] * math.exp(-self.vxi[1] * dt_8)

            SCALE = SCALE * (math.exp(-1 * dt_2 * self.vxi[0]))
            KE = KE * math.exp(-1 * delta_ts * self.vxi[0])
            
            self.xi[0] = self.xi[0] + (dt_2) * self.vxi[0]
            self.xi[1] = self.xi[1] + (dt_2) * self.vxi[1]

            G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
            # print(G_1)
            
            self.vxi[0] = self.vxi[0] * math.exp(-self.vxi[1] * dt_8)
            self.vxi[0] = self.vxi[0] + (dt_4) * G_1
            self.vxi[0] = self.vxi[0] * math.exp(-self.vxi[1] * dt_8)

            G_2 = (self.Q[0] * self.vxi[0] * self.vxi[0] - Units.kB * T) / self.Q[1]
            self.vxi[1] = self.vxi[1] + (G_2 * dt_4)
            # print(self.vxi[1])
            # print("-----")
        
        v_new = v * SCALE
        return v_new

    ## TUCKERMAN PAPER DEFINITION:
    def compute_G(self, KE, ind):
        N_f = self.num_particles
        if ind == 0:
            return (2 * KE - N_f * Units.kB * self.T) / self.Q[0]
        return (self.Q[ind-1] * self.vxi[ind-1]**2 - Units.kB * self.T) / self.Q[ind]

    def step_tuckerman_paper(self, m, v):
        KE = 0.5 * np.sum(m * v**2)
        dt = self.dt
        G_M = self.compute_G(KE, self.M - 1)
        
        self.vxi[self.M - 1] += (G_M * dt / 4)
        
        for i in range(self.M - 2, -1, -1):
            self.vxi[i] *= np.exp(-self.vxi[i + 1] * dt / 8)
            G = self.compute_G(KE, i)
            self.vxi[i] += (G * dt / 4)
            self.vxi[i] *= np.exp(-self.vxi[i + 1] * dt / 8)
        
        SCALE = np.exp(-self.vxi[0] * dt / 2)
        
        vel_new = v * SCALE
        KE *= SCALE**2
        
        for i in range(self.M):
            self.xi[i] += self.vxi[i] * dt / 2
        
        for i in range(0, self.M - 1):
            self.vxi[i] *= np.exp(-self.vxi[i + 1] * dt / 8)
            G = self.compute_G(KE, i)
            self.vxi[i] += (G * dt / 4)
            self.vxi[i] *= np.exp(-self.vxi[i + 1] * dt / 8)
        
        G = self.compute_G(KE, self.M - 1)
        self.vxi[self.M - 1] += (G * dt / 4)
        
        return vel_new

    def step(self, m, v):
        return self.step_someone_elses_code(m, v)
