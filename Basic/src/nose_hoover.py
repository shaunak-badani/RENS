import numpy as np
from .units import Units
from .system import System

class NoseHoover():

    def __init__(self, dt, cfg, M = 1, freq = 1, n_c = 1, w = [1]):
        self.n_c = n_c
        self.w = w
        self.M = M
        self.vxi = np.zeros(M)
        self.xi = np.zeros(M)
        self.dt = dt
        # self.T = cfg.temperature
        self.T = cfg.reduced_temperature
        self.num_particles = cfg.num_particles
        # Q_p1 = (self.num_particles * Units.kB * self.T) / (freq**2)
        # Q_pi = (Units.kB * self.T) / (freq**2)

        # self.Q = np.full(M, Q_pi)
        # self.Q[0] = Q_p1
        self.Q = np.full(M, 0.1)


    def h_prime(self, x, v, m):
        total_system_energy = System.U(x).sum() + 0.5 * np.sum(m * v**2).item()
        total_surrounding_energy = (self.num_particles * self.xi[0]) * Units.kB * self.T + 0.5 * np.sum(self.Q * self.vxi**2)
        return total_system_energy + total_surrounding_energy

    def step(self, m, v):
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
