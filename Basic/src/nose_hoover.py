import numpy as np
from .units import Units

class NoseHoover():

    def __init__(self, dt, cfg, M = 5):
        self.n_c = 1
        self.M = M
        self.v_epsilon = np.zeros(M)
        self.epsilon = np.zeros(M)
        self.Q = np.full(M, 1000)
        self.dt = dt
        self.T = cfg.temperature
        self.num_particles = cfg.num_particles


    def step(self, m, v):
        for i in range(self.n_c):
            KE = 0.5 * np.sum(m * v**2)
            T = self.T
            N_f = self.num_particles
            M = self.M
            delta_ts = self.dt / self.n_c
            SCALE = 1.0

            G_M_1 = (self.Q[M - 2] * self.v_epsilon[M - 2] ** 2 - Units.kB * T) / self.Q[M - 1]
            self.v_epsilon[M - 1] = self.v_epsilon[M - 1] + (delta_ts / 4) * G_M_1

            # 4 part repeat
            for i in range(self.M - 2, 0, -1):
                self.v_epsilon[i] = self.v_epsilon[i] * np.exp(-1 * delta_ts * self.v_epsilon[i + 1] / 8)
                G_i = (self.Q[i - 1] * self.v_epsilon[i-1]**2 - Units.kB * T) / self.Q[i]
                self.v_epsilon[i] = self.v_epsilon[i] + (delta_ts / 4) * G_i
                self.v_epsilon[i] = self.v_epsilon[i] * np.exp(-1 * delta_ts * self.v_epsilon[i + 1] / 8)

            self.v_epsilon[0] = self.v_epsilon[0] * np.exp(-1 * delta_ts * self.v_epsilon[1] / 8)
            G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
            self.v_epsilon[0] = self.v_epsilon[0] + (delta_ts / 4) * G_1
            self.v_epsilon[0] = self.v_epsilon[0] * np.exp(-1 * delta_ts * self.v_epsilon[1] / 8)

            SCALE = SCALE * (np.exp(-1 * delta_ts * self.v_epsilon[0] / 2))
            KE = KE * np.exp(-1 * delta_ts * self.v_epsilon[0])

            for i in range(0, M):
                self.epsilon[i] = self.epsilon[i] + (delta_ts / 2) * self.v_epsilon[i]

            self.v_epsilon[0] = self.v_epsilon[0] * np.exp(-1 * delta_ts * self.v_epsilon[1] / 8)
            G_1 = (2 * KE - N_f * Units.kB * T) / self.Q[0]
            self.v_epsilon[0] = self.v_epsilon[0] + (delta_ts / 4) * G_1
            self.v_epsilon[0] = self.v_epsilon[0] * np.exp(-1 * delta_ts * self.v_epsilon[1] / 8)

            # 4 part repeat, but in reverse direction
            for i in range(M - 1):
                self.v_epsilon[i] = self.v_epsilon[i] * np.exp(-1 * delta_ts * self.v_epsilon[i + 1] / 8)
                G_i = (self.Q[i-1] * self.v_epsilon[i-1]**2 - Units.kB * T) / self.Q[i]
                self.v_epsilon[i] = self.v_epsilon[i] + (delta_ts / 4) * G_i
                self.v_epsilon[i] = self.v_epsilon[i] * np.exp(-1 * delta_ts * self.v_epsilon[i + 1] / 8)
            
            G_M_1 = (self.Q[M-2] * self.v_epsilon[M-2]**2 - Units.kB * T) / self.Q[M - 1]
            self.v_epsilon[M-1] = self.v_epsilon[M - 1] + (delta_ts / 4) * G_M_1


        
        v_new = v * SCALE
        return v_new
        
