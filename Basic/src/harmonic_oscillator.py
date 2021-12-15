from .units import Units

import numpy as np
from .config import Config
from .system import System


class HarmonicOscillator(System):
   
    def __init_velocities(self):
        N = Config.num_particles
        T = Config.T()
        v = np.random.random(size = (N, 1)) - 0.5
        sumv2 = np.sum(self.m * v**2)
        fs = np.sqrt((N * Units.kB * T) / sumv2)
        self.v = v * fs 
        

    def __init__(self, file_io = None):
        N = Config.num_particles
        self.x = np.random.normal(0, 5.0, size = (N, 1))
        self.m = np.full((N, 1), 1) # kg / mol
        self.omega = 1
        self.k = self.omega**2 * self.m
        self.__init_velocities()

        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            self.x = df['x'].dropna().to_numpy().reshape(-1, 1)
            N = self.x.shape[0]
            self.v = df['v'].dropna().to_numpy().reshape(-1, 1)
            self.m = df['m'].dropna().to_numpy().reshape(-1, 1)
            self.N = N
            Config.num_particles = N
        
        
    def pot_energy(self, m, x):
        return self.k * x**2 / 2

    def U(self, x):
        u = self.k * x**2 / 2
        return u.sum()

    
    def __force(self, x):
        return -self.k * x
                    
    def F(self, x):
        f = -self.k * x
        return f
