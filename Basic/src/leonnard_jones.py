from .units import Units

import numpy as np
import random
import math
from .system import System

class LJ(System):

    def __init_velocities(self, cfg):
        N = cfg.num_particles
        T = cfg.temperature

        vel = np.random.uniform(-0.5, 0.5, N*3).reshape(N,3) 
    
        # define center of mass 
        cm_x = np.sum(self.m * vel[:,0] / self.m) / N
        cm_y = np.sum(self.m * vel[:,1] / self.m) / N
        cm_z = np.sum(self.m * vel[:,2] / self.m) / N
        
        # initialize kinetic energy
        ke = 0
        
        # elminate center of mass drift (make zero momentum)
        for i in range(N):    
            vel[i,0] -= cm_x
            vel[i,1] -= cm_y
            vel[i,2] -= cm_z
        
        # obtain kinetic energy from velocity
        ke = 0.5 * np.square(self.m * vel).sum()
        
        # define 'scale velocity'
        T_temp = ke*2 / (3*N)
        scale = math.sqrt(T/T_temp)
        vel = np.multiply(scale, vel)
        # vel = np.loadtxt('../../Nose-Hoover-Chain/vel.txt')

        self.T = T
        self.N = N
        self.v = vel
        

    def __init__(self, rc = 6):
        super().__init__()
        N = Config.num_particles
        self.rc = 6
        V = 1000
        bx = by = bz = V**(1.0 / 3)
        ix, iy, iz = [0, 0 ,0]
        coord=np.zeros(shape=[N,3], dtype="float")
        n_3=int(math.floor(N**(1/3) + 0.5))
        # assign particle postions
        for i in range(N):
            coord[i,0]=float((ix+0.5)*bx/n_3)
            coord[i,1]=float((iy+0.5)*by/n_3)
            coord[i,2]=float((iz+0.5)*bz/n_3)
            ix += 1
            if ix == n_3:
                ix = 0
                iy += 1
            if iy == n_3:
                iy = 0
                iz +=1


        self.x = coord
        # self.x = np.loadtxt('../../Nose-Hoover-Chain/pos.txt')
        cfg.num_particles = self.x.shape[0]
        self.m = np.ones((N, 1), dtype="float")
        self.__init_velocities(cfg)

   
    def U(self, x):
        sigma = 1
        epsilon = 1
        N = self.N
        pe = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                r = x[i] - x[j]
                r_norm = np.linalg.norm(r)
                if r_norm < self.rc:
                    pe += 4 * epsilon * ((sigma / r_norm)**12 - (sigma / r_norm)**6)        
        return pe

                 
    def F(self, x):
        sigma = 1
        epsilon = 1
        N = self.N
        f = np.zeros_like(x)
        for i in range(N - 1):
            for j in range(i + 1, N):
                r = x[i] - x[j]
                r_norm = np.linalg.norm(r)
                F = 48 * epsilon * ((sigma/r_norm)**12 - 0.5 * (sigma/r_norm)**6) / r_norm**2
                f[i] += F * r
                f[j] -= F * r
        return f

    def instantaneous_T(self, v):
        N = self.N
        sumv2 = np.sum(self.m * v**2)
        t = sumv2 / (3 * N * Units.kB)
        # r_t = (Units.kB / Units.epsilon) * t
        return t
