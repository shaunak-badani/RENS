from .units import Units
from .config import Config

import numpy as np
import pandas as pd
import random
import math
from .system import System

class LEPS_I(System):
    a = 0.05
    b = 0.30
    c = 0.05
    dAB = dBC = 4.746
    dAC = 3.445
    r0 = 0.742
    alpha = 1.942

    def __init_velocities(self):
        N = Config.num_particles
        T = Config.T()
        v = np.random.random(size = (N, 2)) - 0.5
        sumv2 = np.sum(self.m * v**2)
        fs = np.sqrt((N * Units.kB * T) / sumv2)

        # Sampling from maxwell boltzmann distribution
        # beta = 1 / (Config.T() * Units.kB)
        # sigma = 1 / np.sqrt(self.m * beta)
        # v = np.random.normal(size = (N, 1), scale = sigma)
        # fs = 1
        self.v = v * fs 

    def __init__(self):

        self.N = Config.num_particles
        x = np.random.uniform(0.5, 1.0, size = (self.N, 1))
        y = np.random.uniform(0.5, 4.0, size = (self.N, 1))
        self.x = np.hstack((x, y))
        self.m = np.ones(shape = (self.N, 1))

        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            self.x = df['x'].dropna().to_numpy().reshape(-1, 2)
            N = self.x.shape[0]

            self.m = df['m'].dropna().to_numpy().reshape(-1, 1)
            self.N = N
            Config.num_particles = N

        self.__init_velocities()

        self.QAB = Q(self.dAB, self.alpha, self.r0)
        self.QBC = Q(self.dBC, self.alpha, self.r0)
        self.QAC = Q(self.dAC, self.alpha, self.r0)
        
        self.JAB = J(self.dAB, self.alpha, self.r0)
        self.JBC = J(self.dBC, self.alpha, self.r0)
        self.JAC = J(self.dAC, self.alpha, self.r0)
        

    def U(self, x):
        '''
        Args:
        x -> (N, 2) array containing positions of N particles
        Returns :
        u : potential energy of the system, scalar
        '''
        rAB = x[:, 0]
        rBC = x[:, 1]
        QAB = self.QAB.value(rAB)
        QBC = self.QBC.value(rBC)
        rAC = rAB + rBC
        QAC = self.QAC.value(rAC)
        
        JAB = self.JAB.value(rAB)
        JBC = self.JBC.value(rBC)
        JAC = self.JAC.value(rAC)
        
        a = self.a
        b = self.b
        c = self.c
        Q_values = (QAB / (1 + a)) + (QBC / (1 + b)) + (QAC / (1 + c)) 
        J_values = (JAB / (1 + a))**2 + (JBC / (1 + b))**2 + (JAC / (1 + c))**2
        J_values = J_values - ((JAB*JBC/((1+a)*(1+b))) + (JBC*JAC/((1+b)*(1+c))) + (JAB*JAC/((1+a)*(1+c))))
        return np.sum(Q_values - np.sqrt(J_values))

    def F(self, x):
        '''
        Args:
        x -> (N, 2) array containing positions of N particles
        Returns :
        u : potential energy of the system, scalar
        '''
        rAB = x[:, 0]
        rBC = x[:, 1]
        a = self.a
        b = self.b
        c = self.c
        rAC = rAB + rBC
        J_AB = self.JAB
        J_BC = self.JBC
        J_AC = self.JAC
        
        # Computing F_x
        F_x = Q(self.dAB, self.alpha, self.r0).der(rAB) / (1 + a)
        F_x += Q(self.dAC, self.alpha, self.r0).der(rAC) / (1 + c)
        
        comp_x = (2 * J_AB.value(rAB) * J_AB.der(rAB) / ((1 + a)**2) + 2 * J_AC.value(rAC) * J_AC.der(rAC) / ((1 + c)**2))
        comp_x -= (J_AB.der(rAB) * J_BC.value(rBC) / ((1 + a)*(1 + b)) + J_BC.value(rBC) * J_AC.der(rAC) / ((1 + b)*(1 + c)))
        comp_x -= ((J_AB.der(rAB) * J_AC.value(rAC) + J_AC.der(rAC) * J_AB.value(rAB)) / ((1 + a) * (1 + c)))
        
        jAB = J_AB.value(rAB)
        jBC = J_BC.value(rBC)
        jAC = J_AC.value(rAC)
        
        J_values = (jAB / (1 + a))**2 + (jBC / (1 + b))**2 + (jAC / (1 + c))**2
        J_values = J_values - ((jAB*jBC/((1+a)*(1+b))) + (jBC*jAC/((1+b)*(1+c))) + (jAB*jAC/((1+a)*(1+c))))
        comp_x *= 1 / (2 * np.sqrt(J_values))
        F_x -= comp_x
        
        # Computing F_y
        F_y = Q(self.dBC, self.alpha, self.r0).der(rBC) / (1 + b)
        F_y += Q(self.dAC, self.alpha, self.r0).der(rAC) / (1 + c)
        
        comp_y = (2 * J_BC.value(rBC) * J_BC.der(rBC) / ((1 + b)**2) + 2 * J_AC.value(rAC) * J_AC.der(rAC) / ((1 + c)**2))
        comp_y -= (J_AB.value(rAB) * J_BC.der(rBC) / ((1 + a)*(1 + b)) + J_AB.value(rAB) * J_AC.der(rAC) / ((1 + a)*(1 + c)))
        comp_y -= ((J_BC.der(rBC) * J_AC.value(rAC) + J_BC.value(rBC) * J_AC.der(rAC)) / ((1 + b) * (1 + c)))
        
        comp_y *= 1 / (2 * np.sqrt(J_values))
        F_y -= comp_y
        return np.array([-F_x, -F_y]).T

class LEPS_II(LEPS_I):
    rAC = 3.742
    kC = 0.2025
    c = 1.154
    
    def __init__(self):
        super().__init__()

        x = np.random.uniform(0.5, 1.0, size = (self.N, 1))
        y = np.random.uniform(0.0, 1.0, size = (self.N, 1))
        self.x = np.hstack((x, y))
        self.m = np.ones(shape = (self.N, 1))

        if Config.rst:
            df = pd.read_csv(Config.rst, sep = ' ')
            self.x = df['x'].dropna().to_numpy().reshape(-1, 2)
            N = self.x.shape[0]

            self.m = df['m'].dropna().to_numpy().reshape(-1, 1)
            self.N = N
            Config.num_particles = N
        
    
    def U(self, r):
        '''
        Args : 
        r -> (N, 2) 
        Return value : 
        potential energy
        '''
        rAB = r[:, 0]
        x = r[:, 1]
        U_normal = super().U(np.array([rAB, self.rAC - rAB]).T)
        
        return U_normal + np.sum(2 * self.kC * (rAB - (self.rAC / 2 - x / self.c))**2)
    
    def F(self, r):
        rAB = r[:, 0]
        x = r[:, 1]
        F_I = super().F(np.array([rAB, self.rAC - rAB]).T)
        F_x = F_I[:, 0] - F_I[:, 1] - 4 * self.kC * (rAB - (self.rAC / 2 - x / self.c))
        
        F_y = -4 * (self.kC / self.c) * (rAB - (self.rAC / 2 - x / self.c)) 
        return np.array([F_x, F_y]).T


class Q:
    def __init__(self, d, alpha, r0):
        self.d = d
        self.alpha = alpha
        self.r0 = r0
    
    def value(self, r):
        d = self.d
        alpha = self.alpha
        r0 = self.r0
        return (d / 2) * (1.5 * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))
    
    def der(self, r):
        d = self.d
        alpha = self.alpha
        r0 = self.r0
        return (-d * alpha / 2) * (3 * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))

class J:
    
    def __init__(self, d, alpha, r0):
        self.d = d
        self.alpha = alpha
        self.r0 = r0
    
    def masked_exponent(self, p):
        capped_exponent = np.zeros_like(p)
        cap = 700
        capped_exponent[p > cap] = cap
        capped_exponent[p < cap] = p
        
        return np.exp(p)
        
    def value(self, r):
        d = self.d
        alpha = self.alpha
        r0 = self.r0
        exp = self.masked_exponent
        return (d / 4) * (exp(-2 * alpha * (r - r0)) - 6 * exp(-alpha * (r - r0)))
    
    def der(self, r):
        d = self.d
        alpha = self.alpha
        r0 = self.r0
        exp = self.masked_exponent
        return (-d * alpha / 2) * (exp(-2 * alpha * (r - r0))  - 3 * exp(-alpha * (r - r0)))
