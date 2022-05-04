from .units import Units

import numpy as np
import random
import math
import os
import pandas as pd
from .config import Config

from .system import System

class TestSystem(System):


    def __init__(self, file_io = None):
        N = 1

        self.x = np.array([[2]])
        self.v = np.array([[2]])

        self.m = np.full((N, 1), 1) # kg / mol

    def U(self, q):
        return np.sum(2 * (q - 2)**2)
        
                    
    def F(self, q):
        return - 4 * (q - 2)
