import numpy as np

class Minimizer:

    def __init__(self, gamma):
        self.gamma = gamma

    def step(self, sys):
        g_k = sys.F(sys.x)
        x = sys.x + self.gamma * (g_k / np.abs(g_k))
        return x