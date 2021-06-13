import numpy as np

class Andersen:

    def __init__(self, cfg, m, nu = 0.01):
        self.m = m
        self.nu = nu
        self.reduced_temperature = cfg.reduced_temperature
        self.num_particles = cfg.num_particles

    
    def fetch_new_velocities(self, v, dt):
        nu = self.nu
        acc_value = nu * dt
        m = self.m
        variance = np.sqrt(self.reduced_temperature)
        v_new = v[:]
        flag = 0
        for i in range(self.num_particles):
            if np.random.random(1).item() < acc_value:
                v_new[i, :] = np.random.normal(0, variance, 1)
                flag = 1
        if flag:
            print("Gaussian velocity replaced")
        return v_new

        
