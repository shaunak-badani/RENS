#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math 
import random
import numpy as np
import matplotlib.pyplot as plt


# # Units

# In[ ]:


BOLTZMANN = 1.380649e-23 
AVOGADRO = 6.02214076e23
KILO = 1e3
RGAS = BOLTZMANN*AVOGADRO
BOLTZ = (RGAS/KILO)  


# # System Configuration

# In[ ]:


d = 1
n = 10
Reduced_Temp = 1
Multiplier = 119.8
T = Reduced_Temp * Multiplier


# # General Functions for MD simulations

# ## Velocity Initialization

# In[ ]:


def generate_velocities(number_particles, system_dimension, masses, temperature):
    h = []
    dof = number_particles * system_dimension
    while(len(h) < dof):
        r2 = 0
        while r2 >= 1 or r2 == 0:
            x = 2 * random.uniform(0, 1) - 1
            y = 2 * random.uniform(0, 1) - 1
            r2 = x**2 + y**2
        mult = math.sqrt(- 2 * math.log(r2) / r2)
        h.extend([x*mult, y*mult])
    vels = np.array(h[:dof]).reshape(-1, 1).astype('float32')
    scaledvels = vels * np.sqrt(BOLTZ * temperature / masses)
    return scaledvels


# ## Integrator

# In[ ]:


def velocity_verlet_step(x, v, m, dt):
    F = np.array(list(map(force, x.flatten()))).reshape(x.shape)
    v = v + (dt / 2) * (F / m)
    x = x + (dt) * v
    F = np.array(list(map(force, x.flatten()))).reshape(x.shape)
    v = v + (dt / 2) * (F / m)
    return x, v


# ##  Potential energy and Force function

# In[ ]:


def pot_energy(x):
    rv = 1e2
    if x >= -2 and x <= -1.25:
        rv = 1 + np.sin(2 * np.pi * x)
    
    if x >= -1.25 and x <= -0.25:
        rv = 2 * (1 + np.sin(2 * np.pi * x))
        
    if x >= -0.25 and x <= 0.75:
        rv = 3 * (1 + np.sin(2 * np.pi * x))
                  
    if x >= 0.75 and x <= 1.75:
        rv = 4 * (1 + np.sin(2 * np.pi * x))
                  
    if x >= 1.75 and x <= 2:
        rv = 5 * (1 + np.sin(2 * np.pi * x))
                  
    return rv


# In[ ]:


def force(x):
    lorge = 60
    if x < -2:
        return lorge
    if x > 2:
        return -1 * lorge
    rv = 0
    if x >= -2 and x <= -1.25:
        rv = -2 * np.pi * np.cos(2 * np.pi * x)
    
    if x >= -1.25 and x <= -0.25:
        rv = -4 * np.pi * np.cos(2 * np.pi * x)
        
    if x >= -0.25 and x <= 0.75:
        rv = -6 * np.pi * np.cos(2 * np.pi * x)
                  
    if x >= 0.75 and x <= 1.75:
        rv = -8 * np.pi * np.cos(2 * np.pi * x)
        
    if x >= 1.75 and x <= 2:
        rv = -10 * np.pi * np.cos(2 * np.pi * x)
                  
    return rv


# In[ ]:


tmp_x = np.linspace(-5, 5, 1000)
U = np.array(list(map(pot_energy, tmp_x)))
f = np.array(list(map(force, tmp_x)))


# In[ ]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))

axes[0].plot(tmp_x, f)
axes[0].set_xlabel(r'$x$')
_ = axes[0].set_ylabel(r'$F$')

axes[1].plot(tmp_x, U)
axes[1].set_xlabel(r'$x$')
_ = axes[1].set_ylabel(r'$U$')


# ## Helper Functions

# In[ ]:


def write_to_file(x, v, step):
    p = ' '.join([str(i) for i in x.flatten()])
    pos_file.write(p)
    pos_file.write("\n")
    
    p = ' '.join([str(i) for i in v.flatten()])
    vel_file.write(p)
    vel_file.write("\n")
    

    
def step(num_steps, x, v, m, dt, file_path):
    global vel_file, pos_file
    vel_file = open("{}/{}".format(file_path, "velocities.txt"), "w")
    pos_file = open("{}/{}".format(file_path, "positions.txt"), "w")
    for i in range(num_steps):
        x_new, v_new = velocity_verlet_step(x, v, m, dt)
        write_to_file(x, v, i)
        x, v = x_new, v_new
    pos_file.close()
    vel_file.close()


# ### Simulation at $T_A$ = 0.30

# In[ ]:


config = {
    'num_particles' : 1,
    'temperature' : 0.30,
    'num_steps' : int(1e7),
    'data_files_path' : '/scratch/shaunak/T_0.3'
}


# In[ ]:


reduced_temperature = config['temperature']
n = config['num_particles']
num_steps = config['num_steps']
m = np.ones((n, 1))
file_path = config['data_files_path']
print("Number of particles = ", n)
print("Temperature = ", reduced_temperature)


# In[ ]:


T = reduced_temperature * Multiplier
vel_a = generate_velocities(n, d, m, T)


# In[ ]:


x = np.random.normal(-1.6, 0.9, size = (n, d))


# In[ ]:


print(x.shape, vel_a.shape)


# In[ ]:


import os
os.system("mkdir -p {}".format(file_path))


# In[ ]:


step(num_steps, x, vel_a, m, 1e-3, file_path)

