#!/usr/bin/env python
# coding: utf-8

import math 
import random
import numpy as np
import matplotlib.pyplot as plt


BOLTZMANN = 1.380649e-23 
AVOGADRO = 6.02214076e23
KILO = 1e3
RGAS = BOLTZMANN*AVOGADRO
BOLTZ = (RGAS/KILO)  


Multiplier = 119.8


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


def velocity_verlet_step(x, v, m, dt):
    F = np.array(list(map(force, x.flatten()))).reshape(x.shape)
    v = v + (dt / 2) * (F / m)
    x = x + (dt) * v
    F = np.array(list(map(force, x.flatten()))).reshape(x.shape)
    v = v + (dt / 2) * (F / m)
    return x, v



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

def pot_energy_config(x):
	U = np.array(list(map(pot_energy, x)))
	#print(U)
	return U.sum(axis = 0).item()


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


def write_to_file(x, v, step, pos_file, vel_file):
    p = ' '.join([str(i) for i in x.flatten()])
    pos_file.write(p)
    pos_file.write("\n")
    
    p = ' '.join([str(i) for i in v.flatten()])
    vel_file.write(p)
    vel_file.write("\n")
    

    
def step(num_steps, x, v, m, dt, p_f, v_f):
    global vel_file, pos_file
    vel_file = open(p_f, "w")
    pos_file = open(v_f, "w")
    for i in range(num_steps):
        x_new, v_new = velocity_verlet_step(x, v, m, dt)
        write_to_file(x, v, i)
        x, v = x_new, v_new
    pos_file.close()
    vel_file.close()



config = {
    'num_particles' : 1,
    'data_files' : '/scratch/shaunak/1D_Run',
	'exchange_period' : 100,
	'num_steps' : int(1e5)
}




import os

def run_sim(reduced_temperature, num_steps):
	n = config['num_particles']
	m = np.ones((n, 1))
	d = 1
	file_path = config['data_files']
	os.system("mkdir -p {}".format(file_path))
	T = reduced_temperature * Multiplier
	vel_a = generate_velocities(n, d, m, T)
	p_f = "{}/p_{}.txt".format(file_path, reduced_temperature)
	v_f = "{}/v_{}.txt".format(file_path, reduced_temperature)
	x = np.random.normal(-1, 1, size = (n, d))
	
	step(num_steps, x, vel_a, m, 1e-3, p_f, v_f)


if __name__ == "__main__":
	reduced_temperature = 2.0
	num_steps = 10
	print("Temperature = ", reduced_temperature)
	run_sim(reduced_temperature, num_steps)

