from mpi4py import MPI
from sim import config
from sim import generate_velocities
from sim import Multiplier
from sim import velocity_verlet_step
from sim import write_to_file
from sim import pot_energy_config
import numpy as np
import os
import math
import random


x = 0
vel_a = 0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

exchange_period = config['exchange_period']
kB = 1

temperatures = [0.05, 0.3, 2.0]

def remd_step():
	global x, vel_a
	# Make the pairs array to determine swap partners

	# Rank 0 is the master process where the pairs array is made.
	pairs = {}
	if rank == 0:
		for i in range(0, size, 2):
			j = (i + 1) % size
			if abs(j - i) != 1:
				continue
			pairs[i] = j
	else:
		pairs = []

	# Broadcast the pairs array
	pairs = comm.bcast(pairs, root = 0)

	# Helper function
	def get_key(val, my_dict):
	    for key, value in my_dict.items():
    		if val == value:
        		return key

	# Key -> Master for the exchange, value -> sends data to master
	if rank in pairs.keys():
		x_j = comm.recv(source = pairs[rank], tag = 11)
		beta_i = 1 / (kB * temperatures[rank])
		beta_j = 1 / (kB * temperatures[pairs[rank]])
		U_i = pot_energy_config(x)
		U_j = pot_energy_config(x_j)
		
		delta =  (beta_i - beta_j) * (U_i - U_j)
		
		exchange = False
		if -1 * delta > 0:
			exchange = True
		else:
			acc_prob = math.exp(-1 * delta)
			acc = min(1, acc_prob)
			if random.random() < acc:
				exchange = True	

		comm.send(exchange, dest = pairs[rank], tag = 11)
		if exchange:
			print("Swap between {} and {} accepted with prob = min(1, exp(- {}))".format(rank, pairs[rank], delta))
			tmp_x = comm.recv(source = pairs[rank], tag = 69)
			comm.send(x, pairs[rank], tag = 11)
			x = tmp_x
			vel_a *= math.sqrt(temperatures[pairs[rank]] / temperatures[rank])
		else:
			print("Swap rejected")

	if rank in pairs.values():
		swap_partner = get_key(rank, pairs)
		comm.send(x, dest = swap_partner, tag = 11)	
		exchange = None
		exchange = comm.recv(source = swap_partner, tag = 11)
		if exchange:
			comm.send(x, dest = swap_partner, tag = 69)
			tmp_x = comm.recv(source = swap_partner, tag = 11)
			x = tmp_x
			vel_a *= math.sqrt(temperatures[swap_partner] / temperatures[rank])
	

def step(num_steps, x, v, m, dt, p_f, v_f):
    global vel_file, pos_file
    vel_file = open(p_f, "w")
    pos_file = open(v_f, "w")
    for i in range(num_steps):
        x_new, v_new = velocity_verlet_step(x, v, m, dt)
        if i % exchange_period == 0:
            remd_step()
            # exchange_x(0, 1)
        write_to_file(x, v, i, pos_file, vel_file)
        #print("i :", i)
        #check_x()
        x, v = x_new, v_new
    pos_file.close()
    vel_file.close()

def run_sim(reduced_temperature, num_steps):
	n = config['num_particles']
	m = np.ones((n, 1))
	d = 1
	file_path = config['data_files']
	os.system("mkdir -p {}".format(file_path))
	T = reduced_temperature * Multiplier
	global vel_a
	vel_a = generate_velocities(n, d, m, T)
	p_f = "{}/p_{}.txt".format(file_path, reduced_temperature)
	v_f = "{}/v_{}.txt".format(file_path, reduced_temperature)
	global x
	x = np.random.normal(-1.6, 1.6, size = (n, d))
	
	step(num_steps, x, vel_a, m, 1e-3, p_f, v_f)

def exchange_x(rank1, rank2):
	global x
	tmp_x = x
	if rank == rank1:
		comm.send(x, dest = rank2, tag = 11)
		tmp_x = comm.recv(source = rank2, tag = 11)
	elif rank == rank2:
		tmp_x = comm.recv(source = rank1, tag = 11)	
		comm.send(x, dest = rank1, tag = 11)
	x = tmp_x
	
	
def check_x():
	global x
	print(rank, x)



num_steps = config['num_steps']
run_sim(temperatures[rank], num_steps)
