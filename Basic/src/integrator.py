from .system import System
from .units import Units
import numpy as np
from .config import Config

class VelocityVerletIntegrator:
    def __init__(self):
        self.dt = 1e-3

       

    def step(self, sys, step, x = None, v = None, m = None):
        if x is None:
            x = sys.x
        if v is None:
            v = sys.v
        if m is None:
            m = sys.m
        F = sys.F(x)
        v_new = v + (self.dt / 2) * (F / m)
        x_new = x + (self.dt) * v_new
        F = sys.F(x_new)
        v_new = v_new + (self.dt / 2) * (F / m)
        return x_new, v_new


class REMDIntegrator(VelocityVerletIntegrator):
    
    def __init__(self):
        self.exchange_period = 1000    
        super().__init__()
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.no_replicas = self.comm.Get_size()
        self.exchange_attempts = 0
    
    def swap_positions(self, x, v, swap_partner, src):
        # If the current rank is not a source, it is a destination
        # print("SWAP PARTNER, SRC : ", swap_partner, src, x, v)
        if src:
            self.comm.send(x, dest = swap_partner, tag = 6)
            self.comm.send(v, dest = swap_partner, tag = 7)
            # print("Src : ", self.comm.rank)
            y_x = self.comm.recv(source = swap_partner, tag = 8)
            y_v = self.comm.recv(source = swap_partner, tag = 9)
            
        else:
            y_x = self.comm.recv(source = swap_partner, tag = 6)
            y_v = self.comm.recv(source = swap_partner, tag = 7)

            # print("not src: ", y_v)
            self.comm.send(x, dest = swap_partner, tag = 8)
            self.comm.send(v, dest = swap_partner, tag = 9)

        return y_x, y_v
    

    def __rex_exchange_as_leader(self, self_energy, peer_rank):
        energy = self.comm.recv(source = peer_rank, tag = 1)
        peer_id = self.comm.recv(source = peer_rank, tag = 2)
        peer_temp = self.comm.recv(source = peer_rank, tag = 3)

        self_temp = Config.T()

        beta_i = 1 / (Units.kB * self_temp)
        beta_j = 1 / (Units.kB * peer_temp)

        delta = (beta_i - beta_j) * (energy - self_energy)

        exchange = True
        metropolis = 1
        if delta > 0:
            metropolis = np.exp(-delta)
            u_rand = np.random.uniform()

            if u_rand >= metropolis:
                exchange = False

        self.comm.send(exchange, dest = peer_rank, tag = 4)
        factor = 1
        if exchange:
            self.comm.send(Config.replica_id, dest = peer_rank, tag = 5)
            factor = self.scale_velocity_factor(Config.T(), Config.temperatures[peer_id])
            # Config.replica_id = peer_id
        return exchange, metropolis, factor
        

    def __rex_exchange_as_follower(self, energy, peer_rank):
        self.comm.send(energy, dest = peer_rank, tag = 1)
        self.comm.send(Config.replica_id, dest = peer_rank, tag = 2)
        self.comm.send(Config.T(), dest = peer_rank, tag = 3)

        exchange = self.comm.recv(source = peer_rank, tag = 4)
        factor = 1
        if exchange:
            peer_id = self.comm.recv(source = peer_rank, tag = 5)
            factor = self.scale_velocity_factor(Config.T(), Config.temperatures[peer_id])

            # Config.replica_id = peer_id
        
        return exchange, 0.69, factor

    
    def step(self, x, v, energy, step_no, file_io):
        
        
        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1        

        y_x = x
        y_v = v
        exchange = False
        if peer_rank >= 0 and peer_rank < self.no_replicas:
            if self.rank > peer_rank:
                exchange, acc_prob, factor = self.__rex_exchange_as_leader(energy, peer_rank)
                file_io.declare_step(step_no)
                file_io.write_exchanges(self.rank, peer_rank, exchange, acc_prob)
                
            else:
                exchange, _, factor = self.__rex_exchange_as_follower(energy, peer_rank)
            
            exchange = True
            if exchange:
                scale_factor = self.scale_velocity_factor(Config.T(), Config.temperatures[peer_rank])
                v *= scale_factor
                y_x, y_v = self.swap_positions(x, v, peer_rank, self.rank > peer_rank)

                # file_io.update_files()
        return y_x, y_v

        
    
    def scale_velocity_factor(self, T_old, T_new):
        
        # Rescale velocities
        # T_current = sys.instantaneous_T(v)
        return np.sqrt(T_new / T_old)
