from .system import System
from .units import Units
import numpy as np

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
        self.exchange_period = 500    
        super().__init__()
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.no_replicas = self.comm.Get_size()
        self.exchange_attempts = 0
    
    def swap_positions(self, sys, swap_partner, src):
        # If the current rank is not a source, it is a destination
        print("SWAP PARTNER, SRC : ", swap_partner, src)
        x = []
        if src:
            self.comm.send(sys.x, dest = swap_partner, tag = 1)
            x = self.comm.recv(source = swap_partner, tag = 2)
            
        else:
            x = self.comm.recv(source = swap_partner, tag = 1)
            self.comm.send(sys.x, dest = swap_partner, tag = 2)
        return x
    

    def __rex_exchange_as_leader(self, self_energy, peer_rank, cfg, sys):
        energy = self.comm.recv(source = peer_rank, tag = 1)

        self_temp = cfg.temperature
        peer_temp = cfg.temperatures[peer_rank]

        u_rand = np.random.uniform()
        delta = 1 / Units.kB * (1 / self_temp - 1 / peer_temp) * (energy - self_energy)
        exchange = True
        if delta > 0:
            metropolis = np.exp(-delta)

            if u_rand >= metropolis:
                exchange = False

        x = sys.x
        v = sys.v
        self.comm.send(exchange, dest = peer_rank, tag = 2)
        if exchange:
            self.comm.send(sys.x, dest = peer_rank, tag = 3)
            self.comm.send(sys.v, dest = peer_rank, tag = 4)
            x = self.comm.recv(source = peer_rank, tag = 5)
            v = self.comm.recv(source = peer_rank, tag = 6)
        
            t_cur = sys.instantaneous_T(v)
            v *= np.sqrt(cfg.temperatures[peer_rank] / t_cur)
        return x, v, exchange

    def __rex_exchange_as_follower(self, energy, peer_rank, cfg, sys):
        self.comm.send(energy, dest = peer_rank, tag = 1)
        x = sys.x[:]
        v = sys.v[:]
        exchange = self.comm.recv(source = peer_rank, tag = 2)
        if exchange:
            x = self.comm.recv(source = peer_rank, tag = 3)
            v = self.comm.recv(source = peer_rank, tag = 4)
            self.comm.send(sys.x, dest = peer_rank, tag = 5)
            self.comm.send(sys.v, dest = peer_rank, tag = 6)

            t_cur = sys.instantaneous_T(v)
            v *= np.sqrt(cfg.temperatures[peer_rank] / t_cur)

        return x, v, exchange

    
    def step(self, sys, cfg, step_no, file_io):
        if step_no % self.exchange_period != 0:
            return sys.x, sys.v
        if self.rank == 0:
            file_io.declare_step(step_no)
        
        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1        

        x = sys.x[:]
        v = sys.v[:]

        energy = sys.U(x)

        if peer_rank >= 0 and peer_rank < self.no_replicas:
            if self.rank > peer_rank:
                x, v, exchange = self.__rex_exchange_as_leader(energy, peer_rank, cfg, sys)
            else:
                x, v, exchange = self.__rex_exchange_as_follower(energy, peer_rank, cfg, sys)

        return x, v

        
    
    def rescale_velocities(self, v, sys, cfg):
        
        # Rescale velocities
        T_current = sys.instantaneous_T(v)
        v_new = v * (cfg.temperature / T_current)
        return v_new