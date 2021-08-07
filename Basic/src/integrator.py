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
    
    def step(self, sys, cfg, step_no, file_io):
        if step_no % self.exchange_period != 0:
            return sys.x, sys.v
        if self.rank == 0:
            file_io.declare_step(step_no)
        
        ## Root makes the swap cycle
        if self.rank == 0:
            k = self.no_replicas // 2
            src = list(range(0, self.no_replicas, 2))[:k]
            dest = list(range(1, self.no_replicas, 2))[:k]
            # if step_no % (2 * self.exchange_period):
            #     k = self.no_replicas // 2
            #     src = list(range(0, self.no_replicas, 2))[:k]
            #     dest = list(range(1, self.no_replicas, 2))[:k]
            # else:
            #     k = (self.no_replicas - 1) // 2
            #     src = list(range(1, self.no_replicas, 2))[:k]
            #     dest = list(range(2, self.no_replicas, 2))[:k]
        else:
            src = []
            dest = []

        src = self.comm.bcast(src, root = 0)
        dest = self.comm.bcast(dest, root = 0)

        x = sys.x[:]
        v = sys.v[:]

        if self.rank in src:
            swap_partner_index = src.index(self.rank)
            swap_partner = dest[swap_partner_index]
            x_j = self.comm.recv(source = swap_partner, tag = 10)
            v_j = self.comm.recv(source = swap_partner, tag = 11)
            t_j = self.comm.recv(source = swap_partner, tag = 12)

            # Determine if the exchange should happen
            U_i = sys.U(sys.x)
            U_j = sys.U(x_j)
            beta_i = 1 / (Units.kB * cfg.temperature)
            beta_j = 1 / (Units.kB * t_j)
            delta = (beta_j - beta_i) * (U_i - U_j)

            exchange = False
            if delta < 0:
                exchange = True
                acc_prob = 1
            else:
                acc_prob = np.exp(-delta)
                if np.random.uniform() < acc_prob:
                    exchange = True

            # Send exchange value to dest            
            self.comm.send(exchange, dest = swap_partner, tag = 14)

            if exchange:
                # Swap position values
                self.comm.send(x, dest = swap_partner, tag = 15)
                self.comm.send(v, dest = swap_partner, tag = 16)
                self.comm.send(cfg.temperature, dest = swap_partner, tag = 17)
                x = x_j
                v = v_j
                # v = self.rescale_velocities(v, sys, cfg)                

            # Send exchange data to root for logging
            self.comm.send(exchange, dest = 0, tag = 18)
            self.comm.send(swap_partner, dest = 0, tag = 19)
            self.comm.send(acc_prob, dest = 0, tag = 20)
            
        if self.rank in dest:
            swap_partner_index = dest.index(self.rank)
            swap_partner = src[swap_partner_index]
            self.comm.send(x, dest = swap_partner, tag = 10)
            self.comm.send(v, dest = swap_partner, tag = 11)
            self.comm.send(cfg.temperature, dest = swap_partner, tag = 12)	

            # Receive 
            exchange = self.comm.recv(source = swap_partner, tag = 14)

            if exchange:
                x = self.comm.recv(source = swap_partner, tag = 15)
                v = self.comm.recv(source = swap_partner, tag = 16)
                t = self.comm.recv(source = swap_partner, tag = 17)
                # v = self.rescale_velocities(v, sys, cfg)



        # Root collects data of all exchanges
        if self.rank == 0:
            if self.rank in src:
                out_exchange = "Yes" if exchange else "No"
                file_io.write_exchanges(self.rank, swap_partner, out_exchange, acc_prob)

                # 0 will always be first element in src
                src.pop(0)

            for exchange_initiator in src:
                exchange = self.comm.recv(source = exchange_initiator, tag = 18)
                dest_rank = self.comm.recv(source = exchange_initiator, tag = 19)
                acc_prob = self.comm.recv(source = exchange_initiator, tag = 20)
                out_exchange = "Yes" if exchange else "No"
                file_io.write_exchanges(exchange_initiator, dest_rank, out_exchange, acc_prob)
                
            file_io.done_step()

        return x, v

        
    
    def rescale_velocities(self, v, sys, cfg):
        
        # Rescale velocities
        T_current = sys.instantaneous_T(v)
        v_new = v * (cfg.temperature / T_current)
        return v_new