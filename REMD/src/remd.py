from mpi4py import MPI
from .units import Units
import numpy as np

class REMD:

    def __init__(self, cfg, no_replicas, exchange_period = 100):
        self.no_replicas = no_replicas
        comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.reduced_temperature = cfg.reduced_temperature
        self.temperature = cfg.temperature
        self.exchange_period = exchange_period
    
    def step(self, step_no, sys, cfg, file_io):
        if step_no % self.exchange_period != 0:
            return
        if self.rank == 0:
            file_io.declare_step(step_no)
        
        ## Root makes the swap cycle
        if self.rank == 0:
            if step_no % (2 * self.exchange_period):
                k = self.no_replicas // 2
                src = list(range(0, self.no_replicas, 2))[:k]
                dest = list(range(1, self.no_replicas, 2))[:k]
            else:
                k = (self.no_replicas - 1) // 2
                src = list(range(1, self.no_replicas, 2))[:k]
                dest = list(range(2, self.no_replicas, 2))[:k]
        else:
            src = []
            dest = []

        src = self.comm.bcast(src, root = 0)
        dest = self.comm.bcast(dest, root = 0)

        if self.rank in src:
            swap_partner_index = src.index(self.rank)
            swap_partner = dest[swap_partner_index]
            x_j = self.comm.recv(source = swap_partner, tag = 11)
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
                self.comm.send(sys.x, dest = swap_partner, tag = 15)
                self.comm.send(cfg.temperature, dest = swap_partner, tag = 16)
                sys.set_x(x_j)
                cfg.temperature = t_j
                cfg.reduced_temperature = (Units.kB / Units.epsilon) * t_j
            
        if self.rank in dest:
            swap_partner_index = dest.index(self.rank)
            swap_partner = src[swap_partner_index]
            self.comm.send(sys.x, dest = swap_partner, tag = 11)	
            self.comm.send(self.temperature, dest = swap_partner, tag = 12)	

            # Receive 
            exchange = self.comm.recv(source = swap_partner, tag = 14)

            if exchange:
                x_i = self.comm.recv(source = swap_partner, tag = 15)
                t_i = self.comm.recv(source = swap_partner, tag = 16)
                sys.set_x(x_i)
                cfg.temperature = t_i
                cfg.reduced_temperature = (Units.kB / Units.epsilon) * t_i

        # Root collects data of all exchanges
        if self.rank == 0:
            if self.rank in src:
                out_exchange = "Yes" if exchange else "No"
                file_io.write_exchanges(self.rank, swap_partner, out_exchange, acc_prob)

                # 0 will always be first element in src
                src.pop(0)

            for exchange_initiator in src:
                exchange = self.comm.recv(source = exchange_initiator, tag = 17)
                dest_rank = self.comm.recv(source = exchange_initiator, tag = 18)
                acc_prob = self.comm.recv(source = exchange_initiator, tag = 19)
                out_exchange = "Yes" if exchange else "No"
                file_io.write_exchanges(exchange_initiator, dest_rank, out_exchange, acc_prob)
                
            file_io.done_step()

        if self.rank in src:
            self.comm.send(exchange, dest = 0, tag = 17)
            self.comm.send(swap_partner, dest = 0, tag = 18)
            self.comm.send(acc_prob, dest = 0, tag = 19)

    def update_environment(self, cfg, nht, sys):
        old_temperature = nht.T
        new_temperature = cfg.temperature

        # Rescale velocities
        v_new = sys.v * (new_temperature / old_temperature)
        sys.set_v(v_new)

        # Set the temperature in the thermostat
        nht.T = new_temperature
        sys.T = new_temperature

        return v_new


            
            





                    
                
        

    

        

    

    
    