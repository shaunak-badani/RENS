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
        if exchange:
            self.comm.send(Config.replica_id, dest = peer_rank, tag = 5)
            Config.replica_id = peer_id
        return exchange, metropolis
        

    def __rex_exchange_as_follower(self, energy, peer_rank):
        self.comm.send(energy, dest = peer_rank, tag = 1)
        self.comm.send(Config.replica_id, dest = peer_rank, tag = 2)
        self.comm.send(Config.T(), dest = peer_rank, tag = 3)

        exchange = self.comm.recv(source = peer_rank, tag = 4)
        if exchange:
            peer_id = self.comm.recv(source = peer_rank, tag = 5)
            Config.replica_id = peer_id
        
        return exchange, 0.69

    
    def step(self, energy, step_no, file_io):
        if step_no % self.exchange_period != 0:
            return
        
        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1        


        exchange = False
        if peer_rank >= 0 and peer_rank < self.no_replicas:
            if self.rank > peer_rank:
                exchange, acc_prob = self.__rex_exchange_as_leader(energy, peer_rank)
                file_io.declare_step(step_no)
                file_io.write_exchanges(self.rank, peer_rank, exchange, acc_prob)
                
            else:
                exchange, _ = self.__rex_exchange_as_follower(energy, peer_rank)
            if exchange:
                file_io.update_files()
        return exchange

        
    
    # def rescale_velocities(self, v, sys, cfg):
        
    #     # Rescale velocities
    #     T_current = sys.instantaneous_T(v)
    #     v_new = v * (cfg.temperature / T_current)
    #     return v_new

class RENSIntegrator(REMDIntegrator):

    def __init__(self):
        super().__init__()

        
        self.attempt_rate = 0.166
        self.t = 0
        self.tau = 1.0

        if hasattr(Config, 'tau'):
            self.tau = Config.tau

        # modes denote what kind of simulation is going on right now
        # mode = 0 => nvt
        # mode = 1 => work simulation
        self.mode = 0
        
    def setup_rens(self, x, v):
        self.T_A = Config.T()
        self.x0 = x
        self.v0 = v

        self.x_prev = x
        self.v_prev = v
        self.t = 0

        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1

        ## Need to exchange temperatures between peer and self
        if peer_rank >= 0 and peer_rank < self.no_replicas:
            if self.rank > peer_rank:
                # send temperature first then receive
                self.comm.send(Config.T(), dest = peer_rank, tag = 1)
                self.T_B = self.comm.recv(source = peer_rank, tag = 2)
            else:
                self.T_B = self.comm.recv(source = peer_rank, tag = 1)
                self.comm.send(Config.T(), dest = peer_rank, tag = 2)


    def lamda(self):
        # Linear protocol
        # returns lambda, der_lamdba
        return (self.t / self.tau), (1 / self.tau)

        # Step like protocol - 2nd paper

    def T_lambda(self):
        l, _ = self.lamda()
        T_A = self.T_A
        T_B = self.T_B
        return T_A + l * (T_B - T_A)

    def attempt(self, x, v):
        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1

        start_work_simulation = False
        if peer_rank >= 0 and peer_rank < self.no_replicas:
            if self.rank > peer_rank:
                rand = np.random.random()
                if rand <= self.attempt_rate * self.dt:
                    start_work_simulation = True
                self.comm.send(start_work_simulation, dest = peer_rank, tag = 1)
            else:
                start_work_simulation = self.comm.recv(source = peer_rank, tag = 1)
            
        if start_work_simulation:
            self.mode = 1
            self.setup_rens(x, v)

    def step(self, sys, step, file_io):
        x = sys.x
        v = sys.v
        F = sys.F
        m = sys.m

        if self.t >= self.tau:
            exchange = self.determine_exchange(step, sys, file_io)
            self.mode = 0
            if not exchange:
                return self.x0, self.v0
            return x, v

        K = sys.K
        _, l_der = self.lamda()
        z =  (self.T_B - self.T_A) * l_der /  (2 * self.T_lambda())
        v = v * np.exp(z * self.dt / 2) + F(x) * (np.exp(z * self.dt / 2) - 1) / (z * m)
        x = x + v * self.dt
        v = v * np.exp(z * self.dt / 2) + F(x) * (np.exp(z * self.dt / 2) - 1) / (z * m)
        self.t += self.dt


        return x, v

    def determine_exchange(self, step, sys, file_io):
        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1

        x = sys.x
        v = sys.v
        U = sys.U
        K = sys.K

        h_b_x = (sys.K(v) + sys.U(x)) / (self.T_B * Units.kB)
        h_a_x = (sys.K(self.v0) + sys.U(self.x0)) / (self.T_A * Units.kB)

        w_a = h_b_x - h_a_x
        exchange = False
        p_acc = 1.0
        arr = []
        if peer_rank >= 0 and peer_rank < self.no_replicas:
            if self.rank > peer_rank:
                w_b = self.comm.recv(source = peer_rank, tag = 1)
                w = w_a + w_b

                if w < 0:
                    exchange = True
                else:
                    p_acc = np.exp(-w)
                    rand = np.random.random()

                    if rand <= p_acc:
                        exchange = True

                arr = [step, self.rank, peer_rank, exchange, p_acc, w, w_a, w_b]
                self.comm.send(arr, dest = peer_rank, tag = 2)
                # file_io.write_exchanges()
            
            else:
                self.comm.send(w_a, dest = peer_rank, tag = 1)
                arr = self.comm.recv(source = peer_rank, tag = 2)
        
        if self.rank % 2 != 0 and len(arr) > 0:
            self.comm.send(arr, dest = 0, tag = 3)

        if self.rank == 0:
            for i in range(1, self.no_replicas + 1, 2):
                arr = self.comm.recv(source = i, tag = 3)
                file_io.write_exchanges(arr)

        return exchange

    # def __rex_exchange_as_leader(self, self_energy, peer_rank):
    #     yb = self.comm.recv(source = peer_rank, tag = 1)
    #     peer_id = self.comm.recv(source = peer_rank, tag = 2)
    #     peer_temp = self.comm.recv(source = peer_rank, tag = 3)

    #     self_temp = Config.T()

    #     beta_i = 1 / (Units.kB * self_temp)
    #     beta_j = 1 / (Units.kB * peer_temp)

    #     delta = (beta_i - beta_j) * (energy - self_energy)

    #     exchange = True
    #     metropolis = 1
    #     if delta > 0:
    #         metropolis = np.exp(-delta)
    #         u_rand = np.random.uniform()

    #         if u_rand >= metropolis:
    #             exchange = False

    #     self.comm.send(exchange, dest = peer_rank, tag = 4)
    #     if exchange:
    #         self.comm.send(Config.replica_id, dest = peer_rank, tag = 5)
    #         Config.replica_id = peer_id
    #     return exchange, metropolis
            
    
