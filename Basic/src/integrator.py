from .system import System
from .units import Units
import numpy as np
from .config import Config

class VelocityVerletIntegrator:
    def __init__(self):
        self.dt = 1e-3

        if hasattr(Config, 'dt'):
            self.dt = Config.dt

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
                arr = [step_no, self.rank, peer_rank, exchange, acc_prob]
                self.comm.send(arr, dest = peer_rank, tag = 10)
                # file_io.declare_step(step_no)
                # file_io.write_exchanges(self.rank, peer_rank, exchange, acc_prob)
                
            else:
                exchange, _, factor = self.__rex_exchange_as_follower(energy, peer_rank)
                arr = self.comm.recv(source = peer_rank, tag = 10)
            
            if exchange:
                scale_factor = self.scale_velocity_factor(Config.T(), Config.temperatures[peer_rank])
                v *= scale_factor
                y_x, y_v = self.swap_positions(x, v, peer_rank, self.rank > peer_rank)

        if self.rank % 2 != 0 and len(arr) > 0:
            self.comm.send(arr, dest = 0, tag = 3)

        if self.rank == 0:
            for i in range(1, self.no_replicas + 1, 2):
                arr = self.comm.recv(source = i, tag = 3)
                file_io.write_exchanges(arr)
        return y_x, y_v
        
    
    def scale_velocity_factor(self, T_old, T_new):
        return np.sqrt(T_new / T_old)
    
class RENSIntegrator(REMDIntegrator):

    def __init__(self, dt):
        super().__init__()

        
        self.attempt_rate = 0.166
        self.t = 0
        self.tau = 1.0

        if hasattr(Config, 'tau'):
            self.tau = Config.tau

        self.nsteps = int(self.tau / dt)
        self.update_interval = 500
        self.dt = dt
        # modes denote what kind of simulation is going on right now
        # mode = 0 => nvt
        # mode = 1 => work simulation
        self.mode = 0

        
    def setup_rens(self, sys, x, v):
        self.T_A = Config.T()
        H = sys.K(v) + sys.U(x)
        kbt = Units.kB * self.T_A
        # print("H : ", H)
        # print("kbt : ", kbt)
        # print("h / kBt : ", H / kbt)
        self.w = - (sys.K(v) + sys.U(x)) / (Units.kB * self.T_A)

        self.t = 0
        self.x0 = x
        self.v0 = v
        self.current_step = 0


        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1
        self.heat = 0


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
        t = self.current_step * self.dt
        return (t / self.tau), (1 / self.tau)

    def T_lambda(self):
        l, _ = self.lamda()
        T_A = self.T_A
        T_B = self.T_B
        return T_A + l * (T_B - T_A)

    def attempt(self, sys, x, v):
        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1

        start_work_simulation = False
        if peer_rank >= 0 and peer_rank < self.no_replicas:
            if self.rank > peer_rank:
                rand = np.random.uniform()
                if rand <= self.attempt_rate * self.dt:
                    start_work_simulation = True
                self.comm.send(start_work_simulation, dest = peer_rank, tag = 1)
            else:
                start_work_simulation = self.comm.recv(source = peer_rank, tag = 1)
            
        if start_work_simulation:
            self.mode = 1
            self.setup_rens(sys, x, v)

    def exchange_phase_space_vectors(self, sys):
        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1

        x = sys.x
        v = sys.v

        y_x = x
        y_v = v

        if peer_rank >= 0 and peer_rank < self.no_replicas:
            if self.rank > peer_rank:
                self.comm.send(x, dest = peer_rank, tag = 5)
                self.comm.send(v, dest = peer_rank, tag = 6)

                y_x = self.comm.recv(source = peer_rank, tag = 7)
                y_v = self.comm.recv(source = peer_rank, tag = 8)
            else:
                y_x = self.comm.recv(source = peer_rank, tag = 5)
                y_v = self.comm.recv(source = peer_rank, tag = 6)

                self.comm.send(x, dest = peer_rank, tag = 7)
                self.comm.send(v, dest = peer_rank, tag = 8)
        
        return y_x, y_v
    
    def andersen_update(self, sys, v, T_lamda):
        N, d = v.shape
        ind = np.random.randint(0, N)
        kBT = Units.kB * T_lamda
        
        beta = 1 / (kBT)
        sigma = 1 / np.sqrt(sys.m[ind] * beta)
        if not Units.arbitrary:
            sigma *= np.sqrt(Units.kJ_mol_TO_J / Units.AMU_TO_KG) * Units.M_S_TO_A_PS
        v_new = v.copy()
        v_new[ind] = np.random.normal(size = d, scale = sigma)
        return v_new

    def step(self, sys, timestep, file_io):
        x = sys.x
        v = sys.v.copy()
        F = sys.F
        m = sys.m

        if self.nsteps == 0:
            v *= np.sqrt(self.T_B / self.T_A)

        if self.current_step >= self.nsteps:
            self.mode = 0
            _, system_dimension = x.shape
            N_f = Config.num_particles * system_dimension
            self.heat += (N_f / 2) * np.log(self.T_B / self.T_A)

            self.w += (sys.K(v) + sys.U(x)) / (Units.kB * self.T_B)
            self.w -= self.heat

            exchange = self.determine_exchange(timestep, sys, file_io)
            x_new, v_new = x[:], v[:]
            if not exchange:
                x_new, v_new = self.x0, self.v0
            else:
                sys.set_v(v)
                x_new, v_new = self.exchange_phase_space_vectors(sys)
            return x_new, v_new
    

        T_lamda = self.T_lambda()
        if self.current_step % self.update_interval == 0:
            v_new = self.andersen_update(sys, v, T_lamda)
            h_new = (sys.K(v_new) + sys.U(x)) / (Units.kB * T_lamda)
            h_old = (sys.K(v) + sys.U(x)) / (Units.kB * T_lamda)
            # print(h_new, " h_new")
            # print(h_old, " h_old")
            self.heat += h_new - h_old
            v = v_new
        else:
            K = sys.K
            _, l_der = self.lamda()
            z =  (self.T_B - self.T_A) * l_der /  (2 * T_lamda)
            v = v * np.exp(z * self.dt / 2) + F(x) * (np.exp(z * self.dt / 2) - 1) / (z * m)
            x = x + v * self.dt
            v = v * np.exp(z * self.dt / 2) + F(x) * (np.exp(z * self.dt / 2) - 1) / (z * m)
        self.current_step += 1
        return x, v

    def determine_exchange(self, timestep, sys, file_io):
        if self.rank % 2 == 0:
            peer_rank = self.rank + 1
        else:
            peer_rank = self.rank - 1

        x = sys.x
        v = sys.v
        U = sys.U
        K = sys.K


        w_a = self.w
        exchange = False
        p_acc = 1.0
        arr = []
        if peer_rank >= 0 and peer_rank < self.no_replicas:
            if self.rank < peer_rank:
                w_b = self.comm.recv(source = peer_rank, tag = 1)
                w = w_a + w_b

                if w < 0:
                    exchange = True
                else:
                    p_acc = np.exp(-w)
                    rand = np.random.random()

                    if rand <= p_acc:
                        exchange = True

                arr = [timestep, self.rank, peer_rank, exchange, p_acc, w, w_a, w_b]
                self.comm.send(arr, dest = peer_rank, tag = 2)
            
            else:
                self.comm.send(w_a, dest = peer_rank, tag = 1)
                arr = self.comm.recv(source = peer_rank, tag = 2)
                exchange = arr[3]
        if self.rank % 2 != 0 and len(arr) > 0:
            self.comm.send(arr, dest = 0, tag = 3)

        if self.rank == 0:
            for i in range(1, self.no_replicas + 1, 2):
                arr = self.comm.recv(source = i, tag = 3)
                file_io.write_exchanges(arr)

        return exchange
