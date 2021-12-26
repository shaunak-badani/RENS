import os
import pandas as pd
import numpy as np
from .config import Config

class FileOperations:

    def __init__(self, first_time = True):

        file_dir = Config.files     
        
        # Constructing Folders
        folder_path = os.path.join(file_dir, Config.run_name)
        self.run_name = Config.run_name
        self.folder_path = folder_path
        self.share_dir = Config.share_dir
        self.ada = Config.ada
        
        os.system("mkdir -p {}".format(folder_path))
        
        # Defining File objects
        pos_file = os.path.join(folder_path, "p.txt")
        vel_file = os.path.join(folder_path, "v.txt")
        scalar_file = os.path.join(folder_path, "scalars.txt")

        file_mode = "w+"
        if not first_time:
            file_mode = "a+"

        self.pos_file = open(pos_file, file_mode)
        self.vel_file = open(vel_file, file_mode)

        self.scalar_file = open(scalar_file, file_mode)
        if first_time and not Config.restart:
            self.scalar_file.write("Step KE PE TE T")
            self.scalar_file.write("\n")

        self.output_period = Config.output_period
        

    def write_vectors(self, x, v, step):
        if step % self.output_period != 0:
            return
        str_x = ' '.join(np.char.mod('%.3f', x.flatten()))
        # print(str_x)
        self.pos_file.write("{} {}".format(step, str_x))
        self.pos_file.write("\n")
        
        str_v = ' '.join(np.char.mod('%.3f', v.flatten()))
        self.vel_file.write("{} {}".format(step, str_v))
        self.vel_file.write("\n")

    def write_scalars(self, ke, pe, T, step):
        if step % self.output_period != 0:
            return
        te = pe + ke
        self.scalar_file.write("{} {} {} {} {}".format(step, ke, pe, te, T))
        self.scalar_file.write("\n")
    
    def write_hprime(self, universe_energy, step):
        if step % self.output_period != 0:
            return
        if not hasattr(self, 'universe_file'):
            universe_path = os.path.join(self.folder_path, "univ_file.txt")
            self.universe_file = open(universe_path, "w+")
            self.universe_file.write("Step Bath_System_Energy \n")
        self.universe_file.write("{} {}".format(step, universe_energy))
        self.universe_file.write("\n")
    
    def write_rst(self, x, v, m, step, xi = None, vxi = None):
        if xi is None and vxi is None:
            data_object = {'x' : x.flatten(), 'v' : v.flatten(), 'm' : m.flatten(), 'step' : step}
        else:
            data_object = {'x' : x.flatten(), 'v' : v.flatten(), 'm' : m.flatten(), 'xi' : xi, 'vxi' : vxi, 'step' : step}
        dict_df = pd.DataFrame({ key:pd.Series(value) for key, value in data_object.items() })
        rst_path = os.path.join(self.folder_path, "end.rst")

        dict_df.to_csv(rst_path, sep = ' ', header = True, index = False)


    def __del__(self):
        self.pos_file.close()
        self.vel_file.close()
        self.scalar_file.close()

        if hasattr(self, 'surr_file'):
            self.surr_file.close()
        share_dir = self.share_dir
        if self.ada:
            from mpi4py import MPI
            
            if MPI.COMM_WORLD.Get_rank() != 0:
                return
            folder_path = os.path.join(Config.files, Config.run_name)
            os.system('rsync -aPs --rsync-path="mkdir -p {} \
            && rsync" {} ada:{}'.format(share_dir, folder_path, share_dir))
            print("Files sent to share1 ")


class FileOperationsREMD(FileOperations):

    def __init__(self, first_time = True):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        root_path = Config.run_name
        Config.run_name += "/{}".format(str(Config.replica_id))
        super().__init__(first_time)
        Config.run_name = root_path
        self.remd_file = os.path.join(Config.files, root_path, "exchanges.txt")
        if rank == 0:
            self.exchanges_file = open(self.remd_file, "w")
            self.exchanges_file.write("Step Src Dest Exchanged Prob")
            self.exchanges_file.write("\n")
            self.exchanges_file.close()
        
    
    def declare_step(self, step_no):
        self.exchanges_file = open(self.remd_file, "a+")
        self.exchanges_file.write("{} ".format(step_no))
        self.exchanges_file.close()
    
    def update_files(self):
        self.pos_file.close()
        self.vel_file.close()
        self.scalar_file.close()

        root_path = Config.run_name
        Config.run_name += "/{}".format(str(Config.replica_id))
        super().__init__(first_time = False)
        Config.run_name = root_path
        

    def write_exchanges(self, src_rank, dest_rank, exchanged, acc_prob):
        self.exchanges_file = open(self.remd_file, "a+")
        self.exchanges_file.write("{0} {1} {2} {3:1.3f}".format(src_rank, dest_rank, exchanged, acc_prob))
        self.exchanges_file.write("\n")
        self.exchanges_file.close()

class FileOperationsRENS(FileOperationsREMD):

    def __init__(self, first_time = True):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        root_path = Config.run_name
        super().__init__(first_time)
        self.remd_file = os.path.join(Config.files, root_path, "exchanges.txt")
        self.exchanges_pandas = pd.DataFrame(columns = ["Step","Src","Dest","Exchanged","Prob","W","W_A","W_B"])

        # if rank == 0:
            # self.exchanges_file = open(self.remd_file, "w")
            # self.exchanges_file.write("Step Src Dest Exchanged Prob W W_A W_B")
            # self.exchanges_file.write("\n")
            # self.exchanges_file.close()

    def write_vectors(self, x, v, step, mode):
        if step % self.output_period != 0:
            return
        str_x = ' '.join(np.char.mod('%.3f', x.flatten()))
        self.pos_file.write("{} {} {}".format(step, str_x, mode))
        self.pos_file.write("\n")

        str_v = ' '.join(np.char.mod('%.3f', v.flatten()))
        self.vel_file.write("{} {} {}".format(step, str_v, mode))
        self.vel_file.write("\n")

    def write_exchanges(self, arr):

        temp_exchanges = pd.DataFrame(columns = ["Step","Src","Dest","Exchanged","Prob","W","W_A","W_B"])
        temp_exchanges = temp_exchanges.append(pd.DataFrame([arr], columns=["Step","Src","Dest","Exchanged","Prob","W","W_A","W_B"]), ignore_index = True)
        self.exchanges_pandas = self.exchanges_pandas.append(temp_exchanges, ignore_index = True)

        # self.exchanges_file.write("{0} {1} {2} {3:1.3f}".format(src_rank, dest_rank, exchanged, acc_prob))
        # self.exchanges_file.write(" {0:1.3f} {1:1.3f} {2:1.3f}".format(w, w_a, w_b))

        
        # self.exchanges_file.write("\n")
        # self.exchanges_file.close()
    
    def __del__(self):

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:

            self.exchanges_pandas.to_csv(self.remd_file, index = False)
        super().__del__()
        
