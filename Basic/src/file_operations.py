import os
import pandas as pd
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
        if first_time:
            self.scalar_file.write("Step KE PE TE T")
            self.scalar_file.write("\n")
        

    def write_vectors(self, x, v, step):
        str_x = ' '.join([str(i) for i in x.flatten()])
        self.pos_file.write("{} {}".format(step, str_x))
        self.pos_file.write("\n")
        
        str_v = ' '.join([str(i) for i in v.flatten()])
        self.vel_file.write("{} {}".format(step, str_v))
        self.vel_file.write("\n")

    def write_scalars(self, ke, pe, T, step):
        te = pe + ke
        self.scalar_file.write("{} {} {} {} {}".format(step, ke, pe, te, T))
        self.scalar_file.write("\n")
    
    def write_surrounding_energy(self, surr_energy, step):
        if not hasattr(self, 'surr_file'):
            surr_file_path = os.path.join(self.folder_path, "surr_file.txt")
            self.surr_file = open(surr_file_path, "w+")
            self.surr_file.write("Step Surrounding_Energy\n")
        self.surr_file.write("{} {}".format(step, surr_energy))
        self.surr_file.write("\n")
    
    def write_rst(self, x, v, m):
        data_object = {'x' : x.flatten(), 'v' : v.flatten(), 'm' : m.flatten()}
        rst_path = os.path.join(self.folder_path, "end.rst")
        pd.DataFrame(data = data_object).to_csv(rst_path, sep = ' ', header = True, index = False)


    def __del__(self):
        self.pos_file.close()
        self.vel_file.close()
        self.scalar_file.close()

        if hasattr(self, 'surr_file'):
            self.surr_file.close()
        share_dir = self.share_dir
        if self.ada:
            os.system('rsync -aPs --rsync-path="mkdir -p {} \
            && rsync" {} ada:{}'.format(share_dir, self.folder_path, share_dir))
            print("Files sent to share1 ")


class FileOperationsREMD(FileOperations):

    def __init__(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        root_path = Config.run_name
        Config.run_name += "/{}".format(str(Config.replica_id))
        super().__init__()
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
