import os
import pandas as pd

class FileOperations:

    def __init__(self, cfg):

        file_dir = cfg.files     
        
        # Constructing Folders
        folder_path = os.path.join(file_dir, cfg.run_name)
        self.run_name = cfg.run_name
        self.folder_path = folder_path
        self.share_dir = cfg.share_dir
        self.ada = cfg.ada
        os.system("mkdir -p {}".format(folder_path))

        # Defining File objects
        pos_file = os.path.join(folder_path, "p.txt")
        vel_file = os.path.join(folder_path, "v.txt")
        scalar_file = os.path.join(folder_path, "scalars.txt")

        self.pos_file = open(pos_file, "w+")
        self.vel_file = open(vel_file, "w+")

        self.scalar_file = open(scalar_file, "w+")
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

    def __init__(self, cfg):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        root_path = cfg.run_name
        cfg.run_name += "/{}".format(str(rank))
        super().__init__(cfg)
        cfg.run_name = root_path
        remd_file = os.path.join(cfg.files, root_path, "exchanges.txt")
        if rank == 0:
            self.exchanges_file = open(remd_file, "w+")
    
    def declare_step(self, step_no):
        self.exchanges_file.write("Step : {}".format(step_no))
        self.exchanges_file.write("\n")
    
    def write_exchanges(self, src_rank, dest_rank, exchanged, acc_prob):
        self.exchanges_file.write("{0} {1} {2} {3:1.3f}".format(src_rank, dest_rank, exchanged, acc_prob))
        self.exchanges_file.write("\n")

    def done_step(self):
        self.exchanges_file.write("\n")
