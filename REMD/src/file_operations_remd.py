import os
from mpi4py import MPI

class FileOperationsREMD:

    def __init__(self, cfg, run_name, exchange_period = 100):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        file_dir = "/scratch/shaunak/default/"
        file_dir = cfg.files       
        
        # Constructing Folders
        folder_path = os.path.join(file_dir, run_name)
        rank_specific_path = os.path.join(folder_path, str(rank))

        self.run_name = run_name
        self.folder_path = folder_path
        self.share_dir = cfg.share_dir
        # self.exchange_period = exchange_period
        self.rank = rank
        os.system("mkdir -p {}".format(folder_path))
        os.system("mkdir -p {}".format(rank_specific_path))

        # Defining File objects
        pos_file = os.path.join(rank_specific_path, "p.txt")
        vel_file = os.path.join(rank_specific_path, "v.txt")
        ke_file = os.path.join(rank_specific_path, "ke.txt")
        pe_file = os.path.join(rank_specific_path, "pe.txt")
        T_file = os.path.join(rank_specific_path, "T.txt")

        remd_file = os.path.join(folder_path, "exchanges.txt")

        self.pos_file = open(pos_file, "w+")
        self.vel_file = open(vel_file, "w+")
        self.ke_file = open(ke_file, "w+")
        self.pe_file = open(pe_file, "w+")
        self.T_file = open(T_file, "w+")

        if rank == 0:
            self.exchanges_file = open(remd_file, "w+")

    def write(self, x, v, ke, pe, T, step):
        str_x = ' '.join([str(i) for i in x.flatten()])
        self.pos_file.write("{} {}".format(step, str_x))
        self.pos_file.write("\n")
        
        str_v = ' '.join([str(i) for i in v.flatten()])
        self.vel_file.write("{} {}".format(step, str_v))
        self.vel_file.write("\n")

        self.ke_file.write("{} {}".format(step, ke))
        self.ke_file.write("\n")

        self.pe_file.write("{} {}".format(step, pe))
        self.pe_file.write("\n")

        self.T_file.write("{} {}".format(step, T))
        self.T_file.write("\n")
    
    def declare_step(self, step_no):
        self.exchanges_file.write("Step : {}".format(step_no))
        self.exchanges_file.write("\n")
    
    def write_exchanges(self, src_rank, dest_rank, exchanged, acc_prob):
        self.exchanges_file.write("{0} {1} {2} {3:1.3f}".format(src_rank, dest_rank, exchanged, acc_prob))
        self.exchanges_file.write("\n")

    def done_step(self):
        self.exchanges_file.write("\n")

    def __del__(self):
        self.pos_file.close()
        self.vel_file.close()
        self.ke_file.close()
        self.pe_file.close()
        self.T_file.close()
        if self.rank == 0:
            self.exchanges_file.close()
        share_dir = self.share_dir

        if self.rank == 0:
            os.system('rsync -aPs --rsync-path="mkdir -p {} \
            && rsync" {} ada:{}'.format(share_dir, self.folder_path, share_dir))
            print("Files sent to share1 ")
