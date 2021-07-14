import os

class FileOperations:

    def __init__(self, cfg):
        file_dir = "../../runs"
        if cfg.ada:
            file_dir = "/scratch/shaunak/default/"
        # file_dir = cfg.files       
        
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
        ke_file = os.path.join(folder_path, "ke.txt")
        pe_file = os.path.join(folder_path, "pe.txt")
        T_file = os.path.join(folder_path, "T.txt")

        self.pos_file = open(pos_file, "w+")
        self.vel_file = open(vel_file, "w+")
        self.ke_file = open(ke_file, "w+")
        self.pe_file = open(pe_file, "w+")
        self.T_file = open(T_file, "w+")
        


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



    def __del__(self):
        self.pos_file.close()
        self.vel_file.close()
        self.ke_file.close()
        self.pe_file.close()
        self.T_file.close()
        share_dir = self.share_dir
        if self.ada:
            os.system('rsync -aPs --rsync-path="mkdir -p {} \
            && rsync" {} ada:{}'.format(share_dir, self.folder_path, share_dir))
            print("Files sent to share1 ")
