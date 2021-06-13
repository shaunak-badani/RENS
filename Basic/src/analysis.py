import os
import matplotlib.pyplot as plt
import numpy as np


class Analysis:
    def __init__(self, cfg, run_name):
        ada_path = os.path.join(cfg.share_dir, run_name)
        file_loc = "/scratch/shaunak/1D_Run/analysis"
        os.system("mkdir -p {}".format(file_loc))
        os.system("rsync -aPs ada:{} {}".format(ada_path, file_loc))
        self.file_path = os.path.join(file_loc, run_name)

        self.images_path = os.path.join(os.getcwd(), "analysis_plots", run_name)
        os.system("mkdir -p {}".format(self.images_path))

    def plot_energy(self):
        file_path = self.file_path
        pe_file = os.path.join(file_path, "pe.txt")
        ke_file = os.path.join(file_path, "ke.txt")
        pe = np.loadtxt(pe_file)
        ke = np.loadtxt(ke_file)

        pe_vals = pe[:, 1]
        ke_vals = ke[:, 1]
        tot_energy = pe_vals + ke_vals
        steps = pe[:, 0]
        plt.plot(steps, pe_vals, label = "PE", linewidth = 3, color = 'blue')
        plt.plot(steps, ke_vals, label = "KE", linewidth = 3, color = 'red')
        plt.plot(steps, tot_energy, label = "Total energy", linewidth = 3, color = 'green')
        plt.legend()
        e_path = os.path.join(self.images_path, "Analysis.png")
        plt.savefig(e_path)

    def plot_temperature(self):
        fig = plt.figure()
        file_path = self.file_path
        T_file = os.path.join(file_path, "T.txt")
        T = np.loadtxt(T_file)

        
        steps = T[:, 0]
        plt.scatter(steps, T[:, 1], s = 3, color = "brown")
        plt.plot(steps, T[:, 1], label = "PE", linewidth = 2, color = 'blue', alpha = 0.5)
        plt.xlabel("No of Steps")
        plt.ylabel("Temperature")
        t_path = os.path.join(self.images_path, "Temperature.png")
        plt.savefig(t_path)




if __name__ == "__main__":
    an = Analysis()
    an.plot_energy()
    an.plot_temperature()
        
