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
        
        energies_path = os.path.join(self.images_path, "energies")
        os.system('mkdir -p {}'.format(energies_path))
        
        # Only KE
        fig = plt.figure(figsize = (21, 7))
        plt.plot(steps, ke_vals, linewidth = 1, color = 'red', alpha = 0.5)
        plt.scatter(steps, ke_vals, s = 2, c='red', alpha = 0.2)
        plt.yticks(fontsize = 22.5)
        plt.xticks(fontsize = 22.5)
        
        plt.xlabel("Steps", fontsize = 30)
        ylabel = plt.ylabel("Kinetic \n Energy", labelpad = 60, fontsize = 30)
        ylabel.set_rotation(0)
        ke_path = os.path.join(energies_path, "KE.png")
        plt.savefig(ke_path)
        plt.close()
        
        # Only PE
        fig = plt.figure(figsize = (21, 7))
        plt.plot(steps, pe_vals, linewidth = 1, color = 'blue', alpha = 0.5)
        plt.scatter(steps, pe_vals, s = 2, c='blue', alpha = 0.2)
        plt.yticks(fontsize = 22.5)
        plt.xticks(fontsize = 22.5)
        
        plt.xlabel("Steps", fontsize = 30)
        ylabel = plt.ylabel("Potential \n Energy", labelpad = 60, fontsize = 30)
        ylabel.set_rotation(0)
        pe_path = os.path.join(energies_path, "PE.png")
        plt.savefig(pe_path)
        plt.close()
        
        # Only TE
        fig = plt.figure(figsize = (21, 7))
        plt.plot(steps, tot_energy, linewidth = 1, color = 'green', alpha = 0.5)
        plt.scatter(steps, tot_energy, s = 2, c='green', alpha = 0.2)
        plt.yticks(fontsize = 22.5)
        plt.xticks(fontsize = 22.5)
        
        plt.xlabel("Steps", fontsize = 30)
        ylabel = plt.ylabel("Total \n Energy", labelpad = 60, fontsize = 30)
        ylabel.set_rotation(0)
        te_path = os.path.join(energies_path, "TE.png")
        plt.savefig(te_path)
        plt.close()
        
        # Collective plot
        fig = plt.figure(figsize = (21, 7))
        lw = 3
        plt.plot(steps, tot_energy, label="Total", linewidth = lw, color = 'green', alpha = 0.5)
        plt.plot(steps, ke_vals, label="Kinetic", linewidth = lw, color = 'red', alpha = 0.5)
        plt.plot(steps, pe_vals, label="Potential", linewidth = lw, color = 'blue', alpha = 0.5)
        
        plt.yticks(fontsize = 22.5)
        plt.xticks(fontsize = 22.5)
        
        plt.xlabel("Steps", fontsize = 30)
        ylabel = plt.ylabel("Total \n Energy", labelpad = 60, fontsize = 30)
        ylabel.set_rotation(0)
        
        plt.legend(fontsize = 25)
        e_path = os.path.join(self.images_path, "All_energies.png")
        plt.savefig(e_path)
        plt.close()

    def plot_temperature(self, temp):
        fig = plt.figure(figsize = (21, 7))
        file_path = self.file_path
        T_file = os.path.join(file_path, "T.txt")
        T = np.loadtxt(T_file)

        
        steps = T[:, 0]
        plt.plot(steps, T[:, 1], label = "Temperature", linewidth = 3, color = 'maroon', alpha = 0.5)
        plt.axhline(y = temp, linewidth = 3, color = 'red')

        plt.yticks(fontsize = 22.5)
        plt.xticks(fontsize = 22.5)
        
        plt.xlabel("No of Steps", fontsize = 30)
        ylabel = plt.ylabel("Temp \n erature", labelpad = 60, fontsize = 30)
        ylabel.set_rotation(0)
        t_path = os.path.join(self.images_path, "Temperature.png")
        plt.savefig(t_path)
        plt.close()
    
    def initialize_bins(self, bin_boundaries):
        self.bin_boundaries = bin_boundaries
        
    def load_positions_and_velocities(self):
        file_path = self.file_path
        positions = os.path.join(file_path, "p.txt")
        velocities = os.path.join(file_path, "v.txt")
        
        pos_file_buffer = open(positions, "r")
        vel_file_buffer = open(velocities, "r")
            
        pos, vel = [], []

        lines = pos_file_buffer.readlines()
        for line in lines:
            l = line.split(' ')
            pos.append([float(i) for i in l])
        pos = np.array(pos)
        steps = pos[:, 0]
        pos = pos[:, 1:]

        lines = vel_file_buffer.readlines()
        for line in lines:
            l = line.split(' ')
            vel.append([float(i) for i in l])
        vel = np.array(vel)
        vel = vel[:, 1:]
        
        self.vel = vel
        self.steps = steps
        self.pos = pos
        
        pos_file_buffer.close()
        vel_file_buffer.close()
    
    def well_x_time(self, num_particles):
        pos = self.pos
        separate_images_path = os.path.join(self.images_path, "well_x_time")
        os.system('mkdir -p {}'.format(separate_images_path))

        particle_no = 0
        if not hasattr(self, 'bin_boundaries'):
            print("Initialize bin boundaries first !")
            return
        bin_boundaries = self.bin_boundaries
        well_counts = []
        
        wells = np.arange(1, len(bin_boundaries))
        steps = pos.shape[0]
        time = np.arange(steps)
        
        for particle_no in range(num_particles):
            well_no = np.zeros(steps)
            fig = plt.figure(figsize = (21, 7))
            for i in wells:
                lower_bound = bin_boundaries[i - 1]
                upper_bound = bin_boundaries[i]
                lies_in_well = np.logical_and(pos[:, particle_no] >= lower_bound, \
                                              pos[:, particle_no] < upper_bound)
                ind = np.where(lies_in_well)[0]
                well_no[ind] = i
            im_path = os.path.join(separate_images_path, "{}.png".format(str(particle_no)))
            
            for l in wells:
                plt.axhline(y = l, linewidth = 2, color = 'goldenrod', alpha = 0.9)
            plt.plot(time, well_no, color = 'midnightblue', linewidth = 3)
            plt.yticks(wells)
            plt.ylim(wells[0] - 1, wells[-1] + 1)
            
            plt.yticks(fontsize = 22.5)
            plt.xticks(fontsize = 22.5)

            plt.xlabel("Steps", fontsize = 30)
            ylabel = plt.ylabel("Well \n No.", labelpad = 60, fontsize = 30)
            ylabel.set_rotation(0)

            plt.savefig(im_path)
            plt.close()

        
    def well_histogram(self, num_particles):
        bin_boundaries = self.bin_boundaries
        pos = self.pos
        total_wells = [0] * (len(bin_boundaries) - 1)
        separate_images_path = os.path.join(self.images_path, "well_hist")
        os.system('mkdir -p {}'.format(separate_images_path))
        wells = np.arange(1, len(bin_boundaries))
        text_fs = 20
        ticks_fs = 15
        
        for particle_no in range(num_particles):
            well_counts = []
            fig = plt.figure(figsize = (5, 10))
            for i in wells:
                lower_bound = bin_boundaries[i - 1]
                upper_bound = bin_boundaries[i]
                lies_in_well = np.logical_and(pos[:, particle_no] >= lower_bound, \
                                     pos[:, particle_no] < upper_bound)
                well_counts.append(np.count_nonzero(lies_in_well))
            total_wells = np.add(total_wells, well_counts)
            im_path = os.path.join(separate_images_path, "{}.png".format(str(particle_no)))
            plt.bar(wells, well_counts, width = 0.4, color = 'limegreen')
            
            plt.yticks(fontsize = ticks_fs)
            plt.xticks(wells, fontsize = ticks_fs)

            plt.xlabel("Well No.", fontsize = text_fs)
            ylabel = plt.ylabel("# times \n particle in \n well", labelpad = 60, fontsize = text_fs)
            ylabel.set_rotation(0)
            
            
            plt.savefig(im_path)
            plt.close()
            
        fig = plt.figure(figsize = (5, 10))

        plt.bar(wells, total_wells, width = 0.4, color = 'limegreen')

        plt.yticks(fontsize = ticks_fs)
        plt.xticks(wells, fontsize = ticks_fs)

        plt.xlabel("Well No.", fontsize = text_fs)
        ylabel = plt.ylabel("# Counts of \n all particles \n over time", labelpad = 60, fontsize = text_fs)
        ylabel.set_rotation(0)
        
        well_hist_path = os.path.join(self.images_path, "collective_well_hist.png")
        plt.savefig(well_hist_path)
        plt.close()
        
    
    def vel_norm_x_time(self, num_particles):
        vel_n = os.path.join(self.images_path, "vel_norm")
        os.system('mkdir -p {}'.format(vel_n))
        
        for particle_no in range(num_particles):
            vel_norm = np.abs(self.vel[:, particle_no])
            fig = plt.figure(figsize = (21, 7))
            plt.scatter(self.steps, vel_norm, s = 3, c='fuchsia', alpha = 0.2)
            plt.yticks(fontsize = 22.5)
            plt.xticks(fontsize = 22.5)

            plt.xlabel("Steps", fontsize = 30)
            ylabel = plt.ylabel("Velocity \n Norm", labelpad = 60, fontsize = 30)
            ylabel.set_rotation(0)
            vel_norm_path = os.path.join(vel_n, "{}.png".format(str(particle_no)))
            plt.savefig(vel_norm_path)
            plt.close()

    def plot_hprime(self):
        file_path = self.file_path
        h_prime_path = os.path.join(file_path, "sys_surr_energy.txt")
        h_prime = np.loadtxt(h_prime_path)
        
        
        fig = plt.figure(figsize = (21, 7))
        steps = h_prime[:, 0]
        h_prime = h_prime[:, 1]
        plt.axhline(y = h_prime.mean(), color = 'red')
        plt.axhline(y = h_prime.mean() + 0.2, color = 'red')
        plt.axhline(y = h_prime.mean() - 0.2, color = 'red')
        plt.plot( steps, h_prime, linewidth = 6, color = 'blue', alpha = 0.5, label = "H' ")
        
        
        plt.yticks(fontsize = 22.5)
        plt.xticks(fontsize = 22.5)
        
        plt.xlabel("Steps", fontsize = 30)
        plt.legend()
        ylabel = plt.ylabel("H`", labelpad = 60, fontsize = 30)
        ylabel.set_rotation(0)
        prime_path = os.path.join(self.images_path, "H_prime.png")
        plt.savefig(prime_path)
        plt.close()




if __name__ == "__main__":
    an = Analysis()
    an.plot_energy()
    an.plot_temperature()
        
