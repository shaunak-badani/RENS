import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import integrate
from .units import Units


class Analysis:
    def __init__(self, cfg, run_name):
        file_loc = "../../runs"
        if cfg.ada:
            ada_path = os.path.join(cfg.share_dir, run_name)
            file_loc = "/scratch/shaunak/1D_Run/analysis"
            os.system("mkdir -p {}".format(file_loc))
            os.system("rsync -aPs ada:{} {}".format(ada_path, file_loc))
        # if cfg.run_type == "remd":
        #     primary_replica = cfg.primary_replica
        #     self.file_path = os.path.join(file_loc, str(primary_replica))
        self.file_path = os.path.join(file_loc, run_name)

        self.images_path = os.path.join(os.getcwd(), "analysis_plots", run_name)
        os.system("mkdir -p {}".format(self.images_path))

    def __get_scalars(self):
        file_path = self.file_path
        scalar_file = os.path.join(file_path, "scalars.txt")
        scalars = pd.read_csv(scalar_file, sep = ' ')
        return scalars

    def plot_energy(self):
        scalars = self.__get_scalars()

        pe = scalars["PE"].to_numpy()
        ke = scalars["KE"].to_numpy()
        pe_vals = pe
        ke_vals = ke

        tot_energy = scalars["TE"]
        steps = scalars["Step"]
        
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
        scalars = self.__get_scalars()
        T = scalars["T"].to_numpy()

        
        steps = scalars["Step"].to_numpy()
        plt.plot(steps, T, label = "Temperature", linewidth = 3, color = 'maroon', alpha = 0.5)
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
        surr_path = os.path.join(file_path, "surr_file.txt")
        if not os.path.isfile(surr_path):
            print("No surrounding energy has been noted during the simulation!\n")
            return
        surr_energies = pd.read_csv(surr_path, sep = ' ')
        scalars = self.__get_scalars()
        h_prime = scalars["TE"].to_numpy() + surr_energies["Surrounding_Energy"].to_numpy()
        
        
        fig = plt.figure(figsize = (21, 7))
        steps = surr_energies["Step"].to_numpy()
        plt.axhline(y = h_prime.mean(), color = 'red')
        # plt.axhline(y = h_prime.mean() + 0.2, color = 'red')
        # plt.axhline(y = h_prime.mean() - 0.2, color = 'red')
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

    def velocity_distribution(self):
        self.load_positions_and_velocities()
        vel_bins = np.arange(-5, 5, 0.5)
        vel_count = np.zeros_like(vel_bins)
        particle_index = 0
        for v in self.vel[:, particle_index]:
            ind = np.argmin(np.abs(vel_bins - v))
            vel_count[ind] += 1
        plt.plot(vel_bins, vel_count)
        plt.scatter(vel_bins, vel_count)
        for i in vel_bins:
            
            plt.axvline(x = i, linewidth = 0.1)
        vel_dist = os.path.join(self.images_path, "vel_dist.png")
        plt.savefig(vel_dist)
         
        plt.close()
    
    def plot_probs(self, pot_energy, temperature, primary_replica):
        if not hasattr(self, 'bin_boundaries'):
            print("First Initialize bin boundaries!")
            return
        bin_boundaries = self.bin_boundaries
        boltzmann_integrand = np.empty(len(bin_boundaries) - 1)
        beta = 1 / (Units.kB * temperature)
        wells = np.arange(1, len(bin_boundaries))
        for i in wells:
            boltzmann_integrand[i - 1] = integrate.quad(lambda x: np.exp(-beta * pot_energy(x)), bin_boundaries[i-1], bin_boundaries[i])[0]
        boltzmann_integrand /= boltzmann_integrand.sum()

        well_counts = np.zeros_like(wells, dtype="float")

        root_dir = self.file_path
        self.file_path = os.path.join(self.file_path, str(primary_replica))
        self.load_positions_and_velocities()
        self.file_path = root_dir
        particle_no = 0

        pos = self.pos
        for i in wells:
            lower_bound = bin_boundaries[i - 1]
            upper_bound = bin_boundaries[i]
            lies_in_well = np.logical_and(pos[:, particle_no] >= lower_bound, \
                                    pos[:, particle_no] < upper_bound)
            well_counts[i-1] = np.count_nonzero(lies_in_well)
        
        well_counts /= well_counts.sum()
        colors = ['r', 'g', 'b', 'm']

        for index, line in enumerate(boltzmann_integrand):
            plt.axhline(y = line, linewidth = 3, label='#{}'.format(index + 1), color = colors[index])
        plt.scatter(0, well_counts[0], color=colors[0])
        plt.scatter(0, well_counts[1], color=colors[1])
        plt.scatter(0, well_counts[2], color=colors[2])
        plt.scatter(0, well_counts[3], color=colors[3])
        plt.xlabel("Run number")
        # plt.ylim([0.2, 0.3])
        plt.ylabel("Probability of particle in well")

        im_path = os.path.join(self.images_path, "boltzmann.png")
        plt.savefig(im_path)
        plt.close()




if __name__ == "__main__":
    an = Analysis()
    an.plot_energy()
    an.plot_temperature()
        
