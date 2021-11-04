import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import integrate
from .units import Units
from .config import Config


class Analysis:
    def __init__(self):
        file_loc = "../../runs"
        if Config.ada:
            ada_path = os.path.join(Config.share_dir, Config.run_name)
            file_loc = "/scratch/shaunak/1D_Run/analysis"
            os.system("mkdir -p {}".format(file_loc))
            os.system("rsync -aPs ada:{} {}".format(ada_path, file_loc))
        # if cfg.run_type == "remd":
        #     primary_replica = cfg.primary_replica
        #     self.file_path = os.path.join(file_loc, str(primary_replica))
        self.file_path = os.path.join(file_loc, Config.run_name)
        print(self.file_path)
        self.images_path = os.path.join(os.getcwd(), "analysis_plots", Config.run_name)
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
        # print(pos)
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
    

    def plot_hprime(self):
        file_path = self.file_path
        univ_path = os.path.join(file_path, "univ_file.txt")
        if not os.path.isfile(univ_path):
            print("No surrounding energy has been noted during the simulation!\n")
            return
        univ_energies = pd.read_csv(univ_path, sep = ' ')
        h_prime = univ_energies["Bath_System_Energy"].to_numpy()
        
        
        fig = plt.figure(figsize = (21, 7))
        steps = univ_energies["Step"].to_numpy()
        plt.axhline(y = h_prime.mean(), color = 'red')
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
        particle_index = 0
        vel_count, be = np.histogram(self.vel[:, particle_index], bins = 30, density = True)
        vel_count = vel_count.astype('float')
        vel_count /= vel_count.sum()
        vel_bins = (be[1:] + be[:-1])/2

        m = 1
        expected_dist = np.exp(-vel_bins**2 / (2 * m * Units.kB * Config.T()))
        expected_dist /= expected_dist.sum()
        plt.plot(vel_bins, expected_dist)
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
        markers=["+", "x", "o", "s"]
        print(well_counts)
        boltzmann_integrand /= boltzmann_integrand.sum()


        for index, line in enumerate(boltzmann_integrand):
            plt.axhline(y = line, linewidth = 3, color = colors[index])
            plt.scatter(0, well_counts[index], marker = markers[index], color = colors[index], label='#{}'.format(index + 1) )
        plt.xlabel("Run number")
        plt.ylabel("Probability of particle in well")
        plt.legend()

        im_path = os.path.join(self.images_path, "boltzmann.png")
        plt.savefig(im_path)
        plt.close()

    def plot_free_energy(self, pot_energy):
        if not hasattr(self, 'pos'):
            self.load_positions_and_velocities()
            
        root_dir = self.file_path
        self.file_path = os.path.join(self.file_path, str(Config.primary_replica))
        self.load_positions_and_velocities()
        self.file_path = root_dir
        particle_no = 0
        
        pos = self.pos

        linspace = np.linspace(-2, 2.25, 1000)
        U = np.array([pot_energy(i) for i in linspace])
        plt.plot(linspace, U)
        plt.xlabel("Position")
        plt.ylabel("Energy")

        probs, bin_edges = np.histogram(self.pos[:, particle_no], bins = 40)
        probs = probs.astype('float')
        probs /= probs.sum()
        Config.replica_id = Config.primary_replica
        free_energy = -Units.kB * Config.T() * np.log(probs)
        free_energy -= free_energy.min()
        coords = (bin_edges[1:] + bin_edges[:-1]) / 2
        plt.plot(coords, free_energy, lw = 2, color='brown')

        im_path = os.path.join(self.images_path, "Free.png")
        plt.savefig(im_path)
        plt.close()

    def plot_prob_distribution(self, pot_energy, temperature, primary_replica):
        if not hasattr(self, 'pos'):
            root_dir = self.file_path
            print(Config.primary_replica)
            self.file_path = os.path.join(self.file_path, str(Config.primary_replica))
            self.load_positions_and_velocities()
            self.file_path = root_dir
            
        particle_no = 0
        linspace = np.linspace(-2, 2.25, 1000)
        probs, bin_edges = np.histogram(self.pos[:, particle_no], bins = 500, density = True)
        coords = (bin_edges[1:] + bin_edges[:-1]) / 2

        U = np.array([pot_energy(i) for i in linspace])
        beta = 1 / (Config.T() * Units.kB)
        Z = integrate.quad(lambda x: np.exp(-beta * pot_energy(x)), linspace.min(), linspace.max())[0]

        Config.replica_id = Config.primary_replica
        expected_prob_dist = (1 / Z) * np.exp(-beta * U)
        plt.figure(figsize = (5, 4))
        plt.plot(linspace, expected_prob_dist, color = 'green', label = 'Expected Prob density')
        plt.xlabel(r'$x$', fontsize = 15)
        axy = plt.ylabel(r'$\rho (x)$', fontsize = 15)
        axy.set_rotation(0)

        plt.scatter(coords, probs, s = 10, color='purple', label = 'MD Simulatiion Prob density')
        plt.legend()
        im_path = os.path.join(self.images_path, "Prob_dist.png")
        
        plt.savefig(im_path)
        plt.close()


    def pos_x_time(self):
        self.load_positions_and_velocities()

        fig = plt.figure(figsize = (20, 3))

        plt.plot(self.steps, self.pos, lw = 0.5, color="black")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.xticks([])
        plt.ylabel("Position")


        im_path = os.path.join(self.images_path, "pos_x_time.png")
        plt.savefig(im_path)
        plt.close()
        

if __name__ == "__main__":
    an = Analysis()
    an.plot_energy()
    an.plot_temperature()
        
