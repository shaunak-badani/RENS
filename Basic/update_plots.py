from src.config import Config
from src.analysis import Analysis

import numpy as np

# Doesnt matter, we need to create the cfg object
# to get share dir
num_particles = 10
num_steps = int(1e7)
reduced_temperature = 2.0


if __name__ == "__main__":
    cfg = Config(num_particles, num_steps, reduced_temperature)
    run_names = ['longer_nvt', 'longer_nvt_corrected']
    # run_names = ['main_nve', 'main_nvt', 'nvt_0.3', 'nvt_corrected', 'nvt_m2']
    for run_name in run_names:
        print(" Analyzing {}".format(run_name))
        an = Analysis(cfg, run_name)
        an.plot_energy()
        an.plot_temperature()

        an.initialize_bins([-np.inf, -0.75, 0.25, 1.25, np.inf])
        an.load_positions_and_velocities()
        an.plot_energy()
        an.well_x_time(num_particles)
        an.well_histogram(num_particles)
        an.vel_norm_x_time(num_particles)
        print("Analysis Done. \n")
        
