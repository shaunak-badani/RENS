import json
import argparse
import os
import numpy as np

from src.system import System
from src.config import Config
from src.file_operations import FileOperations
from src.integrator import VelocityVerletIntegrator
from src.analysis import Analysis
from ensembles import Ensemble


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help = "optional config file")
args = parser.parse_args()

cfg = Config()

if args.config:
    cfg.import_from_file(args.config)

if __name__ == "__main__":

    ensemble = Ensemble(cfg)
    ensemble.run_simulation()
    
    if cfg.analyze:
        print("Analyzing .... \n")
        an = Analysis(cfg, cfg.run_name)
        an.plot_energy()
        an.plot_temperature(cfg.temperature)

        an.initialize_bins([-np.inf, -0.75, 0.25, 1.25, np.inf])
        an.load_positions_and_velocities()
        an.well_x_time(cfg.num_particles)
        an.well_histogram(cfg.num_particles)

        if cfg.run_type == 'nvt':
            an.plot_hprime()
        print("Analysis Done. \n")
        
