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


if args.config:
    Config.import_from_file(args.config)

if __name__ == "__main__":

    ensemble = Ensemble()
    ensemble.run_simulation()
    
    if Config.analyze:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank != 0:
            exit()

        print("Analyzing .... \n")
        an = Analysis()

        # if Config.run_type == 'remd':
        #     for i in range(1000):
        #         local_file_path = os.path.join(an.file_path, str(i))

        #         if not os.path.isdir(local_file_path):
        #             break
        #         an_local = Analysis(cfg, cfg.run_name + "/{}".format(i))
        #         an_local.initialize_bins([-np.inf, -0.75, 0.25, 1.25, np.inf])
        #         an_local.plot_energy()
        #         an_local.plot_temperature(cfg.temperatures[i])
        #         # an_local.load_positions_and_velocities()
        #         # an_local.well_histogram(cfg.num_particles)
        #         # an_local.velocity_distribution()
        #         # an_local.plot_hprime()

                


    #         an.initialize_bins([-np.inf, -0.75, 0.25, 1.25, np.inf])
    #         an.plot_probs(ensemble.sys.pot_energy, cfg.temperatures[cfg.primary_replica], cfg.primary_replica)
    #         exit()
        an.plot_energy()
        an.plot_temperature(Config.T)

    #     an.initialize_bins([-np.inf, -0.75, 0.25, 1.25, np.inf])
    #     an.load_positions_and_velocities()
    #     an.well_x_time(cfg.num_particles)
    #     an.well_histogram(cfg.num_particles)
        an.velocity_distribution()
        
    #     if cfg.run_type == 'nvt':
    #         an.plot_hprime()
    #     print("Analysis Done. \n")
        
