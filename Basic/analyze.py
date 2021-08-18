import argparse
import os
import numpy as np

from src.config import Config
from src.analysis import Analysis
from src.system import System

from mpi4py import MPI



parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help = "optional config file")
args = parser.parse_args()


if args.config:
    Config.import_from_file(args.config)

print("Analyzing .... \n")
if Config.run_type == 'remd':
    an = Analysis()
    an.initialize_bins([-np.inf, -0.75, 0.25, 1.25, np.inf])
    an.plot_probs(System().pot_energy, Config.T(), Config.primary_replica)
    an.plot_free_energy(System().pot_energy)
    path = an.file_path
    replicas = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(replicas)
    for replica in replicas:
        root_path = Config.run_name
        Config.run_name += "/{}".format(str(replica))
        an = Analysis()
        an.pos_x_time()
        an.plot_temperature(Config.T())
        Config.run_name = root_path
    exit()

an = Analysis()
an.plot_energy()
an.plot_temperature(Config.T())
an.pos_x_time()
an.velocity_distribution()

if Config.run_type == 'nvt':
    an.plot_hprime()
