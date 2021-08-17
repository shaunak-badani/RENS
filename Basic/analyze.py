import argparse

from src.config import Config
from src.analysis import Analysis


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help = "optional config file")
args = parser.parse_args()


if args.config:
    Config.import_from_file(args.config)


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank != 0:
    exit()

print("Analyzing .... \n")
an = Analysis()
# an.initialize_bins([-np.inf, -0.75, 0.25, 1.25, np.inf])
an.plot_energy()
an.plot_temperature(Config.T)
an.pos_x_time()
an.velocity_distribution()

if Config.run_type == 'nvt':
    an.plot_hprime()