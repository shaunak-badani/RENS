import argparse
import os
import numpy as np

from src.config import Config
from src.analysis import Analysis
from src.analysis import NVE_Analysis
from src.analysis import NVT_Analysis
from src.analysis import REMD_Analysis
from src.analysis import RENS_Analysis


from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help = "optional config file")
args = parser.parse_args()

if args.config:
    Config.import_from_file(args.config)

print("Analyzing .... \n")

if Config.run_type == 'nve' or Config.run_type == 'minimize':
    analysis_object = NVE_Analysis()
    
if Config.run_type == 'nvt':
    analysis_object = NVT_Analysis()

if Config.run_type == 'remd':
    analysis_object = REMD_Analysis()

if Config.run_type == 'rens':
    analysis_object = RENS_Analysis()

analysis_object.analyze()