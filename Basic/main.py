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