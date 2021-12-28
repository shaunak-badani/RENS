from .units import Units
from mpi4py import MPI
import json
import os

class Config:
    
    files = '/scratch/shaunak/1D_Run'
    share_dir = '/share1/shaunak/1D_Run/'

    num_particles = 1
    num_steps = 10
    temperature = 2.0
    run_name = 'default'
    run_type = 'nve'
    analyze = True
    ada = False
    system = '1D_Leach'
    rst = None
    output_period = 10
    restart = False

    # FOR NVT
    thermostat = 'nh'

    # FOR REMD
    temperatures = []
    rsts = []
    primary_replica = 0
    replica_id = MPI.COMM_WORLD.Get_rank()

    def T():
        if len(Config.temperatures) > 0:
            t = Config.temperatures[Config.replica_id]
        else:
            t = Config.temperature
        t *= (Units.epsilon / Units.kB)
        return t
    

    @staticmethod
    def import_from_file(file_name):
        try:
            with open(file_name) as json_data_file:
                data = json.load(json_data_file)
            if 'num_particles' in data:
                Config.num_particles = data['num_particles']

            if 'num_steps' in data:
                Config.num_steps = data['num_steps']

            if 'share_dir' in data:
                Config.share_dir = data['share_dir']

            if 'temperatures' in data:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                if(rank >= len(data['temperatures'])):
                    print("Run the program with number of processes equal to number of replicas! Idjot")
                Config.temperatures = data['temperatures']
               

            if 'temperature' in data:               
                Config.temperature = data['temperature']


            if 'run_type' in data:
                Config.run_type = data['run_type']

            if 'analyze' in data:
                Config.analyze = data['analyze']
            
            if 'ada' in data:
                Config.ada = data['ada']
            
            if 'system' in data:
                Config.system = data['system']

            if 'rst' in data:
                Config.rst = data['rst']

            if 'rsts' in data:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                Config.rst = data['rsts'][rank]

            Config.run_name = os.path.splitext(file_name)[0]

            if not Config.ada:
                Config.files = "../../runs"
            
            if 'primary_replica' in data:
                Config.primary_replica = data['primary_replica']
            
            if 'output_period' in data:
                Config.output_period = data['output_period']

            if 'restart' in data:
                Config.restart = data['restart']

            if 'thermostat' in data:
                Config.thermostat = data['thermostat']

        except FileNotFoundError:
            print("No such file {}".format(file_name))

        except json.decoder.JSONDecodeError:
            print("Bad JSON file")

    @staticmethod
    def print_config():
        print("T = {}".format(Config.T()))