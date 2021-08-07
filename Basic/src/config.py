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
    primary_replica = 0

    def import_from_file(self, file_name):
        try:
            with open(file_name) as json_data_file:
                data = json.load(json_data_file)
            if 'num_particles' in data:
                self.num_particles = data['num_particles']

            if 'num_steps' in data:
                self.num_steps = data['num_steps']

            if 'share_dir' in data:
                self.share_dir = data['share_dir']

            if 'temperature' in data:

                if(isinstance(data['temperature'], list)):
                    self.temperatures = data['temperature']
                    comm = MPI.COMM_WORLD
                    rank = comm.Get_rank()
                    if(rank >= len(data['temperature'])):
                        print("Run the program with number of processes equal to number of replicas! Idjot")
                    else:
                        self.temperature = data['temperature'][rank]
                else:
                    self.temperature = data['temperature']


            if 'run_type' in data:
                self.run_type = data['run_type']

            if 'analyze' in data:
                self.analyze = data['analyze']
            
            if 'ada' in data:
                self.ada = data['ada']
            
            if 'system' in data:
                self.system = data['system']

            if 'rst' in data:
                self.rst = data['rst']

            self.run_name = os.path.splitext(file_name)[0]

            if not self.ada:
                self.files = "../../runs"
            
            if 'primary_replica' in data:
                self.primary_replica = data['primary_replica']

        except FileNotFoundError:
            print("No such file {}".format(file_name))

        except json.decoder.JSONDecodeError:
            print("Bad JSON file")