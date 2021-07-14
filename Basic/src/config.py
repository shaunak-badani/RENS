from .units import Units
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
                self.temperature = data['temperature']

            if 'run_type' in data:
                self.run_type = data['run_type']

            if 'analyze' in data:
                self.analyze = data['analyze']
            
            if 'ada' in data:
                self.ada = data['ada']

            self.run_name = os.path.splitext(file_name)[0]

        except FileNotFoundError:
            print("No such file {}".format(file_name))

        except json.decoder.JSONDecodeError:
            print("Bad JSON file")