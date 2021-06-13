from .units import Units

class Config:
    
    __multiplier = (Units.epsilon / Units.kB)
    # num_particles = 10
    # num_steps = int(1e5)
    # reduced_temperature = 2.0
    files = '/scratch/shaunak/1D_Run'
    share_dir = '/share1/shaunak/1D_Run/'

    def __init__(self, num_particles, num_steps, reduced_temperature):
        self.num_particles = num_particles
        self.num_steps = num_steps
        self.reduced_temperature = reduced_temperature
        self.temperature = reduced_temperature * self.__multiplier

