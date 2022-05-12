import unittest
import os
import pandas as pd
import numpy as np

class TestOtherSystems(unittest.TestCase):

    def test_free(self):
        os.system('python3 main.py -c tests/OtherSystems/jsons/free.json')
        file_loc = 'tests/OtherSystems/runs/free_nve'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        self.assertTrue(scalars['TE'].std() < 1e-13)

    def test_harmonic(self):
        os.system('python3 main.py -c tests/OtherSystems/jsons/harmonic.json')
        file_loc = 'tests/OtherSystems/runs/harmonic_nve'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        self.assertTrue(np.allclose(scalars['TE'], 2.615172, atol = 0))

    def test_lj_nve(self):
        os.system('python3 main.py -c tests/OtherSystems/jsons/lj.json')
        file_loc = 'tests/OtherSystems/runs/lj_nve'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        standard_deviation = np.std(scalars['TE'])
        self.assertTrue(standard_deviation < 1e-3)

    def test_lj_nvt(self):
        os.system('python3 main.py -c tests/OtherSystems/jsons/lj_nvt.json')
        file_loc = 'tests/OtherSystems/runs/lj_nvt'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        standard_deviation = np.std(scalars['T'])
        self.assertTrue(standard_deviation < 9)

        univ = pd.read_csv(os.path.join(file_loc, 'univ_file.txt'), sep = ' ') 
        standard_deviation = np.std(univ['Bath_System_Energy'])
        self.assertTrue(standard_deviation < 0.05)
    
    def test_leps_1(self):
        os.system('python3 main.py -c tests/OtherSystems/jsons/leps_1.json')
        file_loc = 'tests/OtherSystems/runs/leps_1'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        standard_deviation = np.std(scalars['TE'])
        self.assertTrue(standard_deviation < 1e-5)

    def test_leps_2(self):
        os.system('python3 main.py -c tests/OtherSystems/jsons/leps_2.json')
        file_loc = 'tests/OtherSystems/runs/leps_2'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        standard_deviation = np.std(scalars['TE'])
        self.assertTrue(standard_deviation < 1e-5)
    
    def test_leps_2_mod(self):
        os.system('python3 main.py -c tests/OtherSystems/jsons/leps_2_mod.json')
        file_loc = 'tests/OtherSystems/runs/leps_2_mod'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        standard_deviation = np.std(scalars['TE'])
        self.assertTrue(standard_deviation < 1e-3)


    @classmethod
    def tearDownClass(self):
        pass
        os.system('rm -rf tests/OtherSystems/runs')
