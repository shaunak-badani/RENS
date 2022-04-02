import unittest
import os
import pandas as pd
import numpy as np

class Test1DLeach(unittest.TestCase):

    def test_nve_leach(self):
        os.system('python3 main.py -c tests/1D_Leach/jsons/nve_1D.json')
        file_loc = 'tests/1D_Leach/leach_runs/nve_leach'
        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ')
        self.assertTrue(np.allclose(scalars['TE'], 25.789, 1e-4))

    def test_langevin_leach(self):
        '''
        This test is stochastic and may fail sometimes
        '''
        os.system('python3 main.py -c tests/1D_Leach/jsons/langevin_1D.json')
        file_loc = 'tests/1D_Leach/leach_runs/langevin_leach'

        vels = np.loadtxt(os.path.join(file_loc, 'v.txt'))
        p, be = np.histogram(vels[:, 1], density = True)
        coords = 0.5 * (be[1:] + be[:-1])
        sigma = np.sqrt(2)
        expected = np.exp(-coords**2 / (2 * sigma**2)) /  np.sqrt(2 * np.pi * sigma**2)
        self.assertTrue(np.allclose(p, expected, atol = 0.05, rtol = 0))
        

    @classmethod
    def tearDownClass(self):
        os.system('rm -rf tests/1D_Leach/leach_runs')


