import unittest
import os
import pandas as pd
import numpy as np

class TestMuller(unittest.TestCase):

    def test_muller_nve(self):
        os.system('python3 main.py -c tests/Muller/jsons/muller.json')
        file_loc = 'tests/Muller/runs/muller_nve'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        self.assertTrue(np.allclose(scalars['TE'], -100.9, atol = 1e-1))
    
    def test_muller_nvt(self):
        os.system('python3 main.py -c tests/Muller/jsons/muller_nvt.json')
        file_loc = 'tests/Muller/runs/muller_nvt'

        vel = np.loadtxt(os.path.join(file_loc, 'v.txt'))

        p, be = np.histogram(vel[:, 2], density = True)
        sigma = np.sqrt(2)
        coords = 0.5 * (be[1:] + be[:-1])
        expected = np.exp(-coords**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        self.assertTrue(np.allclose(expected, p, atol = 3.3 * 1e-2))
        

        p, be = np.histogram(vel[:, 1], density = True)
        sigma = np.sqrt(2)
        coords = 0.5 * (be[1:] + be[:-1])
        expected = np.exp(-coords**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        self.assertTrue(np.allclose(expected, p, atol = 1.8 * 1e-2))

        univ = pd.read_csv(os.path.join(file_loc, 'univ_file.txt'), sep = ' ') 
        self.assertTrue(np.allclose(univ['Bath_System_Energy'], -100.93, atol = 0.02))

    @classmethod
    def tearDownClass(self):
        os.system('rm -rf tests/Muller/runs/')

if __name__ == '__main__':
    unittest.main()
