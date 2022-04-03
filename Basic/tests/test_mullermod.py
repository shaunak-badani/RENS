import unittest
import os
import pandas as pd
import numpy as np

class TestMullerMod(unittest.TestCase):

    def test_mullermod_nve(self):
        os.system('python3 main.py -c tests/MullerMod/jsons/mullermod.json')
        file_loc = 'tests/MullerMod/runs/nve'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        self.assertTrue(np.allclose(scalars['TE'], -26.22, atol = 1e-2))
    
    def test_mullermod_nvt(self):
        os.system('python3 main.py -c tests/MullerMod/jsons/mullermod_nvt.json')
        file_loc = 'tests/MullerMod/runs/nvt'

        vel = np.loadtxt(os.path.join(file_loc, 'v.txt'))

        p, be = np.histogram(vel[:, 1], density = True)
        sigma = np.sqrt(2)
        coords = 0.5 * (be[1:] + be[:-1])
        expected = np.exp(-coords**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        self.assertTrue(np.allclose(expected, p, atol = 2.2 * 1e-2))

        p, be = np.histogram(vel[:, 2], density = True)
        coords = 0.5 * (be[1:] + be[:-1])
        expected = np.exp(-coords**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        self.assertTrue(np.allclose(expected, p, atol = 1.6 * 1e-2))
        
        univ = pd.read_csv(os.path.join(file_loc, 'univ_file.txt'), sep = ' ') 
        self.assertTrue(np.allclose(univ['Bath_System_Energy'], -134.32, atol = 0.009))

    @classmethod
    def tearDownClass(self):
        os.system('rm -rf tests/MullerMod/runs/')

if __name__ == '__main__':
    unittest.main()
