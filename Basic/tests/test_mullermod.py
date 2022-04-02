import unittest
import os
import pandas as pd
import numpy as np

class TestMuller(unittest.TestCase):

    def test_mullermod_nve(self):
        os.system('python3 main.py -c tests/Muller/jsons/muller.json')
        file_loc = 'tests/Muller/runs/muller_nve'

        scalars = pd.read_csv(os.path.join(file_loc, 'scalars.txt'), sep = ' ') 
        self.assertTrue(np.allclose(scalars['TE'], -100.9, atol = 1e-1))

    @classmethod
    def tearDownClass(self):
        os.system('rm -rf tests/Muller/runs/')

if __name__ == '__main__':
    unittest.main()
