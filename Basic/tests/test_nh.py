import unittest
import os
import pandas as pd
import numpy as np

class TestNoseHoover(unittest.TestCase):

    def test_nosehoover_written_example(self):
        os.system('python3 main.py -c tests/1D_Leach/jsons/nh_test.json')
        file_loc = 'tests/NH_test/H_prime'
        univ = pd.read_csv(os.path.join(file_loc, 'univ_file.txt'), sep = ' ')
        self.assertTrue(np.allclose(univ['Bath_System_Energy'], 2, atol = 1.5e-4, rtol = 0))

    def test_nosehoover_leach(self):
        os.system('python3 main.py -c tests/1D_Leach/jsons/nh_1D.json')
        file_loc = 'tests/NH_test/nh_leach'
        univ = pd.read_csv(os.path.join(file_loc, 'univ_file.txt'), sep = ' ')
        self.assertTrue(np.allclose(univ['Bath_System_Energy'], 25.789, atol = 3.6e-3, rtol = 0))

        vel = np.loadtxt(os.path.join(file_loc, 'v.txt'))
        particle_no = 3
        p, be = np.histogram(vel[:, 1 + particle_no], density = True)
        coords = 0.5 * (be[1:] + be[:-1])
        sigma = np.sqrt(2)
        expected = np.exp(-coords**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        self.assertTrue(np.allclose(expected, p, atol = 0.025))
        
    @classmethod
    def tearDownClass(self):
        pass
        os.system('rm -rf tests/NH_test')

if __name__ == '__main__':
    unittest.main()