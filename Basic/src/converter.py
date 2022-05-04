import numpy as np
import pandas as pd
import sys

traj = np.loadtxt(sys.argv[1])[:, 1:]

N = traj.shape[1] // 3
f = open(sys.argv[2], "w")
f.write('{}\n'.format(str(N)))
f.write('\n')

atoms = ['Ar'] * N
for pos in traj:
    positions = pos.reshape(-1, 3)
    data_object = {'atom': atoms, 'r_x' : positions[:, 0], 'r_y' : positions[:, 1], 'r_z' : positions[:, 2]}
    dict_df = pd.DataFrame({ key:pd.Series(value) for key, value in data_object.items() })
    dict_df.to_csv(f, sep = " ", header = False, mode = 'a', float_format = '%.3f', index = False)
    f.write('{}\n'.format(str(N)))
    
    f.write('\n')

print("Done")
