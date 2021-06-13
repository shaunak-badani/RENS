## Run :

python3 <file_name.py>

- Uses code from `src` folder
- Saves plots in `analysis_plots/<run_name>` folder
- Convention : Keep file name and the `run_name` same

## Runs : 

| Run name  | Description   |
|-----------|---------------|
|`longer_nvt_corrected` | NVT simulation with Q = 1000, xi = 0, v_xi = 0, n_steps = 1e7 |
|`longer_nvt` | NVT simulation with Q = 1, xi = 1, v_xi = 1 |
|`main_nve` | NVE simulation with Initial T = 2.0 |
|`main_nvt` | NVT simulation with T = 2.0   |
|`nvt_0.3` | NVT simulation with T = 0.3   |
|`nvt_corrected` | NVT simulation with Q = 1000, xi = 0, v_xi = 0, n_steps = 1e5   |
|`nvt_m2` | NVT simulation with M = 2  |


