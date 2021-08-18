#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=1d_slurm.out

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/shaunak/miniconda2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/shaunak/miniconda2/etc/profile.d/conda.sh" ]; then
        . "/home/shaunak/miniconda2/etc/profile.d/conda.sh"
    else
        export PATH="/home/shaunak/miniconda2/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate MD
mpirun -np 2 python3 main.py -c remd.json


