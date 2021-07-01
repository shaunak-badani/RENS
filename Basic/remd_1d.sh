#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 4
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=NONE

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
mpiexec -n 4 python main_remd.py
