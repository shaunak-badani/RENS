#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
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

name="NVE"

jupyter nbconvert --to script ${name}.ipynb
mv ${name}.txt ${name}.py

TEMP=2.0
FOLDER_NAME="T_${TEMP}"
# Make sure this path matches the path where the positions and velocities are stored in scratch in the notebook
TXT_PATH="/scratch/shaunak/${FOLDER_NAME}"
DATA_FILES="${TXT_PATH}/*.txt"

python3 ${name}.py
rsync -aPs --rsync-path="mkdir -p /share1/shaunak/1D_system/${FOLDER_NAME} && rsync" $DATA_FILES ada:"/share1/shaunak/1D_system/${FOLDER_NAME}/"




