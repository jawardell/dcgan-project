#!/bin/bash
SLURM_ARRAY_TASK_ID=0

export PATH=/data/users2/jwardell1/miniconda3/bin:$PATH

source /data/users2/jwardell1/miniconda3/etc/profile.d/conda.sh

CONDA_PATH=`which conda`

eval "$(${CONDA_PATH} shell.bash hook)"
conda activate /data/users2/jwardell1/miniconda3/envs/dcgan

PARAMS_FILE=/data/users2/jwardell1/dcgan-project/slurm_scripts/parameter_combinations.txt
IFS=$'\n'
params_combinations_array=($(cat $PARAMS_FILE))

IFS=$' '
current_params=(${params_combinations_array[$SLURM_ARRAY_TASK_ID]})
lr=${current_params[0]}
bs=${current_params[1]}
wd=${current_params[2]}

echo "lr - $lr"
echo "bs - $bs"
echo "wd - $wd"

python /data/users2/jwardell1/dcgan-project/python_scripts/supervised_baseline.py $lr $bs $wd
