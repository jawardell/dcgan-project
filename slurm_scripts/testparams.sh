#!/bin/bash

PARAMS_FILE=`pwd`/parameter_combinations.txt
IFS=$'\n'
params_combinations_array=($(cat $PARAMS_FILE))
num_combinations=$(cat $PARAMS_FILE | wc -l)

for (( i = 0; i < $num_combinations; i++ )); do
	IFS=$' '
	current_params=(${params_combinations_array[$i]})
	lr=${current_params[0]}
	bs=${current_params[1]}
	wd=${current_params[2]}

	echo "SLURM_ARRAY_TASK_ID - $i"
	echo "lr - $lr"
	echo "bs - $bs"
	echo "wd - $wd"
done
