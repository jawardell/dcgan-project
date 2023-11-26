#!/bin/bash

effective_batch_sizes=(512 1024 2048 4096)
num_views=(2 4 8 16)

for (( i=0; i<50; i++ ));do
	eff_batch_size=${effective_batch_sizes[$((( $i % 4 )))]}
	n_groups=${num_views[$((( $i / 4 )))]}
	batch_size=$(( eff_batch_size / n_groups ))
	echo "SLURM_ARRAY_TASK_ID- ${i}  eff_batch_size- ${eff_batch_size}  n_groups- ${n_groups}  batch_size- ${batch_size}"
done
