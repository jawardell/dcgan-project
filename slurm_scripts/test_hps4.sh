#!/bin/bash

lrs=(0.1 0.01 0.001 0.0001)
wds=(0.04 0.004 0.0004)
bss=(64 128 256)

for (( i=0; i<50; i++ ));do
	LR=${lrs[$(($i % 4))]}
	WEIGHT_DECAY=${wds[$((($i / 4) % 4))]}
	BATCH_SIZE=${bss[$((($i / 16) % 6))]}
	echo "SLURM_ARRAY_TASK_ID- ${i}  LR- ${LR}  WEIGHT_DECAY- ${WEIGHT_DECAY}  BATCH_SIZE- ${BATCH_SIZE}" >> hp.txt
done
cat hp.txt | column -t
