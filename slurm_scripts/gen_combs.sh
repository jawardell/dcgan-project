#!/bin/bash

lrs=(0.1 0.01 0.001 0.0001)
bss=(64 128 256)
wds=(0.4 0.04 0.004 0.0004)

for (( i = 0; i < 4; i++ )); do
	for (( j = 0; j < 3; j++ )); do
		for (( k = 0; k < 4; k++ )); do
			lr=${lrs[${i}]}
			bs=${bss[${j}]}
			wd=${wds[${k}]}
			echo "$lr $bs $wd" >> parameter_combinations.txt
		done
	done
done

