#!/bin/bash

a=1
for ((i=1; i<=4; i++)); do      
    for j in 16 32 64 128 256 512 1024; do
        awk "NR>=$[i+a] && NR<=$[i+a]" ${j}proc_mb >> ${i}ref_mb
    done
done
