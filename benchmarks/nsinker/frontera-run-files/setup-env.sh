#!/bin/bash

# this file is sourced right before the job is run on the cluster and contains
# configurations specific to the run.

echo "starting on `date` with n=$SLURM_NTASKS at `pwd`"

. $SCRATCH/enable3.sh || exit 1

echo nodelist is $SLURM_JOB_NODELIST
echo node count is $SLURM_JOB_NUM_NODES

export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "DEAL_II_NUM_THREADS=$DEAL_II_NUM_THREADS"
