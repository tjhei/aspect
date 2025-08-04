#!/bin/bash
set -e

# MPI hello world to see if mpi4py works

cd mpi-hello-world
mpirun -n 2 python test.py
cd ..
