#!/bin/bash

# Number of processors to run on
NP=4

# adaptive refinement:
echo "----Adaptive refinement----"
cp adaptive_test.prm temp.prm
echo "set Output directory = output/mf/adaptive" >> temp.prm
mpirun -n $NP ./aspect temp.prm | grep "iteration\|DoFs"
rm -f temp.prm

