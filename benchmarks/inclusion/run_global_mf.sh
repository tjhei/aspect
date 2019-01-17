#!/bin/bash

# Number of processors to run on
NP=4

# global refinement:
echo "----Global refinement----"
for r in "4" "5" "6" "7" "8"
do
  echo "ref $r:"
  cp global_test.prm temp.prm
  echo "subsection Mesh refinement" >> temp.prm
  echo "set Initial global refinement = $r" >> temp.prm
  echo "end" >> temp.prm
  echo "set Output directory = output/mf/global/ref$r/" >> temp.prm
  mpirun -n $NP ./aspect temp.prm | grep "iteration\|DoFs"
  rm -f temp.prm
done
