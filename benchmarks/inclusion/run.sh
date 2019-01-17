#!/bin/bash

# Number of processors to run on
NP=4

# averaging scheme:
for avg in "harmonic average"
do
  echo "----Averaging scheme: $avg----"

  echo "----Global refinement----"
  for r in "4" "5" "6" "7" "8"
  do
    echo "ref $r:"
    cp global.prm.base temp.prm
    echo "set Output directory = output/mb/global/ref$r" >> temp.prm
    echo "subsection Material model" >> temp.prm
    echo "set Material averaging = $avg" >> temp.prm
    echo "end" >> temp.prm
    echo "subsection Mesh refinement" >> temp.prm
    echo "set Initial global refinement = $r" >> temp.prm
    echo "end" >> temp.prm
    mpirun -n $NP ./aspect temp.prm | grep "Solving Stokes system\|DoFs" 
    rm -f temp.prm
  done
echo
echo

# adaptive refinement:
  echo "----Adaptive refinement----"
  cp adaptive.prm.base temp.prm
  echo "set Output directory = output/mb/adaptive" >> temp.prm
  echo "subsection Material model" >> temp.prm
  echo "set Material averaging = $avg" >> temp.prm
  echo "end" >> temp.prm
  mpirun -n $NP ./aspect temp.prm | grep "Solving Stokes system\|DoFs"
  rm -f temp.prm
done
