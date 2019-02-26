#!/bin/bash

Nprocs=32 

echo "MatrixFree, 4 sinkers, 1e4 visc jump"
for refinement in 4 5 6 7; do
  cp nsinker_mf.prm temp.prm

  echo "subsection Mesh refinement" >> temp.prm
  echo "  set Initial global refinement = $refinement" >> temp.prm
  echo "end" >> temp.prm
        
  echo "subsection Postprocess" >> temp.prm
  echo "  set List of postprocessors = memory statistics" >> temp.prm
  echo "end" >> temp.prm

  echo "set Output directory = output/global-mf-stampede-${Nprocs}-${refinement}" >> temp.prm
  echo "set Timing output directory = timings-on-stampede" >> temp.prm
  mpirun -np $Nprocs ./aspect temp.prm 
  rm temp.prm
done


echo "MatrixBased, 4 sinkers, 1e4 visc jump"
for refinement in 4 5 6 7; do
  cp nsinker_mb.prm temp.prm

  echo "subsection Mesh refinement" >> temp.prm
  echo "  set Initial global refinement = $refinement" >> temp.prm
  echo "end" >> temp.prm
        
  echo "subsection Postprocess" >> temp.prm
  echo "  set List of postprocessors = memory statistics" >> temp.prm
  echo "end" >> temp.prm

  echo "set Output directory = output/global-mb-stampede-${Nprocs}-${refinement}" >> temp.prm
  echo "set Timing output directory = timings-on-stampede/" >> temp.prm
  mpirun -np $Nprocs ./aspect temp.prm 
  rm temp.prm
done

