#!/bin/bash

for averaging in none; do # arithmetic/geometric/harmonic average
  for nsinkers in 4 8 16; do
    for viscosity in 1e1 1e3 1e5 1e7 ; do
      for refinement in   4 5 6 ; do
        echo "subsection Material model" > current.prm
        echo "  set Material averaging = $averaging" >> current.prm
        echo "  subsection NSinker" >> current.prm
        echo "    set Number of sinkers = $nsinkers" >> current.prm
        echo "    set Dynamic viscosity ratio = $viscosity" >> current.prm
        echo "  end" >> current.prm
        echo "end" >> current.prm
        echo "subsection Solver parameters">>current.prm
		echo "subsection Stokes solver parameters">>current.prm
			echo "set Stokes solver type= block GMG">>current.prm
			echo "set Use full A block as preconditioner=true">>current.prm
			echo "set Use weighted BFBT for Schur complement=true">>current.prm
			echo "set Number of cheap Stokes solver steps=500">>current.prm
			echo "set Maximum number of expensive Stokes solver steps=0">>current.prm
		echo "end">>current.prm
        echo "subsection AMG parameters">>current.prm
		echo "set AMG aggregation threshold = 0.02">>current.prm
	echo "end">>current.prm	
	echo "end">>current.prm
        echo "subsection Mesh refinement" >> current.prm
        echo "  set Initial global refinement = $refinement" >> current.prm
        echo "end" >> current.prm
	echo "subsection Discretization">>current.prm
		echo "set Use locally conservative discretization=false">>current.prm
	echo "end">>current.prm

	current_model="averaging${averaging}_nsinkers${nsinkers}_viscosity${viscosity}_refinement${refinement}"
        echo "set Output directory = home/quang-hoang/Documents/aspect/benchmarks/nsinker/diag_A_diag_mp/output-${current_model}" >> current.prm
        echo "Starting ${current_model}"
        cat nsinker.prm current.prm | mpirun -np 32 ./aspect-release --
      done
    done
  done
done
