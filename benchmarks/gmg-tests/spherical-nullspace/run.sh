#!/usr/bin/env bash

NP=${NP:-8}

for method in GMG-GC GMG-LS AMG; do
  for case_name in free-surface free-slip partial-free-slip; do
    cat "${case_name}.prm" > current.prm

    if [[ ${method} == AMG ]]; then
      cat amg.prm.part >> current.prm
    elif [[ ${method} == GMG-LS ]]; then
      cat gmg-local-smoothing.prm.part >> current.prm
    fi

    mpirun -n "${NP}" ./aspect-release current.prm 2>&1 | tee "log-${method}-${case_name}.txt"
  done
done
