```{tags}
category:benchmark
feature:solver-comparison
```

(sec:benchmarks:gmg-tests)=
# Linear solver convergence tests

A collection of models to examine the performance and capabilities of
the GMG linear solver. Currently, the folder only contains a single test
set. This will be expanded in the future to include other geometries
and problems.


## Test 1: nsinker-box-3d

The 3d box (nsinker) problem with various combinations of boundary conditions:

- no-slip: all 6 sides no slip
- free-slip: all sides free slip
- partial-free-slip: left and front free slip, rest no slip
- partial-set: left: free slip; front: x set to smooth function, y=0; other: no slip
- open: no slip on all sides except top, which is open
- periodic: periodic in x direction; no slip otherwise
- free-surface: free surface on top with smooth topography, no slip bottom, free slip rest

The common test problem settings are:
- Q2-Q1
- tolerance: 1e-8
- 4 sinkers
- viscosity ratio: 1e4

Stokes GMRES iterations on the final mesh with 7.8m Stokes DoFs (10.3m total DoFs) with adaptive refinement:

| boundary condition | GMG global coarsening | GMG local smoothing | AMG |
|---|---|---|---|
| no-slip | 37 | 39 | 70 |
| free-slip | 61 | 73 | 172 |
| partial-free-slip | 51 | 53 | 105 |
| partial-set | 32 | 33 | 62 |
| open | 42 | 44 | 88 |
| periodic | 33 | - | 85 |
| free-surface | 67 | 78 | 195 |

Currently unsupported cases:
- GMG-LS: no support for periodic boundaries


# Test 2: spherical-nullspace

A 2d spherical shell computation with adaptive refinement. This model
checks the handling of the rotational nullspace with different
combinations of boundary conditions:

Mesh deformation iterations on the last timestep reached, and Stokes iterations
from the latest timestep with a completed Stokes solve in each log:

| boundary condition | solver | mesh deformation iterations | Stokes iterations | status |
|---|---|---:|---:|---|
| free-slip | GMG global coarsening | - | 8 | completed |
| free-slip | GMG local smoothing | - | 12 | completed |
| free-slip | AMG | - | 42 | completed |
| partial-free-slip | GMG global coarsening | - | 10 | completed |
| partial-free-slip | GMG local smoothing | - | 14 | completed |
| partial-free-slip | AMG | - | 73 | completed |
| free-surface | GMG global coarsening | 98 | 17 | Error at timestep 3* |
| free-surface | GMG local smoothing | 36 | 19 | Error at timestep 9* |
| free-surface | AMG | 17 | 152 | Advection solver failed at timestep 2* |

`-` means no mesh deformation solve was present. `*` The log ends with an
error before the last timestep completed. For these runs, the Stokes count is
from the most recent timestep before the error that completed a Stokes solve.
