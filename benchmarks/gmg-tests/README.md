
# Test 1: box-3d-bc/

3d box (nsinker) with various combinations of boundary conditions:

no-slip: all 6 sides no slip
free-slip: all sides free slip
partial-free-slip: left and front free slip, rest no slip
partial-set: left: free slip; front: x set to smooth function, y=0; other: no slip
open: no slip on all sides except top, which is open
periodic: periodic in x direction; no slip otherwise
free-surface: periodic in x, free surface on top, no slip otherwise

Stokes: Q2Q1, 7.8m DoFs (adaptively refined)
tolerance: 1e-8
4 sinkers, viscosity ratio: 1e4

Stokes GMRES iterations on the final (largest) mesh:

| boundary condition | GMG global coarsening | GMG local smoothing | AMG |
|---|---|---|---|
| no-slip | 36 | 39 | 70 |
| free-slip | 61 | 72 | 172 |
| partial-free-slip | 50 | 52 | 105 |
| partial-set | 31 | 33 | 61 |
| open | 41 | 43 | 88 |
| periodic | 33 | - | 84 |
| free-surface | ? | ? | ? |

without number: crashes or error with "not supported"
