
Test 1: box-3d-bc/

3d box (nsinker) with various combinations of boundary conditions:

no-slip: all 6 sides no slip
free-slip: all sides free slip
partial-free-slip: left and front free slip, rest no slip
partial-set: left: free slip; front: x set to smooth function, y=0; other: no slip
open: no slip on all sides except top, which is open
periodic: periodic in x direction; no slop otherwise

7.8m DoFs (adaptively refined)
