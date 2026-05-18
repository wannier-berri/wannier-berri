This is a more complete example, showing the entire pipeline:
fom ASE, GPAW
through wannier90
to wannierberri and SOT computations

!!! Generates a lot of temporary data : either remove the data
or copy the scripts and run them in a different directory !!!

!!! for 1_scf+nscf+write_w90.py, If re-running only certain steps,
change the corresponding flags at the top of the file!!!

the fat bands show some local response, but torkance comes out to 0 in all directions
due to the inversion symmetry of the bulk MnTe crystal (as far as I understand)