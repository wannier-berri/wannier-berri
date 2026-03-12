#!/usr/bin/env python3
import numpy as np
import wannierberri as wberri
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code = True



if local_code:
    if 'wannierberri' not in os.listdir():
        os.symlink("../../wannierberri", "wannierberri")
else:
    if 'wannierberri' in os.listdir():
        os.remove('wannierberri')






SYM = wberri.point_symmetry

Efermi = np.linspace(12., 13., 1001)
w90data = wberri.w90files.Wannier90data.from_w90_files(
    '../../tests/data/Fe_Wannier90/Fe', 
    files=['mmn', 'eig', 'chk'],)
system = wberri.System_R.from_w90data(w90data, berry=True)

generators = [SYM.Inversion, SYM.C4z, SYM.TimeReversal * SYM.C2x]
system.set_pointgroup(generators)
grid = wberri.Grid(system=system, NKdiv=16, NKFFT=16)

wberri.ray_init()

wberri.run(system,
           grid=grid,
           calculators={
               "ahc": wberri.calculators.static.AHC(Efermi=Efermi, tetra=False),
           },
           adpt_num_iter=0,
           fout_name='Fe',
           suffix="w90-ray",
           restart=False,
            )
