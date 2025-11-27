#!/usr/bin/env python3
import numpy as np
import wannierberri as wberri
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code = False
num_proc = 4



if local_code:
    if 'wannierberri' not in os.listdir():
        os.symlink("../wannierberri", "wannierberri")
else:
    if 'wannierberri' in os.listdir():
        os.remove('wannierberri')





SYM = wberri.point_symmetry

# Efermi=np.linspace(12.,13.,1001)
Efermi = np.linspace(-0.5, 0.5, 1001)
system = wberri.System_fplo('../tests/data/Fe_FPLO/+hamdata', berry=False, spin=False)

generators = [SYM.Inversion, SYM.C4z, SYM.TimeReversal * SYM.C2x]
system.set_pointgroup(generators)
grid = wberri.Grid(system, length=300, length_FFT=50)

wberri.ray_init()
wberri.run(system,
           grid=grid,
           calculators={
               "ahc": wberri.calculators.static.AHC(Efermi=Efermi, tetra=False),
           },
           adpt_num_iter=0,
           fout_name='Fe',
           suffix="fplo",
           restart=False,
            )
