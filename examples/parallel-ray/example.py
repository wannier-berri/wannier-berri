#!/usr/bin/env python3
import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'

## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code=True


import os

if local_code:
   if 'wannierberri' not in os.listdir() :
       os.symlink("../../wannierberri","wannierberri")
else:
   if 'wannierberri' in os.listdir() :
       os.remove('wannierberri')



import wannierberri as wberri

import numpy as np


SYM=wberri.point_symmetry

Efermi=np.linspace(12.,13.,1001)
system=wberri.system.System_w90('../../tests/data/Fe_Wannier90/Fe',berry=True)

generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
system.set_pointgroup(generators)
grid=wberri.Grid(system,NKdiv=16,NKFFT=16)

parallel = wberri.Parallel(method='ray',cluster=True)


wberri.run(system,
            grid=grid,
            calculators = {
                "ahc":wberri.calculators.static.AHC(Efermi=Efermi,tetra=False),
                          }, 
            parallel=parallel,
            adpt_num_iter=0,
            fout_name='Fe',
            suffix = "w90-ray",
            restart=False,
            )
