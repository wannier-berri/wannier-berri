#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'

## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code=False
num_proc=4


import os

if local_code:
   if 'wannierberri' not in os.listdir() :
       os.symlink("../wannierberri","wannierberri")
else:
   if 'wannierberri' in os.listdir() :
       os.remove('wannierberri')


import wannierberri as wberri

import numpy as np


SYM=wberri.symmetry

Efermi=np.linspace(-0.5,0.5,1001)
system=wberri.System_fplo('../tests/data/Fe_FPLO/+hamdata',berry=False,spin=False)

generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
system.set_symmetry(generators)
grid=wberri.Grid(system,length=300,length_FFT=50)
parallel=wberri.parallel.Parallel(num_cpus=num_proc)

wberri.integrate(system,
            grid=grid,
            Efermi=Efermi, 
            smearEf=10,
            quantities=["ahc"], # ,"dos","cumdos"],
#            quantities=["dos","cumdos"],
            parallel=parallel,
            adpt_num_iter=10,
            parameters = {'tetra':False,"external_terms":False},
            fout_name='Fe_fplo',
            restart=False,
            )


# This line is not actually needed here and is added just for illustrative purpose.
# It is needed only when one wants to close and reopen a new parallel object.
parallel.shutdown()

