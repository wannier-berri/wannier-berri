#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'

## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code=True
num_proc=4


import os

if local_code:
   if 'wannierberri' not in os.listdir() :
       os.symlink("../../../wannierberri","wannierberri")
else:
   if 'wannierberri' in os.listdir() :
       os.remove('wannierberri')


import wannierberri as wberri

import numpy as np


SYM=wberri.symmetry

Efermi=np.linspace(17.,18.,1001)
system=wberri.System_w90('Fe',berry=True,spin=True,morb=True)

#generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
#system.set_symmetry(generators)
grid=wberri.Grid(system,length=30,length_FFT=15)
parallel=wberri.Parallel(method="ray",num_cpus=num_proc)

for sym in False,True: # the order is important here       
    if sym: 
        system.symmetrize(
            proj = ['Fe:s','Fe:p','Fe:d'],
            atom_name = ['Fe'],
            positions = [[0,0,0]],
            magmom = [[0,0,1.]],soc=True
            )


    wberri.integrate(system,
            grid=grid,
            Efermi=Efermi, 
            smearEf=100,
            quantities=["ahc","dos","cumdos","spin","Morb","conductivity_ohmic"],
            parallel=parallel,
            adpt_num_iter=0,
            parameters = {'tetra':False,'external_terms':True},
            #specific_parameters = {'dos^2':{'tetra':False}},
            fftlib='fftw', #default.  alternative  option - 'numpy'
            fout_name='Fe-sym',suffix = f'sym={sym}+ext',
            restart=False,
            )


# Shutdown Parallel object.
# This line is not actually needed here and is added just for illustrative purpose.
# It is needed only when one wants to close and reopen a new parallel object.
parallel.shutdown()

