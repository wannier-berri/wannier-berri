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
       os.symlink("../wannierberri","wannierberri")
else:
   if 'wannierberri' in os.listdir() :
       os.remove('wannierberri')

if 'Fe_tb.dat' not in os.listdir():
    os.system('tar -xvf ../data/Fe_tb.dat.tar.gz') 


import wannierberri as wberri

import numpy as np


SYM=wberri.symmetry

Efermi=np.linspace(12.,13.,1001)
system=wberri.System_tb(tb_file='Fe_tb.dat',berry=True)

generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
system.set_symmetry(generators)
grid=wberri.Grid(system,length=100,length_FFT=15)
parallel=wberri.Parallel(method="ray",num_cpus=num_proc)

wberri.integrate(system,
            grid=grid,
            Efermi=Efermi, 
            smearEf=10,
            quantities=["ahc","dos^1","dos^2","cumdos"],
            parallel=parallel,
            adpt_num_iter=10,
            parameters = {'tetra':True},
            specific_parameters = {'dos^2':{'tetra':False}},
            fftlib='fftw', #default.  alternative  option - 'numpy'
            fout_name='Fe',
            restart=False,
            )


wberri.tabulate(system,
             grid=grid,
             quantities=["berry"],
             frmsf_name='Fe',
             parallel=parallel,
             ibands=np.arange(4,10),
             Ef0=12.6)


# Shutdown Parallel object.
# This line is not actually needed here and is added just for illustrative purpose.
# It is needed only when one wants to close and reopen a new parallel object.
parallel.shutdown()

