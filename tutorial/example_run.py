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
grid=wberri.Grid(system,length=30,length_FFT=15)
#parallel=wberri.Parallel(method="ray",num_cpus=num_proc)
parallel=wberri.Parallel() # serial execution

wberri.run(system,
            grid=grid,
            calculators = {
                "ahc_calc":wberri.calculators.AHC(Efermi,tetra=True)
                          }, 
            parallel=parallel,
            adpt_num_iter=10,
            fout_name='Fe',
            restart=False,
            )

parallel.shutdown()

