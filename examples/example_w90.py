#!/usr/bin/env python3

import wannierberri as wberri
import numpy as np



import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'

## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code=True
num_proc=4

from time import time

import os

if local_code:
   if 'wannierberri' not in os.listdir() :
       os.symlink("../wannierberri","wannierberri")
else:
   if 'wannierberri' in os.listdir() :
       os.remove('wannierberri')




SYM=wberri.point_symmetry

Efermi=np.linspace(12.,13.,11)
omega = np.linspace(0,1.,1001)
system=wberri.system.System_w90('../tests/data/Fe_Wannier90/Fe',berry=True)

generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
system.set_pointgroup(generators)
grid=wberri.Grid(system,length=30,length_FFT=15)

#parallel=wberri.Serial() # serial execution
parallel=wberri.Parallel() # parallel with  "ray",num_cpus - auto)
param_tabulate = {'ibands':np.arange(4,10)}



wberri.run(system,
            grid=grid,
            calculators = {
                "ahc":wberri.calculators.static.AHC(Efermi=Efermi,tetra=False),
                 "tabulate":wberri.calculators.TabulatorAll({
                            "Energy":wberri.calculators.tabulate.Energy(),
                            "berry":wberri.calculators.tabulate.BerryCurvature(),
                                  }, 
                                       ibands = np.arange(4,10)),
                 "opt_conductivity" : wberri.calculators.dynamic.OpticalConductivity(Efermi=Efermi,omega=omega),
#                 "shc_ryoo" : wberri.calculators.dynamic.SHC(Efermi=Efermi,omega=omega),
#                  "SHC": wberri.fermiocean_dynamic.SHC(Efermi=Efermi,omega=omega,SHC_type="qiao")
                          }, 
            parallel=parallel,
            adpt_num_iter=0,
            fout_name='Fe',
            suffix = "run",
            restart=False,
            )

parallel.shutdown()

