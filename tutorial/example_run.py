#!/usr/bin/env python3
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

if 'Fe_tb.dat' not in os.listdir():
    os.system('tar -xvf ../data/Fe_tb.dat.tar.gz') 


import wannierberri as wberri

import numpy as np


SYM=wberri.symmetry

Efermi=np.linspace(12.,13.,11)
omega = np.linspace(0,1.,1001)
system=wberri.System_tb(tb_file='Fe_tb.dat',berry=True)

generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
system.set_symmetry(generators)
grid=wberri.Grid(system,length=15,length_FFT=15)
#parallel=wberri.Parallel(method="ray",num_cpus=num_proc)

parallel=wberri.Parallel() # serial execution
#parallel=wberri.Parallel(method="ray",num_cpus=num_proc)
param_tabulate = {'ibands':np.arange(4,10)}



t0 = time()
wberri.run(system,
            grid=grid,
            calculators = {
#                "ahc":wberri.calculators.static.AHC(Efermi=Efermi,tetra=False),
#                 "tabulate":wberri.calculators.TabulatorAll({
#                            "Energy":wberri.calculators.Energy(),
#                            "berry":wberri.calculators.BerryCurvature(),
#                                  }, 
#                                       ibands = np.arange(4,10))
                 "opt_conductivity" : wberri.calculators.dynamic.OpticalConductivity(Efermi=Efermi,omega=omega),
#                  "SHC": wberri.fermiocean_dynamic.SHC(Efermi=Efermi,omega=omega,SHC_type="qiao")
                          }, 
            parallel=parallel,
            adpt_num_iter=0,
            fout_name='Fe',
            suffix = "run",
            restart=False,
            )
t_run = time()-t0


t0=time()
if False:
    wberri.integrate(system,
            grid=grid,
            Efermi=Efermi, 
            omega = omega,
            smearEf=10,
            quantities=["opt_conductivity"],
            parallel=parallel,
            adpt_num_iter=0,
#            parameters = {'tetra':True},
#            specific_parameters = {'dos^2':{'tetra':False}},
#            parameters_K = {'fftlib':'fftw'}, #default.  alternative  option - 'numpy'
            fout_name='Fe',
            suffix = "integrate",
            restart=False,
            )
t_int=time()-t0






print (f"time for \nrun       : {t_run} \nintegrate : {t_int} \n ratio      {t_run/t_int}")

parallel.shutdown()

