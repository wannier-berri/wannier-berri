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
import pickle

SYM=wberri.symmetry

Efermi=np.linspace(12.,13.,11)
omega = np.linspace(0,1.,1001)
system=wberri.System_tb(tb_file='Fe_tb.dat',berry=True)

generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x] 
system.set_symmetry(generators)
grid=wberri.Grid(system,length=30,length_FFT=15)



if True:  # change to False if you want just to handle the pre-computed result
    #parallel=wberri.Serial() # serial execution
    parallel=wberri.Parallel() # parallel with  "ray",num_cpus - auto)
    result = wberri.run(system,
            grid=grid,
            calculators = {
                "ahc":wberri.calculators.static.AHC(Efermi=Efermi,tetra=False),
                 "tabulate":wberri.calculators.TabulatorAll({
                            "Energy":wberri.calculators.tabulate.Energy(),
                            "berry":wberri.calculators.tabulate.BerryCurvature(),
                            "connection":wberri.calculators.tabulateOD.BerryConnection(),
                                  }, 
                                       ibands = np.arange(4,10),jbands=np.arange(10,14)),
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
    with open("result.pickle","wb") as f:
        pickle.dump(result,f)
else:
    with open("result.pickle","rb") as f:
        result = pickle.load(f)
    

connection = result.results["tabulate"].results["connection"].data
connection = connection.reshape(tuple(grid.dense)+connection.shape[1:])
print (connection.shape)


