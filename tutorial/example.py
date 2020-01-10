#!/usr/bin/env python3


## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code=True #False
num_proc=2

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

seedname="Fe"
NK=96

name=seedname
Efermi=np.linspace(12.,13.,1001)
system=wberri.System(tb_file='Fe_tb.dat',getAA=True)

generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]


if False:
   wberri.tabulate(system,
             NK=NK,
             quantities=["V","berry"],
             symmetry_gen=generators,
             fout_name=name,
             numproc=num_proc,
             Ef0=0,
             restart=False)


wberri.integrate(system,
            NK=NK,
            Efermi=Efermi, 
            smearEf=10,
            quantities=["ahc","dos"],#,"ahc_band"],
            numproc=num_proc,
            adpt_num_iter=0,
            fout_name=name,
            symmetry_gen=generators,
            restart=False,
#             parameters={"degen_thresh":0.1}
            )
