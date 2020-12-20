#!/usr/bin/env python3
num_proc=0

import wannierberri as wberri

import numpy as np

SYM=wberri.symmetry

Efermi=np.linspace(11.1299,21.1299,101)
omega=np.linspace(0.0,0.0,1)

system=wberri.System_w90(seedname='pt',SHCqiao=True,SHCryoo=True,use_ws=False,transl_inv=False)

generators=[]
system.set_symmetry(generators)
grid=wberri.Grid(system,NK=np.array([9,9,9]))

wberri.integrate(system,
            grid=grid,
            Efermi=Efermi, 
            omega=omega,
#            smearEf=0.1,
#            smearW=0.1,
            quantities=["opt_SHCqiao","opt_SHCryoo"],
            numproc=num_proc,
            adpt_num_iter=0,
            fout_name='pt',
            restart=False
            )