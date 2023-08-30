#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'

num_proc=16

import wannierberri as wberri
import numpy as np

system=wberri.System_tb(tb_file='Fe_tb.dat',getAA=True)

generators=['Inversion','C4z','TimeReversal*C2x']
system.set_symmetry(generators)

grid=wberri.Grid(system,length=100)

wberri.integrate(system,
            grid=grid,
            Efermi=np.linspace(12.,13.,1001),
            smearEf=10,
            quantities=["ahc","dos","cumdos"],
            numproc=num_proc,
            adpt_num_iter=10,
            libfft='fftw', #default.  alternative  option - 'numpy'
            fout_name='Fe',
            restart=False,
            )

wberri.tabulate(system,
             grid=grid,
             quantities=["berry"],
             fout_name='Fe',
             numproc=num_proc,
             ibands=np.arange(4,10),
             Ef0=12.6)
