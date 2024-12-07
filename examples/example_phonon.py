#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'

## these linesline if you want to use the git version of the code, instead of the one installed by pip
local_code=False
num_proc=4

from time import time

import os


import wannierberri as wberri

import numpy as np


SYM=wberri.point_symmetry

#Efermi = np.linspace(12.,13.,101)
omega  = np.linspace(-0.01,0.1,1101)
system = wberri.System_Phonon_QE('../tests/data/Si_phonons/si',asr=True)

generators=["C4z","C4x","TimeReversal"]
system.set_pointgroup(generators)
grid=wberri.Grid(system,length=10,NKFFT=4)

#parallel=wberri.Serial() # serial execution
parallel=wberri.Parallel(num_cpus=4) # parallel with  "ray",num_cpus - auto)


t0 = time()
wberri.run(system,
            grid=grid,
            calculators = {
                "dos":wberri.calculators.static.DOS(Efermi=omega,tetra=False),
                "dos_tetra":wberri.calculators.static.DOS(Efermi=omega,tetra=True),
                "cumdos_tetra":wberri.calculators.static.CumDOS(Efermi=omega,tetra=True),
                          }, 
            parallel=parallel,
            use_irred_kpt=False,
            symmetrize=True,
            adpt_num_iter=0,
            fout_name='bn',
            suffix = "nosym",
            restart=False,
            )
t_run = time()-t0

wberri.run(system,
            grid=grid,
            calculators = {
                "dos":wberri.calculators.static.DOS(Efermi=omega,tetra=False),
                "dos_tetra":wberri.calculators.static.DOS(Efermi=omega,tetra=True),
                "cumdos_tetra":wberri.calculators.static.CumDOS(Efermi=omega,tetra=True),
                          }, 
            parallel=parallel,
            use_irred_kpt=True,
            symmetrize=True,
            adpt_num_iter=0,
            fout_name='bn',
            suffix = "sym",
            restart=False,
            )


path=wberri.Path(system,
                k_nodes=[
 [ 0.0000, 0.0000, 0.0000], #30 !Gamma
 [ -0.500, 0.0000, -0.500], #10 !X
 [ 0.0000, 0.3750, -0.375], #20 !K
 [ 0.0000, 0.0000, 0.0000], #30 !Gamma
 [ 0.0000, 0.5000, 0.0000] ] , #1 !L
                 labels=["G","X","K","G","M"],
                 length=200 )


result = wberri.run(system,
            grid=path,
            calculators = {
                 "tabulate":wberri.calculators.TabulatorAll({
                            "Energy":wberri.calculators.tabulate.Energy(),
                                  },mode="path"), 
                          }, 
            parallel=parallel,
            use_irred_kpt=True,
            symmetrize=True,
            adpt_num_iter=0,
            fout_name='bn',
            suffix = "sym",
            restart=False,
            )
path_result = result.results["tabulate"]

path_result.plot_path_fat( path,
              quantity=None,
              save_file="Si_phonons.pdf",
#              Eshift=EF,
              Emin=-0.01,  Emax=0.1,
              iband=None,
              mode="fatband",
              fatfactor=20,
              cut_k=False,
              close_fig=True,
              show_fig=False,
              label = "WB"
              )


parallel.shutdown()

