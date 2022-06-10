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


SYM=wberri.symmetry

#Efermi = np.linspace(12.,13.,101)
omega  = np.linspace(-10,30.,1001)
system = wberri.System_Phonon_QE('../tests/data/MgB2_phonons/mgb2')

generators=["C3z","TimeReversal"]
system.set_symmetry(generators)
grid=wberri.Grid(system,length=10,NKFFT=4)

#parallel=wberri.Serial() # serial execution
parallel=wberri.Parallel() # parallel with  "ray",num_cpus - auto)


t0 = time()
wberri.run(system,
            grid=grid,
            calculators = {
                "dos":wberri.calculators.static.DOS(Efermi=omega,tetra=False),
                "dos_tetra":wberri.calculators.static.DOS(Efermi=omega,tetra=True),
                "cumdos_tetra":wberri.calculators.static.CumDOS(Efermi=omega,tetra=True),
#                 "tabulate":wberri.calculators.TabulatorAll({
#                            "Energy":wberri.calculators.tabulate.Energy(),
#                            "berry":wberri.calculators.tabulate.BerryCurvature(),
#                                  }, 
#                                       ibands = np.arange(4,10)),
#                 "opt_conductivity" : wberri.calculators.dynamic.OpticalConductivity(Efermi=Efermi,omega=omega),
#                 "shc_ryoo" : wberri.calculators.dynamic.SHC(Efermi=Efermi,omega=omega),
#                  "SHC": wberri.fermiocean_dynamic.SHC(Efermi=Efermi,omega=omega,SHC_type="qiao")
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
#                 "tabulate":wberri.calculators.TabulatorAll({
#                            "Energy":wberri.calculators.tabulate.Energy(),
#                            "berry":wberri.calculators.tabulate.BerryCurvature(),
#                                  }, 
#                                       ibands = np.arange(4,10)),
#                 "opt_conductivity" : wberri.calculators.dynamic.OpticalConductivity(Efermi=Efermi,omega=omega),
#                 "shc_ryoo" : wberri.calculators.dynamic.SHC(Efermi=Efermi,omega=omega),
#                  "SHC": wberri.fermiocean_dynamic.SHC(Efermi=Efermi,omega=omega,SHC_type="qiao")
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
        [0.0000, 0.0000, 0.0000 ],   #  G
        [0.000 , 0.0000, 0.5000],   #  A
        [1/3, 1/3, 0.5],   #  H
        [1/3, 1/3, 0.0],   #  K
        [0.0000, 0.0000, 0.000  ] ] , #  G
                 labels=["G","A","H","K","G"],
                 length=200 )



result = wberri.run(system,
            grid=path,
            calculators = {
                 "tabulate":wberri.calculators.TabulatorAll({
                            "Energy":wberri.calculators.tabulate.Energy(),
#                            "berry":wberri.calculators.tabulate.BerryCurvature(),
                                  },mode="path"), 
#                                       ibands = np.arange(4,10)),
#                 "opt_conductivity" : wberri.calculators.dynamic.OpticalConductivity(Efermi=Efermi,omega=omega),
#                 "shc_ryoo" : wberri.calculators.dynamic.SHC(Efermi=Efermi,omega=omega),
#                  "SHC": wberri.fermiocean_dynamic.SHC(Efermi=Efermi,omega=omega,SHC_type="qiao")
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
              save_file="MgB2_phonons.pdf",
#              Eshift=EF,
#              Emin=-10,  Emax=50,
              iband=None,
              mode="fatband",
              fatfactor=20,
              cut_k=False,
              close_fig=True,
              show_fig=False,
              label = "WB"
              )


parallel.shutdown()

