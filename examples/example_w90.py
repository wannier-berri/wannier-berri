#!/usr/bin/env python

import wannierberri as wberri
import wannierberri.calculators as calculators
import wannierberri.w90files as w90files
from wannierberri.symmetry import point_symmetry as SYM
from wannierberri.system import System_R
import numpy as np
import ray


import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
ray.init(num_cpus=4)


Efermi = np.linspace(12., 13., 11)
omega = np.linspace(0, 1., 1001)

seedname = '../tests/data/Fe_Wannier90/Fe'
try:
    for ext in ['mmn', 'eig', 'chk', 'bkvec']:
        if not os.path.isfile(f"{seedname}.{ext}.npz"):
            raise FileNotFoundError(f"{seedname}.{ext}.npz not found")
    w90data = w90files.WannierData.from_npz(seedname=seedname, files=['mmn', 'eig', 'chk', 'bkvec'],)
except FileNotFoundError as e:
    print(f"npz files not found, reading from w90 files and creating npz files for next time : {e}")
    w90data = w90files.WannierData.from_w90_files(seedname=seedname,
                                        files=['mmn', 'eig', 'chk'],)
    w90data.to_npz(seedname=seedname)
system = System_R.from_w90data(w90data=w90data, berry=True)

generators = [SYM.Inversion, SYM.C4z, SYM.TimeReversal * SYM.C2x]
system.set_pointgroup(generators)
grid = wberri.Grid(system=system, length=30, length_FFT=15)


param_tabulate = {'ibands': np.arange(4, 10)}


wberri.run(system,
           grid=grid,
           calculators={
               "ahc": calculators.static.AHC(Efermi=Efermi, tetra=False),
               "tabulate": calculators.TabulatorAll({
                   "Energy": calculators.tabulate.Energy(),
                   "berry": calculators.tabulate.BerryCurvature(),
               },
                   ibands=np.arange(4, 10)),
               "opt_conductivity": calculators.dynamic.OpticalConductivity(Efermi=Efermi, omega=omega),
               #                 "shc_ryoo" : calculators.dynamic.SHC(Efermi=Efermi,omega=omega),
               #                  "SHC": calculators.dynamic.SHC(Efermi=Efermi,omega=omega,SHC_type="qiao")
           },
           adpt_num_iter=0,
           fout_name='Fe',
           suffix="run",
           restart=False,
            )
