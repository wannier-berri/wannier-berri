#!/usr/bin/env python3

import wannierberri as wberri
import numpy as np
import ray


import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
ray.init(num_cpus=4)


SYM = wberri.point_symmetry

Efermi = np.linspace(12., 13., 11)
omega = np.linspace(0, 1., 1001)

seedname = '../tests/data/Fe_Wannier90/Fe'
try:
    for ext in ['mmn','eig','chk','bkvec']:
        if not os.path.isfile(f"{seedname}.{ext}.npz"):
            raise FileNotFoundError(f"{seedname}.{ext}.npz not found")
    w90data = wberri.w90files.Wannier90data.from_npz(seedname=seedname, files=['mmn','eig','chk','bkvec'],)
except FileNotFoundError as e:
    print(f"npz files not found, reading from w90 files and creating npz files for next time : {e}")
    w90data = wberri.w90files.Wannier90data.from_w90_files(seedname=seedname,
                                        files=['mmn','eig','chk'],)
    w90data.to_npz(seedname=seedname)
system = wberri.system.System_w90(w90data=w90data, berry=True)

generators = [SYM.Inversion, SYM.C4z, SYM.TimeReversal * SYM.C2x]
system.set_pointgroup(generators)
grid = wberri.Grid(system=system, length=30, length_FFT=15)


param_tabulate = {'ibands': np.arange(4, 10)}


wberri.run(system,
           grid=grid,
           calculators={
               "ahc": wberri.calculators.static.AHC(Efermi=Efermi, tetra=False),
               "tabulate": wberri.calculators.TabulatorAll({
                   "Energy": wberri.calculators.tabulate.Energy(),
                   "berry": wberri.calculators.tabulate.BerryCurvature(),
               },
                   ibands=np.arange(4, 10)),
               "opt_conductivity": wberri.calculators.dynamic.OpticalConductivity(Efermi=Efermi, omega=omega),
               #                 "shc_ryoo" : wberri.calculators.dynamic.SHC(Efermi=Efermi,omega=omega),
               #                  "SHC": wberri.fermiocean_dynamic.SHC(Efermi=Efermi,omega=omega,SHC_type="qiao")
           },
           adpt_num_iter=0,
           fout_name='Fe',
           suffix="run",
           restart=False,
            )
