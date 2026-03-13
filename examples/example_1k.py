#!/usr/bin/env python3
import numpy as np
import wannierberri as wberri
import wannierberri.formula as formula
import wannierberri.w90files as w90files
from wannierberri.symmetry import point_symmetry as SYM
from wannierberri.system import System_R
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'




Efermi = np.linspace(12., 13., 11)
omega = np.linspace(0, 1., 1001)
system = System_R.from_w90data(w90files.WannierData.from_w90_files('../tests/data/Fe_Wannier90/Fe'), berry=True)

generators = [SYM.Inversion, SYM.C4z, SYM.TimeReversal * SYM.C2x]
system.set_pointgroup(generators)


result = wberri.evaluate_k(
    system,
    k=[0.1, 0.2, 0.3],
    quantities=["energy", "berry_curvature"],
    formula={"ham": formula.covariant.Hamiltonian,
             #    "vel" :formula.covariant.Velocity,
             #    "mass":formula.covariant.InvMass,
             #    "morb":formula.covariant.morb
                                        },
    param_formula={"morb": {"external_terms": False}},
    iband=[4, 5]
)
print(result)
