#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'

import wannierberri as wberri

import numpy as np


SYM=wberri.point_symmetry

Efermi=np.linspace(12.,13.,11)
omega = np.linspace(0,1.,1001)
system=wberri.system.System_w90('../tests/data/Fe_Wannier90/Fe',berry=True)

generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
system.set_pointgroup(generators)


result = wberri.evaluate_k(
                            system,
                            k=[0.1,0.2,0.3],
                            quantities = ["energy","berry_curvature"],
                            formula = {     "ham" :wberri.formula.covariant.Hamiltonian,
                                        #    "vel" :wberri.formula.covariant.Velocity,
                                        #    "mass":wberri.formula.covariant.InvMass,
                                        #    "morb":wberri.formula.covariant.morb
                                        },
                            param_formula = {"morb":{"external_terms":False}},
                            iband=[4,5]
                        )
print (result)



