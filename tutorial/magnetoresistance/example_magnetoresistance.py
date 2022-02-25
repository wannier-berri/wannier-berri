#!/usr/bin/env python3
import wannierberri as wberri
import wannierberri.calculators.static as calc
import numpy as np

model = wberri.models.Chiral(delta=2, hop1=1, hop2=1./3*0.,  phi=np.pi/10, hopz_vert=0.2,hopz_left=0,hopz_right=0)
system = wberri.System_PythTB(model,use_wcc_phase = True)
system.set_spin([-1,1],axis=[0,0,1])
system.set_symmetry(['C3z'])


parallel=wberri.Parallel(method="ray",num_cpus=8)
noexter = {"kwargs_formula":{'external_terms' : False}}
Efermi=np.linspace(-5,5,1001)
param = dict(Efermi=Efermi,tetra=True)

calculators = {}
# the keys of this dictionary may be any name that you whant to appear in the file names
calculators["Ohmic"] =  calc.Ohmic(**param)
calculators["dos"] =    calc.DOS(**param)
calculators["cumdos"] = calc.CumDOS(**param)
calculators["ahc"]=     calc.AHC(**param,**noexter)
calculators["MR_11_fsurf"]=calc.MagnetoResistanceBerryFermiSurface(**param,**noexter)
calculators["MR_11_zeeman_fsurf"] = calc.MagnetoResistanceZeemannFermiSurface(**param,kwargs_formula={'external_terms' : False,'spin':True, 'orb':True})

for length in 10,:
    grid=wberri.Grid(system,length=length)  # the spacing of the k-grid will be approximately 2pi/length
    result = wberri.run(system,
            grid=grid,
            calculators = calculators,
            parallel=parallel,
            adpt_num_iter=0,
            fout_name='Chiral',
            suffix = f"length={length}",
            restart=False,
            )



