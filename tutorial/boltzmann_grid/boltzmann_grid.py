#!/usr/bin/env python3
import pickle
import wannierberri as wberri
import wannierberri.calculators.tabulate as caltab
import numpy as np

model = wberri.models.Chiral(delta=2, hop1=1, hop2=1./3,  phi=np.pi/10, hopz=0.2)
system = wberri.System_PythTB(model,use_wcc_phase = True)
system.set_spin([-1,1],axis=[1,-1,0])
system.set_symmetry(["C3z"])

grid=wberri.Grid(system,length=30)
#parallel=wberri.Parallel() # serial execution
parallel=wberri.Parallel(method="ray",num_cpus=4)
noexter = {"kwargs_formula":{'external_terms' : False}}


if False :
    result = wberri.run(system,
            grid=grid,
            calculators = {
#                "ahc":wberri.calculators.static.AHC(Efermi=Efermi,tetra=False),
                 "tabulate":wberri.calculators.TabulatorAll({
                            "Energy":caltab.Energy(),
                            "berry":caltab.BerryCurvature(**noexter),
                            "velocity":caltab.Velocity(),
                            "morb":caltab.morb(**noexter),
                            "dermorb":caltab.Der_morb(**noexter),
                            "spin" : caltab.Spin(),
                            "derspin" : caltab.Der_Spin()
                                  }, 
#                                       ibands = np.arange(4,10)
                            )
#                 "opt_conductivity" : wberri.calculators.dynamic.OpticalConductivity(Efermi=Efermi,omega=omega),
#                 "shc_ryoo" : wberri.calculators.dynamic.SHC(Efermi=Efermi,omega=omega),
#                  "SHC": wberri.fermiocean_dynamic.SHC(Efermi=Efermi,omega=omega,SHC_type="qiao")
                          }, 
            parallel=parallel,
            adpt_num_iter=0,
            fout_name='Chiral',
            suffix = "",
            restart=False,
            )

    with open("tab_result.pickle","wb") as f:
        pickle.dump(result,f)




if True:

    with open("tab_result.pickle","rb") as f:
        result = pickle.load(f)
        tab_result = result.results['tabulate']
        bgrid = wberri.boltzmann_grid.Boltzmann_grid(tab_result,tau = 1e-9,mu=1.0,kBT=0.5)


    for Ez in [] : #0.01,0.1,1.0,2.0:
        E = [0,0,Ez]
        c = bgrid.current(E=E,n_iter = 5)
        print (f"E={E} V/m  c = {c}  A/m^2 , sigma_zz = {c[2]/E[2]/100:.5e} S/cm")

    for Ey in 0.01,0.1,1.0,2.0:
        E = [0,Ey,0]
        c = bgrid.current(E=E,n_iter = 5)
        print (f"E={E} V/m  c = {c}  A/m^2 , sigma_xy = {c[0]/E[1]/100:.5e} S/cm")

