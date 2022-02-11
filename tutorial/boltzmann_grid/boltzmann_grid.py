#!/usr/bin/env python3
import pickle
import wannierberri as wberri
import wannierberri.calculators.tabulate as caltab
import numpy as np
from scipy.constants import elementary_charge,  Boltzmann

model = wberri.models.Chiral(delta=2, hop1=1, hop2=1./3,  phi=np.pi/10, hopz=0.2)
system = wberri.System_PythTB(model,use_wcc_phase = True)
system.set_spin([-1,1],axis=[0,0,1])
system.set_symmetry(["C3z"])

#parallel=wberri.Parallel() # serial execution
parallel=wberri.Parallel(method="ray",num_cpus=8)
noexter = {"kwargs_formula":{'external_terms' : False}}


if False :
    for length in 25,50,100,150,200:
        grid=wberri.Grid(system,length=length)
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
                            do_write_frmsf = False
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

        with open(f"tab_result_{length}.pickle","wb") as f:
            pickle.dump(result,f)



if False:
    for length in 25,50:
        grid=wberri.Grid(system,length=length)
        wberri.integrate(system,
            grid=grid,
            Efermi=np.linspace(-8,8,401), 
            smearEf=0.5 * (elementary_charge/Boltzmann),
            quantities=["Hall_classic","Hall_classic_fsurf"],  #["dos","conductivity_ohmic","ahc"],
            parallel=parallel,
            adpt_num_iter=0,
            parameters = {'tetra':True,'external_terms':False},
            parameters_K = {'fftlib':'fftw'}, #default.  alternative  option - 'numpy'
            fout_name='Chiral',
            suffix = f"length={length}",
            restart=False,
            )
    

if True:

    with open("tab_result_50.pickle","rb") as f:
        result = pickle.load(f)
        tab_result = result.results['tabulate']
        bgrid = wberri.boltzmann_grid.Boltzmann_grid(tab_result,tau = 1e-12,mu=2.0,kBT=0.5,
                    anomalous_velocity = False,
                    lorentz_force = True , 
                    phase_space_kdot = False,
                    last_term_kdot = False,
                    last_term_rdot = False,
                    zeeman_orb  = False,
                    zeeman_spin = False
            )


    # test linear Ohmic conductivity
    for Ez in []:#0.01,0.1,1.0,2.0:
        E = [0,0,Ez]
        c = bgrid.current(E=E,n_iter = 5)
        print (f"E={E} V/m, j = {c}  A/m^2 , sigma_zz = {c[2]/E[2]/100:.5e} S/cm")

    # test AHC
    for Ey in []: # 0,0.01,0.1,1.0,2.0:
        E = [0,Ey,0]
        print (f"E={E} V/m, j = {c}  A/m^2 , sigma_xy = {c[0]/E[1]/100:.5e} S/cm")

    # test classical Hall effect
    for Ey in 0.,0.01,0.1,1.0,2.0:
        E = [0,Ey,0]
        for Bz in 0.,0.01,0.1,1.0,2.0:
            B = [0,0,Bz]
            c = bgrid.current(E=E,B=B,n_iter = 5)
            print (f"E={E} V/m , B={B} T , j = {c}  A/m^2 , sigma_xy,z = {c[0]/Ey/Bz/100:.5e} S/cm/T")


