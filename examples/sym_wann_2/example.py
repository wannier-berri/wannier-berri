"""
This example symmetrizes a system with the same spacegroup using two different methods:
1. Using the old sym_wann.py module 
2. Using the SymmetrizerSAWF class (sym_wann_2)
"""

import shutil
from irrep.spacegroup import SpaceGroup
from matplotlib import pyplot
import numpy as np
import wannierberri as wberri
from wannierberri.system import System_w90
from wannierberri.symmetry.sawf import SymmetrizerSAWF
from wannierberri.wannierise.projections import Projection
from time import time

path_data = "../../tests/data/Fe_sym_Wannier90/Fe_sym."
for ext in ["amn", "mmn", "eig", "win", "chk"]:
  shutil.copyfile(path_data+ext, "./Fe_sym."+ext)

system_1 = System_w90("Fe_sym", berry = True, silent=True)

t10 = time()
system_1.symmetrize(proj=["Fe:sp3d2;t2g"], atom_name=["Fe"], positions=[[0,0,0]], magmom=[[0,0,1]], soc=True)
t11=time()
system_2 = System_w90("Fe_sym", berry = True, silent=True)
# 
t20 = time()
symmetrizer = SymmetrizerSAWF()
spacegroup = SpaceGroup( cell = ( system_2.real_lattice, [[0,0,0]], [0] ) , magmom = [[0,0,1]] , include_TR=True)
symmetrizer.set_spacegroup(spacegroup)
# spacegroup.show()
t21 = time()
t221 = time()
proj1 = Projection( position_num=np.array([0,0,0]),orbital = "sp3d2" , spacegroup=spacegroup)
proj2 = Projection( position_num=np.array([0,0,0]),orbital = "t2g" , spacegroup=spacegroup)
t222 = time()
# proj2 = Projection( (np.array([0,0,0]),"t2g" , spacegroup=spacegroup)
symmetrizer.set_spacegroup(spacegroup)
t223 = time()
symmetrizer.set_D_wann_from_projections( projections=[proj1,proj2])
t22 = time()
system_2.symmetrize2(symmetrizer)
t23 = time()

tabulators = { "Energy": wberri.calculators.tabulate.Energy(),
               "berry_int" : wberri.calculators.tabulate.BerryCurvature(kwargs_formula={"external_terms":False, "internal_terms":True}),
               "berry_ext" : wberri.calculators.tabulate.BerryCurvature(kwargs_formula={"external_terms":True, "internal_terms":False}),
            #    "spin" : wberri.calculators.tabulate.Spin(),
             }

tab_all_path = wberri.calculators.TabulatorAll(
                    tabulators,
                    ibands = np.arange(0,18),
                    mode = "path"
                        )

path=wberri.Path(system_1,
                 k_nodes=[
        [0.0000, 0.0000, 0.0000 ],   #  G
        [0.500 ,-0.5000, -0.5000],   #  H
        [0.7500, 0.2500, -0.2500],   #  P
        [0.5000, 0.0000, -0.5000],   #  N
        [0.0000, 0.0000, 0.000  ] ] , #  G
                 labels=["G","H","P","N","G"],
                 length=200 )   # length [ Ang] ~= 2*pi/dk


n_alpha = 2
fig_berry_int, axes_berry_int = pyplot.subplots(1, n_alpha, figsize=(4*n_alpha, 5))
fig_berry_ext, axes_berry_ext = pyplot.subplots(1, n_alpha, figsize=(4*n_alpha, 5))

if True:
  for i,system in enumerate([system_1, system_2]):

    result=wberri.run(system,
                  grid=path,
                  calculators = {"tab": tab_all_path},
                  print_Kpoints = False)
    
    parameters_plot = dict(Eshift=18,
              Emin=-5,  Emax=5,
              iband=None,
              mode="fatband",
              fatfactor=20,
              cut_k=False,
              close_fig=False,
              show_fig=False,
    )

    result.results["tab"].plot_path_fat( path,
              quantity="berry_int",
              component="z",
            axes=axes_berry_int[i],
            **parameters_plot
          )
    
    axes_berry_int[i].set_title(f"{i+1}")

    result.results["tab"].plot_path_fat( path,
                quantity="berry_ext",
                component="z",
                axes=axes_berry_ext[i],
                **parameters_plot
            )
    axes_berry_ext[i].set_title(f"{i+1}")

fig_berry_ext.savefig("Fe_berry_ext.png")

fig_berry_int.savefig("Fe_berry_int.png")

print ("Wannier centers system_1")
print (system_1.wannier_centers_cart)
print ("Wannier centers system_2")
print (system_2.wannier_centers_cart)
setRvec1 = set([tuple(x) for x in system_1.iRvec])
setRvec2 = set([tuple(x) for x in system_2.iRvec])
    
print (f"system_1.iRvec = \n{system_1.iRvec}")
print (f"system_2.iRvec = \n{system_2.iRvec}")

print (f"system_1.nRvec = \n{system_1.nRvec}")
print (f"system_2.nRvec = \n{system_2.nRvec}")


for R in system_1.iRvec:
    iR = tuple(R)
    iR0_1 = system_1.index_R[iR]
    iR0_2 = system_2.index_R[iR]

    H01 = system_1.Ham_R[:,:,iR0_1]
    H02 = system_2.Ham_R[:,:,iR0_2]
    diff = abs(H01-H02).max()

    if diff > 1e-10:
      print(f"iR0_1 = {iR0_1}, iR0_2 = {iR0_2}")
      print(f"H0_diff = \n{abs(diff).max()}")
      print(f"cRvec_diff = \n{system_1.cRvec[iR0_1]-system_2.cRvec[iR0_2]}")

print (f"time symmetrize system_1 = {t11-t10}")
print (f"time symmetrize system_2 = {t23-t20}, where time to create spacegroup = {t21-t20}, time to create symmetrizer = {t22-t21} and time to symmetrize = {t23-t22}")
print (f"{t221-t21}, {t222-t221}, {t223-t222}, {t22-t223}")