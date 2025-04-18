#!/usr/bin/env python3

import copy

from matplotlib import pyplot
import wannierberri as wberri
import numpy as np


seedname = "../../tests/data/Fe_sym_Wannier90/Fe_sym"
system0 = wberri.system.System_w90(seedname, berry=True, spin=True, )

system0.symmetrize(
    proj=['Fe:sp3d2;t2g'],
    atom_name=['Fe'],
    positions=np.array([[0, 0, 0]]),
    magmom=[[0., 0., -2.31]],
    soc=True,
)

system1 = copy.deepcopy(system0)

system0.symmetrize(
    proj=['Fe:sp3d2;t2g'],
    atom_name=['Fe'],
    positions=np.array([[0, 0, 0]]),
    magmom=[[0., 0., 0]],
    soc=True,
)

interpolator = wberri.system.interpolate.SystemInterpolator(system0, system1)

tabulators = { "Energy": wberri.calculators.tabulate.Energy(),
               "berry" : wberri.calculators.tabulate.BerryCurvature(),
               "spin" : wberri.calculators.tabulate.Spin(),
             }

tab_all_path = wberri.calculators.TabulatorAll(
                    tabulators,
                    ibands = np.arange(0,18),
                    mode = "path"
                        )

path=wberri.Path(system0,
                 k_nodes=[
        [0.0000, 0.0000, 0.0000 ],   #  G
        [0.500 ,-0.5000, -0.5000],   #  H
        [0.7500, 0.2500, -0.2500],   #  P
        [0.5000, 0.0000, -0.5000],   #  N
        [0.0000, 0.0000, 0.000  ] ] , #  G
                 labels=["G","H","P","N","G"],
                 length=200 )   # length [ Ang] ~= 2*pi/dk


n_alpha = 11
fig_spin, axes_spin = pyplot.subplots(1, n_alpha, figsize=(4*n_alpha, 5))
fig_berry, axes_berry = pyplot.subplots(1, n_alpha, figsize=(4*n_alpha, 5))
for i,alpha in enumerate(np.linspace(0, 1, n_alpha)):
    new_system = interpolator.interpolate(alpha)
    result=wberri.run(new_system,
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
              quantity="spin",
              component="z",
                axes=axes_spin[i],
                **parameters_plot
          )
    result.results["tab"].plot_path_fat( path,
              quantity="berry",
              component="z",
            axes=axes_berry[i],
            **parameters_plot
          )
    
    axes_spin[i].set_title(f"alpha={alpha}")
    axes_berry[i].set_title(f"alpha={alpha}")
    

fig_spin.savefig("Fe_spin.png")
fig_berry.savefig("Fe_berry.png")
    
    
