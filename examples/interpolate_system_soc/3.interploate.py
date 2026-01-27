#!/usr/bin/env python3

from matplotlib import pyplot
import wannierberri as wberri
import numpy as np

from wannierberri.parallel import Parallel

parallel = Parallel(num_cpus=16)

system1 = wberri.system.system_soc.SystemSOC.from_npz("system_soc_m2.2")
system1.set_soc_axis(theta=0.0, phi=0.0, alpha_soc=1.0)

system0  = wberri.system.system_soc.SystemSOC.from_npz("system_soc_m0.0")
system0.set_soc_axis(theta=0.0, phi=0.0, alpha_soc=1.0)

interpolator = wberri.system.interpolate.SystemInterpolatorSOC(system0, system1)

tabulators = {"Energy": wberri.calculators.tabulate.Energy(),
              "berry": wberri.calculators.tabulate.BerryCurvature(),
              "spin": wberri.calculators.tabulate.Spin(),
             }

tab_all_path = wberri.calculators.TabulatorAll(
    tabulators,
    ibands=np.arange(0, 18),
    mode="path"
)

path = wberri.Path(system0,
                 nodes=[
                     [0.0000, 0.0000, 0.0000],  # G
                     [0.500, -0.5000, -0.5000],  # H
                     [0.7500, 0.2500, -0.2500],  # P
                     [0.5000, 0.0000, -0.5000],  # N
                     [0.0000, 0.0000, 0.000]],  # G
                 labels=["G", "H", "P", "N", "G"],
    length=1000)   # length [ Ang] ~= 2*pi/dk

Efermi=9.2

all_alpha =  [-1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2, ]
# all_alpha = np.arange(0, 2.01, 0.5)
n_alpha = len(all_alpha)
nrows=3
ncols=int(np.ceil(n_alpha/nrows))
param_fig = dict(sharex=True, sharey=True, nrows=nrows, ncols=ncols, figsize=(20,20))
fig_spin, axes_spin = pyplot.subplots(**param_fig)
fig_berry, axes_berry = pyplot.subplots(**param_fig)
for i, alpha in enumerate(all_alpha):
    ix, ij = divmod(i, axes_spin.shape[1])
    ax_spin = axes_spin[ix, ij]
    ax_berry = axes_berry[ix, ij]
    print (f"Interpolating for alpha={alpha}")
    if alpha < 0.01:
        new_system = system0
        title = "System with m=0.0"
    elif alpha > 1.01:
        new_system = system1
        title = "System with m=2.2"
    else:
        new_system = interpolator.interpolate(alpha)
        title = f"Interpolated system:\n {alpha:.2} of m=0.0 \n {1-alpha:.2f} of m=2.2"
    # print(f"New system : {new_system} of type {type(new_system)}, system_up: {new_system.system_up}, system_down: {new_system.system_down}")
    # # new_system.set_soc_axis(theta=0.0, phi=0.0, alpha_soc=1.0)
    # print(f"New system : {new_system} of type {type(new_system)}, system_up: {new_system.system_up}, system_down: {new_system.system_down}")
    result = wberri.run(new_system,
                  grid=path,
                    parallel=parallel,
                  calculators={"tab": tab_all_path},
                  print_Kpoints=False)

    parameters_plot = dict(Eshift=Efermi,
              Emin=-5, Emax=5,
              iband=None,
              mode="color",
              fatfactor=0.5,
              cut_k=False,
              close_fig=False,
              show_fig=False,
    )
    result.results["tab"].plot_path_fat(path,
              quantity="spin",
              component="z",
                axes=ax_spin,
        **parameters_plot
    )
    result.results["tab"].plot_path_fat(path,
              quantity="berry",
              component="z",
            axes=ax_berry,
        **parameters_plot
    )

    ax_spin.set_ylabel("Spin (z)")
    ax_berry.set_ylabel("Berry curvature (z)")
    ax_spin.set_title(title)
    ax_berry.set_title(title)
    ax_spin.set_ylim([-5, 4])
    ax_berry.set_ylim([-5, 4])


fig_spin.savefig("Fe_spin.png")
fig_berry.savefig("Fe_berry.png")
