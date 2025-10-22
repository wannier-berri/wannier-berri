import functools
from matplotlib import pyplot as plt
import numpy as np
from wannierberri.grid import Path
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.system.system_soc import SystemSOC
from wannierberri.parallel import Parallel, Serial

# parallel = Parallel(num_cpus=16)
parallel = Serial()

system_soc = SystemSOC.from_npz("system_soc")
theta=0
phi=0
system_soc.set_soc_axis(theta=theta, phi=phi, units="degrees")

path = Path(system_soc,
            nodes=[
                [0.0, 0.0, 0.0],
                [0.5, -0.5, -0.5],
                [0.75, 0.25, -0.25],
                [0.5, 0.0, -0.5],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.75, 0.25, -0.25],
                [0.5, 0, 0]],
            labels=["G ", "H ", "P ", "N ", "G ", "H ", "N ", "G ", "P ", "N"],
            length=1000)   # length [ Ang] ~= 2*pi/dk


evaluate_k_path_loc = functools.partial(
    evaluate_k_path,
    path=path,
    parallel=parallel,
)


bands_soc = evaluate_k_path_loc(system_soc, quantities=["spin"])

EF = 9.22085

fig, axes = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(30, 50))

bands_soc.plot_path_fat(path=path,
                       Eshift=EF,
                       quantity="spin",
                       component="z",
                       mode="color",
                       label=f"soc_nscf, Sz, th={theta}, phi={phi}", 
                       axes=axes[1],
                       fatmax=4,
                        linecolor="orange",
                        close_fig=False,
                        show_fig=False,
                        kwargs_line=dict(linestyle='-', lw=0.0),
)


bands_up = evaluate_k_path_loc(system_soc.system_up)
bands_dw = evaluate_k_path_loc(system_soc.system_down)

bands_up.plot_path_fat(path=path,
                       Eshift=EF,
                       axes=axes[0],
                       label="spin-up",
                       linecolor="red",
                       close_fig=False,
                       show_fig=False,
)

bands_dw.plot_path_fat(path=path,
                       label="spin-dw",
                       axes=axes[0],
                       Eshift=EF,
                       linecolor="blue",
                       close_fig=False,
                       show_fig=False,)





# plt.ylim(-0.6, 0.6)
plt.ylim(-1.7, 1.7)
plt.ylim(-15,40)
plt.savefig("bands.png")
