import functools
from matplotlib import pyplot as plt
import wannierberri as wberri
from wannierberri.grid import Path
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.system.system_soc import SystemSOC

wberri.ray_init()

system_soc = SystemSOC.from_npz("system_soc")
theta = 90
phi = 90
system_soc.set_soc_axis(theta=theta, phi=phi, units="degrees")

kz = 0.35 / system_soc.recip_lattice[2, 2]
path = Path(system=system_soc,
            nodes=[
                [2 / 3, -1 / 3, 0],
                [0, 0, 0],
                [-2 / 3, 1 / 3, 0],
                None,
                [-0.5, 0, kz],
                [0, 0, kz],
                [0.5, 0, kz],
            ],
            labels=[r"${\rm K}\leftarrow$",
                    r"$\Gamma$",
                    r"$\rightarrow{\rm K}$",
                    r"$\overline{\rm M}\leftarrow$",
                    r"$\overline{\Gamma}$",
                    r"$\rightarrow\overline{\rm M}$"],
            length=200)   # length [ Ang] ~= 2*pi/dk



evaluate_k_path_loc = functools.partial(
    evaluate_k_path,
    path=path,
)



bands_soc = evaluate_k_path_loc(system_soc, quantities=["spin"])

bands_soc = evaluate_k_path(system_soc, path=path, quantities=["spin"])

EF = 6.885145845031927

fig, axes = plt.subplots(4, 1, sharey=True, sharex=True, figsize=(15, 30))

for i in range(3):
    component = "xyz"[i]
    bands_soc.plot_path_fat(path=path,
                       Eshift=EF,
                       quantity="spin",
                       component=component,
                       mode="color",
                       label=f"soc_nscf, S{component}, th={theta}, phi={phi}",
                        axes=axes[1 + i],
                       fatmax=10,
                        linecolor="orange",
                        close_fig=False,
                        show_fig=False,
                        kwargs_line=dict(linestyle='-', lw=0.0),
    )


bands_up = evaluate_k_path(system_soc.system_up, path=path,)
bands_dw = evaluate_k_path(system_soc.system_down, path=path,)

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




bands_soc.plot_path_fat(path=path,
                       label="soc",
                       Eshift=EF,
                       axes=axes[0],
                        linecolor="black",
                        kwargs_line=dict(linestyle='--', lw=0.5),
                        close_fig=False,
                        show_fig=False,)




# plt.ylim(0,7)
plt.ylim(-8, 1)
plt.savefig("bands-MnTe-wannier.png")
