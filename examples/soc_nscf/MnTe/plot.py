from matplotlib import pyplot as plt
import numpy as np
from wannierberri.system.system_R import System_R
from wannierberri.grid import Path
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.w90files.soc import SOC
from wannierberri.system.system_soc import SystemSOC
from wannierberri.w90files.chk import CheckPoint as CHK
from wannierberri.parallel import Parallel, Serial

system_dw = System_R().load_npz("system_dw")
system_up = System_R().load_npz("system_up")

parallel=Parallel(num_cpus=16)
# _interlaced()


phi_deg = 90
theta_deg=90

soc = SOC.from_gpaw("mnte-nscf.gpw", calc_overlap=True)

# print(soc.overlap)

# exit()

chk_up = CHK.from_npz("system_up.chk.npz")
chk_dw = CHK.from_npz("system_dw.chk.npz")
system_soc = SystemSOC(system_up=system_up, system_down=system_dw,)
system_soc.set_soc_R(soc, chk_up=chk_up, chk_down=chk_dw,
                    #  alpha_soc=0.3,
                     theta=theta_deg/180*np.pi,
                     phi=phi_deg/180*np.pi)


# path = Path(system_dw,
#             nodes=[
#                 [1/3, 1/3, 0],
#                 [0, 0, 0],
#                 [0, 0 , 1/2],
#                 [1/3, 1/3, 0],
#                 [1/3,1/3,1/2],
#                 [0,1/2,0],
#                 [0,1/2,1/2]],
#             labels=["K", "G", "A", "K", "H ", "M", "L"],
#             length=1000)   # length [ Ang] ~= 2*pi/dk

kz = 0.35/system_dw.recip_lattice[2,2]
path = Path(system_dw,
            nodes=[
                [2/3, -1/3, 0],
                [0, 0, 0],
                [-2/3,1/3,0],
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


print(system_dw.recip_lattice)
print(np.dot(path.K_list, system_dw.recip_lattice))

# exit()
bands_soc = evaluate_k_path(system_soc, path=path, quantities=["spin"])

# EF = 9.22085
EF=6.885145845031927

fig, axes = plt.subplots(4, 1, sharey=True, sharex=True, figsize=(15,30))

for i in range(3):
    component = "xyz"[i]
    bands_soc.plot_path_fat(path=path,
                       Eshift=EF,
                       quantity="spin",
                       component=component,
                       mode="color",
                       label=f"soc_nscf, S{component}, th={theta_deg}, phi={phi_deg}",
                        axes=axes[1+i],
                       fatmax=10,
                        linecolor="orange",
                        close_fig=False,
                        show_fig=False,
                        kwargs_line=dict(linestyle='-', lw=0.0),
    )

# bands_spinor = evaluate_k_path(system_spinor, path=path, quantities=["spin"])
# bands_spinor.plot_path_fat(path=path,
#                        Eshift=EF,
#                        quantity="spin",
#                        component="z",
#                        mode="color",
#                        label="spinor",
#                        axes=axes[1],
#                        fatmax=4,
#                         linecolor="orange",
#                         close_fig=False,
#                         show_fig=False,
#                         kwargs_line=dict(linestyle='-', lw=0.0),
# )

# axes[1].set_colorbar()

bands_up = evaluate_k_path(system_up, path=path,)
bands_dw = evaluate_k_path(system_dw, path=path,)

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

# bands_spinor.plot_path_fat(path=path,
#                        label="spinor",
#                        Eshift=EF,
#                        axes=axes[2],
#                           linecolor="black",
#                           kwargs_line=dict(linestyle='--', lw=0.5),
#                           close_fig=False,
#                           show_fig=False,)




# plt.ylim(0,7)
plt.ylim(-8,1)
plt.savefig("bands.png")
