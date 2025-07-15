from matplotlib import pyplot as plt
from wannierberri.system.system_R import System_R
from wannierberri.grid import Path
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.w90files.soc import SOC
from wannierberri.system.system_soc import SystemSOC
from wannierberri.w90files.chk import CheckPoint as CHK

system_dw = System_R().load_npz("system_dw")
system_up = System_R().load_npz("system_up")
system_spinor = System_R().load_npz("system_spinor")

soc = SOC.from_gpaw("Fe-nscf.gpw")
chk_up = CHK.from_npz("Fe-spin-0.chk.npz")
chk_dw = CHK.from_npz("Fe-spin-1.chk.npz")
system_soc = SystemSOC(system_up=system_up, system_down=system_dw,)
system_soc.set_soc_R(soc, chk_up=chk_up, chk_down=chk_dw)


path = Path(system_dw,
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
            length=300)   # length [ Ang] ~= 2*pi/dk






bands_up = evaluate_k_path(system_up, path=path,)
bands_dw = evaluate_k_path(system_dw, path=path,)
bands_spinor = evaluate_k_path(system_spinor, path=path,)
bands_soc = evaluate_k_path(system_soc, path=path,)

EF = 9.22085

fig, axes = plt.subplots(2, 1, sharey=True, sharex=True)

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

bands_spinor.plot_path_fat(path=path,
                       label="spinor",
                       Eshift=EF,
                       axes=axes[0],
                          linecolor="black",
                          kwargs_line=dict(linestyle='--', lw=1),
                          close_fig=False,
                          show_fig=False,)

bands_spinor.plot_path_fat(path=path,
                       label="spinor",
                       Eshift=EF,
                       axes=axes[1],
                          linecolor="black",
                          kwargs_line=dict(linestyle='-', lw=1),
                          close_fig=False,
                          show_fig=False,)

bands_soc.plot_path_fat(path=path,
                       Eshift=EF,
                       label="soc_nscf",
                       axes=axes[1],
                        linecolor="orange",
                        close_fig=False,
                        show_fig=False,
                        kwargs_line=dict(linestyle='--', lw=1),
)


plt.ylim(-0.6, 0.6)
# plt.ylim(-1.7,1.7)
plt.savefig("bands.pdf")
