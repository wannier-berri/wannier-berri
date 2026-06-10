import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wberri
from pathlib import Path
from gpaw import GPAW

from wannierberri.system import System_R
from wannierberri.system.system_soc import SystemSOC
from wannierberri.w90files.soc import SOC
from wannierberri.w90files.chk import CheckPoint as CHK
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.formula.formula import Matrix_ln
from wannierberri.calculators.tabulate import Tabulator

def run_fe_sot_fat_bands():
    HERE = Path(__file__).parent.resolve()
    DATA_DIR = HERE / "../../tests/data/Fe_gpaw"

    print("Constructing SystemSOC from collinear components...")
    system_up = System_R.from_npz(str(DATA_DIR / "system_up"))
    system_dw = System_R.from_npz(str(DATA_DIR / "system_dw"))
    
    cell = dict(positions=[[0, 0, 0]], typat=[1], magmoms_on_axis=[1])
    system_soc = SystemSOC(system_up=system_up, system_down=system_dw, cell=cell, silent=True)

    print("Extracting SOC matrix from GPAW wavefunctions...")
    calc_nscf = GPAW(str(DATA_DIR / 'Fe-nscf.gpw'), txt=None)
    soc = SOC.from_gpaw(calc_nscf)

    # Explicit band selection matching the test system configuration
    selected_bands_up = np.load(DATA_DIR / "system_up.chk.npz")['selected_bands']
    selected_bands_down = np.load(DATA_DIR / "system_dw.chk.npz")['selected_bands']
    soc.select_bands(selected_bands_up=selected_bands_up, selected_bands_down=selected_bands_down)

    print("Loading Wannier90 checkpoints...")
    chk_up = CHK.from_npz(str(DATA_DIR / "system_up.chk.npz"))
    chk_down = CHK.from_npz(str(DATA_DIR / "system_dw.chk.npz"))

    print("Integrating SOC into Wannier basis...")
    system_soc.set_soc_R(soc=soc, chk_up=chk_up, chk_down=chk_down)

    # for (0,0) only Z component should be non-0. for (90,90), only Y, etc. an interesting, visually satisfying test.
    theta_deg, phi_deg = 90, 90
    system_soc.set_soc_axis(theta=theta_deg, phi=phi_deg, units="degrees", alpha_soc=1.0)
    
    print(f"Setting SOT operators for theta={theta_deg}, phi={phi_deg}...")
    system_soc.set_torque_operators_R(theta=theta_deg, phi=phi_deg, units="degrees")

    # Set up the SOT formula to be used in the Tabulator
    class SOTformula(Matrix_ln):
        def __init__(self, data_K, **kwargs):
            T_obj = data_K.covariant('SOT')
            super().__init__(matrix=T_obj.matrix, **kwargs)

    pts = {"G": [0, 0, 0], "H": [0.5, -0.5, -0.5], "P": [0.75, 0.25, -0.25], "N": [0.5, 0.0, -0.5]}
    path = wberri.grid.Path.from_nodes(
        system=system_soc, 
        nodes=[pts[p] for p in "GHPNGP"],
        labels=list("GHPNGP"), 
        length=1000
    )
    
    print("Evaluating k-path...")
    bands_sot = evaluate_k_path(system_soc, path=path, tabulators={"SOT": Tabulator(SOTformula)})

    print("Plotting fat bands...")
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 12))

    for i, comp in enumerate(["x", "y", "z"]):
        bands_sot.plot_path_fat(path=path, quantity="SOT", component=comp, 
                                axes=axs[i], fig=fig, 
                                show_fig=False, close_fig=False)
        axs[i].set_ylabel(f"$T_{comp}$")

    # Plot Total Amplitude squared
    bands_sot.plot_path_fat(path=path, quantity="SOT", component="sq", 
                            mode="color", axes=axs[3], fig=fig,
                            show_fig=False, close_fig=False)
    axs[3].set_ylabel("$|T|^2$")

    plt.tight_layout()
    fig.savefig(HERE / "sot_fat_bands_Fe.png", dpi=300, bbox_inches="tight")
    print("Execution successful. Saved plot to sot_fat_bands_Fe.png")
    
if __name__ == "__main__":
    run_fe_sot_fat_bands()