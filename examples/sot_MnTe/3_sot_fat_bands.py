import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wberri
from pathlib import Path
from gpaw import GPAW

from wannierberri.system import System_R
from wannierberri.system.system_soc import SystemSOC
from wannierberri.w90files.soc import SOC
from wannierberri.w90files.chk import CheckPoint as CHK
from wannierberri.symmetry.sawf import SymmetrizerSAWF as SAWF
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.formula.formula import Matrix_ln
from wannierberri.calculators.tabulate import Tabulator

def run_mnte_fat_bands():
    HERE = Path(__file__).parent.resolve()
    npz_path = HERE / "MnTe_system_soc"
    
    # 1. Load or Build the SystemSOC object
    if npz_path.exists():
        print("Loading existing SystemSOC from disk...")
        system_soc = SystemSOC.from_npz(str(npz_path))
    else:
        print("Constructing SystemSOC from collinear components...")
        system_up = System_R.from_npz(str(HERE / "system_up"))
        system_dw = System_R.from_npz(str(HERE / "system_dw"))

        system_soc = SystemSOC(system_up=system_up, system_down=system_dw, silent=False)

        # Mandatory for AFM symmetry detection (Magnetic Point Group)
        system_soc.set_cell(
            positions=[[0, 0, 0], [0, 0, 1/2], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]],
            typat=[1, 1, 2, 2],
            magmoms_on_axis=[4.7, -4.7, 0, 0]
        )

        print("Extracting SOC matrix from GPAW wavefunctions (this will take a minute)...")
        calc_nscf = GPAW(str(HERE / 'mnte-nscf.gpw'), txt=None)
        soc = SOC.from_gpaw(calc_nscf)

        print("Loading Wannier90 checkpoints (U matrices)...")
        chk_up = CHK.from_npz(str(HERE / "system_up.chk.npz"))
        chk_down = CHK.from_npz(str(HERE / "system_dw.chk.npz"))

        print("Integrating SOC into Wannier basis...")
        system_soc.set_soc_R(
            soc=soc,
            chk_up=chk_up,
            chk_down=chk_down,
            theta=0, phi=0, alpha_soc=1.0
        )
        
        print(f"Saving compiled SystemSOC to {npz_path}...")
        system_soc.to_npz(str(npz_path))

    # 2. Evaluate SOT Fat Bands
    system_soc.set_torque_operators_R(theta=0, phi=0) 

    class SOTformula(Matrix_ln):
        def __init__(self, data_K, **kwargs):
            T_obj = data_K.covariant('SOT')
            super().__init__(matrix=T_obj.matrix, **kwargs)

    pts = {
        "G": [0, 0, 0], 
        "M": [0.5, 0, 0], 
        "K": [1/3, 1/3, 0], 
        "A": [0, 0, 0.5],
        "L": [0.5, 0, 0.5],
        "H": [1/3, 1/3, 0.5]
    }
    
    path_labels = "GMKGALHA"
    path = wberri.grid.Path.from_nodes(
        system=system_soc, 
        nodes=[pts[p] for p in path_labels],
        labels=list(path_labels), 
        length=1000
    )
    
    print("Evaluating k-path...")
    bands_sot = evaluate_k_path(system_soc, path=path, tabulators={"SOT": Tabulator(SOTformula)})

    print("Plotting results...")
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 12))

    for i, comp in enumerate(["x", "y", "z"]):
        bands_sot.plot_path_fat(path=path, quantity="SOT", component=comp, 
                                axes=axs[i], fig=fig, 
                                show_fig=False, close_fig=False)
        axs[i].set_ylabel(f"$T_{comp}$")

    bands_sot.plot_path_fat(path=path, quantity="SOT", component="sq", 
                            mode="color", axes=axs[3], fig=fig,
                            show_fig=False, close_fig=False)
    axs[3].set_ylabel("$|T|^2$")

    plt.tight_layout()
    fig.savefig(HERE / "sot_fat_bands_MnTe.png", dpi=300, bbox_inches="tight")
    print("Done! Saved plot to sot_fat_bands_MnTe.png")
    
if __name__ == "__main__":
    run_mnte_fat_bands()