import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wberri
from pathlib import Path
from wannierberri.system.system_soc import SystemSOC
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.formula.formula import Matrix_ln
from wannierberri.calculators.tabulate import Tabulator

"""
Example use of wannierberri to compute the spin-orbit torque (SOT) fat bands in Fe
using precomputed data from GPAW.
"""

def run_fe_sot_example():
    HERE = Path(__file__).parent.resolve()
    DATA_DIR = HERE / "../../tests/data/Fe_system_soc"
    system_soc = SystemSOC.from_npz(str(DATA_DIR / "system_soc"))
    
    # Set the torque operators for the SOT calculation.
    system_soc.set_torque_operators_R(theta=0, phi=0) 

    # Set up the SOT formula to be used in the Tabulator
    class SOTformula(Matrix_ln):
        def __init__(self, data_K, **kwargs):
            # 1. Fetch the pre-built Matrix_ln object from covariant()
            T_obj = data_K.covariant('SOT')
            
            # 2. Extract the raw numpy array from the object
            raw_matrix = T_obj.matrix
            
            # 3. Pass the raw array to properly initialize this subclass
            super().__init__(matrix=raw_matrix, **kwargs)

    # 2. Path setup & Evaluation
    pts = {"G": [0,0,0], "H": [0.5,-0.5,-0.5], "P": [0.75,0.25,-0.25], "N": [0.5,0.0,-0.5]}
    path = wberri.grid.Path.from_nodes(
        system=system_soc, 
        nodes=[pts[p] for p in "GHPNGP"],
        labels=list("GHPNGP"), 
        length=1000
    )
    
    # setup tabulator on the fly
    bands_sot = evaluate_k_path(system_soc, path=path, tabulators={"SOT": Tabulator(SOTformula)})

    # 1. Initialize figure
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 12))

    # 2. Plot Cartesian components (0, 1, 2)
    for i, comp in enumerate(["x", "y", "z"]):
        bands_sot.plot_path_fat(path=path, quantity="SOT", component=comp, 
                                axes=axs[i], fig=fig, 
                                show_fig=False, close_fig=False)
        axs[i].set_ylabel(f"$T_{comp}$")

    # 3. Plot Total Amplitude squared using the "color" mode
    # component="sq" triggers np.linalg.norm(data)**2 internally
    bands_sot.plot_path_fat(path=path, quantity="SOT", component="sq", 
                            mode="color", axes=axs[3], fig=fig,
                            show_fig=False, close_fig=False)

    axs[3].set_ylabel("$|T|^2$")

    plt.tight_layout()
    fig.savefig(HERE / "sot_fat_bands_Fe.png", dpi=300, bbox_inches="tight")
    
if __name__ == "__main__":
    run_fe_sot_example()