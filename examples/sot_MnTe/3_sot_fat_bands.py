import matplotlib.pyplot as plt
import wannierberri as wberri
from pathlib import Path

from wannierberri.system.system_soc import SystemSOC
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.formula.formula import Matrix_ln
from wannierberri.calculators.tabulate import Tabulator

Emin = 6
Emax = 7.5

def run_mnte_fat_bands():
    HERE = Path(__file__).parent.resolve()
    npz_path = HERE / "MnTe_system_soc"

    print("Loading existing SystemSOC from disk...")
    system_soc = SystemSOC.from_npz(str(npz_path))
    system_soc.set_soc_axis(theta=0, phi=0, alpha_soc=1.0, units="degrees", torque=True)

    class SOTformula(Matrix_ln):
        def __init__(self, data_K, **kwargs):
            T_obj = data_K.covariant('SOT')
            super().__init__(matrix=T_obj.matrix, **kwargs)

    pts = {
        "G": [0, 0, 0],
        "M": [0.5, 0, 0],
        "K": [1 / 3, 1 / 3, 0],
        "A": [0, 0, 0.5],
        "L": [0.5, 0, 0.5],
        "H": [1 / 3, 1 / 3, 0.5]
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
        data = bands_sot.get_data(quantity="SOT", component=comp)
        axs[i].set_title(f"SOT component {comp.upper()} min = {data.min():.2e}, max = {data.max():.2e}")   
        axs[i].set_ylabel(f"$T_{comp}$")
        axs[i].set_ylim(Emin, Emax)

    bands_sot.plot_path_fat(path=path, quantity="SOT", component="sq",
                            mode="color", axes=axs[3], fig=fig,
                            show_fig=False, close_fig=False)
    axs[3].set_ylabel("$|T|^2$")
    axs[3].set_ylim(Emin, Emax)

    plt.tight_layout()
    fig.savefig(HERE / "sot_fat_bands_MnTe.png", dpi=300, bbox_inches="tight")
    print("Done! Saved plot to sot_fat_bands_MnTe.png")


if __name__ == "__main__":
    run_mnte_fat_bands()
