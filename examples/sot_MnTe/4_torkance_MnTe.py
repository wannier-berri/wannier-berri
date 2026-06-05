import numpy as np
import wannierberri as wberri
from pathlib import Path
from gpaw import GPAW

from wannierberri.system import System_R
from wannierberri.system.system_soc import SystemSOC
from wannierberri.w90files.soc import SOC
from wannierberri.w90files.chk import CheckPoint as CHK
from wannierberri.calculators.static import TorkanceOdd, TorkanceEven


def run_mnte_torkance():
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
            positions=[[0, 0, 0], [0, 0, 1 / 2], [1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]],
            typat=[1, 1, 2, 2],
            magmoms_on_axis=[4.7, -4.7, 0, 0]
        )

        print("Extracting SOC matrix from GPAW wavefunctions...")
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

    # 2. Setup grid and calculate Torkance
    print("\nSetting up grid and calculating Torkance...")
    grid = wberri.Grid(system_soc, length=20)

    # Center grid around the computed Fermi level (6.61 eV)
    Efermi = np.linspace(4.6, 8.6, 5)

    torkance_even = TorkanceEven(Efermi=Efermi, Emin=4.6, Emax=8.6)
    torkance_odd = TorkanceOdd(Efermi=Efermi, Emin=4.6, Emax=8.6)

    results = wberri.run(
        system_soc,
        grid=grid,
        calculators={"torkance_even": torkance_even, "torkance_odd": torkance_odd},
        use_irred_kpt=False,
        symmetrize=False
    )

    print("\n-------------------------------------------")
    print("Torkance Results for alpha-MnTe:")
    print("-------------------------------------------\n")

    # np.set_printoptions(precision=4, suppress=True, linewidth=120)
    for part in "even", "odd":
        res = results.results[f"torkance_{part}"]
        E_f = res.Energies[0]
        torkance_tensor = res.data

        print(f"--- {part.capitalize()} Torkance ---")
        for i, e in enumerate(E_f):
            print(f"Fermi Energy: {e:.4f} eV")
            print(torkance_tensor[i])
            print()


if __name__ == "__main__":
    run_mnte_torkance()
