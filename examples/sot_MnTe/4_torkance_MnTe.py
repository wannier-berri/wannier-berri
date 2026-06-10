import numpy as np
import wannierberri as wberri
from pathlib import Path

from wannierberri.system.system_soc import SystemSOC
from wannierberri.calculators.static import TorkanceOdd, TorkanceEven


def run_mnte_torkance():
    HERE = Path(__file__).parent.resolve()
    npz_path = HERE / "MnTe_system_soc"

    print("Loading existing SystemSOC from disk...")
    system_soc = SystemSOC.from_npz(str(npz_path))
    system_soc.set_soc_axis(theta=0, phi=0, alpha_soc=1.0, units="degrees", torque=True)


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
