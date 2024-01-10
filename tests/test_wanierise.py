import wannierberri as wberri
import numpy as np


def test_disentangle(system_Fe_W90_disentangle, system_Fe_W90_proj_ws):
    """
    tests that the two systems give similar eigenvalues aalong the path
    """
    tabulators = {"Energy": wberri.calculators.tabulate.Energy(),
                  }

    tab_all_path = wberri.calculators.TabulatorAll(
        tabulators,
        ibands=np.arange(0, 18),
        mode="path"
    )

    # all kpoints given in reduced coordinates
    path = wberri.Path(system_Fe_W90_disentangle,
                       k_nodes=[
                           [0.0000, 0.0000, 0.0000],  # G
                           [0.500, -0.5000, -0.5000],  # H
                           [0.7500, 0.2500, -0.2500],  # P
                           [0.5000, 0.0000, -0.5000],  # N
                           [0.0000, 0.0000, 0.000]
                       ],  # G
                       labels=["G", "H", "P", "N", "G"],
                       length=200)  # length [ Ang] ~= 2*pi/dk

    energies = []
    for system in system_Fe_W90_disentangle, system_Fe_W90_proj_ws:
        print("Wannier Centers\n", np.round(system.wannier_centers_reduced, decimals=4))
        result = wberri.run(system,
                        grid=path,
                        calculators={"tabulate": tab_all_path},
                        print_Kpoints=False)
        energies.append(result.results["tabulate"].get_data(quantity="Energy", iband=np.arange(0, 18)))

    select = energies[1] < 18
    diff = abs(energies[1][select] - energies[0][select])
    # the precidsion is not very high here, although the two codes are assumed to do the same. Not sure why..
    d, acc = np.max(diff), 0.2
    assert d < acc, f"the interpolated bands differ from w90 interpolation by max {d}>{acc}"
    d, acc = np.mean(diff) / diff.size, 0.05
    assert d < acc, f"the interpolated bands on average differ from w90 interpolation by {d}>{acc}"
