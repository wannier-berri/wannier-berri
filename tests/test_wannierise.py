from pytest import approx
import wannierberri as wberri
import numpy as np
import subprocess
from matplotlib import pyplot as plt
import os
import shutil
from common import TMP_DATA_DIR


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

    select = energies[1] < 15
    diff = abs(energies[1][select] - energies[0][select])
    # the precidsion is not very high here, although the two codes are assumed to do the same. Not sure why..
    d, acc = np.max(diff), 0.1
    assert d < acc, f"the interpolated bands differ from w90 interpolation by max {d}>{acc}"
    d, acc = np.mean(diff) / diff.size, 0.05
    assert d < acc, f"the interpolated bands on average differ from w90 interpolation by {d}>{acc}"


def test_disentangle_sym():
    systems = {}

    # Just fot Reference : run the Wannier90 with sitesym, but instead of frozen window use outer window
    # to exclude valence bands
    # os.system("wannier90.x diamond")
    tmp_data_rel = os.path.join(os.path.relpath(TMP_DATA_DIR), "diamond")
    prefix_data = os.path.join("./" "data", "diamond", "diamond")
    os.makedirs(tmp_data_rel, exist_ok=True)
    prefix = os.path.join(tmp_data_rel, "diamond")
    prefix_dis = os.path.join(tmp_data_rel, "diamond_disentangled")
    for ext in ["mmn", "amn", "dmn", "eig", "win"]:
        shutil.copy(prefix_data + "." + ext, prefix + "." + ext)
    print("prefix = ", prefix)
    subprocess.run(["wannier90.x", prefix])
    # record the system from the Wanier90 output ('diamond.chk' file)
    systems["w90"] = wberri.system.System_w90(seedname=prefix)
    # Read the data from the Wanier90 inputs
    w90data = wberri.system.Wannier90data(seedname=prefix)
    # Now disentangle with sitesym and frozen window (the part that is not implemented in Wanier90)
    w90data.disentangle(
        froz_min=-8,
        froz_max=20,
        num_iter=1000,
        conv_tol=1e-10,
        mix_ratio=1.0,
        print_progress_every=20,
        sitesym=True
    )
    wannier_centers = w90data.chk._wannier_centers
    wannier_spreads = w90data.chk._wannier_spreads
    wannier_spreads_mean = np.mean(wannier_spreads)
    assert wannier_spreads == approx(wannier_spreads_mean, abs=1e-9)
    assert wannier_spreads == approx(0.39864755, abs=1e-7)
    assert wannier_centers == approx(np.array([[0, 0, 0],
                                               [0, 0, 1],
                                               [0, 1, 0],
                                               [1, 0, 0]
                                               ]).dot(w90data.chk.real_lattice) / 2,
                                     abs=1e-6)

    systems["disentangled"] = wberri.system.System_w90(w90data=w90data)

    # If needed - perform maximal localization using Wannier90

    # first generate the reduced files - where num_bands is reduced to num_wann,
    # by taking the optimized subspace
    w90data_reduced = w90data.get_disentangled(files=["eig", "mmn", "amn", "dmn"])
    w90data_reduced.write(prefix_dis, files=["eig", "mmn", "amn", "dmn"])
    # Now write the diamond_disentangled.win file
    # first take the existing file
    win_file = wberri.system.w90_files.WIN(seedname=prefix)
    # and modify some parameters
    win_file["num_bands"] = win_file["num_wann"]
    win_file["dis_num_iter"] = 0
    win_file["num_iter"] = 1000
    del win_file["dis_froz_win"]
    del win_file["dis_froz_max"]
    win_file["site_symmetry"] = True
    win_file.write(prefix_dis)
    subprocess.run(["wannier90.x", prefix_dis])
    systems["mlwf"] = wberri.system.System_w90(seedname=prefix_dis)

    # Now calculate bandstructure for each of the systems
    # for creating a path any of the systems will do the job
    system0 = list(systems.values())[0]
    path = wberri.Path(system0, k_nodes=[[0, 0, 0], [1 / 2, 0, 0]], labels=['G', 'L'], length=100)
    tabulator = wberri.calculators.TabulatorAll(tabulators={}, mode='path')
    calculators = {'tabulate': tabulator}

    linecolors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    linestyles = ['-', '--', ':', '-.', ] * 4
    energies = {}
    for key, sys in systems.items():
        result = wberri.run(sys, grid=path, calculators=calculators)
        result.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False,
                                                linecolor=linecolors.pop(0), label=key,
                                                kwargs_line={"ls": linestyles.pop(0)})
        energies[key] = result.results['tabulate'].get_data(quantity="Energy", iband=np.arange(0, 4))
    plt.savefig(os.path.join(tmp_data_rel, "diamond_disentangled.png"))
    for k1 in energies:
        for k2 in energies:
            if k1 == k2:
                continue
            diff = abs(energies[k1] - energies[k2])
            # the precidsion is not very high here, although the two codes are assumed to do the same. Not sure why..
            d, acc = np.max(diff), 0.0005
            assert d < acc, f"the interpolated bands {k1} and {k2} differ from w90 interpolation by max {d}>{acc}"

    # One can see that results do not differ much. Also, the maximal localization does not have much effect.
