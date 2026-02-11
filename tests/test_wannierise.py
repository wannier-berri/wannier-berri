import copy
from pytest import approx
import pytest
import scipy
import wannierberri as wberri
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil

from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.system.system_w90 import System_w90

from .common import OUTPUT_DIR, ROOT_DIR, REF_DIR
from wannierberri.symmetry.sawf import SymmetrizerSAWF


# Switch to FAlse, to test purely serial execution
PARALLEL = True



@pytest.mark.parametrize("outer_window", [None,
                                          (-100, 100, "select_bands"),
                                          (-10, 40, "select_bands"),
                                          (-10, 22, "select_bands"),
                                          (-100, 100, "outer"),
                                          (-10, 40, "outer"),
                                          (-10, 22, "outer"),
                                          ])
def test_wannierise(outer_window):
    systems = {}

    cwd = os.getcwd()

    tmp_dir = os.path.join(OUTPUT_DIR, "diamond")

    # Check if the directory exists
    if os.path.exists(tmp_dir):
        # Remove the directory and all its contents
        shutil.rmtree(tmp_dir)
        print(f"Directory {tmp_dir} has been removed.")
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    data_dir = os.path.join(ROOT_DIR, "data", "diamond")
    prefix = "diamond"
    for ext in ["mmn", "amn", "eig", "win", "sawf.npz"]:
        shutil.copy(os.path.join(data_dir, prefix + "." + ext),
                    os.path.join(tmp_dir, prefix + "." + ext))
    for i in range(8):
        shutil.copy(os.path.join(data_dir, f"UNK{i + 1:05d}.1"),
                    os.path.join(tmp_dir, f"UNK{i + 1:05d}.1"))
    print("prefix = ", prefix)
    symmetrizer = SymmetrizerSAWF.from_npz(prefix + ".sawf.npz")

    # Read the data from the Wanier90 inputs
    w90data = wberri.w90files.Wannier90data.from_w90_files(seedname=prefix, readfiles=["amn", "mmn", "eig", "win", "unk"])
    w90data.set_symmetrizer(symmetrizer=symmetrizer)
    outer_min = -np.inf
    outer_max = np.inf
    if outer_window is not None:
        if outer_window[2] == "select_bands":
            w90data.select_bands(win_min=outer_window[0], win_max=outer_window[1])
        elif outer_window[2] == "outer":
            outer_min = outer_window[0]
            outer_max = outer_window[1]
    print(f"num_bands: eig:{w90data.eig.NB}, mmn:{w90data.mmn.NB}, amn:{w90data.amn.NB}")
    # Now wannierise with sitesym and frozen window (the part that is not implemented in Wanier90)
    w90data.wannierise(
        froz_min=-8,
        froz_max=20,
        outer_min=outer_min,
        outer_max=outer_max,
        num_iter=1000,
        conv_tol=1e-10,
        mix_ratio_z=0.8,
        mix_ratio_u=1,
        print_progress_every=20,
        sitesym=True,
        localise=True,
        parallel=PARALLEL,
    )
    wannier_centers = w90data.chk.wannier_centers_cart
    wannier_spreads = w90data.chk.wannier_spreads
    wannier_spreads_mean = np.mean(wannier_spreads)
    assert wannier_spreads == approx(wannier_spreads_mean, abs=1e-9)
    assert wannier_spreads == approx(0.39864755, abs=1e-5)
    assert wannier_centers == approx(np.array([[0, 0, 0],
                                            [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ]).dot(w90data.chk.real_lattice) / 2,
        abs=1e-6)
    sc_origin, sc_basis, WF, rho = w90data.plotWF(select_WF=[1, 2], reduce_r_points=[3, 9, 1])
    assert WF.shape == (2, 6, 2, 18)
    assert rho.shape == (2, 6, 2, 18)
    wf_file_name = "WF_12_red.npy"
    np.save(wf_file_name, WF)
    ref = np.load(os.path.join(REF_DIR, wf_file_name))
    assert WF == approx(ref)

    sc_origin, sc_basis, WF, rho = w90data.plotWF(select_WF=[1, 2], reduce_r_points=[1, 1, 1])
    lattice = w90data.chk.real_lattice
    assert sc_origin == approx(-lattice.sum(axis=0))
    assert sc_basis == approx(2 * lattice)
    assert WF.shape == (2, 18, 18, 18)
    assert rho.shape == (2, 18, 18, 18)
    assert rho.sum(axis=(1, 2, 3)) == approx(1)

    systems["wberri"] = wberri.system.System_w90(w90data=w90data)


    systems["wberri_symmetrized"] = copy.deepcopy(systems["wberri"])


    systems["wberri_symmetrized"].symmetrize2(symmetrizer=symmetrizer)

    # Now calculate bandstructure for each of the systems
    # for creating a path any of the systems will do the job
    system0 = list(systems.values())[0]
    path = wberri.Path.from_nodes(system0, nodes=[[0, 0, 0], [1 / 2, 0, 0]], labels=['G', 'L'], length=100)
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
    np.save(os.path.join(OUTPUT_DIR, "bands_wannierize.npy"), energies["wberri_symmetrized"])
    energies_ref = np.load(os.path.join(REF_DIR, "bands_wannierize.npy"))
    plt.savefig("bands.png")
    for k1 in energies:
        print(f"comparing {k1} and reference")
        diff = abs(energies[k1] - energies_ref)
        # the precidsion is not very high here, although the two codes are assumed to do the same. Not sure why..
        d, acc = np.max(diff), 0.0005
        assert d < acc, f"the interpolated bands {k1}  differ from reference by max {d}>{acc}"
    # One can see that results do not differ much. Also, the maximal localization does not have much effect.
    os.chdir(cwd)



spreads_Fe_spd_444_win50 = np.array([1.48094878, 1.44313864, 1.65060907, 1.61414571, 1.65053543,
            1.61438103, 1.65053543, 1.61438103, 0.4680118, 0.43165047,
            0.40912432, 0.39499786, 0.40912432, 0.39499786, 0.46803275,
            0.43165531, 0.40912598, 0.39500838])

spreads_Fe_spd_444_nowin = np.array([1.4903928, 1.45260483, 1.62040443, 1.58683755, 1.62042416,
       1.586817, 1.62042416, 1.586817, 0.4686682, 0.4321233,
       0.40920197, 0.39504559, 0.40920197, 0.39504559, 0.46868826,
       0.43212589, 0.40920177, 0.39505669])

spreads_Fe_spd_444_win50_outer = np.array([1.49368614, 1.43535665, 1.75385611, 1.70652668, 1.75385869,
       1.70653457, 1.75385869, 1.70653457, 0.46611582, 0.4308129,
       0.40815878, 0.39383537, 0.40815878, 0.39383537, 0.46613583,
       0.43081535, 0.40815827, 0.39384666])


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("include_TR", [True, False])
@pytest.mark.parametrize("use_window", [False, "select_bands", "outer"])
def test_sitesym_Fe(include_TR, use_window, parallel):
    path_data = os.path.join(ROOT_DIR, "data", "Fe-444-sitesym")
    w90data = wberri.w90files.Wannier90data.from_w90_files(seedname=path_data + "/Fe", readfiles=["amn", "eig", "mmn", "win"], read_npz=True)

    symmetrizer = SymmetrizerSAWF.from_npz(path_data + f"/Fe_TR={include_TR}.sawf.npz")
    w90data.set_symmetrizer(symmetrizer)
    outer_min, outer_max = -np.inf, np.inf
    if use_window == "select_bands":
        w90data.select_bands(win_min=-8, win_max=50)
    elif use_window == "outer":
        outer_min, outer_max = -8, 50
    froz_max = 30
    w90data.wannierise(init="amn",
                       froz_min=-8,
                       froz_max=froz_max,
                       outer_min=outer_min,
                       outer_max=outer_max,
                       print_progress_every=20,
                       num_iter=40,
                       conv_tol=1e-6,
                       mix_ratio_z=1.0,
                       localise=True,
                       sitesym=True,
                       parallel=parallel,
                       )
    assert np.allclose(w90data.wannier_centers_cart, 0, atol=1e-6), f"wannier_centers differ from 0 by {np.max(abs(w90data.wannier_centers_cart))} \n{w90data.wannier_centers_cart}"
    spreads = w90data.chk.wannier_spreads
    print(f"spreads: {repr(spreads)}")
    assert np.all(spreads < 2)
    atol = 1e-8
    assert spreads[4] == approx(spreads[6], abs=atol)
    assert spreads[5] == approx(spreads[7], abs=atol)
    assert spreads[10] == approx(spreads[12], abs=atol)
    assert spreads[11] == approx(spreads[13], abs=atol)
    spreads_ref = {"select_bands": spreads_Fe_spd_444_win50,
                   "outer": spreads_Fe_spd_444_win50_outer,
                   False: spreads_Fe_spd_444_nowin}[use_window]
    assert spreads == approx(spreads_ref, abs=0.01)
    system = wberri.system.System_w90(w90data=w90data, berry=True)

    # all kpoints given in reduced coordinates
    path = wberri.Path.from_nodes(system,
                    nodes=[
                        [0.0000, 0.0000, 0.0000],  # G
                        [0.500, -0.5000, -0.5000],  # H
                        [0.7500, 0.2500, -0.2500],  # P
                        [0.5000, 0.0000, -0.5000],  # N
                        [0.0000, 0.0000, 0.000]
                    ],  # G
        labels=["G", "H", "P", "N", "G"],
        nk=[21] * 5)   # length [ Ang] ~= 2*pi/dk

    result_path = wberri.evaluate_k_path(system,
                    path=path,
                    parallel=parallel,
                    print_Kpoints=False)
    EF = 12.6
    A = np.loadtxt(os.path.join(path_data, "Fe_bands_pw.dat"))
    energies_ref = np.copy(A[:, 1].reshape(-1, 81)[:18].T)

    bohr_ang = scipy.constants.physical_constants['Bohr radius'][0] / 1e-10
    alatt = 5.4235 * bohr_ang
    A[:, 0] *= 2 * np.pi / alatt
    A[:, 1] = A[:, 1] - EF
    plt.scatter(A[:, 0], A[:, 1], c="black", s=5)
    kpoints = result_path.kpoints
    print(f"kpoints: \n{kpoints}")
    kpoints_path = path.get_kpoints()
    print(f"kpoints_path: \n{kpoints_path}")
    diff = abs(kpoints - kpoints_path)
    diff -= np.round(diff)  # account for periodicity
    assert np.allclose(diff, 0, atol=1e-5), f"kpoints from path and result differ by {np.max(abs(diff))}"

    energies = result_path.get_data(quantity="Energy", iband=np.arange(0, 18))

    np.save(os.path.join(OUTPUT_DIR, f"Fe_bands-{include_TR}.npy"), energies)
    np.savetxt(os.path.join(OUTPUT_DIR, f"Fe_bands-{include_TR}.dat"), energies)

    atol = 0.7
    nk = energies.shape[0]
    energies_diff = np.abs(energies - energies_ref)
    energies_diff[energies_ref > 13] = 0  # ignore the high energy bands
    for ik in range(nk):
        if ik % 20 == 0:
            _atol = 0.01
        else:
            _atol = atol
        assert np.allclose(energies_diff[ik], 0, atol=_atol), \
            f"energies at ik={ik} differ by {np.max(abs(energies_diff[ik]))} more than {_atol}" +\
            f"energies: {energies[ik]}\nref: {energies_ref[ik]}"

    result_path.plot_path_fat(
        path,
        quantity=None,
        Eshift=EF,
        Emin=-10, Emax=50,
        iband=None,
        mode="fatband",
        fatfactor=20,
        cut_k=False,
        linecolor="red",
        close_fig=False,
        show_fig=False,
        label=f"TR={include_TR}",
    )

    plt.ylim(-10, 20)
    plt.hlines(froz_max - EF, 0, A[-1, 0], linestyles="dashed")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fe_bands-{include_TR}.pdf"))
    plt.close()


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("outer_window", [(-100, 100, ""), (-15, 15, "-outer")])
def test_graphene_freeze_bands(outer_window, parallel):
    # This test is to check that the frozen bands are actually frozen. We take the graphene example and freeze the first valence band, which should be well separated from the rest of the bands. Then we check that the spread of this band is not changed by the wannierisation procedure.
    path_data = os.path.join(ROOT_DIR, "data", "graphene_gpaw")
    w90data = wberri.w90files.Wannier90data.from_npz(seedname=path_data + "/graphene-wb", files=["amn", "eig", "mmn", "symmetrizer", "chk"], irreducible=True)
    print(f"files in w90data: {w90data._files}")
    w90data.wannierise(init="amn",
                    print_progress_every=20,
                    outer_min=outer_window[0],
                    outer_max=outer_window[1],
                    frozen_states={0: [1,], 2: [3, 4]},
                    num_iter=50,
                    conv_tol=1e-10,
                    mix_ratio_z=0.8,
                    localise=True,
                    sitesym=True,
                    parallel=parallel,
                    )
    spreads = w90data.chk.wannier_spreads
    print(f"spreads: {repr(spreads)}")
    assert np.all(spreads < 1)
    assert np.all(spreads > 0.3)
    system = System_w90(w90data=w90data, berry=True)

    kpoints = {
        'G': [0.0, 0.0, 0.0],
        'K': [1 / 3, 1 / 3, 0.0],
        'M': [0.5, 0.0, 0.0],
    }

    path_labels = "GKMG"

    path, bands_wannier = evaluate_k_path(system,
                                    nodes=[kpoints[label] for label in path_labels],
                                    labels=list(path_labels),
                                    length=100,
                                    return_path=True)  # length~=2pi/dk
    bands_dft = np.load(os.path.join(path_data, "graphene-bands-dft.npz"))
    ik_G_dft, ik_K_dft = None, None
    for ik, k in enumerate(bands_dft["kpts"]):
        if np.allclose(k, kpoints['G'], atol=1e-5):
            ik_G_dft = ik
        elif np.allclose(k, kpoints['K'], atol=1e-5):
            ik_K_dft = ik
    assert ik_G_dft is not None, "G point not found in DFT k-points"
    assert ik_K_dft is not None, "K point not found in DFT k-points"
    ik_G_wan, ik_K_wan = None, None
    for ik, k in enumerate(path.K_list):
        if np.allclose(k, kpoints['G'], atol=1e-5):
            ik_G_wan = ik
        elif np.allclose(k, kpoints['K'], atol=1e-5):
            ik_K_wan = ik
    assert ik_G_wan is not None, "G point not found in Wannier k-points"
    assert ik_K_wan is not None, "K point not found in Wannier k-points"
    print(f"ik_G_dft={ik_G_dft}, ik_K_dft={ik_K_dft}, ik_G_wan={ik_G_wan}, ik_K_wan={ik_K_wan}")

    bands_wannier = bands_wannier.results["Energy"].data
    bands_dft_G = bands_dft["energies"][ik_G_dft][[1,]]  # the first valence band is frozen, so we take the second one as reference
    bandsw_wannier_G = bands_wannier[ik_G_wan][[0,]]  # the first band should correspond to the frozen band
    assert np.allclose(bandsw_wannier_G, bands_dft_G, atol=0.01), f"G point energies differ from reference by {np.max(abs(bandsw_wannier_G - bands_dft_G))}"
    bands_dft_K = bands_dft["energies"][ik_K_dft][[3, 4]]  # the first valence band is frozen, so we take the second and third one as reference
    bands_wannier_K = bands_wannier[ik_K_wan][[0, 1]]  # the first two bands should correspond to the frozen band and the first conduction band, which are well separated from the rest of the bands at K point
    assert np.allclose(bands_wannier_K, bands_dft_K, atol=0.01), f"K point energies differ from reference by {np.max(abs(bands_wannier_K - bands_dft_K))}"

    bands_wannier_ref = np.load(os.path.join(path_data, f"graphene-bands-wannier{outer_window[2]}.npz"))
    assert np.allclose(path.K_list, bands_wannier_ref["kpts"]), f"kpts differ from reference by {np.max(abs(bands_wannier['kpts'] - bands_wannier_ref['kpts']))}"
    assert np.allclose(bands_wannier, bands_wannier_ref["energies"], atol=0.01), f"energies differ from reference by {np.max(abs(bands_wannier - bands_wannier_ref['energies']))}"
