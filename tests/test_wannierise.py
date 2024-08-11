import irrep
from pytest import approx
import wannierberri as wberri
import numpy as np
import subprocess
from matplotlib import pyplot as plt
import os
import shutil

from wannierberri.wannierise.projections import Projection
from common import OUTPUT_DIR, ROOT_DIR
from wannierberri.w90files import DMN, WIN
from wannierberri.w90files.Dwann import Dwann
from irrep.bandstructure import BandStructure



def test_wanierise():
    systems = {}

    # Just fot Reference : run the Wannier90 with sitesym, but instead of frozen window use outer window
    # to exclude valence bands
    # os.system("wannier90.x diamond")
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
    prefix_dis = "diamond_disentangled"
    for ext in ["mmn", "amn", "dmn", "eig", "win"]:
        shutil.copy(os.path.join(data_dir, prefix + "." + ext),
                    os.path.join(tmp_dir, prefix + "." + ext))
    print("prefix = ", prefix)
    subprocess.run(["wannier90.x", prefix])
    # record the system from the Wanier90 output ('diamond.chk' file)
    systems["w90"] = wberri.system.System_w90(seedname=prefix)
    # Read the data from the Wanier90 inputs
    w90data = wberri.w90files.Wannier90data(seedname=prefix)
    # Now disentangle with sitesym and frozen window (the part that is not implemented in Wanier90)
    w90data.wannierise(
        froz_min=-8,
        froz_max=20,
        num_iter=1000,
        conv_tol=1e-10,
        mix_ratio_z=0.8,
        mix_ratio_u=1,
        print_progress_every=20,
        sitesym=True,
        localise=True
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

    systems["wberri"] = wberri.system.System_w90(w90data=w90data)

    # If needed - perform maximal localization using Wannier90

    # first generate the reduced files - where num_bands is reduced to num_wann,
    # by taking the optimized subspace
    w90data_reduced = w90data.get_disentangled(files=["eig", "mmn", "amn", "dmn"])
    w90data_reduced.wannierise(
        froz_min=-8,
        froz_max=20,
        num_iter=1000,
        conv_tol=1e-10,
        mix_ratio_z=0.8,
        mix_ratio_u=1,
        print_progress_every=20,
        sitesym=True,
        localise=True
    )


    w90data_reduced.write(prefix_dis, files=["eig", "mmn", "amn", "dmn"])
    # Now write the diamond_disentangled.win file
    # first take the existing file
    win_file = wberri.w90files.WIN(seedname=prefix)
    # and modify some parameters
    win_file["num_bands"] = win_file["num_wann"]
    win_file["dis_num_iter"] = 0
    win_file["num_iter"] = 1000
    del win_file["dis_froz_win"]
    del win_file["dis_froz_max"]
    win_file["site_symmetry"] = True
    win_file.write(prefix_dis)
    subprocess.run(["wannier90.x", prefix_dis])
    systems["wberri+mlwf"] = wberri.system.System_w90(seedname=prefix_dis)

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
    plt.savefig("bands.png")
    for k1 in energies:
        for k2 in energies:
            if k1 == k2:
                continue
            diff = abs(energies[k1] - energies[k2])
            # the precidsion is not very high here, although the two codes are assumed to do the same. Not sure why..
            d, acc = np.max(diff), 0.0005
            assert d < acc, f"the interpolated bands {k1} and {k2} differ from w90 interpolation by max {d}>{acc}"

    # One can see that results do not differ much. Also, the maximal localization does not have much effect.
    os.chdir(cwd)


def test_create_dmn():
    data_dir = os.path.join(ROOT_DIR, "data", "diamond")
    tmp_dmn_path = os.path.join(OUTPUT_DIR, "diamond")

    bandstructure = irrep.bandstructure.BandStructure(prefix = data_dir + "/di", Ecut=100,
                                                      code="espresso", 
                                                    from_sym_file=data_dir+"/diamond.sym"
                                                      )
    dmn = DMN(empty=True)
    dmn.from_irrep(bandstructure)
    projection = Projection(position_num=[[0,0,0],[0,0,1/2],[0,1/2,0],[1/2,0,0]], orbital='s', spacegroup=bandstructure.spacegroup)
    print("all positions:",projection.positions)
    dmn.set_D_wann_from_projections(projections_obj=[projection])
    dmn.to_w90_file(tmp_dmn_path)
    

    dmn_ref = DMN(seedname=os.path.join(data_dir, "diamond"),read_npz=False)
    dmn_new = DMN(seedname=tmp_dmn_path, read_npz=False)

    for key in ['NB', "num_wann", "NK", "NKirr", "kptirr", "kptirr2kpt", "kpt2kptirr"]:
        assert np.all(getattr(dmn_ref, key) == getattr(dmn_new, key)), (
                f"key {key} differs between reference and new DMN file\n"
                f"reference: {getattr(dmn_ref, key)}\n"
                f"new: {getattr(dmn_new, key)}\n"
                )

    assert np.all(dmn_ref.D_wann_block_indices == dmn_new.D_wann_block_indices), (
            f"D_wann_block_indices differs between reference and new DMN file\n"
            f"reference: {dmn_ref.D_wann_block_indices}\n"
            f"new: {dmn_new.D_wann_block_indices}\n"
            )
    for ikirr in range(dmn_ref.NKirr):
        assert np.all(dmn_ref.d_band_block_indices[ikirr] == dmn_new.d_band_block_indices[ikirr]), (
            f"d_band_block_indices differs  at ikirr={ikirr} between reference and new DMN file\n"
            f"reference: {dmn_ref.d_band_block_indices}\n"
            f"new: {dmn_new.d_band_block_indices}\n"
            )
    
        for isym in range(dmn_ref.Nsym):
            for blockref,blocknew in zip(dmn_ref.D_wann_blocks[ikirr][isym], dmn_new.D_wann_blocks[ikirr][isym]):
                assert blockref == approx(blocknew, abs=1e-6), f"D_wann at ikirr = {ikirr}, isym = {isym} differs between reference and new DMN file by a maximum of {np.max(np.abs(blockref - blocknew))} > 1e-6"
            for blockref,blocknew in zip(dmn_ref.d_band_blocks[ikirr][isym], dmn_new.d_band_blocks[ikirr][isym]):
                assert blockref == approx(blocknew, abs=1e-6), f"d_band at ikirr = {ikirr}, isym = {isym} differs between reference and new DMN file by a maximum of {np.max(np.abs(blockref - blocknew))} > 1e-6"
    
