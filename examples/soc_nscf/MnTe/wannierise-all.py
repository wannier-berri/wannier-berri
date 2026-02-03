import pickle
from gpaw import GPAW
# import numpy as np
import ray
from wannierberri.system.system_w90 import System_w90
from wannierberri.w90files.amn import AMN
from wannierberri.w90files.w90data import Wannier90data
from irrep.bandstructure import BandStructure
from irrep.spacegroup import SpaceGroup

from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.symmetry.sawf import SymmetrizerSAWF as SAWF


calc_gpaw = GPAW("mnte-nscf.gpw")

ray.init(num_cpus=16)


positions_Mn = [[0, 0, 0],
                [0, 0, 1 / 2]]

positions_Te = [[1 / 3, 2 / 3, 1 / 4],
                [2 / 3, 1 / 3, 3 / 4]]



def get_wannierised(prefix, spin_channel, save_name=None):
    try:
        bandstructure = pickle.load(open(f"bandstructure-spin-{spin_channel}.pkl", "rb"))
    except FileNotFoundError:
        bandstructure = BandStructure(code="gpaw",
                                calculator_gpaw=calc_gpaw,
                                Ecut=200,
                                normalize=True,
                                spinor=False,
                                spin_channel=spin_channel,
                                )
        pickle.dump(bandstructure, open(f"bandstructure-spin-{spin_channel}.pkl", "wb"))
    sg = bandstructure.spacegroup
    sg = SpaceGroup.from_cell(real_lattice=sg.real_lattice, positions=sg.positions, spinor=False,
                              typat=[1, 2, 3, 3],)
    bandstructure.spacegroup = sg

    proj_Mn1_d = Projection(position_num=positions_Mn[0], orbital='d', spacegroup=sg)
    proj_Mn2_d = Projection(position_num=positions_Mn[1], orbital='d', spacegroup=sg)
    # proj_Mn1_s = Projection(position_num=positions_Mn[0], orbital='s', spacegroup=sg)
    # proj_Mn2_s = Projection(position_num=positions_Mn[1], orbital='s', spacegroup=sg)
    # proj_Mn1_p = Projection(position_num=positions_Mn[0], orbital='p', spacegroup=sg)
    # proj_Mn2_p = Projection(position_num=positions_Mn[1], orbital='p', spacegroup=sg)
    proj_Te_s = Projection(position_num=positions_Te, orbital='s', spacegroup=sg)
    proj_Te_p = Projection(position_num=positions_Te, orbital='p', spacegroup=sg)

    # proj_set = ProjectionsSet([
    #     proj_Mn1_d,
    #     proj_Mn2_d,
    #     proj_Mn1_s,
    #     proj_Mn2_s,
    #     proj_Mn1_p,
    #     proj_Mn2_p,
    #     proj_Te_s,
    #     proj_Te_p])


    if spin_channel == 0:
        proj_set = ProjectionsSet([proj_Mn1_d, proj_Te_s, proj_Te_p])
    elif spin_channel == 1:
        proj_set = ProjectionsSet([proj_Mn2_d, proj_Te_s, proj_Te_p])
    else:
        raise ValueError("spin_channel must be 0 or 1")


    amn = AMN.from_bandstructure(bandstructure, projections=proj_set)
    try:
        symmetrizer = SAWF.from_npz(f"symmetrizer-spin-{spin_channel}.npz")
    except FileNotFoundError:
        symmetrizer = SAWF.from_irrep(bandstructure,
                                        unitary_params={'error_threshold': 0.1,
                                                        'warning_threshold': 0.01,
                                                        'nbands_upper_skip': 8})
        symmetrizer.to_npz(f"symmetrizer-spin-{spin_channel}.npz")
    symmetrizer.set_D_wann_from_projections(proj_set)

    w90data = Wannier90data.from_w90_files(prefix, readfiles=["win", "eig", "mmn"],
                                             read_npz=True)
    w90data.set_file("amn", amn, overwrite=True)
    w90data.set_file("symmetrizer", symmetrizer)
    # w90data.select_bands(win_min=-10,
    #                      win_max=50)

    w90data.wannierise(
        froz_min=-10,
        froz_max=7,
        num_iter=500,
        print_progress_every=100,
        sitesym=True,
        localise=True,

    )
    System_w90(w90data=w90data, symmetrize=True, berry=True).save_npz(save_name)
    w90data.get_file('chk').to_npz(save_name + ".chk.npz")


get_wannierised("mnte-spin-0", spin_channel=0, save_name="system_up")
get_wannierised("mnte-spin-1", spin_channel=1, save_name="system_dw")
