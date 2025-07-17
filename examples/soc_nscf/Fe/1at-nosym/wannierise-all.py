from gpaw import GPAW
import numpy as np
import ray
from wannierberri.system.system_w90 import System_w90
from wannierberri.w90files.amn import AMN
from wannierberri.w90files.w90data import Wannier90data
from irrep.bandstructure import BandStructure

from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.symmetry.sawf import SymmetrizerSAWF as SAWF


calc_gpaw = GPAW("Fe-nscf.gpw")

ray.init()


def get_wannierised(prefix, spin_channel, spinor=False, save_name=None):
    bandstructure = BandStructure(code="gpaw",
                              calculator_gpaw=calc_gpaw,
                                Ecut=200,
                                normalize=True,
                                spinor=spinor,
                                spin_channel=spin_channel,
                                magmom=[[0, 0, 1]] if spinor else None
                                )
    sg = bandstructure.spacegroup
    proj_s = Projection(position_num=[[0, 0, 0]], orbital='s', spacegroup=sg)
    proj_p = Projection(position_num=[[0, 0, 0]], orbital='p', spacegroup=sg)
    proj_d = Projection(position_num=[[0, 0, 0]], orbital='d', spacegroup=sg)
    proj_sp3d2 = Projection(position_num=[[0, 0, 0]], orbital='sp3d2', spacegroup=sg)
    proj_t2g = Projection(position_num=[[0, 0, 0]], orbital='t2g', spacegroup=sg)

    # proj_set = ProjectionsSet([proj_s, proj_p, proj_d])
    proj_set = ProjectionsSet([proj_sp3d2, proj_t2g])

    amn = AMN.from_bandstructure(bandstructure, projections=proj_set)
    symmetrizer = SAWF().from_irrep(bandstructure,
                                    unitary_params={'error_threshold': 0.1,
                                                    'warning_threshold': 0.01,
                                                    'nbands_upper_skip': 8 * (2 if spinor else 1)})
    symmetrizer.set_D_wann_from_projections(proj_set)

    w90data = Wannier90data().from_w90_files(prefix, readfiles=["win", "eig", "mmn", "amn"],
                                             read_npz=False)
    w90data.set_file("amn", amn, overwrite=True)
    w90data.set_file("symmetrizer", symmetrizer)
    w90data.select_bands(win_min=-100,
                         win_max=50)

    w90data.wannierise(
        froz_min=-np.inf,
        froz_max=17,
        num_iter=500,
        print_progress_every=10,
        sitesym=True,
        localise=True,

    )
    System_w90(w90data=w90data, symmetrize=False).save_npz(save_name)
    w90data.get_file('chk').to_npz(prefix + ".chk.npz")


get_wannierised("Fe-spin-0", spin_channel=0, save_name="system_dw")
get_wannierised("Fe-spin-1", spin_channel=1, save_name="system_up")
get_wannierised("Fe-spinors", spin_channel=None, spinor=True, save_name="system_spinor", )
