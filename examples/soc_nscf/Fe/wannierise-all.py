from gpaw import GPAW
import numpy as np
import ray
import wannierberri as wberri
from wannierberri.system import System_R
from wannierberri.w90files.amn import AMN
from wannierberri.w90files.wandata import WannierData
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
                                include_TR=True,
                                magmom=[[0, 0, 1]] if spinor else None
                                )
    sg = bandstructure.spacegroup
    proj_sp3d2 = Projection(position_num=[[0, 0, 0]], orbital='sp3d2', spacegroup=sg)
    proj_t2g = Projection(position_num=[[0, 0, 0]], orbital='t2g', spacegroup=sg)
    proj_set = ProjectionsSet([proj_sp3d2, proj_t2g])

    # proj_set = ProjectionsSet([proj_s, proj_p, proj_d])
    # proj_s = Projection(position_num=[[0, 0, 0]], orbital='s', spacegroup=sg)
    # proj_p = Projection(position_num=[[0, 0, 0]], orbital='p', spacegroup=sg)
    # proj_d = Projection(position_num=[[0, 0, 0]], orbital='d', spacegroup=sg)

    amn = AMN.from_bandstructure(bandstructure, projections=proj_set)
    symmetrizer = SAWF.from_irrep(bandstructure,
                                  unitary_params={'error_threshold': 0.1,
                                                  'warning_threshold': 0.01,
                                                  'nbands_upper_skip': 8 * (2 if spinor else 1)})
    symmetrizer.set_D_wann_from_projections(proj_set)

    wandata = WannierData.from_w90_files(prefix, files=["win", "eig", "mmn", "amn"])
    wandata.set_file("amn", amn, overwrite=True)
    wandata.set_file("symmetrizer", symmetrizer)

    wberri.wannierise(
        wandata=wandata,
        froz_min=-np.inf,
        froz_max=17,
        outer_min=-100,
        outer_max=50,
        num_iter=500,
        print_progress_every=10,
        sitesym=True,
        localise=True,

    )
    System_R.from_wannierdata(wandata=wandata, symmetrize=False, berry=True).save_npz(save_name)
    wandata.get_file('chk').to_npz(save_name + ".chk.npz")


get_wannierised("Fe-spin-0", spin_channel=0, save_name="system_up")
get_wannierised("Fe-spin-1", spin_channel=1, save_name="system_dw")
get_wannierised("Fe-spinors", spin_channel=None, spinor=True, save_name="system_spinor", )
