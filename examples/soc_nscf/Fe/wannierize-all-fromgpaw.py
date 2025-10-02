from gpaw import GPAW
import numpy as np
import ray
from wannierberri.system.system_w90 import System_w90
from wannierberri.w90files.w90data import Wannier90data

from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.symmetry import get_spacegroup_from_gpaw


calc_gpaw = GPAW("Fe-nscf.gpw")

# ray.init()


def get_wannierised(spin_channel, save_name=None):
    sg = get_spacegroup_from_gpaw(calc_gpaw)
    sg.show()
    proj_sp3d2 = Projection(position_num=[[0, 0, 0]], orbital='sp3d2', spacegroup=sg)
    proj_t2g = Projection(position_num=[[0, 0, 0]], orbital='t2g', spacegroup=sg)
    proj_set = ProjectionsSet([proj_sp3d2, proj_t2g])
    w90data = Wannier90data().from_gpaw(calculator=calc_gpaw,
                                        spin_channel=spin_channel,
                                        spacegroup=sg,
                                        projections=proj_set,
                                        files=("mmn", "eig", "amn", "symmetrizer"),
                                        unitary_params={'error_threshold': 0.1,
                                                    'warning_threshold': 0.01,
                                                    'nbands_upper_skip': 10}
                                    )

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
    System_w90(w90data=w90data, symmetrize=False, berry=True).save_npz(save_name)
    w90data.get_file('chk').to_npz(save_name + ".chk.npz")


get_wannierised(spin_channel=0, save_name="system_up")
# get_wannierised("Fe-spin-1", spin_channel=1, save_name="system_dw")
# get_wannierised("Fe-spinors", spin_channel=None, spinor=True, save_name="system_spinor", )
