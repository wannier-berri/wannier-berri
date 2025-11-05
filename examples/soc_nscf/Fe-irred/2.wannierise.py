import numpy as np
from gpaw import GPAW
from irrep.spacegroup import SpaceGroup
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.w90files.w90data_soc import Wannier90dataSOC
from wannierberri.system.system_soc import SystemSOC

gpaw_calc = GPAW( "Fe-nscf-irred-444.gpw")
sg = SpaceGroup.from_gpaw(gpaw_calc)
projection_sp3d2 = Projection(position_num=[0, 0, 0], orbital='sp3d2', spacegroup=sg)
projection_t2g = Projection(position_num=[0, 0, 0], orbital='t2g', spacegroup=sg)
proj_set = ProjectionsSet([projection_sp3d2, projection_t2g])

# path = os.path.join(OUTPUT_DIR, "Fe-gpaw-soc-irred")
# os.makedirs(path, exist_ok=True)
w90data = Wannier90dataSOC.from_gpaw(
    calculator=gpaw_calc,
    projections=proj_set,
    mp_grid=(4,4,4),
    # read_npz_list=[],
    spacegroup=sg,
    files=["mmn", "eig", "amn", "symmetrizer","soc", "mmn_ud", "mmn_du"],
)

w90data.select_bands(win_min=-100,
                        win_max=50)

w90data.wannierise(
    froz_min=-np.inf,
    froz_max=17,
    num_iter=50,
    print_progress_every=10,
    sitesym=True,
    localise=True,
)

theta = 0
phi = 0

system = SystemSOC.from_wannier90data_soc(w90data=w90data, berry=True, silent=False)
system.set_soc_axis(theta=theta, phi=phi, alpha_soc=1.0)
system.save_npz("system_soc")

