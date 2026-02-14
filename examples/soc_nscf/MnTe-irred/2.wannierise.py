from gpaw import GPAW
from irrep.spacegroup import SpaceGroup
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.w90files.w90data_soc import Wannier90dataSOC
from wannierberri.system.system_soc import SystemSOC

gpaw_calc = GPAW("MnTe-nscf-irred-664.gpw")
sg = SpaceGroup.from_gpaw(gpaw_calc)

positions_Mn = [[0, 0, 0],
                [0, 0, 1 / 2]]

positions_Te = [[1 / 3, 2 / 3, 1 / 4],
                [2 / 3, 1 / 3, 3 / 4]]

proj_Mn1_d = Projection(position_num=positions_Mn[0], orbital='d', spacegroup=sg)
proj_Mn2_d = Projection(position_num=positions_Mn[1], orbital='d', spacegroup=sg)
# proj_Mn1_s = Projection(position_num=positions_Mn[0], orbital='s', spacegroup=sg)
# proj_Mn2_s = Projection(position_num=positions_Mn[1], orbital='s', spacegroup=sg)
# proj_Mn1_p = Projection(position_num=positions_Mn[0], orbital='p', spacegroup=sg)
# proj_Mn2_p = Projection(position_num=positions_Mn[1], orbital='p', spacegroup=sg)
# proj_Te_s = Projection(position_num=positions_Te, orbital='s', spacegroup=sg)
# proj_Te_p = Projection(position_num=positions_Te, orbital='p', spacegroup=sg)

proj_Te_sp2 = Projection(position_num=positions_Te, orbital='sp2', spacegroup=sg, xaxis=[0, -1, 0], rotate_basis=True)
proj_Te_pz = Projection(position_num=positions_Te, orbital='pz', spacegroup=sg)

proj_set_up = ProjectionsSet([proj_Mn1_d, proj_Te_sp2, proj_Te_pz])
proj_set_down = ProjectionsSet([proj_Mn2_d, proj_Te_sp2, proj_Te_pz])

w90data = Wannier90dataSOC.from_gpaw(
    seedname="wannier_soc",
    calculator=gpaw_calc,
    projections_up=proj_set_up,
    projections_down=proj_set_down,
    mp_grid=(6, 6, 4),
    spacegroup=sg,
)

w90data.select_bands(win_min=-10,
                     win_max=50)

w90data.wannierise(
    froz_min=-10,
    froz_max=7,
    num_iter=0,
    print_progress_every=50,
    sitesym=True,
    localise=True,
)

theta = 90
phi = 90

system = SystemSOC.from_wannier90data_soc(w90data=w90data, berry=True, silent=False)
system.set_soc_axis(theta=theta, phi=phi, alpha_soc=1.0, units='degrees')
system.save_npz("system_soc")
