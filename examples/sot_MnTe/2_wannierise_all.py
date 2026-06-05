from wannierberri.symmetry.projections import Projection, ProjectionsSet
import os

from gpaw import GPAW
# import numpy as np
from wannierberri.system.system_soc import SystemSOC
from wannierberri.w90files.wandata_soc import WannierDataSOC
from irrep.spacegroup import SpaceGroup
import ray
ray.init(num_cpus=16)


calc_gpaw = GPAW("mnte-nscf.gpw")
sg = SpaceGroup.from_gpaw(calc_gpaw)
positions_Mn = sg.positions[:2]
positions_Te = sg.positions[2:]
print(f"Mn positions: {positions_Mn}")
print(f"Te positions: {positions_Te}")
print(f"sg.spinor: {sg.spinor}")


proj_Mn1_d = Projection(position_num=positions_Mn[0], orbital='d', spacegroup=sg)
proj_Mn2_d = Projection(position_num=positions_Mn[1], orbital='d', spacegroup=sg)
proj_Te_s = Projection(position_num=positions_Te, orbital='s', spacegroup=sg)
proj_Te_p = Projection(position_num=positions_Te, orbital='p', spacegroup=sg)


proj_set_up = ProjectionsSet([proj_Mn1_d, proj_Te_s, proj_Te_p])
proj_set_down = ProjectionsSet([proj_Mn2_d, proj_Te_s, proj_Te_p])


seedname = "wandata/mnte"
try:
    files = ["amn", "mmn", "eig", "symmetrizer", "soc"]
    wandata = WannierDataSOC.from_npz(seedname, nspin=2, files=files)
    for file in ["soc"]:
        assert wandata.has_file(file), f"File {file} not found in {seedname}.npz"
    for file in ["amn", "mmn", "eig", "symmetrizer"]:
        for updw in ["up", "down"]:
            if updw == "up":
                data = wandata.data_up
            else:
                data = wandata.data_down
        if not data.has_file(file):
            print(f"Warning: File {file} not found in {seedname}.npz, data {updw} but it is needed for wannierisation. ")
except (FileNotFoundError, AssertionError) as e:
    print(f"Could not load WannierDataSOC from {seedname}.npz, creating new one from GPAW calculation. Reason: \n{e}")
    exit()
    os.makedirs(os.path.dirname(seedname), exist_ok=True)
    wandata = WannierDataSOC.from_gpaw(calc_gpaw,
                                    files=["amn", "mmn", "eig", "symmetrizer", "soc"],
                                    projections_up=proj_set_up,
                                    projections_down=proj_set_down,
                                    mp_grid=(6, 6, 4),
                                    )
    wandata.to_npz(seedname)

wandata.wannierise(
    froz_min=-10,
    froz_max=7,
    # outer_min=-10,
    # outer_max=50,
    num_iter=500,
    print_progress_every=100,
    sitesym=True,
    localise=True,
)


system_soc = SystemSOC.from_wannierdata(wandata)

system_soc.to_npz("MnTe_system_soc")
