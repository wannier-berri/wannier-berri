from wannierberri.workflow.workflow_gpaw import WorkflowGpaw
from .common import OUTPUT_DIR
import numpy as np
from ase import Atoms
from irrep.spacegroup import SpaceGroup

do_scf = False
do_nscf_irred = True
seed = "diamond"



a = 3.227

lattice = a * (np.ones((3, 3)) - np.eye(3)) / 2
positions = np.array([[0, 0, 0],
                      [1 / 4, 1 / 4, 1 / 4]])
atoms = Atoms(
    "C2", cell=lattice, pbc=[1, 1, 1], scaled_positions=positions
)


def test_wrkflow():
    workflow = WorkflowGpaw(ase_atoms=atoms, path=OUTPUT_DIR)
    # workflow.run_scf()
    # workflow.run_nscf()
    workflow.run_nscf_kpath(dk=1)
