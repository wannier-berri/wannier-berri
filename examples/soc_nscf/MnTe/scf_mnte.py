from gpaw import GPAW, PW, MixerSum
from ase import Atoms
import numpy as np

a = 4.134
c = 6.652

lattice = a * np.array([[1, 0, 0], [-1 / 2, np.sqrt(3) / 2, 0], [0, 0, c / a]])
positions = np.array(
    [
        [0, 0, 0],
        [0, 0, 1 / 2],
        [1 / 3, 2 / 3, 1 / 4],
        [2 / 3, 1 / 3, 3 / 4],
    ]
)
typeat = [1, 1, 2, 2]
magmom = [[0, 1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]]

atoms = Atoms(
    "Mn2Te2", cell=[a, a, c, 90, 90, 120], pbc=[1, 1, 1], scaled_positions=positions
)

m = 4.7
magmoms = np.zeros(4)
magmoms[0] += m
magmoms[1] -= m
atoms.set_initial_magnetic_moments(magmoms)

calc = GPAW(
    mode=PW(600),
    xc="PBE",
    kpts={"size": [6, 6, 4], "gamma": True},
    convergence={"density": 1e-6},
    mixer=MixerSum(0.25, 8, 100),
    setups={"Mn": ":d,4.0"},
    txt="MnTe_scf_norelax.txt"
)

atoms.calc = calc
atoms.get_potential_energy()

# from ase.filters import FrechetCellFilter
# from ase.optimize import LBFGS

# filter = FrechetCellFilter(atoms=atoms, mask=[1, 1, 1, 0, 0, 0])
# opt = LBFGS(filter)
# opt.run(fmax=25e-3)

calc.write("scf_norelax.gpw", mode="all")
