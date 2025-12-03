#  parallel execution: 
#   gpaw -P 16 python Fe-gpaw.py
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW
from irrep.spacegroup import SpaceGroup



a = 2.87
m = 0

seed = "Fe"

lattice = a * np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]]) / 2

for m in 0.0, 2.2:
    seed = f'Fe_m{m:.1f}'
    fe = Atoms('Fe',
        scaled_positions=[(0, 0, 0)],
        magmoms=[m],
        cell=lattice,
        pbc=True)

    calc = GPAW(mode=PW(600),
            kpts={'size': (8, 8, 8), },
            txt=f'{seed}.txt')

    fe.calc = calc
    e = fe.get_potential_energy()
    calc.write(f'{seed}-gs.gpw')
    sg = SpaceGroup.from_gpaw(calc)
    kpoints_irred = sg.get_irreducible_kpoints_grid([4,4,4], sg)
    calc_nscf = calc.fixed_density(
    kpts=kpoints_irred,
    nbands=40,
    convergence={'bands': 32},
    txt=f'{seed}-nscf.txt')
    calc_nscf.write(f'{seed}-nscf-irred-444.gpw', mode='all')
