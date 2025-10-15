import numpy as np
from ase import Atoms
from gpaw import GPAW, PW
from wannierberri.symmetry import get_irreducible_kpoints_grid
from irrep.spacegroup import SpaceGroup


do_scf = False
do_nscf = True


a = 2.87
m = 2.2

seed = "Fe"

lattice = a * np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]]) / 2

fe = Atoms('Fe',
           scaled_positions=[(0, 0, 0)],
           magmoms=[m],
           cell=lattice,
           pbc=True)

if do_scf:
    calc = GPAW(mode=PW(600),
                kpts={'size': (8, 8, 8), },
                txt=f'{seed}.txt')

    fe.calc = calc
    e = fe.get_potential_energy()
    calc.write(f'{seed}-gs.gpw')
else:
    calc = GPAW(f'{seed}-gs.gpw', txt=None)

if do_nscf:
    sg = SpaceGroup.from_gpaw(calc)
    kpoints_irred = get_irreducible_kpoints_grid([2, 2, 2], sg)
    calc_nscf = calc.fixed_density(
        kpts=kpoints_irred,
        nbands=40,
        convergence={'bands': 32},
        txt=f'{seed}-nscf.txt')
    calc_nscf.write(f'{seed}-nscf-irred-222.gpw', mode='all')
else:
    calc_nscf = GPAW(f'{seed}-nscf-irred-222.gpw', txt=None)
