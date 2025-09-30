import numpy as np
from ase import Atoms
from gpaw import GPAW, PW

do_scf = False
do_nscf = True
do_write_w90 = True


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
    calc = GPAW(mode=PW(400),
                kpts={'size': (4, 4, 4), },
                txt=f'{seed}.txt')

    fe.calc = calc
    e = fe.get_potential_energy()
    calc.write(f'{seed}-gs.gpw')
else:
    calc = GPAW(f'{seed}-gs.gpw', txt=None)

if do_nscf:
    calc_nscf = calc.fixed_density(
        kpts={'size': (2, 2, 2), 'gamma': True},
        symmetry='off',
        nbands=30,
        convergence={'bands': 26},
        txt=f'{seed}-nscf.txt')
    calc_nscf.write(f'{seed}-nscf.gpw', mode='all')
else:
    calc_nscf = GPAW(f'{seed}-nscf.gpw', txt=None)


if do_write_w90:
    import os
    from gpaw.wannier90 import Wannier90
    for ispin in 0, 1, :
        spin_name = f'spin-{ispin}'
        seed_wan = f"{seed}-{spin_name}"
        w90 = Wannier90(calc_nscf,
                        seed=seed_wan,
                        # bands=range(40),
                        spinors=False,
                        spin=ispin,
                        orbitals_ai=None,  # [[0], [0, 1, 2,3], [0,1,2,3]]
                        )

        w90.write_input()
        os.system(f'wannier90.x -pp {seed_wan}')
        w90.write_eigenvalues()
        w90.write_overlaps()
