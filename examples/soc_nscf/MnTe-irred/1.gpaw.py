import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, MixerSum
from irrep.spacegroup import SpaceGroup


seed = "MnTe"
do_scf = True
do_nscf = True




if do_scf:
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
    atoms = Atoms("Mn2Te2",
                  cell=lattice,
                  scaled_positions=positions,
                  pbc=[1, 1, 1])

    m = 4.7
    magmoms = np.zeros(4)
    magmoms[0] = +m
    magmoms[1] = -m
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
    calc.write("{seed}-scf.gpw", mode="all")
else:
    calc = GPAW(f'{seed}-scf.gpw', txt=None)

if do_nscf:
    sg = SpaceGroup.from_gpaw(calc)
    kpoints_irred = sg.get_irreducible_kpoints_grid([6, 6, 4])
    calc_nscf = calc.fixed_density(
        kpts=kpoints_irred,
        symmetry={'symmorphic': False},
        nbands=60,
        convergence={'bands': 40},
        txt=f'{seed}-nscf.txt')
    calc_nscf.write(f'{seed}-nscf-irred-664.gpw', mode='all')
else:
    calc_nscf = GPAW(f'{seed}-nscf-irred-664.gpw', txt=None)
