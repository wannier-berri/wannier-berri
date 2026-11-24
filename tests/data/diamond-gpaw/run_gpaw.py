import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, MixerSum
from irrep.spacegroup import SpaceGroup
do_scf = False
do_nscf_irred = True
seed = "diamond"




if do_scf:
    a = 3.227

    lattice = a * (np.ones((3, 3)) - np.eye(3)) / 2
    positions = np.array(
        [
            [0, 0, 0],
            [1 / 4, 1 / 4, 1 / 4],
        ]
    )
    typat = [1, 1]

    atoms = Atoms(
        "C2", cell=lattice, pbc=[1, 1, 1], scaled_positions=positions
    )


    calc = GPAW(
        mode=PW(500),
        xc="PBE",
        symmetry={'symmorphic': False},
        kpts={"size": [8, 8, 8], "gamma": True},
        convergence={"density": 1e-6},
        mixer=MixerSum(0.25, 8, 100),
        txt=f"{seed}-scf.txt"
    )

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(f"{seed}-scf.gpw", mode="all")


if do_nscf_irred:
    calc = GPAW(f'{seed}-scf.gpw', txt=None)
    sg = SpaceGroup.from_gpaw(calc)
    sg.show()
    irred_kpt = sg.get_irreducible_kpoints_grid((4, 4, 4))
    calc_nscf_irred = calc.fixed_density(
        kpts=irred_kpt,
        symmetry={'symmorphic': False},
        nbands=20,
        convergence={'bands': 20},
        txt=f'{seed}-nscf-irred.txt')
    calc_nscf_irred.write(f'{seed}-nscf-irred.gpw', mode='all')
else:
    calc_nscf_irred = GPAW(f'{seed}-nscf-irred.gpw', txt=None)
