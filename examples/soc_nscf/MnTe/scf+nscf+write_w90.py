import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, MixerSum

do_scf = False
do_nscf = False
do_write_w90 = True
seed = "mnte"




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
    calc.write("scf_norelax.gpw", mode="all")

if do_nscf:
    calc = GPAW('scf_norelax.gpw', txt=None)
    calc_nscf = calc.fixed_density(
        kpts={'size': (6, 6, 4), 'gamma': True},
        symmetry='off',
        nbands=60,
        convergence={'bands': 40},
        txt=f'{seed}-nscf.txt')
    calc_nscf.write(f'{seed}-nscf.gpw', mode='all')
else:
    calc_nscf = GPAW(f'{seed}-nscf.gpw', txt=None)


if do_write_w90:
    import os
    from gpaw.wannier90 import Wannier90
    for ispin in 0, 1:
        spin_name = f'spin-{ispin}' if ispin < 2 else 'spinors'
        seed_wan = f"{seed}-{spin_name}"
        w90 = Wannier90(calc_nscf,
                        seed=seed_wan,
                        # bands=range(40),
                        spinors=(ispin == 2),
                        spin=ispin if ispin < 2 else 0,
                        orbitals_ai=None,  # [[0], [0, 1, 2,3], [0,1,2,3]]
                        )

        w90.write_input()
        os.system(f'wannier90.x -pp {seed_wan}')
        # if ispin != 2:
        # w90.write_wavefunctions()
        # os.mkdir(f"UNK-{seed_wan}")
        # os.system(f"mv UNK0* UNK-{seed_wan}")
        # w90.write_projections()
        w90.write_eigenvalues()
        w90.write_overlaps()
