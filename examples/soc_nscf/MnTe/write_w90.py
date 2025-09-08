from gpaw import GPAW

do_nscf = False
do_write_w90 = True

seed = "mnte"


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
    for ispin in 0,1 :
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
