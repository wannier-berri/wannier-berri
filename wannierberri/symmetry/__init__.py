from . import point_symmetry
from irrep.spacegroup import SpaceGroup


def getSpaceGroup1(lattice, spinor=False, time_reversal=False):
    from irrep.spacegroup import SpaceGroup
    import numpy as np
    if not time_reversal:
        return SpaceGroup(Lattice=lattice, spinor=spinor, rotations=[np.eye(3)], translations=[np.zeros(3)], time_reversals=[False], number=1,
                          name="trivial", spinor_rotations=[np.eye(2)])
    else:
        return SpaceGroup(Lattice=lattice, spinor=spinor, rotations=[np.eye(3)] * 2, translations=[np.zeros(3)], time_reversals=[False, True], number=1,
                          name="trivial+TR", spinor_rotations=[np.eye(2)] * 2)


def get_spacegroup_from_gpaw(calculator, spinor=False, include_TR=True,
                             typat=None,
                             magmoms=None, symprec=1e-5):
    from gpaw import GPAW
    if isinstance(calculator, str):
        calculator = GPAW(calculator)
    else:
        assert isinstance(calculator, GPAW)
    lattice = calculator.atoms.cell
    typat_read = calculator.atoms.get_atomic_numbers()
    if typat is None:
        typat = typat_read
    elif len(typat) != len(typat_read):
        raise ValueError("Length of typat does not match the number of atoms in the calculator.")
    positions = calculator.atoms.get_scaled_positions()
    return SpaceGroup.from_cell(real_lattice=lattice,
                               positions=positions,
                               typat=typat,
                               spinor=spinor,
                               include_TR=include_TR,
                               symprec=symprec,
                               magmoms=magmoms)
