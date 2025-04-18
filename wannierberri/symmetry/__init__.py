from . import point_symmetry


def getSpaceGroup1(lattice, spinor=False, time_reversal=False):
    from irrep.spacegroup import SpaceGroupBare
    import numpy as np
    if not time_reversal:
        return SpaceGroupBare(Lattice=lattice, spinor=spinor, rotations=[np.eye(3)], translations=[np.zeros(3)], time_reversals=[False], number=1,
                             name="trivial", spinor_rotations=[np.eye(2)])
    else:
        return SpaceGroupBare(Lattice=lattice, spinor=spinor, rotations=[np.eye(3)] * 2, translations=[np.zeros(3)], time_reversals=[False, True], number=1,
                             name="trivial+TR", spinor_rotations=[np.eye(2)] * 2)
