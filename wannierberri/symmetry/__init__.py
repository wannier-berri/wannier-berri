import numpy as np
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


def get_spacegroup_from_gpaw(calculator,
                             symprec_magmom=0.05,
                             include_TR=True,
                             symprec=1e-5,
                             typat=None,
                             magmoms=None):
    """Get the spacegroup of a GPAW calculator (non-spinor only).

    Parameters
    ----------
    calculator : GPAW
        The GPAW calculator.
    symprec_magmom : float
        The precision for distinguishing different magnetic moments.
    symprec : float
        The symmetry precision for spacegroup detection.
    include_TR : bool
        Whether to include time-reversal symmetry.
    typat : list of int, optional
        The typat to use for spacegroup detection. If None, the atomic numbers are used,
        and if magmoms is also None, the magnetic moments are used to distinguish different types of atoms.
        The magnetic moments are rounded to the nearest integer and mapped to consecutive integers starting from 1.
        For example, if the magnetic moments are [2.1, -2.1, 0.0, 2.9], they are rounded to [2, -2, 0, 3],
        and then mapped to [2, 1, 0, 3] (since -2 is the smallest unique value, it is mapped to 1).
        The final typat will be atomic_number*1000 + mapped_magnetic_moment.
    magmoms : list of float, optional
        The magnetic moments to use for spacegroup detection. If None, the magnetic moments from the calculator are used.

    Returns
    -------
    spacegroup : irrep.spacegroup.SpaceGroup
        The detected spacegroup.
    """
    lattice = calculator.atoms.cell
    if typat is None:
        typat = calculator.atoms.get_atomic_numbers()
        if magmoms is None:
            magmoms = calculator.get_magnetic_moments()
            assert magmoms.shape == (len(calculator.atoms),)
        magmoms = np.round(magmoms / symprec_magmom).astype(int).tolist()
        magmoms_set = set(magmoms)
        if len(magmoms_set) > 1:
            magmom_map = {m: i + 1 for i, m in enumerate(sorted(magmoms_set))}
            typat = [typat[i] * 1000 + magmom_map[mag] for i, mag in enumerate(magmoms)]
    else:
        assert len(typat) == len(calculator.atoms), "typat should have the same length as the number of atoms"
    print("typat used for spacegroup detection (accounting magmoms):", typat)
    positions = calculator.atoms.get_scaled_positions()
    return SpaceGroup.from_cell(real_lattice=lattice,
                               positions=positions,
                               typat=typat,
                               spinor=False,
                               include_TR=include_TR,
                               symprec=symprec,
                               magmoms=magmoms)
