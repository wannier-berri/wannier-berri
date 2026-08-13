import os
import spglib
from gpaw import GPAW
from irrep.spacegroup import SpaceGroup
from wannierberri.symmetry.wyckoff_position import WyckoffPosition
from .common import ROOT_DIR


try:
    spglib.error.OLD_ERROR_HANDLING = False
except AttributeError:
    pass


def test_contains_position():
    """
    Checks that two equivalent positions in the diamond structure are contained in the same Wyckoff position. 
    """
    data_dir = os.path.join(ROOT_DIR, "data", "diamond-gpaw")
    calc = GPAW(os.path.join(data_dir, "diamond-nscf-irred.gpw"), txt=None)
    cell = calc.atoms.cell
    atomic_positions = calc.atoms.get_scaled_positions()
    numbers = calc.atoms.numbers
    lattice = (cell, atomic_positions, numbers)
    spacegroup = SpaceGroup.from_cell(cell=lattice)
    atoms_wpos = WyckoffPosition(
        position_str="0.5,0.5,0.5",
        spacegroup=spacegroup,
    )
    _, std_positions, _ = spglib.standardize_cell(
        lattice,
        to_primitive=True,
    )
    assert atoms_wpos.num_points == 2
    assert atoms_wpos.contains_position(std_positions[0]) == []
    assert atoms_wpos.contains_position(std_positions[1]) == []
