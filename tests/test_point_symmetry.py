"""Test symmetry objects"""

from copy import deepcopy
import numpy as np
import pytest
from packaging.version import parse as pversion
from wannierberri import point_symmetry as sym
from .common_systems import symmetries_GaAs, symmetries_Fe


@pytest.fixture
def check_pointgroup_equal():

    def _inner(g1, g2):
        assert g1.size == g2.size, F"Symmetry group size is different fIRST: {g1.size}, SECOND: {g2.size}"
        for s in g1.symmetries:
            assert s in g2.symmetries, f"Symmetry {s} in group 1 is not in group 2"

    return _inner


def test_symmetry_group():
    assert sym.PointGroup([sym.Inversion, sym.TimeReversal]).size == 4
    assert sym.PointGroup([sym.C3z, sym.C6z]).size == 6
    assert sym.PointGroup([sym.Inversion, sym.C4z, sym.TimeReversal * sym.C2x]).size == 16


def test_symmetry_as_dict(check_pointgroup_equal):
    for sg1 in (sym.PointGroup([sym.Inversion, sym.TimeReversal]),
               sym.PointGroup([sym.C3z, sym.C6z]),
               sym.PointGroup([sym.Inversion, sym.C4z, sym.TimeReversal * sym.C2x])
               ):
        sg2 = sym.PointGroup(dictionary=sg1.as_dict())
        check_pointgroup_equal(sg1, sg2)


def test_symmetry_group_failure():
    # sym.Group should fail for this generator
    with pytest.raises(RuntimeError):
        c = np.cos(0.1)
        s = np.sin(0.1)
        sym.PointGroup([sym.PointSymmetry(np.array([[1, 0, 0], [0, c, s], [0, -s, c]]))])


def test_symmetry_spglib_GaAs(system_GaAs_W90, check_pointgroup_equal):
    system_explicit = deepcopy(system_GaAs_W90)
    system_explicit.set_pointgroup(symmetries_GaAs)

    system_spglib = deepcopy(system_GaAs_W90)
    positions = [[0., 0., 0.], [0.25, 0.25, 0.25]]
    labels = ["Ga", "As"]
    system_spglib.set_structure(positions, labels)
    system_spglib.set_pointgroup_from_structure()

    check_pointgroup_equal(system_explicit.pointgroup, system_spglib.pointgroup)


def test_symmetry_spglib_Fe(system_Fe_W90, check_pointgroup_equal):
    system_explicit = deepcopy(system_Fe_W90)

    # Magnetic symmetries involving time-reversal is not implemented in spglib.
    # So, we exclude symmetries involving time reversal from the generators.
    import spglib
    if pversion(spglib.__version__) < pversion("2"):
        symmetries_Fe_except_TR = [sym for sym in symmetries_Fe if not sym.TR]
        system_explicit.set_pointgroup(symmetries_Fe_except_TR)
    else:
        system_explicit.set_pointgroup(symmetries_Fe)

    system_spglib = deepcopy(system_Fe_W90)
    positions = [[0., 0., 0.]]
    labels = ["Fe"]
    magnetic_moments = [[0., 0., 1.]]
    system_spglib.set_structure(positions, labels, magnetic_moments)
    system_spglib.set_pointgroup_from_structure()

    try:
        sg1 = system_explicit.pointgroup
        sg2 = system_spglib.pointgroup
        check_pointgroup_equal(sg1, sg2)
    except AssertionError as err:
        raise RuntimeError(f"groups are not equal {err}\n explicit: \n---------\n{sg1}\n---------\n---------\n{sg2}\n---------\n")

    # Raise error if magnetic_moments is set to a number, not a 3d vector
    with pytest.raises(Exception):
        system_spglib.set_structure(positions, labels, [1.])
