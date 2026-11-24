import os
from irrep.bandstructure import BandStructure
from fractions import Fraction
import numpy as np
import sympy
from wannierberri.symmetry.sawf import SymmetrizerSAWF
import pytest
from wannierberri.symmetry.projections_searcher import (
    EBRsearcher, grey_group_parts, unitary_spacegroup, get_all_possible_irreps_conj)
from wannierberri.symmetry.projections import Projection, ProjectionsSet

from .common import OUTPUT_DIR, ROOT_DIR


@pytest.fixture(scope="module")
def diamond_setup():
    """(symmetrizer, trial_projections) for diamond, keyed by include_TR."""
    cache = {}

    def _get(TR):
        if TR not in cache:
            data_dir = os.path.join(ROOT_DIR, "data", "diamond")
            bs = BandStructure.from_espresso(prefix=data_dir + "/di", Ecut=100,
                                             include_TR=TR)
            symmetrizer = SymmetrizerSAWF.from_irrep(bs)
            symmetrizer.to_npz(OUTPUT_DIR + f"/diamond-only-bands_{TR}.sawf.npz")
            x, y, z = sympy.symbols('x y z')
            F12, F14, F18 = Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)
            WP = [[0, 0, 0], [x, 0, 0], [F12] * 3, [F14] * 3, [F18] * 3, [0, x, z]]
            projections = ProjectionsSet()
            for w in WP:
                p = ",".join(str(c) for c in w)
                projections.add(Projection(position_sym=p, orbital='s',
                                           spacegroup=bs.spacegroup))
            cache[TR] = (symmetrizer, projections)
        return cache[TR]

    return _get


@pytest.mark.parametrize("TR", [True, False])
def test_find_projections_diamond(TR, diamond_setup):

    symmetrizer, trial_projections = diamond_setup(TR)

    print("trial_projections")
    print(trial_projections.write_with_multiplicities(orbit=False))

    ebrsearcher = EBRsearcher(
        symmetrizer=symmetrizer,
        trial_projections_set=trial_projections,
        froz_min=-10,
        froz_max=30,
        outer_min=-20,
        outer_max=50,
        debug=True
    )

    combinations = ebrsearcher.find_combinations(num_wann_max=10)
    assert len(combinations) == 1
    assert np.all(combinations[0] == [0, 0, 0, 1, 0, 0]), f"combinations[0] = {combinations[0]}, expected [0,0,0,1,0,0]"

    ebrsearcher = EBRsearcher(
        symmetrizer=symmetrizer,
        trial_projections_set=trial_projections,
        froz_min=-10,
        froz_max=20,
        outer_min=-20,
        outer_max=25,
        debug=True
    )

    combinations = ebrsearcher.find_combinations(num_wann_max=10)
    assert len(combinations) == 1
    assert np.all(combinations[0] == [1, 0, 0, 0, 0, 0]), f"combinations[0] = {combinations[0]}, expected [1,0,0,0,0,0]"


def test_grey_group_split(diamond_setup):
    sg_grey = diamond_setup(True)[0].spacegroup
    sg_plain = diamond_setup(False)[0].spacegroup

    isym_unitary, is_grey = grey_group_parts(sg_grey)
    assert is_grey
    # a grey group is G + G·1', so exactly half the operations are unitary
    assert len(isym_unitary) == len(sg_grey.symmetries) // 2
    assert len(isym_unitary) == len(sg_plain.symmetries)

    sg_u = unitary_spacegroup(sg_grey, isym_unitary)
    assert sg_grey.symmetries is not sg_u.symmetries  # original untouched

    isym_all, is_grey_plain = grey_group_parts(sg_plain)
    assert not is_grey_plain
    assert isym_all == list(range(len(sg_plain.symmetries)))
    assert unitary_spacegroup(sg_plain, isym_all) is sg_plain


def test_isym_little_unitary(diamond_setup):
    sym_TR, projections = diamond_setup(True)
    sym_plain, _ = diamond_setup(False)
    searcher = EBRsearcher(symmetrizer=sym_TR, trial_projections_set=projections,
                           froz_min=-10, froz_max=30, outer_min=-20, outer_max=50)
    sg = sym_TR.spacegroup
    for ik, little in enumerate(searcher.isym_little):
        assert little == sorted(little)
        assert not any(sg.symmetries[i].time_reversal for i in little)
        assert set(little) <= set(sym_TR.isym_little[ik])
    assert any(len(a) < len(b) for a, b in
               zip(searcher.isym_little, sym_TR.isym_little))
    # the grey path must reconstruct the colourless group's little groups exactly
    assert sym_TR.NKirr == sym_plain.NKirr
    assert ([len(l) for l in searcher.isym_little] ==
            [len(l) for l in sym_plain.isym_little])


# minimal Irrep-like objects for testing
class _Op:
    def __init__(self, rotation, translation, time_reversal):
        self.rotation, self.translation = np.asarray(rotation), np.asarray(translation)
        self.time_reversal = time_reversal


class _SG:
    def __init__(self, symmetries):
        self.symmetries = symmetries


def test_grey_group_parts_classification():
    E, I, ZERO = np.eye(3), -np.eye(3), np.zeros(3)
    # type I: no antiunitary operations
    assert grey_group_parts(_SG([_Op(E, ZERO, False), _Op(I, ZERO, False)])) == ([0, 1], False)
    # type II: pure 1' present
    assert grey_group_parts(_SG([_Op(E, ZERO, False), _Op(I, ZERO, False),
                                 _Op(E, ZERO, True), _Op(I, ZERO, True)])) == ([0, 1], True)
    # type III: TR only on non-identity operations
    assert grey_group_parts(_SG([_Op(E, ZERO, False), _Op(I, ZERO, True)])) == ([0], False)
    # type IV: antiunitary identity carries a translation
    assert grey_group_parts(_SG([_Op(E, ZERO, False), _Op(E, [.5, 0, 0], True)])) == ([0], False)
    # integer translation is a lattice vector, i.e. still pure 1'
    assert grey_group_parts(_SG([_Op(E, ZERO, False),
                                 _Op(E, [1, 0, 0], True)])) == ([0], True)


def test_get_all_possible_irreps_conj_defaults(diamond_setup):
    """The bare call must reproduce the type-I path."""
    sym_plain, _ = diamond_setup(False)
    irreps = get_all_possible_irreps_conj(sym_plain)
    assert len(irreps) == sym_plain.NKirr
    for ik, ir in enumerate(irreps):
        assert ir.shape[1] == len(sym_plain.isym_little[ik])
