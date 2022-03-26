"""Test symmetry objects"""

import numpy as np
import pytest

import wannierberri as wberri
import wannierberri.symmetry as sym

def test_symmetry_group():
    assert sym.Group([sym.Inversion, sym.TimeReversal]).size == 4
    assert sym.Group([sym.C3z, sym.C6z]).size == 6
    assert sym.Group([sym.Inversion, sym.C4z, sym.TimeReversal*sym.C2x]).size == 16

def test_symmetry_group_failure():
    # sym.Group should fail for this generator
    with pytest.raises(RuntimeError):
        c = np.cos(0.1)
        s = np.sin(0.1)
        y = sym.Group([sym.Symmetry(np.array([[1, 0, 0], [0, c, s], [0, -s, c]]))])
