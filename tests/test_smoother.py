"""Test the Smoother classes"""

import numpy as np
import pytest
from pytest import approx

from wannierberri.smoother import (
    AbstractSmoother,
    VoidSmoother,
    FermiDiracSmoother,
    GaussianSmoother,
    get_smoother,
)


def test_smoother():
    np.random.seed(123)

    # Instantiation of an abstract class should fail
    with pytest.raises(TypeError):
        AbstractSmoother()

    # Test VoidSmoother. A do-nothing smoother
    sm_void = VoidSmoother()
    data = np.random.rand(3, 4)
    assert sm_void(data) == approx(data, abs=1E-10)

    # Test nontrivial smoothers
    e = np.arange(-1.0, 1.1, 0.1)
    data = np.zeros((len(e), ))
    data[10] = 1.0
    for smoother, param in [(GaussianSmoother, 0.1), (FermiDiracSmoother, 1000.0)]:
        sm = smoother(e, param)
        target = sm._broaden(e) * sm.dE
        target /= np.sum(target[10 - sm.NE1:10 + sm.NE1 + 1])
        assert sm(data)[9:12] == approx(target[9:12], abs=1E-10)
        assert np.all(sm(data)[:10 - sm.NE1] == 0.0)
        assert np.all(sm(data)[10 + sm.NE1 + 1:] == 0.0)

    # Test nontrivial smoothers for high-dimensional data
    e = np.arange(-1.0, 1.1, 0.1)
    data_1d = np.random.rand(len(e))
    data_3d = np.zeros((2, len(e), 3))
    data_3d[...] = data_1d[None, :, None]
    for smoother, param in [(GaussianSmoother, 0.1), (FermiDiracSmoother, 1000.0)]:
        sm = smoother(e, param)
        sm_data_1d = sm(data_1d)
        sm_data_3d = sm(data_3d, axis=1)
        assert sm_data_3d - sm_data_1d[None, :, None] == approx(0.0, abs=1E-10)

    # Test get_smoother
    assert get_smoother(e, None) == VoidSmoother()
    assert get_smoother(None, None) == VoidSmoother()
    assert get_smoother([0.0], 1.0) == VoidSmoother()
    assert get_smoother(e, 0.1, "Gaussian") == GaussianSmoother(e, 0.1)
    assert get_smoother(e, 0.1, "Fermi-Dirac") == FermiDiracSmoother(e, 0.1)
    assert get_smoother(e, 0.2, "Gaussian") != GaussianSmoother(e, 0.1)
    assert get_smoother(e + 0.1, 0.2, "Gaussian") != GaussianSmoother(e, 0.1)
    assert get_smoother(e, 0.1, "Gaussian") != FermiDiracSmoother(e, 0.1)
