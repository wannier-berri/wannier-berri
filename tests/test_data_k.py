"""Test the Data_K object."""

import numpy as np
from pytest import approx

import wannierberri as wberri
from wannierberri.grid.__Kpoint import KpointBZparallel
from wannierberri.data_K import get_data_k


def test_fourier(system_Fe_W90):
    """Compare slow FT and FFT."""
    system = system_Fe_W90

    k = np.array([0.1, 0.2, -0.3])

    grid = wberri.Grid(system, NKFFT=[4, 3, 2], NKdiv=1, use_symmetry=False)

    dK = 1. / grid.div
    NKFFT = grid.FFT
    factor = 1. / np.prod(grid.div)

    kpoint = KpointBZparallel(K=k, dK=dK, NKFFT=NKFFT, factor=factor, symgroup=None)

    assert kpoint.Kp_fullBZ == approx(k / grid.FFT)

    data_fftw = get_data_k(system, kpoint.Kp_fullBZ, grid=grid, Kpoint=kpoint, npar=0, fftlib='fftw', use_symmetry=False)
    data_slow = get_data_k(system, kpoint.Kp_fullBZ, grid=grid, Kpoint=kpoint, npar=0, fftlib='slow', use_symmetry=False)
    data_numpy = get_data_k(system, kpoint.Kp_fullBZ, grid=grid, Kpoint=kpoint, npar=0, fftlib='numpy', use_symmetry=False)

    test_fields = ["E_K", "D_H", "A_H", "dEig_inv"]

    for field in test_fields:
        assert getattr(data_fftw, field) == approx(getattr(data_slow,
                                                           field)), "fftw  does not match slow for {} ".format(field)
        assert getattr(data_numpy, field) == approx(getattr(data_slow,
                                                            field)), "numpy does not match slow for {}".format(field)
        assert getattr(data_numpy, field) == approx(getattr(data_fftw,
                                                            field)), "numpy does not match fftw for {}".format(field)

    test_fields = ['Ham']

    for field in test_fields:
        for der in 0, 1, 2:
            assert data_fftw.Xbar(field, der) == approx(data_slow.Xbar(
                field, der)), "fftw  does not match slow for {}_bar_der{} ".format(field, der)
            assert data_numpy.Xbar(field, der) == approx(data_slow.Xbar(
                field, der)), "numpy does not match slow for {}_bar_der{} ".format(field, der)
            assert data_numpy.Xbar(field, der) == approx(data_fftw.Xbar(
                field, der)), "numpy does not match fftw for {}_bar_der{} ".format(field, der)

    # TODO: Allow gauge degree of freedom
