"""test auxilary functions"""

import numpy as np
from pytest import approx
from wannierberri.formula.covariant import _spin_velocity_einsum_opt


def test_spin_velocity_einsum_opt():

    nw = 5
    nk = 6
    for i in range(10):
        A = np.random.random((nk, nw, nw, 3))
        B = np.random.random((nk, nw, nw, 3))
        C = np.random.random((nk, nw, nw, 3, 3))
        C1 = np.copy(C)
        _spin_velocity_einsum_opt(C, A, B)
        # Optimized version of C += np.einsum('knls,klma->knmas', A, B). Used in shc_B_H.
        C1 += np.einsum('knls,klma->knmas', A, B)
        assert C1 == approx(C)
