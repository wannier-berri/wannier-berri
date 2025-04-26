"""test auxilary functions"""

import numpy as np
from pytest import approx
from wannierberri.formula.covariant import _spin_velocity_einsum_opt
from wannierberri.utility import vectorize


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


def test_vectorized_eigh():
    nw = 5
    nk = 6
    for i in range(10):
        H = np.random.random((nk, nw, nw)) + 1j * np.random.random((nk, nw, nw))
        H = H + H.transpose(0, 2, 1).conj()
        EV = vectorize(np.linalg.eigh, H)
        for h, ev in zip(H, EV):
            e1, v1 = ev
            e2, v2 = np.linalg.eigh(h)
            assert e1 == approx(e2)


def test_vectorized_matrix_prod():
    nw = 5
    nk = 6
    for i in range(10):
        A = np.random.random((nk, nw, nw)) + 1j * np.random.random((nk, nw, nw))
        B = np.random.random((nk, nw, nw)) + 1j * np.random.random((nk, nw, nw))
        C = vectorize(np.dot, A, B)
        for a, b, c in zip(A, B, C):
            c1 = np.dot(a, b)
            assert c == approx(c1)
