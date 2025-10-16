"""test auxilary functions"""

import numpy as np
from pytest import approx
import pytest
from wannierberri.formula.covariant import _spin_velocity_einsum_opt
from wannierberri.symmetry.sym_wann_2 import _rotate_matrix
from wannierberri.utility import cached_einsum, vectorize, arr_to_string


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
        C1 += cached_einsum('knls,klma->knmas', A, B)
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


@pytest.mark.parametrize("fmt", ["{:8.4f}", "8.4f"])
def test_arr_to_string(fmt):
    a = np.array([[1.123456789, 2.123456789], [3.123456789, 4.123456789]])
    s = arr_to_string(a, fmt=fmt)
    expected = "  1.1235   2.1235 \n  3.1235   4.1235 "
    assert s == expected, f"Got: \n{s}, expected: \n{expected}"


def test_rotate_matrix():
    for num_wann in 1, 2, 5, 7:
        for num_cart in 0, 1, 2, 3:
            shape_LR = (num_wann, num_wann)
            shape_X = (num_wann,) * 2 + (3,) * num_cart
            L = np.random.rand(*shape_LR) + 1j * np.random.rand(*shape_LR)
            R = np.random.rand(*shape_LR) + 1j * np.random.rand(*shape_LR)
            X = np.random.rand(*shape_X) + 1j * np.random.rand(*shape_X)
            Y = _rotate_matrix(X, L, R)
            assert Y.shape == X.shape
            Z = cached_einsum("ij,jk...,kl->il...", L, X, R)
            assert np.allclose(Y, Z), f"for num_wann={num_wann}, num_cart={num_cart}, the difference is {np.max(np.abs(Y - Z))} Y.shape={Y.shape} X.shape = {X.shape}\nX={X}\nY={Y}\nZ={Z}"
