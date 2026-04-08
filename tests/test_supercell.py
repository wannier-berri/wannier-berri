"""Tests for system_supercell: fold_system, spin_double_system, add_scattering."""

import itertools

import numpy as np
import pytest

from wannierberri.fourier.rvectors import Rvectors
from wannierberri.system.system_R import System_R
from wannierberri.system.system_supercell import (
    add_scattering,
    enumerate_subcells,
    fold_system,
    spin_double_system,
)


def _make_system(iRvec, ham, wc_cart, real_lattice):
    system = System_R(periodic=(True, True, False), silent=True)
    system.real_lattice = np.array(real_lattice)
    system.num_wann = ham.shape[1]
    system.wannier_centers_cart = np.array(wc_cart)
    system.rvec = Rvectors(lattice=system.real_lattice, iRvec=np.array(iRvec))
    system.set_R_mat("Ham", ham)
    return system


def _graphene_primitive(t=1.0):
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2, 0.0])
    a3 = np.array([0.0, 0.0, 10.0])
    lattice = np.array([a1, a2, a3])
    wc = np.array([[0.0, 0.0, 0.0], [1.0 / 3, 1.0 / 3, 0.0]]) @ lattice

    iRvec = np.array([
        [0, 0, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0],
    ])
    ham = np.zeros((len(iRvec), 2, 2), dtype=complex)
    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        i = next(k for k, r in enumerate(iRvec) if tuple(r) == R)
        ham[i, 0, 1] = -t
    for R in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
        i = next(k for k, r in enumerate(iRvec) if tuple(r) == R)
        ham[i, 1, 0] = -t
    return _make_system(iRvec, ham, wc, lattice)


def _ham_k(system, k_red):
    H = system.get_R_mat("Ham")
    iR = system.rvec.iRvec
    phase = np.exp(2j * np.pi * iR @ k_red)
    return np.einsum("R,Rij->ij", phase, H)


def test_graphene_sqrt3_dirac_at_sc_gamma():
    sys_prim = _graphene_primitive(t=1.0)
    M = np.array([[2, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=int)
    sc = fold_system(sys_prim, M)

    assert sc.num_wann == 6

    eig_sc = np.sort(np.linalg.eigvalsh(_ham_k(sc, np.zeros(3))))

    # Primitive k-points folding to sc-Γ: k_prim = M^{-T} g, g ∈ Z^3 / M^T Z^3
    MinvT = np.linalg.inv(M.T.astype(float))
    prim_eig = []
    for g in enumerate_subcells(M.T):
        prim_eig.extend(np.linalg.eigvalsh(_ham_k(sys_prim, MinvT @ g)))
    prim_eig = np.sort(prim_eig)

    np.testing.assert_allclose(eig_sc, prim_eig, atol=1e-10)
    np.testing.assert_allclose(eig_sc, [-3, 0, 0, 0, 0, 3], atol=1e-10)
