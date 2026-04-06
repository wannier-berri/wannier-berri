"""Tests for wannierberri.system.system_supercell — fold_system & add_scattering."""

import numpy as np
import pytest

from wannierberri.fourier.rvectors import Rvectors
from wannierberri.system.system_R import System_R
from wannierberri.system.system_supercell import (
    _build_wannier_centres_sc,
    _fold_matrix,
    _fold_scattering,
    _supercell_rvectors,
    add_scattering,
    enumerate_subcells,
    fold_system,
)


# ------------------------------------------------------------------
# Helper: build a minimal primitive System_R for testing
# ------------------------------------------------------------------


def _make_system(iRvec, ham, wc_cart, real_lattice, extra_matrices=None):
    """Create a minimal System_R for testing."""
    system = System_R(periodic=(True, True, False), silent=True)
    system.real_lattice = np.array(real_lattice)
    system.num_wann = ham.shape[1]
    system.wannier_centers_cart = np.array(wc_cart)
    system.rvec = Rvectors(lattice=system.real_lattice, iRvec=np.array(iRvec))
    system.set_R_mat("Ham", ham)
    if extra_matrices:
        for key, val in extra_matrices.items():
            system.set_R_mat(key, val)
    return system


# ------------------------------------------------------------------
# enumerate_subcells
# ------------------------------------------------------------------


class TestEnumerateSubcells:
    def test_identity(self):
        subcells = enumerate_subcells(np.eye(3, dtype=int))
        assert len(subcells) == 1
        np.testing.assert_array_equal(subcells[0], [0, 0, 0])

    def test_diagonal_2x2x1(self):
        subcells = enumerate_subcells(np.diag([2, 2, 1]))
        assert len(subcells) == 4

    def test_diagonal_3x1x1(self):
        subcells = enumerate_subcells(np.diag([3, 1, 1]))
        assert len(subcells) == 3
        # Should be [0,0,0], [1,0,0], [2,0,0]
        for i in range(3):
            assert any(np.all(s == [i, 0, 0]) for s in subcells)

    def test_non_diagonal(self):
        M = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
        subcells = enumerate_subcells(M)
        assert len(subcells) == 4

    def test_rejects_singular(self):
        with pytest.raises(ValueError, match="non-singular"):
            enumerate_subcells(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]))

    def test_negative_entries(self):
        """M with negative entries should still enumerate correctly."""
        subcells = enumerate_subcells(np.diag([-2, 1, 1]))
        assert len(subcells) == 2

    def test_off_diagonal_negative(self):
        M = np.array([[2, -1, 0], [0, 2, 0], [0, 0, 1]])
        subcells = enumerate_subcells(M)
        assert len(subcells) == 4

    def test_rejects_non_3x3(self):
        with pytest.raises(ValueError, match="3x3"):
            enumerate_subcells(np.diag([2, 2]))


# ------------------------------------------------------------------
# _supercell_rvectors
# ------------------------------------------------------------------


class TestSupercellRvectors:
    def test_identity(self):
        M = np.eye(3, dtype=int)
        subcells = enumerate_subcells(M)
        iRvec_prim = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        R_sc = _supercell_rvectors(iRvec_prim, M, subcells)
        assert {tuple(r) for r in R_sc} == {tuple(r) for r in iRvec_prim}

    def test_contains_origin(self):
        M = np.diag([2, 2, 1])
        subcells = enumerate_subcells(M)
        iRvec_prim = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        R_sc = _supercell_rvectors(iRvec_prim, M, subcells)
        assert any(np.all(r == 0) for r in R_sc)


# ------------------------------------------------------------------
# _fold_matrix
# ------------------------------------------------------------------


class TestFoldMatrix:
    def test_identity_preserves(self):
        M = np.eye(3, dtype=int)
        subcells = enumerate_subcells(M)
        iRvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        prim_lookup = {tuple(r): i for i, r in enumerate(iRvec)}
        R_sc = _supercell_rvectors(iRvec, M, subcells)

        nwann = 2
        rng = np.random.default_rng(42)
        X_R = rng.standard_normal((3, nwann, nwann)) + 0j
        X_sc = _fold_matrix(X_R, iRvec, prim_lookup, R_sc, M, subcells, nwann)

        sc_lookup = {tuple(r): i for i, r in enumerate(R_sc)}
        for R, ip in prim_lookup.items():
            np.testing.assert_allclose(X_sc[sc_lookup[R]], X_R[ip], atol=1e-12)

    def test_2x1x1_diagonal_blocks(self):
        """R=0 diagonal blocks should contain primitive R=0 matrix."""
        M = np.diag([2, 1, 1])
        subcells = enumerate_subcells(M)
        nwann = 2

        iRvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        prim_lookup = {tuple(r): i for i, r in enumerate(iRvec)}
        R_sc = _supercell_rvectors(iRvec, M, subcells)

        X_R = np.zeros((3, nwann, nwann), dtype=complex)
        X_R[0] = np.array([[1.0, 0.2], [0.2, 2.0]])
        X_R[1] = np.array([[0.5, 0.1], [0.1, 0.3]])
        X_R[2] = X_R[1].conj().T

        X_sc = _fold_matrix(X_R, iRvec, prim_lookup, R_sc, M, subcells, nwann)
        iR0 = next(i for i, r in enumerate(R_sc) if np.all(r == 0))
        for s in range(len(subcells)):
            sl = slice(s * nwann, (s + 1) * nwann)
            np.testing.assert_allclose(X_sc[iR0, sl, sl], X_R[0], atol=1e-12)

    def test_extra_dimensions(self):
        """AA-like matrix with shape [nR, nw, nw, 3]."""
        M = np.diag([2, 1, 1])
        subcells = enumerate_subcells(M)
        iRvec = np.array([[0, 0, 0]])
        prim_lookup = {(0, 0, 0): 0}
        R_sc = _supercell_rvectors(iRvec, M, subcells)

        X_R = np.array([[[[1.0, 2.0, 3.0]]]], dtype=complex)
        X_sc = _fold_matrix(X_R, iRvec, prim_lookup, R_sc, M, subcells, 1)
        assert X_sc.shape == (len(R_sc), 2, 2, 3)

    def test_gamma_eigenvalues_1d_chain(self):
        """Folded 2x supercell of 1D chain: E(k=0_sc) = E(0) and E(pi)."""
        M = np.diag([2, 1, 1])
        subcells = enumerate_subcells(M)
        eps, t = 1.0, -0.5

        iRvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        prim_lookup = {tuple(r): i for i, r in enumerate(iRvec)}
        R_sc = _supercell_rvectors(iRvec, M, subcells)

        X_R = np.zeros((3, 1, 1), dtype=complex)
        X_R[0, 0, 0] = eps
        X_R[1, 0, 0] = t
        X_R[2, 0, 0] = t

        X_sc = _fold_matrix(X_R, iRvec, prim_lookup, R_sc, M, subcells, 1)
        H_gamma = X_sc.sum(axis=0)
        E_sc = np.sort(np.linalg.eigvalsh(H_gamma))
        E_expected = np.sort([eps + 2 * t, eps - 2 * t])  # cos(0)=1, cos(pi)=-1
        np.testing.assert_allclose(E_sc, E_expected, atol=1e-12)


# ------------------------------------------------------------------
# _build_wannier_centres_sc
# ------------------------------------------------------------------


class TestBuildWannierCentresSc:
    def test_tiling(self):
        subcells = np.array([[0, 0, 0], [1, 0, 0]])
        lattice = np.array([[3.0, 0, 0], [0, 4.0, 0], [0, 0, 20.0]])
        wc = np.array([[0.5, 0.5, 0.0]])
        wc_sc = _build_wannier_centres_sc(subcells, lattice, wc)
        assert wc_sc.shape == (2, 3)
        np.testing.assert_allclose(wc_sc[0], [0.5, 0.5, 0.0], atol=1e-12)
        np.testing.assert_allclose(wc_sc[1], [3.5, 0.5, 0.0], atol=1e-12)


# ------------------------------------------------------------------
# _fold_scattering
# ------------------------------------------------------------------


class TestFoldScattering:
    def test_uniform_on_site(self):
        """V_kk = V0 * I  =>  all subcell diagonals get V0."""
        grid_shape = (2, 1, 1)
        M = np.diag([2, 1, 1])
        grid_arr = np.array(grid_shape)
        norb = 1
        nk = 2

        V0 = 0.3
        V_kk = V0 * np.eye(nk).reshape(nk, nk, 1, 1)
        V_grid = V_kk.reshape(*grid_shape, *grid_shape, norb, norb)
        T_R = np.fft.ifftn(np.fft.fftn(V_grid, axes=(0, 1, 2)), axes=(3, 4, 5))

        subcells = enumerate_subcells(M)
        R_sc = np.array([[0, 0, 0]])

        dH = _fold_scattering(T_R, R_sc, subcells, M, grid_arr, norb)
        expected = V0 * np.eye(2).reshape(1, 2, 2)
        np.testing.assert_allclose(dH, expected, atol=1e-12)


# ------------------------------------------------------------------
# fold_system (integration)
# ------------------------------------------------------------------


class TestFoldSystem:
    def test_identity(self):
        iRvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        nwann = 2
        ham = np.zeros((3, nwann, nwann), dtype=complex)
        ham[0] = np.diag([1.0, 2.0])
        ham[1] = np.array([[0.1, 0.2], [0.3, 0.4]])
        ham[2] = ham[1].conj().T

        lattice = np.array([[3.0, 0, 0], [0, 4.0, 0], [0, 0, 20.0]])
        wc = np.array([[0.0, 0.0, 0.0], [1.5, 2.0, 0.0]])
        system = _make_system(iRvec, ham, wc, lattice)

        sc = fold_system(system, np.eye(3, dtype=int))
        assert sc.num_wann == nwann
        E_prim = np.sort(np.linalg.eigvalsh(ham.sum(axis=0)))
        E_sc = np.sort(np.linalg.eigvalsh(sc.get_R_mat("Ham").sum(axis=0)))
        np.testing.assert_allclose(E_sc, E_prim, atol=1e-10)

    def test_2x1x1_eigenvalues(self):
        """1D chain folded into 2x supercell."""
        t = -0.5
        iRvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        ham = np.zeros((3, 1, 1), dtype=complex)
        ham[1, 0, 0] = t
        ham[2, 0, 0] = t

        lattice = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 20.0]])
        wc = np.array([[0.0, 0.0, 0.0]])
        system = _make_system(iRvec, ham, wc, lattice)

        sc = fold_system(system, np.diag([2, 1, 1]))
        assert sc.num_wann == 2
        E = np.sort(np.linalg.eigvalsh(sc.get_R_mat("Ham").sum(axis=0)))
        np.testing.assert_allclose(E, [2 * t, -2 * t], atol=1e-10)

    def test_all_matrices_folded(self):
        iRvec = np.array([[0, 0, 0]])
        nwann = 2
        ham = np.eye(nwann, dtype=complex).reshape(1, nwann, nwann)
        AA = np.zeros((1, nwann, nwann, 3), dtype=complex)
        AA[0, 0, 0, :] = [0.1, 0.2, 0.0]
        AA[0, 1, 1, :] = [0.3, 0.4, 0.0]

        lattice = np.eye(3) * 3.0
        wc = np.zeros((nwann, 3))
        system = _make_system(iRvec, ham, wc, lattice, extra_matrices={"AA": AA})

        sc = fold_system(system, np.diag([2, 1, 1]))
        assert sc.has_R_mat("Ham")
        assert sc.has_R_mat("AA")
        assert sc.get_R_mat("AA").shape[1:] == (sc.num_wann, sc.num_wann, 3)

    def test_wannier_centres_tiled(self):
        iRvec = np.array([[0, 0, 0]])
        ham = np.ones((1, 1, 1), dtype=complex)
        lattice = np.array([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 20.0]])
        wc = np.array([[0.5, 0.5, 0.0]])
        system = _make_system(iRvec, ham, wc, lattice)

        sc = fold_system(system, np.diag([2, 1, 1]))
        wc_sc = sc.wannier_centers_cart
        assert wc_sc.shape == (2, 3)
        np.testing.assert_allclose(wc_sc[0], [0.5, 0.5, 0.0], atol=1e-10)
        np.testing.assert_allclose(wc_sc[1], [2.5, 0.5, 0.0], atol=1e-10)

    def test_supercell_lattice(self):
        iRvec = np.array([[0, 0, 0]])
        ham = np.ones((1, 1, 1), dtype=complex)
        lattice = np.array([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 20.0]])
        wc = np.zeros((1, 3))
        system = _make_system(iRvec, ham, wc, lattice)

        M = np.diag([2, 3, 1])
        sc = fold_system(system, M)
        np.testing.assert_allclose(sc.real_lattice, M @ lattice, atol=1e-10)

    def test_from_supercell_classmethod(self):
        """System_R.from_supercell should give the same result."""
        iRvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        ham = np.zeros((3, 1, 1), dtype=complex)
        ham[0, 0, 0] = 1.0
        ham[1, 0, 0] = -0.3
        ham[2, 0, 0] = -0.3

        lattice = np.eye(3)
        wc = np.zeros((1, 3))
        system = _make_system(iRvec, ham, wc, lattice)

        M = np.diag([2, 1, 1])
        sc1 = fold_system(system, M)
        sc2 = System_R.from_supercell(system, M)

        np.testing.assert_allclose(
            sc2.get_R_mat("Ham"), sc1.get_R_mat("Ham"), atol=1e-12
        )


# ------------------------------------------------------------------
# add_scattering (integration)
# ------------------------------------------------------------------


class TestAddScattering:
    def test_adds_to_ham(self):
        iRvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        ham = np.zeros((3, 1, 1), dtype=complex)
        ham[0, 0, 0] = 1.0
        ham[1, 0, 0] = -0.3
        ham[2, 0, 0] = -0.3

        lattice = np.eye(3)
        wc = np.zeros((1, 3))
        system = _make_system(iRvec, ham, wc, lattice)

        M = np.diag([2, 1, 1])
        sc = fold_system(system, M)
        H_before = sc.get_R_mat("Ham").copy()

        V0 = 0.1
        V_kk = V0 * np.eye(2).reshape(2, 2, 1, 1)
        add_scattering(sc, V_kk, (2, 1, 1), M)

        H_after = sc.get_R_mat("Ham")
        dH = H_after - H_before
        iR0 = next(i for i, r in enumerate(sc.rvec.iRvec) if np.all(r == 0))
        for s in range(2):
            assert abs(dH[iR0, s, s] - V0) < 1e-10

    def test_validation_bad_vkk_shape(self):
        iRvec = np.array([[0, 0, 0]])
        ham = np.ones((1, 1, 1), dtype=complex)
        system = _make_system(iRvec, ham, np.zeros((1, 3)), np.eye(3))
        sc = fold_system(system, np.eye(3, dtype=int))

        with pytest.raises(ValueError, match="nk, nk, norb, norb"):
            add_scattering(sc, np.zeros((1, 1, 1)), (1, 1, 1), np.eye(3, dtype=int))

    def test_validation_grid_mismatch(self):
        iRvec = np.array([[0, 0, 0]])
        ham = np.ones((1, 1, 1), dtype=complex)
        system = _make_system(iRvec, ham, np.zeros((1, 3)), np.eye(3))
        sc = fold_system(system, np.eye(3, dtype=int))

        with pytest.raises(ValueError, match="k-points"):
            add_scattering(sc, np.zeros((2, 2, 1, 1)), (1, 1, 1), np.eye(3, dtype=int))

    def test_validation_grid_shape_length(self):
        iRvec = np.array([[0, 0, 0]])
        ham = np.ones((1, 1, 1), dtype=complex)
        system = _make_system(iRvec, ham, np.zeros((1, 3)), np.eye(3))
        sc = fold_system(system, np.eye(3, dtype=int))

        with pytest.raises(ValueError, match="length 3"):
            add_scattering(sc, np.zeros((1, 1, 1, 1)), (1, 1), np.eye(3, dtype=int))
