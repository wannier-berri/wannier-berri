"""Tests for wannierberri.system.system_supercell — fold_system."""

import numpy as np

from wannierberri.fourier.rvectors import Rvectors
from wannierberri.system.system_R import System_R
from wannierberri.system.system_supercell import enumerate_subcells, fold_system


def _make_system(iRvec, ham, wc_cart, real_lattice):
    """Create a minimal System_R for testing."""
    system = System_R(periodic=(True, True, False), silent=True)
    system.real_lattice = np.array(real_lattice)
    system.num_wann = ham.shape[1]
    system.wannier_centers_cart = np.array(wc_cart)
    system.rvec = Rvectors(lattice=system.real_lattice, iRvec=np.array(iRvec))
    system.set_R_mat("Ham", ham)
    return system


class TestGrapheneSqrt3:
    """Fold a tight-binding graphene model into the √3×√3 R30 supercell."""

    @staticmethod
    def _graphene_primitive(t=1.0):
        a1 = np.array([1.0, 0.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3) / 2, 0.0])
        a3 = np.array([0.0, 0.0, 10.0])
        lattice = np.array([a1, a2, a3])

        # Sites A=(0,0,0), B=(1/3,1/3,0) in reduced coords
        wc = np.array([[0.0, 0.0, 0.0], [1.0 / 3, 1.0 / 3, 0.0]]) @ lattice

        # Nearest neighbors of A live in cells R = (0,0,0), (-1,0,0), (0,-1,0)
        iRvec = np.array([
            [0, 0, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        nR = len(iRvec)
        ham = np.zeros((nR, 2, 2), dtype=complex)
        # H_AB(R) = -t for R in {0, -a1, -a2}
        for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
            i = next(k for k, r in enumerate(iRvec) if tuple(r) == R)
            ham[i, 0, 1] = -t
        # Hermitian conjugate: H_BA(-R) = -t*
        for R in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
            i = next(k for k, r in enumerate(iRvec) if tuple(r) == R)
            ham[i, 1, 0] = -t

        return _make_system(iRvec, ham, wc, lattice)

    def _ham_k(self, system, k_red):
        H = system.get_R_mat("Ham")
        iR = system.rvec.iRvec
        phase = np.exp(2j * np.pi * iR @ k_red)
        return np.einsum("R,Rij->ij", phase, H)

    def test_dirac_point_folds_to_sc_gamma(self):
        sys_prim = self._graphene_primitive(t=1.0)
        # √3×√3 R30: M = [[2,1,0],[-1,1,0],[0,0,1]], det = 3
        M = np.array([[2, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=int)
        sc = fold_system(sys_prim, M)

        assert sc.num_wann == 6

        # Primitive k-points folding to sc-Γ are {Γ, K, K'}
        H_sc_gamma = self._ham_k(sc, np.zeros(3))
        eig_sc = np.sort(np.linalg.eigvalsh(H_sc_gamma))

        # Primitive k folding to sc-Γ: k_prim = M^{-T} g, g ∈ Z^3 / M^T Z^3
        MinvT = np.linalg.inv(M.T.astype(float))
        prim_eig = []
        for g in enumerate_subcells(M.T):
            k_prim = MinvT @ g
            prim_eig.extend(np.linalg.eigvalsh(self._ham_k(sys_prim, k_prim)))
        prim_eig = np.sort(prim_eig)

        np.testing.assert_allclose(eig_sc, prim_eig, atol=1e-10)
        # Sanity: spectrum is {-3, 0, 0, 0, 0, 3}
        np.testing.assert_allclose(eig_sc, [-3, 0, 0, 0, 0, 3], atol=1e-10)

    def test_random_k_unfolding(self):
        sys_prim = self._graphene_primitive(t=1.0)
        M = np.array([[2, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=int)
        sc = fold_system(sys_prim, M)

        # Pick an arbitrary supercell k; primitive k_prim = M^T k_sc + G_i
        k_sc = np.array([0.13, 0.27, 0.0])
        eig_sc = np.sort(np.linalg.eigvalsh(self._ham_k(sc, k_sc)))

        # k_prim_red = M^{-T} (k_sc_red + g), g ∈ Z^3 / M^T Z^3
        MinvT = np.linalg.inv(M.T.astype(float))
        prim_eig = []
        for g in enumerate_subcells(M.T):
            k_prim = MinvT @ (k_sc + g)
            prim_eig.extend(np.linalg.eigvalsh(self._ham_k(sys_prim, k_prim)))
        prim_eig = np.sort(prim_eig)

        np.testing.assert_allclose(eig_sc, prim_eig, atol=1e-10)
