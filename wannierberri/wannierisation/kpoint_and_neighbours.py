from copy import deepcopy
from functools import cached_property

import scipy
from ..utility import get_max_eig, orthogonalize
import numpy as np


class Kpoint_and_neighbours:
    """ a class to store the data on a single k-point

        Parameters
        ----------
        Mmn : numpy.ndarray(nnb,nb,nb)
            Mmn matrices
        frozen : numpy.ndarray(nb, dtype=bool)
            frozen bands at this k-point
        frozen_nb : list of numpy.ndarray(nnb,nb, dtype=bool)
            frozen bands at neighbours
        wb : numpy.ndarray(nnb, dtype=float)
            weights for each neighbour (b-vector)

        Attributes
        ----------
        data : dict((str,str),list of numpy.ndarray(nnb,nf,nf)
            the data for the Mmn matrix for each pair of subspaces (free/frozen)
        spaces : dict
            the spaces (free/frozen)
        neighbours : list of list of tuple
            list of neighbours for each k-point
        wk : list of numpy.ndarray(nnb)
            list of weights for each neighbour (b-vector)
        """

    def __init__(self, Mmn,
                 frozen, frozen_nb,
                 free, free_nb,
                 wb, bk,
                 ikirr,
                 symmetrizer_Zirr,
                 symmetrizer_Uirr,
                 amn,
                 weight=1,
                 ):
        nnb, nb = Mmn.shape[:2]
        self.nnb = nnb
        self.Mmn = Mmn
        assert Mmn.shape[2] == nb
        assert len(frozen) == nb
        assert len(free) == nb
        assert frozen_nb.shape == (nnb, nb), f"frozen_nb shape {frozen_nb.shape} does not match nnb {nnb} and nb {nb}"
        assert free_nb.shape == (nnb, nb), f"free_nb shape {free_nb.shape} does not match nnb {nnb} and nb {nb}"

        self.num_wann = amn.shape[1]
        self.nband = amn.shape[0]
        self.wb = wb
        self.bk = bk
        self.wbk = wb[:, None] * bk
        self.weight = weight

        self.data = {}
        self.frozen = frozen
        self.nfrozen = sum(frozen)
        self.free = free
        self.selected = frozen | free
        self.NBselected = sum(self.selected)
        self.amn_sel = amn[self.selected, :]
        self.free_nb = free_nb
        self.num_bands_free = sum(self.free)
        self.nWfree = self.num_wann - sum(frozen)
        self.NBfree = sum(self.free)
        self.spaces = {'free': self.free_nb, 'frozen': frozen_nb}
        self.freefree = [Mmn[ib][self.free, :][:, self.free_nb[ib]] for ib in range(nnb)]
        self.freefrozen = [Mmn[ib][self.free, :][:, frozen_nb[ib]] for ib in range(nnb)]
        self.symmmetrizer_Zirr = symmetrizer_Zirr
        self.symmetrizer_Uirr = symmetrizer_Uirr
        self.Zfrozen = self.calc_Z()
        self.Zold = None

        # initialize the U matrix with projections
        amn2 = amn[self.free, :].dot(amn[self.free, :].T.conj())
        self.U_opt_free = get_max_eig(amn2, self.nWfree, self.NBfree)  # nBfee x nWfree marrices
        self.U_opt_full = self.rotate_to_projections(self.U_opt_free)

    def get_U_opt_full(self):
        return self.U_opt_full

    @cached_property
    def sumwb(self):
        return sum(self.wb)

    def update(self, U_nb, wcc_bk_phase, localise=True, mix_ratio=1.0,
               localise_num_iter=10,
               localise_alpha=0.5,
               localise_conv_tol=1e-6,
               ):
        """
        update the Z matrix

        Parameters
        ----------
        U_nb : numpy.ndarray(nnb, nBfree,nWfree) or None
            the U matrix at neighbouring k-points

        Returns
        -------
        numpy.ndarray(NB, nW)
            the updated U matrix
        """
        assert 0 <= mix_ratio <= 1
        if not hasattr(self, 'U_localize'):
            self.U_localize = np.eye(self.num_wann, dtype=complex)
        self.U_nb = deepcopy(U_nb)
        U_nb_free = [self.U_nb[ib][f] for ib, f in enumerate(self.free_nb)]
        Z = self.calc_Z(U_nb_free) + self.Zfrozen
        if self.Zold is not None and mix_ratio != 1:
            Z = mix_ratio * Z + (1 - mix_ratio) * self.Zold
        self.Zold = Z
        self.U_opt_free = get_max_eig(Z, self.nWfree, self.num_bands_free)
        if localise:
            U_opt_full = np.zeros((self.nband, self.num_wann), dtype=complex)
            U_opt_full[self.frozen, range(self.nfrozen)] = 1.
            U_opt_full[self.free, self.nfrozen:] = self.U_opt_free
            U_opt_full = orthogonalize(U_opt_full)
            check = U_opt_full.T.conj().dot(U_opt_full) - np.eye(self.num_wann)
            assert np.allclose(check, 0, atol=1e-6), f"U_opt_full is not unitary : {check}"
            Mmn_loc = np.array([U_opt_full.T.conj() @ self.Mmn[ib].dot(self.U_nb[ib]) *
                                wcc_bk_phase[None, :, ib]
                                for ib in range(self.nnb)])
            Mmn_loc_sumb = sum(mm * wb for mm, wb in zip(Mmn_loc, self.wb)) / sum(self.wb)
            U = np.linalg.inv(Mmn_loc_sumb)
            U = U.T.conj()
            U = orthogonalize(U)
            Mmn_loc_rotated = np.einsum('ji, bjl->bil', U.conj(), Mmn_loc)
            r2_old = get_r2(wb=self.wb, Mmn_rotated=Mmn_loc_rotated, weight=self.weight)
            multiplier = localise_alpha / (2 * self.sumwb)
            for _ in range(localise_num_iter):
                grad = gradOmega(self.wb, Mmn_loc_rotated) * multiplier
                dU = scipy.linalg.expm(grad)
                Mmn_loc_rotated = np.einsum('ji, bjl->bil', dU.conj(), Mmn_loc_rotated)
                U = U.dot(dU)
                check_U = U.T.conj().dot(U) - np.eye(self.num_wann)
                assert np.allclose(check_U, 0, atol=1e-6), f"U is not unitary : {check_U}"
                r2 = get_r2(wb=self.wb, Mmn_rotated=Mmn_loc_rotated, weight=self.weight)
                if abs(r2.sum() - r2_old.sum()) < localise_conv_tol:
                    break
                r2_old = r2
            self.U_opt_full = orthogonalize(U_opt_full.dot(U))

        else:
            self.U_opt_full = self.rotate_to_projections(self.U_opt_free)

        self.U_opt_full = self.symmetrizer_Uirr(self.U_opt_full)
        self.update_Mmn_opt(wcc_bk_phase=wcc_bk_phase)
        return self.U_opt_full, self._wcc, self._r2


    def calc_Z(self, U_nb=None):
        r"""
        calculate the Z matrix for the given U matrix

        Z = \sum_{b} w_{b,k} M_{b,k} M_{b,k}^{\dagger}
        where M_{b,k} = M_{b,k}^{loc} U_{b,k}

        Parameters
        ----------
        U_nb : list of nnb matrices numpy.ndarray(nBfree,nWfree) or None
            the U matrix at neighbouring k-points

        Returns
        -------
        numpy.ndarray(nWfree,nWfree)
            the Z matrix
        """
        if U_nb is None:
            Mmn_loc_opt = self.freefrozen
        else:
            Mmn_loc_opt = [self.freefree[ib].dot(U_nb[ib]) for ib in range(len(self.wb))]
        Z = np.array(sum(wb * mmn.dot(mmn.T.conj()) for wb, mmn in zip(self.wb, Mmn_loc_opt)))
        self.symmmetrizer_Zirr(Z)
        return Z


    def rotate_to_projections(self, U_opt_free):
        """
        rotate the U matrix to the projections of the bands
        to better match the initial guess

        Parameters
        ----------
        U_opt_free : numpy.ndarray(nBfree,nW)
            the optimized U matrix for the free bands and wannier functions

        Returns
        -------
        numpy.ndarray(NB,nW)
            the rotated U matrix for full set of bands and WFs
        """
        U = np.zeros((self.nband, self.num_wann), dtype=complex)
        U[self.frozen, range(self.nfrozen)] = 1.
        U[self.free, self.nfrozen:] = U_opt_free
        U_loc = U[self.selected, :].copy()
        ZV = orthogonalize(U_loc.T.conj().dot(self.amn_sel))
        U[:] = 0
        U[self.selected] = U_loc.dot(ZV)
        return U

    def update_Mmn_opt(self, wcc_bk_phase):
        """
        update the Mmn matrix for the optimized U matrix
        """
        if self.U_opt_full is None or self.U_nb is None:
            return
        UT = self.U_opt_full.T.conj()
        self.Mmn_opt = np.array([UT @ mmn @ Ub for mmn, Ub in zip(self.Mmn, self.U_nb)])
        self._r2, self._wcc = get_r2(wb=self.wb,
                                     Mmn_rotated=self.Mmn_opt,
                                     wcc_bk_phase=wcc_bk_phase,
                                     weight=self.weight, wbk=self.wbk)

    def update_Unb(self, U_nb=None, wcc_bk_phase=None):
        """
        update the U matrix at neighbouring k-points

        Parameters
        ----------
        U_nb : list of nnb matrices numpy.ndarray(nBfree,nWfree)
            the U matrix at neighbouring k-points

        Returns
        -------
        numpy.ndarray(nW, 3)
            the contriibution of the k-point to the WCC
        numpy.ndarray(nW)
            the contribution of the k-point to the r2 (part of the spread)
        """
        if U_nb is not None:
            self.U_nb = U_nb
            self.update_Mmn_opt(wcc_bk_phase=wcc_bk_phase)
        return self._wcc, self._r2


def calA(B):
    """ (B^dagger - B)/2 from MV97"""
    return (B.T.conj() - B) / 2


def calS(B):
    """ (B^dagger + B)/(2i) from MV97"""
    return (B.T.conj() + B) / (2j)


def gradOmega(wb, Mbmn):
    num_wann = Mbmn.shape[1]
    rng = np.arange(num_wann)
    Mbnn = Mbmn[:, rng, rng]
    q_bn = np.angle(Mbnn)
    R = Mbmn * Mbnn[:, None, :].conj()
    Rtilde = Mbmn / Mbnn[:, None, :]
    Tbmn = Rtilde * q_bn[:, None, :]
    return 2 * sum(w * (-calA(R) - calS(T))  for w, T, R in zip(wb, Tbmn, R))


def get_r2(wb, Mmn_rotated, weight, wcc_bk_phase=None, wbk=None):
    num_wann = Mmn_rotated.shape[1]
    rng = np.arange(num_wann)
    Mmn_diag = Mmn_rotated[:, rng, rng]
    if wcc_bk_phase is not None:
        Mmn_diag = Mmn_diag * wcc_bk_phase.T[:, :]
    Mmn_diag_angle = np.angle(Mmn_diag)
    r2 = wb @ (1 - abs(Mmn_diag)**2 + Mmn_diag_angle ** 2) * weight
    if wbk is not None:
        wcc = -Mmn_diag_angle.T @ wbk * weight
        return r2, wcc
    else:
        return r2
