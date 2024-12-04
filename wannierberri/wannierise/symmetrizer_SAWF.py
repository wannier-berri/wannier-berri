from functools import lru_cache
import numpy as np
from ..__utility import get_inverse_block, rotate_block_matrix, orthogonalize


class Symmetrizer:

    """	
    Class to symmetrize the U and Z matrices

    Parameters
    ----------
    Dmn : DMN object
    neighbours : list of NKirr np.ndarray(dtype=int, shape=(NNB,))
        The list of neighbours for each irreducible kpoint. 
        Used to determine the kpoints in full BZ that are neighbours to at least one irreducible kpoint.
        U_opt at other kpoints are not calculated 9set to None)
    free : list of NKirr np.ndarray(dtype=bool, shape=(NB,))
        The list of free bands at each kpoint (True if the band is free, False if it is frozen)
        the default is all bands are free

    Attributes
    ----------
    free : list of NKirr np.ndarray(dtype=bool, shape=(NB,))
    Dmn : DMN object
    kptirr : np.ndarray(dtype=int, shape=(NKirr,))
    NK : int
    NKirr : int
    kptirr2kpt : np.ndarray(dtype=int, shape=(NKirr,Nsym))
    kpt2kptirr : np.ndarray(dtype=int, shape=(NK,))
    Nsym : int
    include_k : np.ndarray(dtype=bool, shape=(NK,))
        marks which kpoints in the full Bz are needed for evaluation of the Z matrix
        - the irreducible points and their neighbours are included
    """

    def __init__(self, Dmn=None, neighbours=None,
                 symmetrize_Z=True
                 ):
        self.Dmn = Dmn
        self.kptirr = Dmn.kptirr
        self.NK = Dmn.NK
        self.NKirr = Dmn.NKirr
        self.kptirr2kpt = Dmn.kptirr2kpt
        self.kpt2kptirr = Dmn.kpt2kptirr
        self.Nsym = Dmn.Nsym
        self.symmetrize_Z = symmetrize_Z
        if neighbours is None:
            self.include_k = np.ones(self.NK, dtype=bool)
        else:
            self.include_k = np.zeros(self.NK, dtype=bool)
            for ik in self.kptirr:
                self.include_k[ik] = True
                self.include_k[neighbours[ik]] = True

    @lru_cache
    def ndegen(self, ikirr):
        return len(set(self.kptirr2kpt[ikirr]))


    def U_to_full_BZ(self, U, all_k=False):
        """
        Expands the U matrix from the irreducible to the full BZ

        Parameters
        ----------
        U : list of NKirr np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The input matrix to be expanded
        all_k : bool
            If True, the U matrices are expanded at all reducible kpoints (self.include_k is ignored)
            if False, the U matrices are expanded only at the irreducible kpoints and their neighbours,
            for the rest of the kpoints, the U matrices are set to None

        Returns
        -------
        U : list of NK np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The expanded matrix. if all_k is False, the U matrices at the kpoints not included in self.include_k are set to None
        """
        Ufull = [None for _ in range(self.NK)]
        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                iRk = self.Dmn.kptirr2kpt[ikirr, isym]
                if Ufull[iRk] is None and (self.include_k[iRk] or all_k):
                    Ufull[iRk] = self.Dmn.rotate_U(U[ikirr], ikirr, isym, forward=True)
        return Ufull

    def get_symmetrizer_Uirr(self, ikirr):
        return Symmetrizer_Uirr(self.Dmn, ikirr)

    def get_symmetrizer_Zirr(self, ikirr, free=None):
        if self.symmetrize_Z:
            if free is None:
                free = np.ones(self.Dmn.NB, dtype=bool)
            return Symmetrizer_Zirr(self.Dmn, ikirr, free=free)
        else:
            return VoidSymmetrizer()


class Symmetrizer_Uirr(Symmetrizer):

    def __init__(self, dmn, ikirr):
        self.ikirr = ikirr
        self.isym_little = dmn.isym_little[ikirr]
        self.nsym_little = len(self.isym_little)
        self.ikpt = dmn.kptirr[ikirr]
        self.d_indices = dmn.d_band_block_indices[ikirr]
        self.D_indices = dmn.D_wann_block_indices
        self.d_band_blocks = dmn.d_band_blocks[ikirr]
        self.D_wann_blocks_inverse = dmn.D_wann_blocks_inverse[ikirr]
        self.nb = dmn.NB
        self.num_wann = dmn.num_wann
        self.time_reversals = dmn.time_reversals



    def rotate_U(self, U, isym):
        # forward = not forward
        Uloc = U.copy()
        if self.time_reversals[isym]:
            Uloc = Uloc.conj()
        Uloc = rotate_block_matrix(Uloc,
                                   lblocks=self.d_band_blocks[isym],
                                   lindices=self.d_indices,
                                   rblocks=self.D_wann_blocks_inverse[isym],
                                   rindices=self.D_indices)
        return Uloc



    def __call__(self, U):
        Usym = sum(self.rotate_U(U, isym) for isym in self.isym_little) / self.nsym_little
        return orthogonalize(Usym)


class Symmetrizer_Zirr(Symmetrizer):

    def __init__(self, dmn, ikirr, free):
        self.ikirr = ikirr
        self.isym_little = dmn.isym_little[ikirr]
        self.nsym_little = len(self.isym_little)
        self.ikpt = dmn.kptirr[ikirr]
        self.nb = dmn.NB
        self.num_wann = dmn.num_wann
        self.time_reversals = dmn.time_reversals

        if free is not None:
            (
                d_band_block_indices_free,
                d_band_blocks_free
            ) = dmn.select_window(dmn.d_band_blocks[ikirr], dmn.d_band_block_indices[ikirr], free)
            d_band_blocks_free_inverse = get_inverse_block(d_band_blocks_free)

            self.lblocks = d_band_blocks_free_inverse
            self.rblocks = d_band_blocks_free
            self.indices = d_band_block_indices_free
        else:
            self.lblocks = dmn.d_band_blocks_inverse,
            self.rblocks = dmn.d_band_blocks
            self.indices = dmn.d_band_block_indices


    def __call__(self, Z):
        # return Z # temporary for testing
        if Z.shape[0] == 0:
            return Z
        else:
            Z_rotated = [self.rotate_Z(Z, isym) for isym in self.isym_little]
            Z[:] = sum(Z_rotated) / self.nsym_little
            return Z

    def rotate_Z(self, Z, isym):
        """
        Rotates the zmat matrix at the irreducible kpoint
        Z = d_band^+ @ Z @ d_band
        """
        Zloc = Z.copy()
        # if self.time_reversals[isym]:
        #     Zloc = Zloc.conj()
        Zloc = rotate_block_matrix(Zloc, lblocks=self.lblocks[isym],
                                 lindices=self.indices,
                                 rblocks=self.rblocks[isym],
                                 rindices=self.indices,
                                )
        if self.time_reversals[isym]:
            Zloc = Zloc.conj()

        return Zloc


class VoidSymmetrizer(Symmetrizer):

    """
    A fake symmetrizer that does nothing
    Just to be able to use the same with and without site-symmetry
    """

    def __init__(self, NK=1):
        self.NKirr = NK
        self.NK = NK
        self.kptirr = np.arange(NK)
        self.kptirr2kpt = self.kptirr[:, None]
        self.kpt2kptirr = np.arange(NK)
        self.Nsym = 1

    def symmetrize_U_kirr(self, U, ikirr):
        return np.copy(U)

    def symmetrize_Z(self, Z):
        return np.copy(Z)

    def symmetrize_Zk(self, Z, ikirr):
        return np.copy(Z)

    def U_to_full_BZ(self, U, all_k=False):
        return np.copy(U)

    def __call__(self, X):
        return np.copy(X)

    def get_symmetrizer_Uirr(self, ikirr):
        return VoidSymmetrizer()

    def get_symmetrizer_Zirr(self, ikirr):
        return VoidSymmetrizer()
