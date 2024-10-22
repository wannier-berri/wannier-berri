from functools import lru_cache
import warnings
import numpy as np
from .utility import orthogonalize


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
    n_iter : int
        The maximum number of iterations for the symmetrization of U and Z
    epsilon : float
        The convergence criterion for the symmetrization of U and Z

    Attributes
    ----------
    free : list of NKirr np.ndarray(dtype=bool, shape=(NB,))
    n_iter : int
    epsilon : float
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
                 free=None,
                 n_iter=20, epsilon=1e-8):
        if free is None:
            self.free = np.ones((Dmn.NK, Dmn.NB), dtype=bool)
        else:
            self.free = free
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.Dmn = Dmn
        self.kptirr = Dmn.kptirr
        self.NK = Dmn.NK
        self.NKirr = Dmn.NKirr
        self.kptirr2kpt = Dmn.kptirr2kpt
        self.kpt2kptirr = Dmn.kpt2kptirr
        self.Nsym = Dmn.Nsym
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


    def symmetrize_U(self, U, to_full_BZ=True, all_k=False):
        """
        Symmetrizes the umat matrix (in-place) at irreducible kpoints
        and treturns the U matrices at the full BZ

        Parameters
        ----------
        U : list of NKirr np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The input matrix to be symmetrized
        to_full_BZ : bool
            If True, the U matrices are expanded to the full BZ in the return
        all_k : bool
            If True, the U matrices are symmetrized at all reducible kpoints (self.include_k is ignored)

        Returns
        -------
        U : list of NKirr(if to_full_BZ=False) or  np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The symmetrized matrix.
        """
        for ikirr in range(self.Dmn.NKirr):
            U[ikirr][:] = self.symmetrize_U_kirr(U[ikirr], ikirr)
        if to_full_BZ:
            U = self.U_to_full_BZ(U, all_k=all_k)
        return U

    def U_to_full_BZ(self, U, all_k=False):
        """
        Expands the U matrix from the irreducible to the full BZ

        Parameters
        ----------
        U : list of NKirr np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The input matrix to be expanded
        all_k : bool
            If True, the U matrices are expanded at all reducible kpoints (self.include_k is ignored)

        Returns
        -------
        U : list of NK np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The expanded matrix.
        """
        Ufull = [None for _ in range(self.NK)]
        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                iRk = self.Dmn.kptirr2kpt[ikirr, isym]
                if Ufull[iRk] is None and (self.include_k[iRk] or all_k):
                    Ufull[iRk] = self.Dmn.rotate_U(U[ikirr], ikirr, isym, forward=True)
        return Ufull

    def symmetrize_Zk(self, Z, ikirr):
        # return Z # temporary for testing
        if Z.shape[0] == 0:
            return Z
        for i in range(self.n_iter):
            Zsym = sum(self.Dmn.rotate_Z(Z, isym, ikirr, self.free[ikirr])
                       for isym in self.Dmn.isym_little[ikirr]) / len(self.Dmn.isym_little[ikirr])
            diff = np.max(abs(Zsym - Z))
            if diff < self.epsilon:
                break
            Z[:] = Zsym
        Z[:] = Zsym
        return Z

    def symmetrize_Z(self, Z):
        """
        symmetrizes Z in-place
        Z(k) <- \sum_{R} d^{+}(R,k) Z(Rk) d(R,k)
        """
        for ikirr, _ in enumerate(Z):
            self.symmetrize_Zk(Z[ikirr], ikirr)
        return Z


    def symmetrize_U_kirr(self, U, ikirr):
        """
        Symmetrizes the umat matrix at the irreducible kpoint

        Parameters:
        - U: The matrix to be symmetrized.
        - Dmn : The DMN object
        - ir: The index of the irreducible kpoint (in the list of irreducible kpoints in dmn)

        Returns:
        - U: The symmetrized matrix.
        """

        Dmn = self.Dmn
        isym_little = Dmn.isym_little[ikirr]
        nsym_little = len(isym_little)
        ikpt = Dmn.kptirr[ikirr]
        nb, nw = U.shape

        for i in range(self.n_iter):
            Usym = sum(Dmn.rotate_U(U, ikirr, isym, forward=True) for isym in isym_little) / nsym_little
            Usym = orthogonalize(Usym)
            diff = np.eye(nw) - U.conj().T @ Usym
            diff = np.sum(np.abs(diff))
            if diff < self.epsilon:
                break
            U = Usym
        else:
            warnings.warn(f'symmetrization of U matrix at irreducible point {ikirr} ({ikpt})' +
                        f' did not converge after {self.n_iter} iterations, diff={diff}' +
                        'Either eps is too small or specified irreps is not compatible with the bands' +
                        f'diff{diff}, eps={self.epsilon}')
        return orthogonalize(Usym)




class VoidSymmetrizer(Symmetrizer):

    """
    A fake symmetrizer that does nothing
    Just to be able to use the same with and without site-symmetry
    """

    def __init__(self, NK):
        self.NKirr = NK
        self.NK = NK
        self.kptirr = np.arange(NK)
        self.kptirr2kpt = self.kptirr[:, None]
        self.kpt2kptirr = np.arange(NK)
        self.Nsym = 1

    def symmetrize_U(self, U, **kwargs):
        return U

    def symmetrize_U_kirr(self, U, ikirr):
        return U

    def symmetrize_Z(self, Z):
        return Z

    def symmetrize_Zk(self, Z, ikirr):
        return Z

    def U_to_full_BZ(self, U):
        return U
