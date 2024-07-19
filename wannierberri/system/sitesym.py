import warnings
import numpy as np
from scipy.linalg import svd


class Symmetrizer:

    def __init__(self, Dmn=None, neighbours=None,
                 free=None,
                  n_iter=100, epsilon=1e-8):
        if free is None:
            self.free = np.ones(Dmn.NB, dtype=bool)
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
        for ikirr in range(self.Dmn.NKirr ):
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
        Ufull = [None]*self.NK
        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                iRk = self.Dmn.kptirr2kpt[ikirr,isym]
                if Ufull[iRk] is None and (self.include_k[iRk] or all_k):
                    Ufull[iRk] = self.Dmn.rotate_U(U[ikirr], ikirr, isym)
        return Ufull

    def symmetrize_Z(self, Z):
        """
        symmetrizes Z in-place
        Z(k) <- \sum_{R} d^{+}(R,k) Z(Rk) d(R,k)
        """
        for ikirr,z in enumerate(Z):
            # Zold=Z[ikirr].copy()
            for i in range(self.n_iter):
                Zsym = sum(self.Dmn.rotate_Z(Z[ikirr], isym, ikirr, self.free[ikirr]) 
                           for isym in self.Dmn.isym_little[ikirr])/len(self.Dmn.isym_little[ikirr])
                diff = np.max(abs(Zsym - Z[ikirr]))
                if diff < self.epsilon:
                    break
                Z[ikirr][:] = Zsym
            Z[ikirr][:] = Zsym
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
            Usym = sum(Dmn.rotate_U(U, ikirr, isym, forward=False) for isym in isym_little)/nsym_little
            diff = np.eye(nw) -  U.conj().T @ Usym
            diff = np.sum(np.abs(diff))
            if diff < self.epsilon:
                break
            U = Usym
        else:
            warnings.warn(f'symmetrization of U matrix at irreducible point {ikirr} ({ikpt})'+
                        f' did not converge after {self.n_iter} iterations, diff={diff}'+
                        'Either eps is too small or specified irreps is not compatible with the bands'+
                        f'diff{diff}, eps={self.epsilon}')
        return  orthogonalize(Usym)



class VoidSymmetrizer(Symmetrizer):
    def __init__(self, NK):
        self.NKirr = NK
        self.NK = NK
        self.kptirr = np.arange(NK)
        self.kptirr2kpt = self.kptirr[:,None]
        self.Nsym = 1

    def symmetrize_U(self, U):
        return U

    def symmetrize_Z(self, Z):
        return Z
    
    def U_to_full_BZ(self, U):
        return U

def orthogonalize(u):
    """
    Orthogonalizes the matrix u using Singular Value Decomposition (SVD).

    Parameters
    ----------
    u : np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
        The input matrix to be orthogonalized.

    Returns
    -------
    u : np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
        The orthogonalized matrix.
    """
    U, _, VT = svd(u, full_matrices=False)
    return U @ VT

import numpy as np


