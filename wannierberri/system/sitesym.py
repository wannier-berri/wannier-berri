import warnings
import numpy as np
from scipy.linalg import svd

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


def symmetrize_U_kirr(U, Dmn, ikirr, n_iter=100, epsilon=1e-8):
    """
    Symmetrizes the umat matrix at the irreducible kpoint

    Parameters:
    - U: The matrix to be symmetrized.
    - Dmn : The DMN object
    - ir: The index of the irreducible kpoint (in the list of irreducible kpoints in dmn)
    
    Returns:
    - U: The symmetrized matrix.
    """
    
    isym_little = Dmn.isym_little[ikirr]
    nsym_little = len(isym_little)
    ikpt = Dmn.kptirr[ikirr]
    nb, nw = U.shape
    # print (f'kpoint {ikpt}, irreducible point {ikirr}, symmetrizing U matrix of shape {U.shape} nbands={nb}, nwann={nw}')
    # print (f'little group symmetries: {isym_little} ({nsym_little}) ')
    # print (f'U matrix at irreducible point {ikirr} ({ikpt})\n {np.round(U,4)}')
    
        
    for i in range(n_iter):
        # print (f'iteration {i}, vialation of symmetries:')
        # for isym in isym_little:
        #     Usym =  Dmn.rotate_U(U, ikirr, isym)
        #     print (f'isym={isym}, violation={np.sum(np.abs(Usym - U))}')
        Usym = sum(Dmn.rotate_U(U, ikirr, isym) for isym in isym_little)/nsym_little
        # print (f'U changed by {np.sum(np.abs(Usym - U))}')
        #Usym = orthogonalize(Usym)
        diff = np.eye(nw) -  U.conj().T @ Usym
        diff = np.sum(np.abs(diff))
        if diff < epsilon:
            # print (f'stmmetrization of U matrix at irreducible point {ikirr} ({ikpt}) converged after {i} iterations, diff={diff}')
            break
        U = Usym
    else:
        warnings.warn(f'symmetrization of U matrix at irreducible point {ikirr} ({ikpt})'+
                      f' did not converge after {n_iter} iterations, diff={diff}'+
                      'Either eps is too small or specified irreps is not compatible with the bands'+
                      f'diff{diff}, eps={epsilon}')
    # print (f'U symmetrized at irreducible point {ikirr} ({ikpt})\n {Usym}')
    # return Usym
    return  orthogonalize(Usym)

def symmetrize_U(U, Dmn, n_iter=100, epsilon=1e-8):
    """
    Symmetrizes the umat matrix at all kpoints
    First, the matrix is symmetrized at the irreducible kpoints, 
    then it is rotated to the other kpoints using the symmetry operations.
    """
    Usym = [None]*Dmn.NK
    for ikirr in range(Dmn.NKirr ):
        ik = Dmn.kptirr[ikirr]
        Usym[ik] = symmetrize_U_kirr(U[ik], Dmn, ikirr, n_iter=n_iter, epsilon=epsilon)
        for isym in range(Dmn.Nsym):
            iRk = Dmn.kptirr2kpt[ikirr,isym]
            if Usym[iRk] is None:
            # if iRk not in Dmn.kptirr:
                Usym[iRk] = Dmn.rotate_U(Usym[ik], ikirr, isym)
                # diff = np.sum(np.abs(Usym[iRk] - U[iRk]))
                # print (f'U rotated from irreducible point {ikirr} ({ik},) to {iRk}  changed by {diff}')
                # if diff > 1e-7:
                #     print (f' Uold = \n{np.round(U[iRk],3)}')
                #     print (f' Unew = \n{np.round(Usym[iRk],3)}')
    return Usym

def symmetrize_Z(Z, Dmn):
    """
    Z(k) <- \sum_{R} d^{+}(R,k) Z(Rk) d(R,k)
    """
    num_kpts = len(Z)
    Zsym = [None]*num_kpts
    for ikirr in range(Dmn.NKirr):
        ik = Dmn.kptirr[ikirr]
        Zsym[ik] = np.zeros(Z[ik].shape, dtype=complex)
        for isym in Dmn.isym_little[ikirr]:
            Zsym[ik] += Dmn.rotate_Z(Z[ik], isym, ikirr)
        Zsym[ik] /= len(Dmn.isym_little[ikirr])

        for isym in range(1, Dmn.Nsym):
            irk = Dmn.kptirr2kpt[ikirr, isym]
            if Zsym[irk] is None:
                Zsym[irk] = Dmn.rotate_Z(Zsym[ik], isym, ikirr)
    return Zsym
           