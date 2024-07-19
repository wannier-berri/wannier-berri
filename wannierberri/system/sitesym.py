import warnings
import numpy as np
from scipy.linalg import svd

def orthogonalize(u):
    """
    Orthogonalizes the matrix u using Singular Value Decomposition (SVD).

    Parameters
    ----------
    u : np.ndarray(dtype=complex, shape = (m,n))
        The input matrix to be orthogonalized.

    Returns
    -------
    u : np.ndarray(dtype=complex, shape = (m,n))
        The orthogonalized matrix.
    """
    U, _, VT = svd(u, full_matrices=True)
    return U @ VT

import numpy as np

def rotate_U(U, Dmn, ikirr, isym):
    """
    Rotates the umat matrix at the irreducible kpoint
    U = D_band^+ @ U @ D_wann
    """
    return Dmn.d_band[ikirr,isym].conj().T @ U @ Dmn.D_wann_dag[ikirr,isym]

def rotate_Z(Z, Dmn, isym, ikirr):
    """
    Rotates the zmat matrix at the irreducible kpoint
    Z = d_band^+ @ Z @ d_band
    """
    return Dmn.d_band[ikirr,isym].conj().T @ Z @ Dmn.d_band[ikirr,isym]

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
    nw, nb = U.shape
    print (f'kpoint {ikpt}, irreducible point {ikirr}, symmetrizing U matrix of shape {U.shape}')
    print (f'little group symmetries: {isym_little} ({nsym_little}) ')
        
    for i in range(n_iter):
        Usym = np.zeros(U.shape, dtype=complex)
        for isym in isym_little:
            Usym +=  rotate_U(U, Dmn, ikirr, isym)
        Usym /= nsym_little
        diff = np.eye(nw) -  U.conj().T @ Usym
        diff = np.sum(np.abs(diff))
        if diff < epsilon:
            print (f'simmetrization of U matrix at irreducible point {ikirr} ({ikpt}) converged after {i} iterations, diff={diff}')
            break
    else:
        warnings.warn(f'simmetrization of U matrix at irreducible point {ikirr} ({ikpt})'
                      f' did not converge after {n_iter} iterations, diff={diff}'
                      'Either eps is too small or specified irreps is not compatible with the bands'
                      f'diff{diff}, eps={epsilon}')
    print (f'U symmetrized at irreducible point {ikirr} ({ikpt})\n {Usym}')
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
                Usym[iRk] = rotate_U(Usym[ik], Dmn, ikirr, isym)
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
            Zsym[ik] += rotate_Z(Z[ik], Dmn, isym, ikirr)
        Zsym[ik] /= len(Dmn.isym_little[ikirr])

        for isym in range(1, Dmn.Nsym):
            irk = Dmn.kptirr2kpt[ikirr, isym]
            if Zsym[irk] is None:
                Zsym[irk] = rotate_Z(Zsym[ik], Dmn, isym, ikirr)
    return Zsym
           