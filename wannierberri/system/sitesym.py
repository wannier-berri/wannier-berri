import warnings
import numpy as np
from scipy.linalg import svd
from numpy.linalg import LinAlgError

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
    U, s, VT = svd(u, full_matrices=True)
    return U @ VT

import numpy as np

def symmetrize_U_kirr(U, Dmn, ikirr, niter=100, epsilon=1e-8):
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
    nw,nb = U.shape
        
    for _ in range(niter):
        Usym = np.zeros(U.shape, dtype=complex)
        for isym in isym_little:
            Usym +=  Dmn.d_band[ikpt,isym,].conj().T @ U @ Dmn.D_wann[ikpt,isym]
        Usym /= nsym_little
        diff = np.eye(nw) -  U.conj().T @ Usym
        diff = np.sum(np.abs(diff))
        if diff < epsilon:
            break
    else:
        warnings.warn('Error in symmetrize_u: not converged'
                      'Error in symmetrize_u: not converged'
                      'Either eps is too small or specified irreps is not compatible with the bands'
                      'diff,eps={diff},{sitesym["symmetrize_eps"]}')
    return  orthogonalize(Usym)