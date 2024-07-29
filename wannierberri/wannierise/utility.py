import sys
from typing import Any
import numpy as np
DEGEN_THRESH = 1e-2  # for safety - avoid splitting (almost) degenerate states between free/frozen  inner/outer subspaces  (probably too much)


def print_centers_and_spreads(w90data, U_opt_full_BZ,
                              spread_functional=None, spreads=None,
                              comment=None):
    """
    print the centers and spreads of the Wannier functions

    Parameters
    ----------
    w90data : Wannier90data
        the data (inputs of wannier90)
    U_opt_free_BZ : list of numpy.ndarray(nBfree,nW)
        the optimized U matrix for the free bands and wannier functions
    """
    if spreads is None:
        if spread_functional is not None:
            wcc1 = spread_functional.get_wcc(U_opt_full_BZ)
            spreads = spread_functional(U_opt_full_BZ, wcc=wcc1)
            print( "wannier centers from spread functional: \n", wcc1)

    w90data.chk.v_matrix = np.array(U_opt_full_BZ)
    w90data.chk._wannier_centers, w90data.chk._wannier_spreads = w90data.chk.get_wannier_centers(w90data.mmn, spreads=True)

    breakline = "-" * 100
    startline = "#" * 100
    endline = startline
    wcc, spread = w90data.chk._wannier_centers, w90data.chk._wannier_spreads
    print(startline)
    if comment is not None:
        print(comment)
        print(breakline)
    print("wannier centers and spreads")
    print(breakline)
    for wcc, spread in zip(wcc, spread):
        wcc = np.round(wcc, 6)
        print(f"{wcc[0]:16.12f}  {wcc[1]:16.12f}  {wcc[2]:16.12f}   |   {spread:16.12f}")
    if spreads is not None:
        print(breakline)
        print(" | ".join(f"{key} = {value:12.8f}" for key, value in spreads.items() if key.startswith("Omega")))
    print(endline)



def print_progress(i_iter, Omega_list, num_iter_converge, 
                   spread_functional=None, spreads=None,
                   w90data=None, U_opt_full_BZ=None):
    """
    print the progress of the disentanglement

    Parameters
    ----------
    Omega_I_list : list of float
        the list of the spread functional
    num_iter_converge : int
        the number of iterations to check the convergence

    Returns
    -------
    float
        the standard deviation of the spread functional over the last `num_iter_converge` iterations
    """

    if spreads is None:
        assert spread_functional is not None
        wcc = spread_functional.get_wcc(U_opt_full_BZ)
        spreads = spread_functional(U_opt_full_BZ, wcc=wcc)

    Omega_list.append(spreads["Omega_tot"])
    Omega = Omega_list[-1]
    if i_iter > 0:
        delta = f"{Omega - Omega_list[-2]:15.8e}"
    else:
        delta = "--"

    if len(Omega_list) > num_iter_converge:
        delta_max = np.abs(Omega_list[-num_iter_converge:] - np.mean(Omega_list[-num_iter_converge:])).max()
        delta_max_str = f"{delta_max:15.8e}"
        slope = (Omega_list[-1] - Omega_list[-num_iter_converge - 1]) / num_iter_converge
        slope_str = f"{slope:15.8e}"
    else:
        delta_max = np.Inf
        delta_max_str = "--"
        slope_str = "--"

    comment = f"iteration {i_iter:4d} Omega= {Omega:15.10f}  delta={delta}, max({num_iter_converge})={delta_max_str}, slope={slope_str}"
    print_centers_and_spreads(w90data, U_opt_full_BZ, spreads=spreads, comment=comment)

    return delta_max


def frozen_nondegen(E, thresh=DEGEN_THRESH, froz_min=np.inf, froz_max=-np.inf):
    """define the indices of the frozen bands, making sure that degenerate bands were not split
    (unfreeze the degenerate bands together)

    Parameters
    ----------
    E : numpy.ndarray(nb, dtype=float)
        the energies of the bands
    thresh : float
        the threshold for the degeneracy

    Returns
    -------
    numpy.ndarray(bool)
        the boolean array of the frozen bands  (True for frozen)
    """
    ind = list(np.where((E <= froz_max) * (E >= froz_min))[0])
    while len(ind) > 0 and ind[0] > 0 and E[ind[0]] - E[ind[0] - 1] < thresh:
        del ind[0]
    while len(ind) > 0 and ind[-1] < len(E) - 1 and E[ind[-1] + 1] - E[ind[-1]] < thresh:
        del ind[-1]
    froz = np.zeros(E.shape, dtype=bool)
    froz[ind] = True
    return froz


def get_max_eig(matrix, nvec, nBfree):
    """ return the nvec column-eigenvectors of matrix with maximal eigenvalues.

    Parameters
    ----------
    matrix : numpy.ndarray(n,n)
        list of matrices
    nvec : int
        number of eigenvectors to return
    nBfree : int
        number of free bands

    Returns
    -------
    numpy.ndarray(n,nvec)
        eigenvectors
    """
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] >= nvec, f"nvec={nvec}, matrix.shape={matrix.shape}"
    e, v = np.linalg.eigh(matrix)
    return v[:, np.argsort(e)[nBfree - nvec:nBfree]]


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
    U, _, VT = np.linalg.svd(u, full_matrices=False)
    return U @ VT


class unique_list(list):
        """	
        A list that only allows unique elements.
        uniqueness is determined by the == operator.
        Thus, non-hashable elements are also allowed.
        unlike set, the order of elements is preserved.
        """

        def __init__(self, iterator=[]):
            super().__init__()
            for x in iterator:
                self.append(x)

        def append(self, item):
            for i in self:
                if i == item:
                    break
            else:
                super().append(item)

        def index(self, value: Any, start = 0, stop = sys.maxsize) -> int:
            for i in range(start, stop):
                if self[i] == value:
                    return i
            raise ValueError(f"{value} not in list")
        
        def __contains__(self, item):
            for i in self:
                if i == item:
                    return True
            return False
        
        def remove(self, value: Any) -> None:
            for i in range(len(self)):
                if self[i] == value:
                    del self[i]
                    return
                
class unique_list_mod1(unique_list):

    def __init__(self, iterator=[], tol=1e-5):
        self.tol = tol
        self.appended_indices = []
        self.last_try_append = -1
        super().__init__(iterator)

    def append(self, item):
        self.last_try_append += 1
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                break
        else:
            list.append(self,item)
            self.appended_indices.append(self.last_try_append)

    def __contains__(self, item):
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                return True
        return False

    def index(self, value: Any, start = 0, stop = sys.maxsize) -> int:
        stop = min(stop, len(self))
        for i in range(start, stop):
            if all_close_mod1(self[i], value):
                return i
        raise ValueError(f"{value} not in list")


                
def all_close_mod1(a,b,tol=1e-5):
    """check if two vectors are equal modulo 1"""
    if  not np.shape(a)==() and not np.shape(b)==() and (np.shape(a) != np.shape(b)):
        return False
    diff = a-b
    return np.allclose(np.round(diff), diff , atol=tol)
