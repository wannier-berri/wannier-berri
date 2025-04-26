#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file 'LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
# ------------------------------------------------------------

import os
from functools import lru_cache
import numpy as np
import warnings

from collections.abc import Iterable
import datetime



def time_now_iso():
    return datetime.datetime.now().isoformat()


alpha_A = np.array([1, 2, 0])
beta_A = np.array([2, 0, 1])
delta_f = np.eye(3)


def conjugate_basis(basis):
    return 2 * np.pi * np.linalg.inv(basis).T


def real_recip_lattice(real_lattice=None, recip_lattice=None):
    if recip_lattice is None:
        if real_lattice is None:
            warnings.warn("usually need to provide either with real or reciprocal lattice."
                          "If you only want to generate a random symmetric tensor - that it fine")
            return None, None
        else:
            recip_lattice = conjugate_basis(real_lattice)
    else:
        if real_lattice is not None:
            assert np.linalg.norm(
                np.array(real_lattice).dot(recip_lattice.T) / (2 * np.pi) -
                np.eye(3)) <= 1e-8, "real and reciprocal lattice do not match"
        else:
            real_lattice = conjugate_basis(recip_lattice)
    return np.array(real_lattice), np.array(recip_lattice)


def clear_cached(obj, properties=()):
    for attr in properties:
        if hasattr(obj, attr):
            delattr(obj, attr)


def str2bool(v):
    v1 = v.strip().lower()
    if v1 in ("f", "false", ".false."):
        return False
    elif v1 in ("t", "true", ".true."):
        return True
    else:
        raise ValueError(f"unrecognized value of bool parameter :`{v}`")




@lru_cache()
def get_head(n):
    if n <= 0:
        return ['  ']
    else:
        return [a + b for a in 'xyz' for b in get_head(n - 1)]


def iterate_nd(size, pm=False, start=None):
    a = -size[0] if pm else (0 if start is None else start[0])
    b = size[0] + 1 if pm else (size[0] if start is None else start[0] + size[0])
    if len(size) == 1:
        return np.array([(i,) for i in range(a, b)])
    else:
        return np.array([(i,) + tuple(j) for i in range(a, b) for j in iterate_nd(size[1:], pm=pm, start=(start[1:]  if start is not None else None))])



def iterate3dpm(size):
    assert len(size) == 3
    return iterate_nd(size, pm=True)



def find_degen(arr, degen_thresh):
    """ finds shells of 'almost same' values in array arr, and returns a list o[(b1,b2),...]"""
    A = np.where(arr[1:] - arr[:-1] > degen_thresh)[0] + 1
    A = [0, ] + list(A) + [len(arr)]
    return [(ib1, ib2) for ib1, ib2 in zip(A, A[1:])]


# def get_angle(sina, cosa):
#     """Get angle in radian from sin and cos."""
#     if abs(cosa) > 1.0:
#         cosa = np.round(cosa, decimals=1)
#     alpha = np.arccos(cosa)
#     if sina < 0.0:
#         alpha = 2.0 * np.pi - alpha
#     return alpha


def angle_vectors(vec1, vec2):
    cos = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    return np.arccos(cos)


def angle_vectors_deg(vec1, vec2):
    angle = angle_vectors(vec1, vec2)
    return int(round(angle / np.pi * 180))


# smearing functions
def Lorentzian(x, width):
    return 1.0 / (np.pi * width) * width ** 2 / (x ** 2 + width ** 2)


def Gaussian(x, width, adpt_smr):
    """
    Compute 1 / (np.sqrt(pi) * width) * exp(-(x / width) ** 2)
    If the exponent is less than -200, return 0.
    An unoptimized version is the following.
        def Gaussian(x, width, adpt_smr):
            return 1 / (np.sqrt(pi) * width) * np.exp(-np.minimum(200.0, (x / width) ** 2))
    """
    inds = abs(x) < width * np.sqrt(200.0)
    output = np.zeros(x.shape, dtype=float)
    if adpt_smr:
        # width is array
        width_tile = np.tile(width, (x.shape[0], 1, 1))
        output[inds] = 1.0 / (np.sqrt(np.pi) * width_tile[inds]) * np.exp(-(x[inds] / width_tile[inds]) ** 2)
    else:
        # width is number
        output[inds] = 1.0 / (np.sqrt(np.pi) * width) * np.exp(-(x[inds] / width) ** 2)
    return output


# auxillary function"
def FermiDirac(E, mu, kBT):
    """here E is a number, mu is an array"""
    if kBT == 0:
        return 1.0 * (E <= mu)
    else:
        res = np.zeros(mu.shape, dtype=float)
        res[mu > E + 30 * kBT] = 1.0
        res[mu < E - 30 * kBT] = 0.0
        sel = abs(mu - E) <= 30 * kBT
        res[sel] = 1.0 / (np.exp((E - mu[sel]) / kBT) + 1)
        return res


def one2three(nk):
    if nk is None:
        return None
    elif isinstance(nk, Iterable):
        assert len(nk) == 3
    else:
        nk = (nk,) * 3
    assert np.all([isinstance(n, (int, np.integer)) and n > 0 for n in nk])
    return np.array(nk)


def remove_file(filename):
    if filename is not None and os.path.exists(filename):
        os.remove(filename)


def vectorize(func, *args, kwargs={}, sum=False, to_array=False):
    """decorator to vectorize the function over the positional arguments

    TODO : make parallel

    Parameters
    ----------
    func : function
        the function to vectorize over all the arguments
    args : list
        list of arguments
    kwargs : dict
        keyword arguments
    to_array : bool
        if True, return the result as numpy array, otherwise as list
    sum : bool
        if True, sum the results (after transforming to numpy array, if to_array is True)

    Returns
    -------
    list or np.array
        list of results of the function applied to all the arguments
    """
    l = [len(a) for a in args]
    assert all([_ == l[0] for _ in l]), f"length of all arguments should be the same, got {l}"
    lst = [func(*a, **kwargs) for a in zip(*args)]
    if to_array:
        lst = np.array(lst)
    if sum:
        lst = sum(lst)
    return lst


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
    try:
        U, _, VT = np.linalg.svd(u, full_matrices=False)
        return U @ VT
    except np.linalg.LinAlgError as e:
        warnings.warn(f"SVD failed with error '{e}', using non-orthogonalized matrix")
        return u


def select_window_degen(E, thresh=1e-2, win_min=np.inf, win_max=-np.inf,
                        include_degen=False,
                        return_indices=False):
    """define the indices of the bands inside the window, making sure that degenerate bands were not split


    Parameters
    ----------
    E : numpy.ndarray(nb, dtype=float)
        the energies of the bands (sorted ascending)
    thresh : float
        the threshold for the degeneracy
    include_degen : bool
        if True, the degenerate bands are included in the window

    Returns
    -------
    numpy.ndarray(bool)
        the boolean array of the frozen bands  (True for frozen)
    """
    NB = len(E)
    ind = list(np.where((E <= win_max) * (E >= win_min))[0])
    if len(ind) == 0:
        if return_indices:
            return []
        else:
            return np.zeros(E.shape, dtype=bool)

    # The upper bound
    for i in range(ind[-1], NB - 1):
        if E[i + 1] - E[i] < thresh:
            if include_degen:
                ind[i + 1] = True
            else:
                ind[i] = False
                break
        else:
            break

    # The lower bound
    for i in range(ind[0], 1, -1):
        if E[i] - E[i - 1] < thresh:
            if include_degen:
                ind[i - 1] = True
            else:
                ind[i] = False
                break
        else:
            break
    if return_indices:
        return ind
    else:
        inside = np.zeros(E.shape, dtype=bool)
        inside[ind] = True
        return inside
