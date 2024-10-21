
from functools import lru_cache
import sys
import warnings
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
            print("wannier centers from spread functional: \n", wcc1)

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
    sys.stdout.flush()
    return delta_max


def select_window_degen(E, thresh=DEGEN_THRESH, win_min=np.inf, win_max=-np.inf,
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


def find_solution_mod1(A, B, max_shift=2):
    """
    Find a solution such that A@x = B mod 1

    Parameters
    ----------
    A : np.ndarray (n,m)
        The matrix of the system.   
    B : np.ndarray (n,)
        The right hand side.
    max_shift : int
        The maximum shift.

    Returns
    -------
    list of np.ndarray
        The shifts that are compatible with the system.
    """
    A = np.array(A)
    B = np.array(B)
    r1 = np.linalg.matrix_rank(A)
    assert (r1 == A.shape[1]), f"overdetermined system {r1} != {A.shape}[1]"
    dim = A.shape[0]
    for shift in get_shifts(max_shift, ndim=dim):
        B_loc = B + shift
        if np.linalg.matrix_rank(np.hstack([A, B_loc[:, None]])) == r1:
            x, residuals, rank, s = np.linalg.lstsq(A, B_loc, rcond=None)
            if len(residuals) > 0:
                assert np.max(np.abs(residuals)) < 1e-7
            assert rank == r1
            return x
    return None


@lru_cache
def get_shifts(max_shift, ndim=3):
    """return all possible shifts of a 3-component vector with integer components
    recursively by number of dimensions

    Parameters
    ----------
    max_shift : int
        The maximum absolute value of the shift.
    ndim : int
        The number of dimensions.

    Returns
    -------
    array_like(n, ndim)
        The shifts. n=(max_shift*2+1)**ndim
        sorted by the norm of the shift
    """
    if ndim == 1:
        shifts = np.arange(-max_shift, max_shift + 1)[:, None]
    else:
        shift_1 = get_shifts(max_shift, ndim - 1)
        shift1 = get_shifts(max_shift, 1)
        shifts = np.vstack([np.hstack([shift_1, [s1] * shift_1.shape[0]]) for s1 in shift1])
    # more probably that equality happens at smaller shift, so sort by norm
    srt = np.linalg.norm(shifts, axis=1).argsort()
    return shifts[srt]


def find_distance_periodic(positions, real_lattice, max_shift=2):
    """
    find the distances between the pairs of atoms in a list of positions
    the distance to the closest image in the periodic lattice is returned

    Parameters
    ----------
    positions : np.ndarray( (num_atoms,3), dtype=float)
        The list of atomic positions in reduced coordinates.
    real_lattice : np.ndarray((3,3), dtype=float)
        The lattice vectors.

    Returns
    -------
    np.ndarray( (num_atoms,num_atoms), dtype=float)
        The distance between the pairs atoms.
    """
    if len(positions) == 0:
        return np.array([[np.inf]])
    positions = np.array(positions) % 1
    shifts = get_shifts(max_shift)
    diff = positions[:, None, None, :] - positions[None, :, None, :] + shifts[None, None, :, :]
    metric = real_lattice @ real_lattice.T
    prod = np.einsum('ijla,ab,ijlb->ijl', diff, metric, diff)

    rng = np.arange(len(positions))
    prod[rng, rng, 0] = np.inf  # distance to itself is not interesting, so the distance to its nearest image is counted

    distances2 = np.min(prod, axis=2)
    return np.sqrt(distances2)
