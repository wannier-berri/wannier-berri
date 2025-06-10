from fractions import Fraction
import warnings
import numpy as np

readstr = lambda F: "".join(c.decode('ascii') for c in F.read_record('c')).strip()


def is_round(A, prec=1e-14):
    # returns true if all values in A are integers, at least within machine precision
    return np.linalg.norm(A - np.round(A)) < prec


def readints(fl, n):
    lst = []
    while len(lst) < n:
        lst += fl.readline().split()
    assert len(lst) == n, f"expected {n} integers, got {len(lst)}"
    return np.array(lst, dtype=int)


def writeints(lst, perlinbe=10):
    """
    returns a string with integers separated by spaces
    each line has at most perline integers
    """
    n = len(lst)
    s = ""
    for i in range(0, n, perlinbe):
        s += " ".join(f"{x:4d}" for x in lst[i:i + perlinbe]) + "\n"
    return s


def get_mp_grid(kpoints):
    """
    Get the Monkhorst-Pack grid from the kpoints
    also check that all the kpoints are on the grid
    and no extra kpoints are present

    Parameters
    ----------
    kpoints : numpy.ndarray(float, shape=(NK, 3))
        the kpoints in reciprocal coordinates

    Returns
    -------
    tuple(int)
        the Monkhorst-Pack grid
    """
    kpoints = np.round(np.array(kpoints), 8) % 1
    assert kpoints.ndim == 2
    assert kpoints.shape[1] == 3
    mp_grid = np.array([None, None, None])
    for i in range(3):
        kfrac = [Fraction(k).limit_denominator(100) for k in kpoints[:, i]]
        kfrac = [k for k in kfrac if k != 0]
        if len(kfrac) == 0:
            mp_grid[i] = 1
        else:
            kmin = min(kfrac)
            assert kmin.numerator == 1, f"numerator of the smallest fraction is not 1 : {kmin}"
            mp_grid[i] = kmin.denominator
    k1 = np.array(kpoints * mp_grid[None, :], dtype=float)
    assert np.allclose(np.round(k1, 6) % 1, 0), (
        f"some kpoints are not on the Monkhorst-Pack grid {mp_grid}:\n {k1}")
    # assert kpoints.shape[0] == np.prod(mp_grid), "some kpoints are missing"
    return tuple(mp_grid)


def convert(A):
    """
    Convert a list of strings (numbers separated by spaces) 
    into a NumPy array of floats.

    Parameters:
    A (list): The list of strings to be converted.

    Returns:
    numpy.ndarray: The NumPy array of floats.
    """
    return np.array([l.split() for l in A], dtype=float)


def str2arraymmn(A):
    a = np.array([l.split()[3:] for l in A], dtype=float)
    return (a[:, 0] + 1j * a[:, 1])



def grid_from_kpoints(kpoints, grid=None):
    """
    Given a list of kpoints in fractional coordinates, return a the size of the grid in each direction
    if some k-points are repeated, they are counted only once
    if some k-points are missing, an error is raised

    Parameters
    ----------
    kpoints : np.array((nk, ndim), dtype=float)
        list of kpoints in fractional coordinates
    grid : tuple(int), optional
        size of the grid in each direction, used to select only points which belong to the grid.
        If None, the grid is calculated from the kpoints, and it is assumed that all kpoints are on the grid.

    Returns
    -------
    grid : tuple(int)
        size of the grid in each
    selected_kpoints : list of int
        indices of the selected kpoints

    Raises
    ------
    ValueError
        if some k-points are missing
    """
    if grid is None:
        grid = tuple(np.lcm.reduce([Fraction(k).limit_denominator(100).denominator for k in kp]) for kp in kpoints.T)
        returngrid = True
    else:
        returngrid = False
    npgrid = np.array(grid)
    print(f"mpgrid = {npgrid}, {len(kpoints)}")
    kpoints_unique = set()
    selected_kpoints = []
    for i, k in enumerate(kpoints):
        if is_round(k * npgrid, prec=1e-5):
            kint = tuple(np.round(k * npgrid).astype(int))
            if kint not in kpoints_unique:
                kpoints_unique.add(kint)
                selected_kpoints.append(i)
            else:
                warnings.warn(f"k-point {k} is repeated")

    num_selected = len(selected_kpoints)
    num_k_grid = np.prod(npgrid)
    if num_selected < num_k_grid:
        raise ValueError(f"Some k-points are missing {num_selected} < {num_k_grid}")
    if num_selected > num_k_grid:
        raise RuntimeError("Some k-points are taken twice - this must be a bug")
    if len(kpoints_unique) < len(kpoints):
        warnings.warn("Some k-points are not on the grid or are repeated")
    if returngrid:
        return grid
    else:
        return selected_kpoints
