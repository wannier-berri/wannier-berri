from fractions import Fraction
import numpy as np


readstr = lambda F: "".join(c.decode('ascii') for c in F.read_record('c')).strip()


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
