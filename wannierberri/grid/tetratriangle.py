
import numpy as np
from numba import njit

from functools import lru_cache
from .triangle import weights_triangle
from .tetrahedron import weights_tetra, _add_principal_3d


@lru_cache
def ones(n):
    return np.ones(n)


# @njit
def get_borders(A, degen_thresh, degen_Kramers=False):
    borders = [0] + list(np.where((A[1:] - A[:-1]) > degen_thresh)[0] + 1) + [len(A)]
    if degen_Kramers:
        borders = [i for i in borders if i % 2 == 0]
    return [[ib1, ib2] for ib1, ib2 in zip(borders, borders[1:])]


# @njit
def get_bands_in_range(emin, emax, Eband, degen_thresh=-1, degen_Kramers=False, Ebandmin=None, Ebandmax=None, select_bands=None):
    if Ebandmin is None:
        Ebandmin = Eband
    if Ebandmax is None:
        Ebandmax = Eband
    bands = []
    for ib1, ib2 in get_borders(Eband, degen_thresh, degen_Kramers=degen_Kramers):
        if select_bands is not None:
            if set(range(ib1, ib2)).intersection(set(select_bands)) == set():
                continue
        if Ebandmax[ib1:ib2].max() >= emin and Ebandmin[ib1:ib2].min() <= emax:
            bands.append([ib1, ib2])
    return bands


# @njit
def get_bands_below_range(emin, Eband, Ebandmax=None):
    if Ebandmax is None:
        Ebandmax = Eband
    add = np.where((Ebandmax < emin))[0]
    if len(add) > 0:
        return add[-1] + 1
    else:
        return 0


# @njit
def get_bands_above_range(emax, Eband, Ebandmin=None):
    if Ebandmin is None:
        Ebandmin = Eband
    add = np.where((Ebandmin > emax))[0]
    if len(add) > 0:
        return add[0]
    else:
        return len(Eband)


@njit
def _edges_in_range(edges, emin, emax):
    """the slice lo:hi + 1 of the edges between which the cumulative weight of values in [emin, emax] changes: it is
    0 at the edges up to lo and 1 at the edges from hi on. Returns (lo, hi), with hi <= lo if nothing changes.
    The margin covers the shift of nearly equal values by `weights_tetra` (up to 3e-12)."""
    lo = max(np.searchsorted(edges, emin, side='right') - 1, 0)
    hi = min(np.searchsorted(edges, emax + 1e-10, side='left'), edges.shape[0] - 1)
    return lo, hi


@njit
def _omega_in_range(omega, emin, emax):
    """the slice lo:hi of the increasing values omega within [emin, emax]"""
    return np.searchsorted(omega, emin, side='left'), np.searchsorted(omega, emax + 1e-10, side='right')


@njit
def omega_intervals(omega):
    """the edges of the intervals around the increasing values omega: at the midpoints between neighbouring values,
    and as far beyond the two ends, so that each value has one interval around it. For example, omega = 0, 0.25,
    0.5, 1 gives the edges -0.125, 0.125, 0.375, 0.75, 1.25. For a single value, the whole axis.

    Used for the averages over the intervals (tetra_delta_average=True), and for flat dE (see `_add_delta_values`)"""
    if omega.shape[0] < 2:
        return np.array([-np.inf, np.inf])
    edges = np.empty(omega.shape[0] + 1)
    edges[1:-1] = (omega[1:] + omega[:-1]) / 2
    edges[0] = 2 * omega[0] - edges[1]
    edges[-1] = 2 * omega[-1] - edges[-2]
    return edges


# a tetrahedron where the spread of dE is below this fraction of the interval of omega around it is taken as flat
FLAT_FRACTION = 1e-3


@njit
def _add_delta_values(res, omega, edges, weight, dE):
    """add weight * delta(dE - omega) over a tetrahedron (see `delta_tetra`) at the values omega to res. If dE is
    flat (its spread below FLAT_FRACTION of the interval of omega around it), its values cannot be sampled at
    the frequencies, and its weight is put into that interval, as with the averages over the intervals."""
    emin, emax = dE.min(), dE.max()
    i = np.searchsorted(edges, (emin + emax) / 2, side='right') - 1
    if 0 <= i < omega.shape[0] and np.isfinite(edges[i + 1] - edges[i]) and \
            emax - emin < FLAT_FRACTION * (edges[i + 1] - edges[i]):
        res[i] += weight / (edges[i + 1] - edges[i])
        return
    lo, hi = _omega_in_range(omega, emin, emax)
    if hi > lo:

        if dE.shape[0] == 4:
            res[lo:hi] += weight * weights_tetra(omega[lo:hi], dE[0], dE[1], dE[2], dE[3], der=1)
        elif dE.shape[0] == 3:
            res[lo:hi] += weight * weights_triangle(omega[lo:hi], dE[0], dE[1], dE[2], der=1)
        else:
            raise ValueError(f"dim={dE.shape[0] - 1} not supported")


@njit
def delta_values_tetra(omega, dE):
    """values of delta(dE - omega) over a tetrahedron (see `delta_tetra`, and `_add_delta_values` for a flat dE) at
    the given increasing values of omega

    omega : array (nomega,); dE : array (npair, 4); returns an array (npair, nomega)"""
    res = np.zeros((dE.shape[0], omega.shape[0]))
    edges = omega_intervals(omega)
    for i in range(dE.shape[0]):
        _add_delta_values(res[i], omega, edges, 1., dE[i])
    return res


@njit
def delta_tetra(omega, e0, e1, e2, e3=None):
    """delta(dE - omega) over a tetrahedron, where dE is linear with the values e0, ..., e3 at its corners, at each
    of the given values of omega: the density of the values of dE within the tetrahedron, normalised to 1, which is
    `weights_tetra` with der=1"""
    if e3 is None:
        return weights_triangle(omega, e0, e1, e2, der=1)
    else:
        return weights_tetra(omega, e0, e1, e2, e3, der=1)


@njit
def delta_weights_tetra(edges, dE):
    """the averages of delta(dE - omega) over a tetrahedron, as a function of omega, over the intervals of omega
    between the given edges, where dE is linear and given at its 4 corners: for each interval, the fraction of the
    tetrahedron where dE is within it, divided by its width. omega itself is not needed, only the edges. (With
    tetra_delta_average=True, the dynamic calculators take the edges from `omega_intervals`, and the average over
    each interval is the result at the frequency inside it.)

    edges : increasing array (nomega + 1,); dE : array (npair, 4); returns an array (npair, nomega). The weights times
    the widths of the intervals sum to the fraction of the tetrahedron where edges[0] < dE < edges[-1]"""
    width = edges[1:] - edges[:-1]
    res = np.zeros((dE.shape[0], width.shape[0]))
    if dE.shape[1] == 4:
        dim = 3
    elif dE.shape[1] == 3:
        dim = 2
    for i in range(dE.shape[0]):
        lo, hi = _edges_in_range(edges, dE[i].min(), dE[i].max())
        if hi <= lo:
            continue
        if dim == 3:
            occ = weights_tetra(edges[lo:hi + 1], dE[i, 0], dE[i, 1], dE[i, 2], dE[i, 3], 0)
        elif dim == 2:
            occ = weights_triangle(edges[lo:hi + 1], dE[i, 0], dE[i, 1], dE[i, 2], 0)
        res[i, lo:hi] = (occ[1:] - occ[:-1]) / width[lo:hi]
    return res


@njit
def reactive_values_tetra(omega, dE):
    """values of the average of P 1/(dE - omega) over a tetrahedron (see `weights_tetra_reactive`) at the given omega

    omega : array (nomega,); dE : array (npair, 4); returns an array (npair, nomega)"""
    res = np.zeros((dE.shape[0], omega.shape[0]))
    dim = dE.shape[1] - 1
    for i, dEi in enumerate(dE):
        if dim == 3:
            _add_principal_3d(res[i], omega, dEi[0], dEi[1], dEi[2], dEi[3])
        else:
            raise NotImplementedError(f"dim={dim} not supported")
    return res
