
import numpy as np
from numba import njit

from .triangle import weights_triangle, _add_principal_2d, occupied_triangles
from .tetrahedron import weights_tetra, _add_principal_3d, occupied_tetrahedra


@njit
def weights_general(efall, dE, der):
    if dE.shape[0] == 4:
        return weights_tetra(efall, dE[0], dE[1], dE[2], dE[3], der=der)
    elif dE.shape[0] == 3:
        return weights_triangle(efall, dE[0], dE[1], dE[2], der=der)


@njit
def _add_principal(res, omega, dE, weight=1):
    if dE.shape[0] == 4:
        _add_principal_3d(res, omega, e=dE, weight=weight)
    elif dE.shape[0] == 3:
        _add_principal_2d(res, omega, e=dE, weight=weight)
    else:
        raise NotImplementedError(f"dim={dE.shape[0] - 1} not supported")


@njit
def occupied_pieces(occupied, efermi):
    if occupied.shape[0] == 4:
        return occupied_tetrahedra(occupied, efermi)
    elif occupied.shape[0] == 3:
        return occupied_triangles(occupied, efermi)


def get_borders(A, degen_thresh, degen_Kramers=False):
    borders = [0] + list(np.where((A[1:] - A[:-1]) > degen_thresh)[0] + 1) + [len(A)]
    if degen_Kramers:
        borders = [i for i in borders if i % 2 == 0]
    return [[ib1, ib2] for ib1, ib2 in zip(borders, borders[1:])]


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


def get_bands_below_range(emin, Eband, Ebandmax=None):
    if Ebandmax is None:
        Ebandmax = Eband
    add = np.where((Ebandmax < emin))[0]
    if len(add) > 0:
        return add[-1] + 1
    else:
        return 0


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
        res[lo:hi] += weight * weights_general(omega[lo:hi], dE, der=1)


@njit
def delta_values_tetra(omega, dE):
    """values of delta(dE - omega) over a tetrahedron (see `delta_tetra`, and `_add_delta_values` for a flat dE) at
    the given increasing values of omega

    omega : array (nomega,); dE : array (npair, 4); returns an array (npair, nomega)"""
    res = np.zeros((dE.shape[0], omega.shape[0]))
    edges = omega_intervals(omega)
    for i in range(dE.shape[0]):
        _add_delta_values(res=res[i], omega=omega, edges=edges, weight=1., dE=dE[i])
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
    for i in range(dE.shape[0]):
        lo, hi = _edges_in_range(edges, dE[i].min(), dE[i].max())
        if hi <= lo:
            continue
        occ = weights_general(edges[lo:hi + 1], dE[i], der=0)
        res[i, lo:hi] = (occ[1:] - occ[:-1]) / width[lo:hi]
    return res


@njit
def reactive_values_tetra(omega, dE):
    """values of the average of P 1/(dE - omega) over a tetrahedron (see `weights_tetra_reactive`) at the given omega

    omega : array (nomega,); dE : array (npair, 4); returns an array (npair, nomega)"""
    res = np.zeros((dE.shape[0], omega.shape[0]))
    for i, dEi in enumerate(dE):
        _add_principal(res[i], omega, dEi)
    return res



@njit
def _occupation_state(energies, efermi):
    """1 if the band is occupied in the whole tetrahedron (as in `occupied_tetrahedra`), 0 if it is empty, 2 if the
    Fermi level cuts it"""
    if efermi >= energies.max():
        return 1
    if efermi <= energies.min():
        return 0
    return 2


@njit
def _add_kernel(res, omega, edges, intervals, kind, weight, dE):
    """add weight times a function of dE - omega over a tetrahedron, with the values d0, ..., d3 of dE at its corners,
    to res: delta(dE - omega) (kind 0: the values at omega, with the intervals around them for a flat dE; kind 1:
    the averages over the intervals between the edges of omega) or P 1/(dE - omega) (kind 2)"""
    if kind == 0:
        _add_delta_values(res=res, omega=omega, edges=intervals, weight=weight, dE=dE)
    elif kind == 1:
        res += weight * delta_weights_tetra(edges=edges, dE=dE.reshape(1, -1))[0]
    else:
        _add_principal(res=res, omega=omega, dE=dE, weight=weight)


@njit
def _add_occupied(res, occupied, dE, efermi, omega, edges, intervals, kind, weight):
    """add weight times the integral over a tetrahedron of theta(efermi - occupied) times the function of dE - omega
    given by `kind` (see `_add_kernel`) to res"""
    bary, volume = occupied_pieces(occupied, efermi)

    for t in range(3):
        if volume[t] > 0:
            # relative to a corner value, so that a constant dE stays exactly constant (P 1/(dE - omega) is zero for
            # a constant dE equal to omega, but large for values that differ by rounding)
            d = dE[0] + bary[t] @ (dE - dE[0])
            _add_kernel(res=res, omega=omega, edges=edges, intervals=intervals, kind=kind, weight=weight * volume[t], dE=d)


@njit
def occupation_kernel(Ei, Ej, sign, efermi, omega, edges, kind):
    """the integral over a tetrahedron of [theta(efermi - Ej) - theta(efermi - Ei)] times the function of
    sign * (Ei - Ej) - omega given by `kind` (see `occupied_kernel`)

    Ei, Ej : arrays (npair, 4), the band energies at the corners; returns an array (npair, nefermi, nomega).
    Only the bands that the Fermi level cuts within the tetrahedron are divided: a band occupied in the whole
    tetrahedron gives the integral over all of it, computed once for all the Fermi levels."""
    res = np.zeros((Ei.shape[0], efermi.shape[0], omega.shape[0]))
    intervals = omega_intervals(omega)
    whole = np.zeros(omega.shape[0])
    for i in range(Ei.shape[0]):
        dE = sign * (Ei[i] - Ej[i])
        have_whole = False
        for ief in range(efermi.shape[0]):
            ef = efermi[ief]
            state_i, state_j = _occupation_state(energies=Ei[i], efermi=ef), _occupation_state(energies=Ej[i], efermi=ef)
            if state_i == state_j and state_i != 2:  # both occupied or both empty
                continue
            for state, occupied, weight in ((state_j, Ej[i], 1.), (state_i, Ei[i], -1.)):
                if state == 1:
                    if not have_whole:
                        whole[:] = 0
                        _add_kernel(res=whole, omega=omega, edges=edges, intervals=intervals, kind=kind, weight=1., dE=dE)
                        have_whole = True
                    res[i, ief] += weight * whole
                elif state == 2:
                    _add_occupied(res=res[i, ief], occupied=occupied, dE=dE, efermi=ef, omega=omega, edges=edges, intervals=intervals, kind=kind, weight=weight)
    return res
