from collections import defaultdict
import numpy as np
from numba import njit

from functools import cached_property, lru_cache

from ..utility import weight_select_bands


@lru_cache
def ones(n):
    return np.ones(n)


@njit
def weights_tetra_der0(efall, e1, e2, e3, e4):
    occ = np.zeros_like(efall)
    denom4 = 1. / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    denom1 = 1. / ((e2 - e1) * (e3 - e1) * (e4 - e1))

    # symmetric expression, using the peak of der=1 at (e3 e4 - e1 e2) / S, here taken relative to e1 (center1):
    # S and the peak computed from the differences of the energies do not lose precision when the spread of the
    # energies is small compared with their values
    S = (e3 - e1) + (e4 - e2)
    center1 = (e3 - e1) * (e4 - e1) / S
    coef_middle = S / ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4 - e2))
    vol_middle = ((e2 - e1) * S + 2 * (e3 - e2) * (e4 - e2)) / S**2
    peak_height = 3 / S


    for i, ef in enumerate(efall):
        ef = efall[i]
        if ef >= e4:
            occ[i] = 1.
        elif ef < e1:
            continue
        elif ef >= e3:  # c3
            occ[i] = 1 - (e4 - ef) ** 3 * denom4
        elif ef >= e2:  # c2  # e2:e3
            y = (ef - e1) - center1
            occ[i] = vol_middle + y * (peak_height - y**2 * coef_middle)
        else:  # c1
            occ[i] = (ef - e1) ** 3 * denom1
    return occ


@njit
def weights_tetra_der1(efall, e1, e2, e3, e4):
    occ = np.zeros_like(efall)
    denom4 = 3. / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    denom1 = 3. / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    S = (e3 - e1) + (e4 - e2)
    center1 = (e3 - e1) * (e4 - e1) / S
    coef_middle = 3 * S / ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4 - e2))
    peak_height = 3 / S
    for i, ef in enumerate(efall):
        if ef <= e1 or ef >= e4:
            continue
        elif ef >= e3:  # c3
            occ[i] = (e4 - ef) ** 2 * denom4
        elif ef >= e2:  # c2  # e2:e3
            y = (ef - e1) - center1
            occ[i] = peak_height - y**2 * coef_middle
        else:  # c1
            occ[i] = (ef - e1) ** 2 * denom1
    return occ


@njit
def weights_tetra_der2(efall, e1, e2, e3, e4):
    occ = np.zeros_like(efall)
    denom4 = -6. / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    denom1 = 6. / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    S = (e3 - e1) + (e4 - e2)
    center1 = (e3 - e1) * (e4 - e1) / S
    coef_middle = -6 * S / ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4 - e2))
    for i, ef in enumerate(efall):
        if ef <= e1 or ef >= e4:
            continue
        elif ef >= e3:  # c3
            occ[i] = (e4 - ef) * denom4
        elif ef >= e2:  # c2  # e2:e3
            y = (ef - e1) - center1
            occ[i] = y * coef_middle
        else:  # c1
            occ[i] = (ef - e1) * denom1
    return occ


@njit
def weights_tetra_der3(efall, e1, e2, e3, e4):
    occ = np.zeros_like(efall)
    denom4 = 6. / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    denom1 = 6. / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    S = (e3 - e1) + (e4 - e2)
    coef_middle = -6 * S / ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4 - e2))
    for i, ef in enumerate(efall):
        if ef <= e1 or ef >= e4:
            continue
        elif ef >= e3:  # c3
            occ[i] = denom4
        elif ef >= e2:  # c2  # e2:e3
            occ[i] = coef_middle
        else:  # c1
            occ[i] = denom1
    return occ


@njit
def space_e1234(e1, e2, e3, e4, diff_min=1e-12):
    e = np.array(sorted([e1, e2, e3, e4]))
    # a dirty trick to avoid divisions by zero
    for i in range(3):
        e[i + 1] = max(e[i + 1], e[i] + diff_min)
    return e[0], e[1], e[2], e[3]


@njit
def weights_tetra(efall, e1, e2, e3, e4, der=0):
    """for a function linear within a tetrahedron, with the values e0, ..., e3 at its corners: the fraction of the
    tetrahedron where it is below each value of efall (der=0), and its derivatives with respect to efall (der=1 is
    the density of the values, normalised to 1; der=2, 3 its derivatives). The formulas of the linear tetrahedron
    method, see e.g. P. E. Bloechl, O. Jepsen and O. K. Andersen, Phys. Rev. B 49, 16223 (1994)."""

    e1, e2, e3, e4 = space_e1234(e1, e2, e3, e4, diff_min=1e-12)
    if der == 0:
        return weights_tetra_der0(efall, e1, e2, e3, e4)
    elif der == 1:
        return weights_tetra_der1(efall, e1, e2, e3, e4)
    elif der == 2:
        return weights_tetra_der2(efall, e1, e2, e3, e4)
    elif der == 3:
        return weights_tetra_der3(efall, e1, e2, e3, e4)
    else:
        raise NotImplementedError(f"der={der} not implemented in weights_tetra_accurate")


# the expansion of the principal part far from the values of dE: its order, and the largest ratio of the spread of
# the values to the distance of omega from their mean where it is used (the terms decrease by at least 3/4 of it)
PRINCIPAL_ORDER = 16
PRINCIPAL_FAR = 0.25


@njit
def _get_moments(e, mean):
    # the moments of x = dE - mean over the tetrahedron: <x^n> = 6 / ((n+1) (n+2) (n+3)) h_n(d), with h_n the
    # complete homogeneous symmetric polynomial of the deviations d of the corner values from the mean
    moments = np.zeros(PRINCIPAL_ORDER + 1)
    moments[0] = 1.
    for k in range(4):
        for l in range(1, PRINCIPAL_ORDER + 1):
            moments[l] += (e[k] - mean) * moments[l - 1]
    n = np.arange(PRINCIPAL_ORDER + 1)
    return 6. / ((n + 1) * (n + 2) * (n + 3)) * moments


@njit
def _get_c_matrix(e1, e2, e3, e4):
    e21, e31, e41, e32, e42, e43 = e2 - e1, e3 - e1, e4 - e1, e3 - e2, e4 - e2, e4 - e3
    c = np.zeros((3, 3))  # the density on each interval, sum_n c[k, n] * u ** n, with u = dE - e[k]
    c[0, 2] = 3 / (e21 * e31 * e41)
    c[1, 0] = 3 * e21 / (e31 * e41)
    c[1, 1] = 6 / (e31 * e41)
    c[1, 2] = -3 * (e31 + e42) / (e32 * e42 * e31 * e41)
    a = 3 / (e41 * e42 * e43)
    c[2, 0] = a * e43 ** 2
    c[2, 1] = -2 * a * e43
    c[2, 2] = a
    return c


@njit
def _get_polynomial(moments, y):
    value = 0.
    for n in range(PRINCIPAL_ORDER, -1, -1):
        value = value * y + moments[n]
    return value


@njit
def _add_principal(res, omega, e0, e1, e2, e3, weight=1.):
    """add weight times the average of P 1/(dE - omega) over a tetrahedron to res (see `weights_tetra_reactive`)"""
    e = np.array(sorted([float(e0), float(e1), float(e2), float(e3)]))
    shift = e[0]
    e = e - shift
    mean = np.sum(e) / 4
    spread = e[3] - e[0]

    moments = _get_moments(e, mean)
    gap = 1e-7 * spread
    for i in range(3):
        e[i + 1] = max(e[i + 1], e[i] + gap)
    e1, e2, e3, e4 = e

    if spread > 0:
        c = _get_c_matrix(e1, e2, e3, e4)

    for i in range(omega.shape[0]):
        D = mean - (omega[i] - shift)
        if spread <= PRINCIPAL_FAR * abs(D):
            if D != 0:  # with all values equal to omega, the principal part is zero
                # <1/(D + x)> = sum_n (-1)^n <x^n> / D^(n+1)
                res[i] += weight * _get_polynomial(moments, -1 / D) / D
        else:
            w0 = omega[i] - shift
            # the logarithms of the distances to the corners; at omega equal to a corner value, the two terms with it, of
            # the neighbouring intervals, cancel (the density is continuous there), so that it is left out
            logs = np.log(np.abs(w0 - e) + 1e-300)
            wk = w0 - e[:3]
            hk = e[1:] - e[:3]
            pwk = c[:, 0] + wk * (c[:, 1] + wk * c[:, 2])
            value = np.sum(hk * (c[:, 1] + c[:, 2] * (hk / 2 + wk)))
            for w, h, pw, log1, log2 in zip(wk, hk, pwk, logs[1:], logs[:-1]):
                if abs(h) < 0.5 * abs(w):  # omega far from the interval: log|1 - h/w| without cancellation
                    value += pw * np.log1p(-h / w)
                else:
                    # near the interval: the logarithms of the distances to the corners, which cancel exactly between
                    # neighbouring intervals at a corner value shared by them
                    value += pw * (log1 - log2)
            res[i] += weight * value


@njit
def weights_tetra_reactive(omega, e0, e1, e2, e3):
    """the average of P 1/(dE - omega) over a tetrahedron, where dE is linear with the values e0, ..., e3 at its
    corners, at each of the given values of omega (the principal part that goes with `delta_tetra`).

    It is the integral of the density of dE (piecewise quadratic, see `delta_tetra`) times 1/(dE - omega), done
    analytically on each interval between the sorted values. If omega is far from the values compared with their
    spread (more than 1/PRINCIPAL_FAR times), the expansion in the spread over the distance is used instead, to the
    order PRINCIPAL_ORDER, with the moments of dE over the tetrahedron; this also covers nearly equal values. Values
    closer than 1e-7 times the spread are separated. The values and omega are taken relative to the smallest value,
    so that this works also for a small spread far from zero. If three corners have the same value, the density
    jumps there, and the principal part diverges logarithmically as omega approaches it; at that value, the result
    is finite but set by the separation of the values."""
    res = np.zeros(omega.shape[0])
    _add_principal(res, omega, e0, e1, e2, e3)
    return res


@njit
def reactive_values_tetra(omega, dE):
    """values of the average of P 1/(dE - omega) over a tetrahedron (see `weights_tetra_reactive`) at the given omega

    omega : array (nomega,); dE : array (npair, 4); returns an array (npair, nomega)"""
    res = np.zeros((dE.shape[0], omega.shape[0]))
    for i, dEi in enumerate(dE):
        _add_principal(res[i], omega, dEi[0], dEi[1], dEi[2], dEi[3])
    return res


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
        occ = weights_tetra(edges[lo:hi + 1], dE[i, 0], dE[i, 1], dE[i, 2], dE[i, 3], 0)
        res[i, lo:hi] = (occ[1:] - occ[:-1]) / width[lo:hi]
    return res


@njit
def delta_tetra(omega, e0, e1, e2, e3):
    """delta(dE - omega) over a tetrahedron, where dE is linear with the values e0, ..., e3 at its corners, at each
    of the given values of omega: the density of the values of dE within the tetrahedron, normalised to 1, which is
    `weights_tetra` with der=1"""
    return weights_tetra(omega, e0, e1, e2, e3, der=1)


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
def _add_delta_values(res, omega, edges, weight, e0, e1, e2, e3):
    """add weight * delta(dE - omega) over a tetrahedron (see `delta_tetra`) at the values omega to res. If dE is
    flat (its spread below FLAT_FRACTION of the interval of omega around it), its values cannot be sampled at
    the frequencies, and its weight is put into that interval, as with the averages over the intervals."""
    emin, emax = min(e0, e1, e2, e3), max(e0, e1, e2, e3)
    i = np.searchsorted(edges, (emin + emax) / 2, side='right') - 1
    if 0 <= i < omega.shape[0] and np.isfinite(edges[i + 1] - edges[i]) and \
            emax - emin < FLAT_FRACTION * (edges[i + 1] - edges[i]):
        res[i] += weight / (edges[i + 1] - edges[i])
        return
    lo, hi = _omega_in_range(omega, emin, emax)
    if hi > lo:
        res[lo:hi] += weight * delta_tetra(omega[lo:hi], e0, e1, e2, e3)


@njit
def delta_values_tetra(omega, dE):
    """values of delta(dE - omega) over a tetrahedron (see `delta_tetra`, and `_add_delta_values` for a flat dE) at
    the given increasing values of omega

    omega : array (nomega,); dE : array (npair, 4); returns an array (npair, nomega)"""
    res = np.zeros((dE.shape[0], omega.shape[0]))
    edges = omega_intervals(omega)
    for i in range(dE.shape[0]):
        _add_delta_values(res[i], omega, edges, 1., dE[i, 0], dE[i, 1], dE[i, 2], dE[i, 3])
    return res


@njit
def occupied_tetrahedra(energies, efermi):
    """the part of a tetrahedron where a linear function is below efermi, divided into at most 3 tetrahedra

    energies : the values at the 4 corners. Returns (bary, volume): bary is an array (3, 4, 4), the barycentric
    coordinates of the corners of each tetrahedron (rows), and volume (3,) their volumes as fractions of the whole
    (zero for the unused ones)"""
    order = np.argsort(energies)
    a = energies[order]
    corner = np.zeros((4, 4))
    for k in range(4):
        corner[k, order[k]] = 1.

    def point(k, l):
        t = (efermi - a[k]) / (a[l] - a[k])
        return (1 - t) * corner[k] + t * corner[l]

    bary = np.zeros((3, 4, 4))
    n = 0
    if efermi >= a[3]:  # also a flat band at the Fermi level, as with the occupations at kBT = 0 (E <= mu)
        bary[0] = np.eye(4)
        n = 1
    elif efermi <= a[0]:
        pass
    elif efermi <= a[1]:
        bary[0, 0], bary[0, 1], bary[0, 2], bary[0, 3] = corner[0], point(0, 1), point(0, 2), point(0, 3)
        n = 1
    elif efermi <= a[2]:  # a prism, with the triangles (0, p02, p03) and (1, p12, p13) at its ends
        p02, p03, p12, p13 = point(0, 2), point(0, 3), point(1, 2), point(1, 3)
        bary[0, 0], bary[0, 1], bary[0, 2], bary[0, 3] = corner[0], p02, p03, corner[1]
        bary[1, 0], bary[1, 1], bary[1, 2], bary[1, 3] = p02, p03, corner[1], p12
        bary[2, 0], bary[2, 1], bary[2, 2], bary[2, 3] = p03, corner[1], p12, p13
        n = 3
    else:  # a prism, with the triangles (0, 1, 2) and (p03, p13, p23) at its ends
        p03, p13, p23 = point(0, 3), point(1, 3), point(2, 3)
        bary[0, 0], bary[0, 1], bary[0, 2], bary[0, 3] = corner[0], corner[1], corner[2], p03
        bary[1, 0], bary[1, 1], bary[1, 2], bary[1, 3] = corner[1], corner[2], p03, p13
        bary[2, 0], bary[2, 1], bary[2, 2], bary[2, 3] = corner[2], p03, p13, p23
        n = 3
    volume = np.zeros(3)
    for i in range(n):
        volume[i] = abs(np.linalg.det(bary[i]))
    return bary, volume


@njit
def _add_kernel(res, omega, edges, intervals, kind, weight, d0, d1, d2, d3):
    """add weight times a function of dE - omega over a tetrahedron, with the values d0, ..., d3 of dE at its corners,
    to res: delta(dE - omega) (kind 0: the values at omega, with the intervals around them for a flat dE; kind 1:
    the averages over the intervals between the edges of omega) or P 1/(dE - omega) (kind 2)"""
    if kind == 0:
        _add_delta_values(res, omega, intervals, weight, d0, d1, d2, d3)
    elif kind == 1:
        res += weight * delta_weights_tetra(edges, np.array([[d0, d1, d2, d3]]))[0]
    else:
        _add_principal(res, omega, d0, d1, d2, d3, weight)


@njit
def _add_occupied(res, occupied, dE, efermi, omega, edges, intervals, kind, weight):
    """add weight times the integral over a tetrahedron of theta(efermi - occupied) times the function of dE - omega
    given by `kind` (see `_add_kernel`) to res"""
    bary, volume = occupied_tetrahedra(occupied, efermi)
    for t in range(3):
        if volume[t] > 0:
            # relative to a corner value, so that a constant dE stays exactly constant (P 1/(dE - omega) is zero for
            # a constant dE equal to omega, but large for values that differ by rounding)
            d = dE[0] + bary[t] @ (dE - dE[0])
            _add_kernel(res, omega, edges, intervals, kind, weight * volume[t], d[0], d[1], d[2], d[3])


@njit
def occupied_kernel(occupied, dE, efermi, omega, edges, kind):
    """the integral over a tetrahedron of theta(efermi - occupied) times a function of dE - omega: delta(dE - omega)
    (kind 0: the values at omega; kind 1: the averages over the intervals between the edges of omega) or
    P 1/(dE - omega) (kind 2), as fractions of the volume of the tetrahedron.

    occupied, dE : the values of the two linear functions at the 4 corners; returns an array (nefermi, nomega)"""
    res = np.zeros((efermi.shape[0], omega.shape[0]))
    intervals = omega_intervals(omega)
    for ief in range(efermi.shape[0]):
        _add_occupied(res[ief], occupied, dE, efermi[ief], omega, edges, intervals, kind, 1.)
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
def occupation_kernel_tetra(Ei, Ej, sign, efermi, omega, edges, kind):
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
            state_i, state_j = _occupation_state(Ei[i], ef), _occupation_state(Ej[i], ef)
            if state_i == state_j and state_i != 2:  # both occupied or both empty
                continue
            for state, occupied, weight in ((state_j, Ej[i], 1.), (state_i, Ei[i], -1.)):
                if state == 1:
                    if not have_whole:
                        whole[:] = 0
                        _add_kernel(whole, omega, edges, intervals, kind, 1., dE[0], dE[1], dE[2], dE[3])
                        have_whole = True
                    res[i, ief] += weight * whole
                elif state == 2:
                    _add_occupied(res[i, ief], occupied, dE, ef, omega, edges, intervals, kind, weight)
    return res


class TetraWeights:
    """the idea is to make a lazy evaluation, i.e. the weights are evaluated only once for a particular ik,ib
       the Fermi level list remains the same throughout calculation"""

    def __init__(self, eCenter, eCorners, kind="single"):
        self.eCenter = eCenter
        self.nk, self.nb = self.eCenter.shape
        assert eCorners.shape[0] == self.nk, f"nk mismatch: {eCorners.shape[0]} != {self.nk}"
        assert eCorners.shape[-1] == self.nb, f"nb mismatch: {eCorners.shape[-1]} != {self.nb}"
        if kind == "single":
            assert eCorners.shape == (self.nk, 4, self.nb), f"eCorners shape mismatch: {eCorners.shape} != {(self.nk, 4, self.nb)}"
            self.all_tetrahedra_vertices = [eCorners.copy()]
        elif kind == "parallelepiped":
            assert eCorners.shape == (self.nk, 2, 2, 2, self.nb), f"eCorners shape mismatch: {eCorners.shape} != {(self.nk, 2, 2, 2, self.nb)}"
            self.all_tetrahedra_vertices = []
            for iface in 0, 1:
                for Eface in eCorners[:, iface, :, :], eCorners[:, :, iface, :], eCorners[:, :, :, iface]:
                    E0 = eCenter
                    E1 = Eface[:, 0, 0]
                    E3 = Eface[:, 1, 1]
                    for E2 in (Eface[:, 0, 1], Eface[:, 1, 0]):
                        self.all_tetrahedra_vertices.append(np.array([E0, E1, E2, E3]).swapaxes(0, 1))
        else:
            raise ValueError(f"Unknown kind of tetrahedra: {kind}")

        self.eFermis = []
        self.eFermi = None
        if self.nk == 0:
            self.null = True
        else:
            self.null = False
            self.weights = []
            Eall = np.concatenate((self.eCenter[:, None, :], eCorners.reshape(self.nk, -1, self.nb)), axis=1)
            self.Emin = Eall.min(axis=1)
            self.Emax = Eall.max(axis=1)

    def weight_1k1b(self, ief, ik, ib, der):
        if self.null:
            return 0.
        if der == -1:
            return 1 - self.weight_1k1b(ief, ik, ib, der=0)
        return self.weight_1k1b_priv(self.eFermis[ief], ik, ib, der=der)

    @cached_property
    def factor_num_tetra(self):
        return 1. / len(self.all_tetrahedra_vertices)

    def weight_1k1b_priv(self, eFermi, ik, ib, der):
        occ = np.zeros(eFermi.shape, dtype=float)
        for tetrahedron in self.all_tetrahedra_vertices:
            evert = tetrahedron[ik, :, ib]
            occ += weights_tetra(eFermi, evert[0], evert[1], evert[2], evert[3], der=der)
        return occ * self.factor_num_tetra

    def __weight_1b(self, ief, ik, ib, der):
        if ib not in self.weights[ief][der][ik]:
            self.weights[ief][der][ik][ib] = self.weight_1k1b(ief, ik, ib, der)
        return self.weights[ief][der][ik][ib]

    def index_eFermi(self, eFermi):
        for i, eF in enumerate(self.eFermis):
            if eF is eFermi:
                return i
        return -1

    def weights_all_band_groups(self, eFermi, der, degen_thresh=-1, degen_Kramers=False, Emin=-np.inf, Emax=np.inf, select_bands=None):
        """
             here  the key of the return dict is a pair of integers (ib1,ib2)
        """
        ief = self.index_eFermi(eFermi)
        #        print ("debug, index_eFermi:",ief)
        if ief < 0:
            ief = len(self.weights)
            self.weights.append(defaultdict(lambda: defaultdict(lambda: {})))
            self.eFermis.append(eFermi)
        res = []
        for ik in range(self.nk):
            bands_in_range = get_bands_in_range(
                eFermi[0],
                eFermi[-1],
                self.eCenter[ik],
                degen_thresh=degen_thresh,
                degen_Kramers=degen_Kramers,
                Ebandmin=self.Emin[ik],
                Ebandmax=self.Emax[ik],
                select_bands=select_bands
            )
            weights = {
                (ib1, ib2): sum(self.__weight_1b(ief, ik, ib, der) for ib in range(ib1, ib2)) / (ib2 - ib1) * weight_select_bands(ib1, ib2, select_bands)
                for ib1, ib2 in bands_in_range
            }

            if der == 0:
                if select_bands is not None:
                    raise NotImplementedError("Selection of bands for Fermi sea is not implemented")
                bandmax = get_bands_below_range(eFermi[0], self.eCenter[ik], Ebandmax=self.Emax[ik])
                bandmin = get_bands_below_range(Emin, self.eCenter[ik], Ebandmax=self.Emax[ik])
                if len(bands_in_range) > 0:
                    bandmax = min(bandmax, bands_in_range[0][0])
                if bandmax > bandmin:
                    weights[(bandmin, bandmax)] = ones(len(eFermi))

            if der == -1:
                if select_bands is not None:
                    raise NotImplementedError("Selection of bands for inverse Fermi sea is not implemented")
                bandmin = get_bands_above_range(eFermi[-1], self.eCenter[ik], Ebandmin=self.Emin[ik])
                bandmax = get_bands_above_range(Emax, self.eCenter[ik], Ebandmin=self.Emin[ik])
                if len(bands_in_range) > 0:
                    bandmin = max(bandmin, bands_in_range[-1][-1])
                if bandmax > bandmin:
                    weights[(bandmin, bandmax)] = ones(len(eFermi))

            res.append(weights)
        return res


    def dynamic_weights(self, ik, pairs, omega, edges=None, type="delta"):
        """weights of delta(E_i - E_j - omega) at the k-point ik, for pairs of band groups ((i1, i2), (j1, j2)),
        averaged over the pairs of bands i1 <= i < i2, j1 <= j < j2. The values at the given increasing omega, or,
        if edges are given, the averages over the intervals of omega between them. Returns an array
        (len(pairs), len(omega))"""
        omega = np.asarray(omega, dtype=float)
        band_i, band_j, group = [], [], []
        for ig, ((i1, i2), (j1, j2)) in enumerate(pairs):
            for i in range(i1, i2):
                for j in range(j1, j2):
                    band_i.append(i)
                    band_j.append(j)
                    group.append(ig)
        res = np.zeros((len(pairs), len(omega)))
        if len(group) == 0 or self.null:
            return res
        if edges is not None:
            edges = np.asarray(edges, dtype=float)
            assert len(edges) == len(omega) + 1
        addition = 0
        assert type in ("delta", "reactive"), f"Unknown type: {type}"
        for vertices in self.all_tetrahedra_vertices:
            dE = np.ascontiguousarray((vertices[ik][:, band_i] - vertices[ik][:, band_j]).T)
            if type == "delta":
                addition = addition + (delta_values_tetra(omega, dE) if edges is None else delta_weights_tetra(edges, dE))
            elif type == "reactive":
                addition = addition + reactive_values_tetra(omega, dE)
        np.add.at(res, np.array(group), addition)
        nband_pairs = np.array([max((i2 - i1) * (j2 - j1), 1) for (i1, i2), (j1, j2) in pairs])  # empty pairs give 0
        return res * self.factor_num_tetra / nband_pairs[:, None]

    def occupation_weights(self, ik, pairs, omega, efermi, type="delta", reverse=False, edges=None):
        """the integrals over the cells around the k-point ik of [theta(efermi - E_j) - theta(efermi - E_i)] times
        delta(E_i - E_j - omega) (type 'delta'; the averages over the intervals between the edges of omega if they are
        given) or P 1/(E_i - E_j - omega) (type 'reactive'), with E_j - E_i instead of E_i - E_j if reverse, for
        pairs of band groups ((i1, i2), (j1, j2)), averaged over the pairs of bands as `dynamic_weights`. Returns an
        array (len(pairs), len(efermi), len(omega))"""
        omega = np.asarray(omega, dtype=float)
        efermi = np.asarray(efermi, dtype=float).reshape(-1)
        if type == "reactive":
            code = 2
        elif type == "delta":
            code = 0 if edges is None else 1
        else:
            raise ValueError(f"Unknown type: {type}")
        if edges is None:
            edges = np.zeros(1)
        else:
            edges = np.asarray(edges, dtype=float)
            assert len(edges) == len(omega) + 1
        sign = -1. if reverse else 1.
        band_i, band_j, group = [], [], []
        for ig, ((i1, i2), (j1, j2)) in enumerate(pairs):
            for i in range(i1, i2):
                for j in range(j1, j2):
                    band_i.append(i)
                    band_j.append(j)
                    group.append(ig)
        res = np.zeros((len(pairs), len(efermi), len(omega)))
        if len(group) == 0 or self.null:
            return res
        addition = 0
        for vertices in self.all_tetrahedra_vertices:
            Ei = np.ascontiguousarray(vertices[ik][:, band_i].T)
            Ej = np.ascontiguousarray(vertices[ik][:, band_j].T)
            addition = addition + occupation_kernel_tetra(Ei, Ej, sign, efermi, omega, edges, code)
        np.add.at(res, np.array(group), addition)
        nband_pairs = np.array([max((i2 - i1) * (j2 - j1), 1) for (i1, i2), (j1, j2) in pairs])  # empty pairs give 0
        return res * self.factor_num_tetra / nband_pairs[:, None, None]
