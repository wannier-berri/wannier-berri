import numpy as np
from numba import njit

from functools import lru_cache



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
    elif der > 3:
        return np.zeros_like(efall)
    else:
        raise ValueError(f"der={der} not supported")


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
def _separate_values(e, gap):
    """separate the 4 sorted values e closer than gap (in place), for the density on the intervals between them:
    each group of such values is spread symmetrically around its mean, with the gap between them; groups that come
    closer than gap are merged. The principal part is a symmetric function of the values, so that for equal values its
    change is of second order in the gap (shifting only the upper values changes it in first order, by about 1e-5 of
    its scale for a gap of 1e-7 of the spread). With omega within about the gap from three nearly equal values, where
    the principal part diverges logarithmically for equal ones, the result depends on the separation."""
    original = e.copy()
    starts = np.ones(4, dtype=np.bool_)  # whether a group starts at each value
    for i in range(1, 4):
        starts[i] = original[i] - original[i - 1] >= gap
    while True:
        start = 0
        for i in range(1, 5):
            if i == 4 or starts[i]:
                m = i - start
                center = np.sum(original[start:i]) / m
                for k in range(m):
                    e[start + k] = center + (k - (m - 1) / 2) * gap
                start = i
        merged = False
        for i in range(1, 4):
            if starts[i] and e[i] - e[i - 1] < gap:
                starts[i] = False
                merged = True
                break
        if not merged:
            return


@njit
def _add_principal_3d(res, omega, e, weight=1.):
    """add weight times the average of P 1/(dE - omega) over a tetrahedron to res (see `weights_tetra_reactive`)"""
    e = np.array(sorted(e))
    shift = e[0]
    e = e - shift
    mean = np.sum(e) / 4
    spread = e[3] - e[0]

    moments = _get_moments(e, mean)
    _separate_values(e, 1e-7 * spread)
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
