from ..utility import njit
import numpy as np


@njit
def weights_triangle_der0(efall, e0, e1, e2):
    occ = np.zeros_like(efall)
    denom1 = 1. / ((e1 - e0) * (e2 - e0))
    denom2 = 1. / ((e2 - e1) * (e2 - e0))
    for i, ef in enumerate(efall):
        if ef < e0:
            continue
        elif ef <= e1:
            occ[i] = (ef - e0)**2 * denom1
        elif ef <= e2:
            occ[i] = 1. - (e2 - ef)**2 * denom2
        else:
            occ[i] = 1.
    return occ


@njit
def weights_triangle_der1(efall, e0, e1, e2):
    occ = np.zeros_like(efall)
    denom1 = 2. / ((e1 - e0) * (e2 - e0))
    denom2 = 2. / ((e2 - e1) * (e2 - e0))
    for i, ef in enumerate(efall):
        if ef < e0:
            continue
        elif ef <= e1:
            occ[i] = (ef - e0) * denom1
        elif ef <= e2:
            occ[i] = (e2 - ef) * denom2
        else:
            occ[i] = 0.
    return occ


@njit
def weights_triangle_der2(efall, e0, e1, e2):
    occ = np.zeros_like(efall)
    denom1 = 2. / ((e1 - e0) * (e2 - e0))
    denom2 = - 2. / ((e2 - e1) * (e2 - e0))
    for i, ef in enumerate(efall):
        if ef < e0:
            continue
        elif ef <= e1:
            occ[i] = denom1
        elif ef <= e2:
            occ[i] = denom2
        else:
            occ[i] = 0.
    return occ


@njit
def space_e123(e0, e1, e2, diff_min=1e-12):
    e = np.array(sorted([e0, e1, e2]))
    # a dirty trick to avoid divisions by zero
    for i in range(2):
        e[i + 1] = max(e[i + 1], e[i] + diff_min)
    return e[0], e[1], e[2]


@njit
def weights_triangle(efall, e0, e1, e2, der=0):
    e0, e1, e2 = space_e123(e0, e1, e2, diff_min=1e-12)
    if der == 0:
        return weights_triangle_der0(efall, e0, e1, e2)
    elif der == 1:
        return weights_triangle_der1(efall, e0, e1, e2)
    elif der == 2:
        return weights_triangle_der2(efall, e0, e1, e2)
    elif der > 2:
        return np.zeros_like(efall)
    else:
        raise NotImplementedError(f"der={der} not implemented in weights_triangle")


PRINCIPAL_ORDER = 16
PRINCIPAL_FAR = 0.25


@njit
def _mean_log(a, b, at_zero):
    """the average of log|t| over the interval between a and b, without cancellation for close values; for a equal
    to b, log|a|, and log(at_zero) for a = b = 0 (see `_add_principal_2d`)"""
    if a == b:
        return np.log(abs(a) if a != 0 else at_zero)
    m = (a + b) / 2
    h = abs(b - a) / 2
    if h >= abs(m):  # the interval contains or touches zero: no cancellation, |a|, |b| <= 2 h
        value = 0.
        if b != 0:
            value += b * np.log(abs(b))
        if a != 0:
            value -= a * np.log(abs(a))
        return value / (b - a) - 1
    r = h / abs(m)
    if r < 0.1:  # (1 + r) log(1 + r) - (1 - r) log(1 - r) = 2 r - sum_k r^(2k+1) / (k (2k + 1))
        series = 0.
        for k in range(10, 0, -1):
            series = series * r ** 2 + 1 / (2 * k * (2 * k + 1))
        return np.log(abs(m)) - series * r ** 2
    return np.log(abs(m)) + ((1 + r) * np.log1p(r) - (1 - r) * np.log1p(-r)) / (2 * r) - 1


@njit
def _add_principal_2d(res, omega, e, weight=1.):
    """add weight times the average of P 1/(dE - omega) over a triangle, with the values e of dE at its corners, to
    res, at each of the given values of omega

    The average is twice the second divided difference of t log|t| at the values t = e - omega, i.e.
    2 (M(t1, t2) - M(t0, t1)) / (t2 - t0) for the sorted values, with M(a, b) the average of log|t| over [a, b]
    (log|a| for a equal to b, so that equal values need no special treatment). If omega is far from the values
    compared with their spread (more than 1/PRINCIPAL_FAR times), the expansion in the spread over the distance is used
    instead, with the values and omega taken relative to the smallest value, so that this works also for a small
    spread far from zero. If two corners have the same value, the density jumps there, and the principal part diverges
    logarithmically as omega approaches it; exactly at that value, the distance is taken as 2.2e-16 (the machine
    epsilon) times the magnitude of the values and their spread, about the spacing of floating-point numbers near them,
    so that the result there is finite and of the order of the results next to it."""
    values = np.sort(np.asarray(e, dtype=np.float64))
    shift = values[0]
    e = values - shift
    mean = np.sum(e) / 3
    spread = e[2]
    # the moments of x = dE - mean over the triangle: <x^n> = 2 / ((n+1) (n+2)) h_n(d), with h_n the complete
    # homogeneous symmetric polynomial of the deviations d of the corner values from the mean
    moments = np.zeros(PRINCIPAL_ORDER + 1)
    moments[0] = 1.
    for k in range(3):
        for l in range(1, PRINCIPAL_ORDER + 1):
            moments[l] += (e[k] - mean) * moments[l - 1]
    n = np.arange(PRINCIPAL_ORDER + 1)
    moments *= 2. / ((n + 1) * (n + 2))

    for i in range(omega.shape[0]):
        D = mean - (omega[i] - shift)
        if spread <= PRINCIPAL_FAR * abs(D):
            if D != 0:  # with all values equal to omega, the principal part is zero
                # <1/(D + x)> = sum_n (-1)^n <x^n> / D^(n+1)
                value = 0.
                y = -1 / D
                for k in range(PRINCIPAL_ORDER, -1, -1):
                    value = value * y + moments[k]
                res[i] += weight * value / D
        else:
            # not relative to the smallest value: exact for omega near a value, while the relative error of the
            # differences is about eps * |omega| / spread for a small spread far from zero, as for tetrahedra
            t = values - omega[i]
            at_zero = 2.2e-16 * (spread + max(abs(values[0]), abs(values[2])))
            res[i] += weight * 2 * (_mean_log(t[1], t[2], at_zero) - _mean_log(t[0], t[1], at_zero)) / spread


@njit
def occupied_triangles(occupied, efermi):
    """the part of a triangle where a linear function is below efermi, divided into at most 2 triangles

    occupied : the values at the 3 corners. Returns (bary, area): bary is an array (3, 3, 3), the barycentric
    coordinates of the corners of each triangle (rows), and area (3,) their areas as fractions of the whole (zero
    for the unused ones; 3 as for `occupied_tetrahedra`)"""
    order = np.argsort(occupied)
    a = occupied[order]
    corner = np.zeros((3, 3))
    for k in range(3):
        corner[k, order[k]] = 1.

    def point(k, l):
        t = (efermi - a[k]) / (a[l] - a[k])
        return (1 - t) * corner[k] + t * corner[l]

    bary = np.zeros((3, 3, 3))
    n = 0
    if efermi >= a[2]:  # also a flat band at the Fermi level, as with the occupations at kBT = 0 (E <= mu)
        bary[0] = np.eye(3)
        n = 1
    elif efermi <= a[0]:
        pass
    elif efermi <= a[1]:
        bary[0, 0], bary[0, 1], bary[0, 2] = corner[0], point(0, 1), point(0, 2)
        n = 1
    else:  # a quadrilateral (0, 1, p12, p02)
        p02, p12 = point(0, 2), point(1, 2)
        bary[0, 0], bary[0, 1], bary[0, 2] = corner[0], corner[1], p12
        bary[1, 0], bary[1, 1], bary[1, 2] = corner[0], p12, p02
        n = 2
    area = np.zeros(3)
    for i in range(n):
        area[i] = abs(np.linalg.det(bary[i]))
    return bary, area
