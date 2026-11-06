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
def _get_polynomial(moments, y):
    value = 0.
    for n in range(PRINCIPAL_ORDER, -1, -1):
        value = value * y + moments[n]
    return value


@njit
def _mean_tlog(a, b):
    """the average of t log|t| over the interval between a and b, without cancellation for close values (a log|a|
    for a equal to b)"""
    if a == b:
        return a * np.log(abs(a)) if a != 0 else 0.
    m = (a + b) / 2
    h = abs(b - a) / 2
    if h >= abs(m):
        # the interval contains or touches zero: no cancellation, with a = h alpha, b = h beta, |alpha|, |beta| <= 2,
        # the average is m (log h - 1/2) + h (beta^2 log|beta| - alpha^2 log|alpha|) / (2 (beta - alpha)), without
        # squares of a and b, which underflow for small values
        alpha, beta = a / h, b / h
        value = 0.
        if beta != 0:
            value += beta ** 2 * np.log(abs(beta))
        if alpha != 0:
            value -= alpha ** 2 * np.log(abs(alpha))
        return m * (np.log(h) - 1 / 2) + h * value / (2 * (beta - alpha))
    r = h / abs(m)
    if r < 0.1:  # m log|m| + sum_j f^(2j)(m) h^2j / (2j+1)!, with f^(2j)(t) = (2j-2)! / t^(2j-1)
        series = 0.
        for j in range(10, 0, -1):
            series = series * r ** 2 + 1 / ((2 * j + 1) * 2 * j * (2 * j - 1))
        return m * (np.log(abs(m)) + series * r ** 2)
    return m * (np.log(abs(m)) + ((1 + r) ** 2 * np.log1p(r) - (1 - r) ** 2 * np.log1p(-r)) / (4 * r) - 1 / 2)


@njit
def _mean_log_triangle(a, b, c, at_zero):
    """the average of log|t| over a triangle, with the sorted values a, b, c of a linear function t at its corners,
    without cancellation for close values (log(at_zero) for a = b = c = 0, see `_add_principal_3d`). It is twice the
    second divided difference of t^2 log|t| / 2 - 3 t^2 / 4, i.e. 2 (T(b, c) - T(a, b)) / (c - a) - 1, with T(a, b)
    the average of t log|t| over [a, b]. If zero is far from the values compared with their spread (more than
    1/PRINCIPAL_FAR times from their mean), the expansion of log|m + x| in x = t - m, with m the mean, is used."""
    if a == c:
        return np.log(abs(a) if a != 0 else at_zero)
    m = (a + b + c) / 3
    if c - a <= PRINCIPAL_FAR * abs(m):
        # the moments of y = x / m over the triangle: <y^n> = 2 / ((n+1) (n+2)) h_n(d), with h_n the complete
        # homogeneous symmetric polynomial of the deviations d of the corner values from the mean, relative to the
        # mean (dimensionless, so that they do not overflow or underflow), and <log|1 + y|> = sum_n (-1)^(n+1) <y^n> / n,
        # the terms decreasing by at least 1/6
        h = np.zeros(PRINCIPAL_ORDER + 1)
        h[0] = 1.
        for d in ((a - m) / m, (b - m) / m, (c - m) / m):
            for n in range(1, PRINCIPAL_ORDER + 1):
                h[n] += d * h[n - 1]
        value = 0.
        for n in range(PRINCIPAL_ORDER, 1, -1):
            value -= (-1) ** n * 2. / ((n + 1) * (n + 2) * n) * h[n]
        return np.log(abs(m)) + value
    return 2 * (_mean_tlog(b, c) - _mean_tlog(a, b)) / (c - a) - 1


@njit
def _add_principal_3d(res, omega, e, weight=1.):
    """add weight times the average of P 1/(dE - omega) over a tetrahedron, with the values e of dE at its corners,
    to res, at each of the given values of omega

    The average is six times the third divided difference of t^2 log|t| / 2 at the values t = e - omega, i.e.
    3 (N(t1, t2, t3) - N(t0, t1, t2)) / (t3 - t0) for the sorted values, with N the average of log|t| over the
    triangle with the given values at its corners (`_mean_log_triangle`), so that equal values need no special
    treatment. If omega is far from the values compared with their spread (more than 1/PRINCIPAL_FAR times), the
    expansion in the spread over the distance is used instead, with the values and omega taken relative to the
    smallest value, so that this works also for a small spread far from zero. If three corners have the same value,
    the density jumps there, and the principal part diverges logarithmically as omega approaches it; exactly at that
    value, the distance is taken as 2.2e-16 (the machine epsilon) times the magnitude of the values and their spread,
    about the spacing of floating-point numbers near them, as for triangles (`_add_principal_2d`); so the value there,
    unlike the values next to it, changes if all values and omega are shifted by the same amount."""
    values = np.sort(np.asarray(e, dtype=np.float64))
    shift = values[0]
    e = values - shift
    mean = np.sum(e) / 4
    spread = e[3]
    moments = _get_moments(e, mean)

    for i in range(omega.shape[0]):
        D = mean - (omega[i] - shift)
        if spread <= PRINCIPAL_FAR * abs(D):
            if D != 0:  # with all values equal to omega, the principal part is zero
                # <1/(D + x)> = sum_n (-1)^n <x^n> / D^(n+1)
                res[i] += weight * _get_polynomial(moments, -1 / D) / D
        else:
            # not relative to the smallest value: exact for omega near a value, while the relative error of the
            # differences is about eps * |omega| / spread for a small spread far from zero, as for triangles
            t = values - omega[i]
            at_zero = 2.2e-16 * (spread + max(abs(values[0]), abs(values[3])))
            res[i] += weight * 3 * (_mean_log_triangle(t[1], t[2], t[3], at_zero)
                                    - _mean_log_triangle(t[0], t[1], t[2], at_zero)) / spread


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
