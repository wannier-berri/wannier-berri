import numpy as np
from ..utility import njit


@njit
def _simplex_value(x, e, inv, coef, k, der, b, right):
    """the value of `weights_simplex` at x in [e_0, e_n], from the right (x + 0) or from the left (x - 0); b is a work
    array (n + 1)"""
    n = e.shape[0] - 1
    if der == 0:
        for j in range(n + 1):
            b[j] = 1. if (x >= e[j] if right else x > e[j]) else 0.
        for m in range(1, n + 1):
            for j in range(n + 1 - m):
                if (x >= e[j + m]) if right else (x > e[j + m]):
                    b[j] = 1.
                elif (x >= e[j]) if right else (x > e[j]):
                    b[j] = ((x - e[j]) * b[j] + (e[j + m] - x) * b[j + 1]) * inv[j, j + m]
        return b[0]
    for j in range(n):
        b[j] = 1. if ((e[j] <= x < e[j + 1]) if right else (e[j] < x <= e[j + 1])) else 0.
    for d in range(1, k + 1):
        for j in range(n - d):
            # each term only where its spline is not zero, so that the factors are between 0 and 1 and the
            # inverses of tiny differences never multiply a large distance
            value = 0.
            if b[j] != 0:
                value += (x - e[j]) * inv[j, j + d] * b[j]
            if b[j + 1] != 0:
                value += (e[j + d + 1] - x) * inv[j + 1, j + d + 1] * b[j + 1]
            b[j] = value
    value = 0.
    for j in range(n - k):
        if b[j] != 0:  # coef[j] may overflow for differences of the values near the smallest floats
            value += coef[j] * b[j]
    return value


@njit
def weights_simplex(efall, e, der=0):
    """for a function linear within a simplex of dimension n (a segment, triangle or tetrahedron), with the sorted
    values e (n + 1) at its corners: the fraction of the simplex where it is below each value of efall (der=0), and
    its derivatives with respect to efall (der=1 is the density of the values, normalised to 1; zero for der > n)

    The density is the B-spline with the knots e, normalised to 1: n B(x) / (e_n - e_0), with B the B-spline of
    degree n - 1, computed by the recursion of Cox and de Boor, with the factors (x - e_j) / (e_(j+d) - e_j) and
    (e_(j+d+1) - x) / (e_(j+d+1) - e_(j+1)) between 0 and 1. The fraction below x satisfies the same kind of
    recursion, F(e_j, ..., e_(j+m)) = ((x - e_j) F(e_j, ..., e_(j+m-1)) + (e_(j+m) - x) F(e_(j+1), ..., e_(j+m)))
    / (e_(j+m) - e_j) for e_j <= x < e_(j+m) (0 below and 1 above). The derivatives of the density are sums of the
    B-splines of lower degree over the knots. So equal and nearly equal values need no special treatment: the terms
    with equal knots are dropped (also for differences so small that their inverse overflows). At equal values, the
    density or its derivatives jump; then der=2, 3 are their regular parts, without the delta functions. At x equal
    to a value, the fraction is that strictly below it (so 0 at the value if all values are equal, as for
    `_add_delta_values`), and the derivatives, which may jump there, are the mean of their limits from below and from
    above. NaN in efall gives NaN (for der <= n)."""
    if der < 0:
        raise ValueError("weights_simplex: der must not be negative")
    n = e.shape[0] - 1
    occ = np.zeros(efall.shape[0])
    if der > n:
        return occ
    # 1 / (e_j - e_i), and 0 for equal values
    inv = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if e[j] > e[i]:
                value = 1. / (e[j] - e[i])
                if np.isfinite(value):
                    inv[i, j] = value
    # the derivative of order der - 1 of the density is sum_j coef[j] B_j, with B_j the B-spline of degree k with
    # the knots e_j, ..., e_(j+k+1), from (B_j)' = k (B_j inv[j, j+k] - B_(j+1) inv[j+1, j+k+1]) for degree k
    k = n - 1
    coef = np.zeros(n + 1)
    coef[0] = n * inv[0, n]
    for r in range(der - 1):
        for j in range(r + 1, 0, -1):
            coef[j] = k * inv[j, j + k] * (coef[j] - coef[j - 1])
        coef[0] = k * inv[0, k] * coef[0]
        k -= 1
    b = np.zeros(n + 1)
    for i in range(efall.shape[0]):
        x = efall[i]
        if np.isnan(x):
            occ[i] = np.nan
        elif x < e[0]:
            continue
        elif x > e[n]:
            if der == 0:
                occ[i] = 1.
        elif der == 0:
            occ[i] = _simplex_value(x, e, inv, coef, k, der, b, False)
        else:
            value = _simplex_value(x, e, inv, coef, k, der, b, True)
            for j in range(n + 1):
                if x == e[j]:
                    value = value / 2 + _simplex_value(x, e, inv, coef, k, der, b, False) / 2
                    break
            occ[i] = value
    return occ


@njit
def weights_triangle(efall, e0, e1, e2, der=0):
    """`weights_simplex` for a triangle, with the values e0, e1, e2 at its corners (in any order)"""
    if der < 0:
        raise NotImplementedError(f"der={der} not implemented in weights_triangle")
    return weights_simplex(efall, np.sort(np.array([e0, e1, e2], dtype=np.float64)), der)


# the expansion of the principal part far from the values of dE: its order, and the largest ratio of the spread of
# the values to the distance of omega from their mean where it is used (the terms decrease by at least 3/4 of it)
PRINCIPAL_ORDER = 16
PRINCIPAL_FAR = 4


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
    compared with their spread (more than PRINCIPAL_FAR times), the expansion in the spread over the distance is used
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
        if spread * PRINCIPAL_FAR <= abs(D):
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


@njit
def weights_tetra(efall, e1, e2, e3, e4, der=0):
    """for a function linear within a tetrahedron, with the values e1, ..., e4 at its corners (in any order): the
    fraction of the tetrahedron where it is below each value of efall (der=0), and its derivatives with respect to
    efall (der=1 is the density of the values, normalised to 1; der=2, 3 its derivatives), as in the linear
    tetrahedron method, see e.g. P. E. Bloechl, O. Jepsen and O. K. Andersen, Phys. Rev. B 49, 16223 (1994).
    Computed by `weights_simplex`, also for equal values."""
    if der < 0:
        raise ValueError(f"der={der} not supported")
    return weights_simplex(efall, np.sort(np.array([e1, e2, e3, e4], dtype=np.float64)), der)


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
def _mean_tlog(a, b, log_a, log_b):
    """the average of t log|t| over the interval [a, b] (a <= b), without cancellation for close values (a log|a|
    for a equal to b), with log_a = log|a| and log_b = log|b| given (not used for a zero value)"""
    if a == b:
        return a * log_a if a != 0 else 0.
    m = (a + b) / 2
    h = abs(b - a) / 2
    if h >= abs(m):
        # the interval contains or touches zero: no cancellation, with a = h alpha, b = h beta, |alpha|, |beta| <= 2,
        # beta - alpha = 2, the average is h (beta^2 log|b| - alpha^2 log|a| - alpha - beta) / 4 (the last terms are
        # -m / 2), without squares of a and b, which underflow for small values, and with the dimensionless factor
        # computed first, so that it does not overflow before the result does for large ones
        alpha, beta = a / h, b / h
        value = -(alpha + beta)
        if b != 0:
            value += beta * beta * log_b
        if a != 0:
            value -= alpha * alpha * log_a
        return h * (value / 4)
    r = h / abs(m)
    if r < 0.1:  # m log|m| + sum_j f^(2j)(m) h^2j / (2j+1)!, with f^(2j)(t) = (2j-2)! / t^(2j-1)
        r2 = r * r
        series = 0.
        for j in range(10, 0, -1):
            series = series * r2 + 1 / ((2 * j + 1) * 2 * j * (2 * j - 1))
        return m * (np.log(abs(m)) + series * r2)
    else:
        return m * (np.log(abs(m)) + ((1 + r) ** 2 * np.log1p(r) - (1 - r) ** 2 * np.log1p(-r)) / (4 * r) - 1 / 2)


@njit
def _mean_log_triangle_close(a, b, c, at_zero):
    """the average of log|t| over a triangle, with the sorted values a, b, c of a linear function t at its corners,
    if they are equal (log(at_zero) for a = b = c = 0, see `_add_principal_3d`) or zero is far from them compared
    with their spread (more than PRINCIPAL_FAR times from their mean m), as (value, True); else (0., False), and
    the average is 2 (T(b, c) - T(a, b)) / (c - a) - 1, with T(a, b) the average of t log|t| over [a, b] (twice the
    second divided difference of t^2 log|t| / 2 - 3 t^2 / 4). Far from zero, the expansion of log|m + x| in
    x = t - m is used."""
    if a == c:
        return np.log(abs(a) if a != 0 else at_zero), True
    m = (a + b + c) / 3
    if (c - a) * PRINCIPAL_FAR <= abs(m):
        # the moments of y = x / m over the triangle: <y^n> = 2 / ((n+1) (n+2)) h_n(d), with h_n the complete
        # homogeneous symmetric polynomial of the deviations d of the corner values from the mean, relative to the
        # mean (dimensionless, so that they do not overflow or underflow), and <log|1 + y|> = sum_n (-1)^(n+1) <y^n> / n,
        # the terms decreasing by at least 1/6. As the deviations add up to zero, h_n = -e2 h_(n-2) + e3 h_(n-3), with
        # e2, e3 their elementary symmetric polynomials
        d1, d2, d3 = (a - m) / m, (b - m) / m, (c - m) / m
        e2 = d1 * d2 + d1 * d3 + d2 * d3
        e3 = d1 * d2 * d3
        g3, g2, g1 = 0., 1., 0.
        value = 0.
        for n in range(2, PRINCIPAL_ORDER + 1):
            g3, g2, g1 = g2, g1, -(e2 * g2 + e3 * g3)
            value -= 2. / (n * (n + 1) * (n + 2)) * g1
        return np.log(abs(m)) + value, True
    else:
        return 0., False


@njit
def _add_principal_3d(res, omega, e, weight=1.):
    """add weight times the average of P 1/(dE - omega) over a tetrahedron, with the values e of dE at its corners,
    to res, at each of the given values of omega

    The average is six times the third divided difference of t^2 log|t| / 2 at the values t = e - omega, i.e.
    3 (N(t1, t2, t3) - N(t0, t1, t2)) / (t3 - t0) for the sorted values, with N the average of log|t| over the
    triangle with the given values at its corners (`_mean_log_triangle_close`, or from the averages T of t log|t|
    over the intervals, `_mean_tlog`, with T(t1, t2) and the logarithms of |t| shared by both triangles), so that
    equal values need no special treatment. If omega is far from the values compared with their spread (more than
    PRINCIPAL_FAR times), the expansion in the spread over the distance is used instead, with the values and omega
    taken relative to the smallest value, so that this works also for a small spread far from zero. If three corners
    have the same value, the density jumps there, and the principal part diverges logarithmically as omega
    approaches it; exactly at that value, the distance is taken as 2.2e-16 (the machine epsilon) times the magnitude
    of the values and their spread, about the spacing of floating-point numbers near them, as for triangles
    (`_add_principal_2d`); so the value there, unlike the values next to it, changes if all values and omega are
    shifted by the same amount."""
    values = np.sort(np.asarray(e, dtype=np.float64))
    shift = values[0]
    e = values - shift
    mean = np.sum(e) / 4
    spread = e[3]
    moments = _get_moments(e, mean)
    at_zero = 2.2e-16 * (spread + max(abs(values[0]), abs(values[3])))

    for i in range(omega.shape[0]):
        D = mean - (omega[i] - shift)
        if spread * PRINCIPAL_FAR <= abs(D):
            if D != 0:  # with all values equal to omega, the principal part is zero
                # <1/(D + x)> = sum_n (-1)^n <x^n> / D^(n+1)
                res[i] += weight * _get_polynomial(moments, -1 / D) / D
        else:
            # not relative to the smallest value: exact for omega near a value, while the relative error of the
            # differences is about eps * |omega| / spread for a small spread far from zero, as for triangles
            t0, t1, t2, t3 = values - omega[i]

            upper, upper_done = _mean_log_triangle_close(t1, t2, t3, at_zero=at_zero)
            lower, lower_done = _mean_log_triangle_close(t0, t1, t2, at_zero=at_zero)
            if not (upper_done and lower_done):
                log1 = np.log(abs(t1)) if t1 != 0 else 0.
                log2 = np.log(abs(t2)) if t2 != 0 else 0.
                T12 = _mean_tlog(t1, t2, log1, log2)
                if not upper_done:
                    log3 = np.log(abs(t3)) if t3 != 0 else 0.
                    upper = 2 * (_mean_tlog(t2, t3, log2, log3) - T12) / (t3 - t1) - 1
                if not lower_done:
                    log0 = np.log(abs(t0)) if t0 != 0 else 0.
                    lower = 2 * (T12 - _mean_tlog(t0, t1, log0, log1)) / (t2 - t0) - 1
            res[i] += weight * 3 * (upper - lower) / spread


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
    0 at the edges up to lo (at most emin, as the cumulative weight is that strictly below the edge, see
    `weights_simplex`) and 1 at the edges from hi on (above emax). Returns (lo, hi), with hi <= lo if nothing
    changes."""
    lo = max(np.searchsorted(edges, emin, side='right') - 1, 0)
    hi = min(np.searchsorted(edges, emax, side='right'), edges.shape[0] - 1)
    return lo, hi


@njit
def _omega_in_range(omega, emin, emax):
    """the slice lo:hi of the increasing values omega within [emin, emax]"""
    return np.searchsorted(omega, emin, side='left'), np.searchsorted(omega, emax, side='right')



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
    the widths of the intervals sum to the fraction of the tetrahedron where edges[0] <= dE < edges[-1]"""
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
