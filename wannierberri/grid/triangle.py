from numba import njit
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


@njit
def _add_principal_2d(res, omega, e, weight=1):
    raise NotImplementedError("principal value integration not implemented for 2D triangles")


@njit
def occupied_triangles(occupied, efermi):
    """return the barycentric coordinates and areas of the occupied triangles in a triangle defined by `occupied`"""
    # raise ValueError("occupied_triangles not implemented yet")  # makes problems in njit compilation, so we just return zeros for now
    return np.zeros((3, 3, 3)), np.zeros(3)
