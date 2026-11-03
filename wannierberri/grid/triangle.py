from .tetrahedron import TetraWeights
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
def weights_triangle_der3(efall, e0, e1, e2):
    return np.zeros_like(efall)


@njit
def space_e1234(e0, e1, e2, diff_min=1e-12):
    e = np.array(sorted([e0, e1, e2]))
    # a dirty trick to avoid divisions by zero
    for i in range(2):
        e[i + 1] = max(e[i + 1], e[i] + diff_min)
    return e[0], e[1], e[2]


@njit
def weights_triangle(efall, e0, e1, e2, der=0):
    e0, e1, e2 = space_e1234(e0, e1, e2, diff_min=1e-12)
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




class TriangleWeights(TetraWeights):


    def __init__(self, eCenter, eCorners, kind="0011"):
        self.eCenter = eCenter
        self.nk, self.nb = self.eCenter.shape
        assert eCorners.shape[0] == self.nk, f"nk mismatch: {eCorners.shape[0]} != {self.nk}"
        assert eCorners.shape[-1] == self.nb, f"nb mismatch: {eCorners.shape[-1]} != {self.nb}"
        if kind == "single":
            assert eCorners.shape == (self.nk, 4, self.nb), f"eCorners shape mismatch: {eCorners.shape} != {(self.nk, 4, self.nb)}"
            self.all_tetrahedra_vertices = [eCorners.copy()]
        elif kind == "0011":
            assert eCorners.shape == (self.nk, 2, 2, self.nb), f"eCorners shape mismatch: {eCorners.shape} != {(self.nk, 2, 2, 2, self.nb)}"
            self.all_tetrahedra_vertices = []
            E1 = eCorners[:, 0, 0]
            E3 = eCorners[:, 1, 1]
            for E2 in (eCorners[:, 0, 1], eCorners[:, 1, 0]):
                self.all_tetrahedra_vertices.append(np.array([E1, E2, E3]).swapaxes(0, 1))
        elif kind == "0101":
            assert eCorners.shape == (self.nk, 2, 2, self.nb), f"eCorners shape mismatch: {eCorners.shape} != {(self.nk, 2, 2, 2, self.nb)}"
            self.all_tetrahedra_vertices = []
            E1 = eCorners[:, 0, 1]
            E3 = eCorners[:, 1, 0]
            for E2 in (eCorners[:, 0, 0], eCorners[:, 1, 1]):
                self.all_tetrahedra_vertices.append(np.array([E1, E2, E3]).swapaxes(0, 1))
        else:
            raise ValueError(f"Unknown kind of triangles: {kind}")

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
        self.weights_function = weights_triangle


    def dynamic_weights(self, ik, pairs, omega, edges=None, type="delta"):
        raise NotImplementedError("dynamic_weights is not implemented for 2D triangles.")
