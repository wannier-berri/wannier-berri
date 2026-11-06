import abc

import numpy as np
from functools import cached_property, lru_cache
from collections import defaultdict
from .tetratriangle import (delta_values_tetra, delta_weights_tetra, get_bands_in_range, get_bands_below_range,
                            get_bands_above_range, reactive_values_tetra, occupation_kernel, weights_general)
from ..utility import cached_einsum, weight_select_bands


@lru_cache
def ones(n):
    return np.ones(n)


class TTWeights(abc.ABC):
    """the idea is to make a lazy evaluation, i.e. the weights are evaluated only once for a particular ik,ib
       and list of Fermi levels (compared by value, see `index_eFermi`)"""

    def __init__(self, eCenter, eCorners, kind="single"):
        self.eCenter = eCenter
        self.nk, self.nb = self.eCenter.shape
        assert eCorners.shape[0] == self.nk, f"nk mismatch: {eCorners.shape[0]} != {self.nk}"
        assert eCorners.shape[-1] == self.nb, f"nb mismatch: {eCorners.shape[-1]} != {self.nb}"
        self.set_kind(kind, eCorners, eCenter)

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

    @abc.abstractmethod
    def set_kind(self, kind, eCorners, eCenter):
        """set the kind of tetrahedra or triangles and store the vertices in self.all_tetrahedra_vertices"""
        pass

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
            occ += weights_general(eFermi, evert, der=der)
        return occ * self.factor_num_tetra

    def __weight_1b(self, ief, ik, ib, der):
        if ib not in self.weights[ief][der][ik]:
            self.weights[ief][der][ik][ib] = self.weight_1k1b(ief, ik, ib, der)
        return self.weights[ief][der][ik][ib]

    def index_eFermi(self, eFermi):
        """the index of the stored Fermi levels equal to eFermi, or -1. They are compared by value (exactly, so that
        an array with NaN never matches), so that the weights are shared between calculators (each of which has its
        own array of the Fermi levels)"""
        for i, eF in enumerate(self.eFermis):
            if eF is eFermi or (eF.shape == eFermi.shape and np.array_equal(eF, eFermi)):
                return i
        return -1

    def weights_all_band_groups(self, eFermi, der, degen_thresh=-1, degen_Kramers=False, Emin=-np.inf, Emax=np.inf, select_bands=None):
        """
             here  the key of the return dict is a pair of integers (ib1,ib2)
        """
        eFermi = np.asarray(eFermi)
        ief = self.index_eFermi(eFermi)
        #        print ("debug, index_eFermi:",ief)
        if ief < 0:
            # a copy, in case the array of the caller is changed later, and in float64: the weights are computed
            # with the stored array, and would be truncated for integers
            stored = np.array(eFermi, dtype=np.float64)
            ief = len(self.weights)
            self.weights.append(defaultdict(lambda: defaultdict(lambda: {})))
            self.eFermis.append(stored)
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


    def dynamic_weights(self, ik, pairs, omega, efermi=None, type="delta", reverse=False, edges=None):
        """the integrals over the cells around the k-point ik of [theta(efermi - E_j) - theta(efermi - E_i)] times
        delta(E_i - E_j - omega) (type 'delta'; the averages over the intervals between the edges of omega if they are
        given) or P 1/(E_i - E_j - omega) (type 'reactive'), with E_j - E_i instead of E_i - E_j if reverse, for
        pairs of band groups ((i1, i2), (j1, j2)), averaged over the pairs of bands as `dynamic_weights`. Returns an
        array (len(pairs), len(efermi), len(omega))"""
        occupation_mode = (efermi is not None)
        omega = np.asarray(omega, dtype=float)
        if occupation_mode:
            efermi = np.asarray(efermi, dtype=float).reshape(-1)
            if type == "reactive":
                code = 2
            elif type == "delta":
                code = 0 if edges is None else 1
            else:
                raise ValueError(f"Unknown type: {type}")
        if edges is not None:
            edges = np.asarray(edges, dtype=float)
            assert len(edges) == len(omega) + 1
        elif occupation_mode:
            edges = np.zeros(1)

        sign = -1. if reverse else 1.

        band_i, band_j, group = [], [], []
        for ig, ((i1, i2), (j1, j2)) in enumerate(pairs):
            for i in range(i1, i2):
                for j in range(j1, j2):
                    band_i.append(i)
                    band_j.append(j)
                    group.append(ig)
        if occupation_mode:
            res = np.zeros((len(pairs), len(efermi), len(omega)))
        else:
            res = np.zeros((len(pairs), len(omega)))
        if len(group) == 0 or self.null:
            return res
        addition = 0
        for vertices in self.all_tetrahedra_vertices:
            if occupation_mode:
                Ei = np.ascontiguousarray(vertices[ik][:, band_i].T)
                Ej = np.ascontiguousarray(vertices[ik][:, band_j].T)
                addition = addition + occupation_kernel(Ei, Ej, sign, efermi, omega, edges, code)
            else:
                dE = np.ascontiguousarray(sign * (vertices[ik][:, band_i] - vertices[ik][:, band_j]).T)
                if type == "delta":
                    addition = addition + (delta_values_tetra(omega, dE) if edges is None else delta_weights_tetra(edges, dE))
                elif type == "reactive":
                    addition = addition + reactive_values_tetra(omega, dE)
        np.add.at(res, np.array(group), addition)
        nband_pairs_inv = 1. / np.array([max((i2 - i1) * (j2 - j1), 1) for (i1, i2), (j1, j2) in pairs])  # empty pairs give 0
        return self.factor_num_tetra * cached_einsum("i...,i->i...", res, nband_pairs_inv)



class TriangleWeights(TTWeights):


    def set_kind(self, kind, eCorners, eCenter=None):
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


class TetraWeights(TTWeights):
    """the idea is to make a lazy evaluation, i.e. the weights are evaluated only once for a particular ik,ib
       and list of Fermi levels (compared by value, see `index_eFermi`)"""

    def set_kind(self, kind, eCorners, eCenter):
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
