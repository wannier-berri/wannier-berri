from collections import defaultdict
import numpy as np
from numba import njit

from functools import lru_cache

from ..utility import weight_select_bands


@lru_cache
def ones(n):
    return np.ones(n)


@njit
def weights_tetra(efall, e0, e1, e2, e3, der=0, accurate=True):
    e = np.array(sorted([e0, e1, e2, e3]))
    # a dirty trick to avoid divisions by zero
    diff_min = 1e-12
    for i in range(3):
        if e[i + 1] - e[i] < diff_min:
            e[i + 1] = e[i] + diff_min
    e1, e2, e3, e4 = e

    nEF = len(efall)
    occ = np.zeros(nEF)

    # the accurate behaviour is a bit slower, but in some cases the faster implementation gives wrong results
    # in particular, when the energies e0,e1,e2,e3 are close to each other
    # TODO : check how to handle this with derivatives
    if accurate and der == 0:
        for i in range(nEF):
            ef = efall[i]
            if ef >= e4:
                occ[i] = 1.
            elif ef < e1:
                occ[i] = 0.
            elif ef >= e3:  # c3
                occ[i] = 1 - ((ef - e4) / (e1 - e4)) * ((ef - e4) / (e2 - e4)) * ((ef - e4) / (e3 - e4))
            elif ef >= e2:  # c2
                a13 = (ef - e1) / (e3 - e1)
                a14 = (ef - e1) / (e4 - e1)
                a23 = (ef - e2) / (e3 - e2)
                a24 = (ef - e2) / (e4 - e2)
                occ[i] = a23 * a24 + a13 * (a14 * (1 - a24) + a24 * (1 - a23))
            else:  # c1
                occ[i] = ((ef - e1) / (e2 - e1)) * ((ef - e1) / (e3 - e1)) * ((ef - e1) / (e4 - e1))
        return occ

    denom3 = 1. / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    denom2 = 1. / ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4 - e2))
    denom1 = 1. / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    #    _denom1 = 1. / ((e2 - e1) * (e3 - e1) * (e4 - e1))

    if der == 0:
        c10 = -e1 ** 3 * denom1
        c30 = -e4 ** 3 * denom3 + 1.
        c20 = (e1 ** 2 * (e3 - e2) * (e4 - e2) - (e2 ** 2 * e4) * (e1 - e3) - e1 * e2 * e3 * (e2 - e4)) * denom2
    if der <= 3:
        c13 = denom1
        c33 = denom3
        c23 = denom2 * (e1 + e2 - e3 - e4)
    if der <= 2:
        c12 = -3 * e1 * denom1
        c22 = (((e3 - e2) * (e4 - e2)) - (e1 - e3) * (2 * e2 + e4) - (e3 + e1 + e2) * (e2 - e4)) * denom2
        c32 = -3 * e4 * denom3
    if der <= 1:
        c11 = 3 * e1 ** 2 * denom1
        c21 = (
            -2 * e1 * ((e3 - e2) * (e4 - e2)) + (2 * e2 * e4 + e2 ** 2) * (e1 - e3) + (
                e1 * e2 + e2 * e3 + e1 * e3) *
            (e2 - e4)
        ) * denom2
        c31 = 3 * e4 ** 2 * denom3

    if der == 0:
        for i in range(nEF):
            ef = efall[i]
            if ef >= e4:
                occ[i] = 1.
            elif ef < e1:
                occ[i] = 0.
            elif ef >= e3:  # c3
                occ[i] = c30 + ef * (c31 + ef * (c32 + c33 * ef))
            elif ef >= e2:  # c2
                occ[i] = c20 + ef * (c21 + ef * (c22 + c23 * ef))
            else:  # c1
                occ[i] = c10 + ef * (c11 + ef * (c12 + c13 * ef))
    elif der == 1:
        for i in range(nEF):
            ef = efall[i]
            if ef >= e4:
                occ[i] = 0.
            elif ef < e1:
                occ[i] = 0.
            elif ef >= e3:  # c3
                occ[i] = c31 + ef * (2 * c32 + 3 * c33 * ef)
            elif ef >= e2:  # c2
                occ[i] = c21 + ef * (2 * c22 + 3 * c23 * ef)
            else:  # c1
                occ[i] = c11 + ef * (2 * c12 + 3 * c13 * ef)
    elif der == 2:
        for i in range(nEF):
            ef = efall[i]
            if ef >= e4:
                occ[i] = 0.
            elif ef < e1:
                occ[i] = 0.
            elif ef >= e3:  # c3
                occ[i] = 2 * c32 + 6 * c33 * ef
            elif ef >= e2:  # c2
                occ[i] = 2 * c22 + 6 * c23 * ef
            else:  # c1
                occ[i] = 2 * c12 + 6 * c13 * ef
    elif der == 3:
        for i in range(nEF):
            ef = efall[i]
            if ef >= e4:
                occ[i] = 0.
            elif ef < e1:
                occ[i] = 0.
            elif ef >= e3:  # c3
                occ[i] = 6 * c33
            elif ef >= e2:  # c2
                occ[i] = 6 * c23
            else:  # c1
                occ[i] = 6 * c13
    return occ


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
def _tetrahedra_parallel(dEcorners):
    """the 12 tetrahedra of a parallelepiped as in `TetraWeightsParal`: for each, the values at its 3 corners other
    than the centre. dEcorners : array (2, 2, 2); returns an array (12, 3)"""
    res = np.empty((12, 3))
    face = np.empty((2, 2))
    n = 0
    for axis in range(3):
        for iface in range(2):
            for a in range(2):
                for b in range(2):
                    if axis == 0:
                        face[a, b] = dEcorners[iface, a, b]
                    elif axis == 1:
                        face[a, b] = dEcorners[a, iface, b]
                    else:
                        face[a, b] = dEcorners[a, b, iface]
            res[n, 0], res[n, 1], res[n, 2] = face[0, 0], face[0, 1], face[1, 1]
            res[n + 1, 0], res[n + 1, 1], res[n + 1, 2] = face[0, 0], face[1, 0], face[1, 1]
            n += 2
    return res


@njit
def delta_weights_parallel(edges, dEcenter, dEcorners):
    """the same as `delta_weights_tetra` over a parallelepiped, divided into 12 tetrahedra as in `TetraWeightsParal`

    dEcenter : array (npair,), dEcorners : array (npair, 2, 2, 2); returns an array (npair, nomega)"""
    width = edges[1:] - edges[:-1]
    res = np.zeros((dEcenter.shape[0], width.shape[0]))
    for i in range(dEcenter.shape[0]):
        emin = min(dEcenter[i], dEcorners[i].min())
        emax = max(dEcenter[i], dEcorners[i].max())
        lo, hi = _edges_in_range(edges, emin, emax)
        if hi <= lo:
            continue
        e = edges[lo:hi + 1]
        occ = np.zeros(e.shape[0])
        for t in _tetrahedra_parallel(dEcorners[i]):
            occ += weights_tetra(e, dEcenter[i], t[0], t[1], t[2], 0)
        res[i, lo:hi] = (occ[1:] - occ[:-1]) / width[lo:hi] / 12.
    return res


@njit
def delta_tetra(omega, e0, e1, e2, e3):
    """delta(dE - omega) over a tetrahedron, where dE is linear with the values e0, ..., e3 at its corners, at each
    of the given values of omega (the density of the values of dE within the tetrahedron, normalised to 1). For
    nearly equal values, they are separated as in `weights_tetra`."""
    e = np.array(sorted([e0, e1, e2, e3]))
    diff_min = 1e-12
    for i in range(3):
        if e[i + 1] - e[i] < diff_min:
            e[i + 1] = e[i] + diff_min
    e1, e2, e3, e4 = e
    res = np.zeros(omega.shape[0])
    for i in range(omega.shape[0]):
        w = omega[i]
        if w <= e1 or w >= e4:
            continue
        elif w < e2:
            res[i] = 3 * (w - e1) ** 2 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
        elif w < e3:
            x = w - e2
            res[i] = (3 * (e2 - e1) + 6 * x - 3 * (e3 - e1 + e4 - e2) * x ** 2 / ((e3 - e2) * (e4 - e2))) / (
                (e3 - e1) * (e4 - e1))
        else:
            res[i] = 3 * (e4 - w) ** 2 / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    return res


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
def delta_values_parallel(omega, dEcenter, dEcorners):
    """the same as `delta_values_tetra` over a parallelepiped, divided into 12 tetrahedra as in `TetraWeightsParal`

    dEcenter : array (npair,), dEcorners : array (npair, 2, 2, 2); returns an array (npair, nomega)"""
    res = np.zeros((dEcenter.shape[0], omega.shape[0]))
    edges = omega_intervals(omega)
    for i in range(dEcenter.shape[0]):
        for t in _tetrahedra_parallel(dEcorners[i]):
            _add_delta_values(res[i], omega, edges, 1. / 12, dEcenter[i], t[0], t[1], t[2])
    return res


class TetraWeights:
    """the idea is to make a lazy evaluation, i.e. the weights are evaluated only once for a particular ik,ib
       the Fermi level list remains the same throughout calculation"""

    def __init__(self, eCenter, eCorners):
        self.eCenter = eCenter
        self.eCorners = eCorners
        assert self.eCenter.shape == (self.eCorners.shape[0], self.eCorners.shape[-1])
        self.nk, self.nb = self.eCenter.shape
        self.eFermis = []
        self.eFermi = None
        if self.nk == 0:
            self.null = True
        else:
            self.null = False
            #        self.eFermis = []
            self.weights = []
            Eall = np.concatenate((self.eCenter[:, None, :], self.eCorners.reshape(self.nk, -1, self.nb)), axis=1)
            self.Emin = Eall.min(axis=1)
            self.Emax = Eall.max(axis=1)

    def weight_1k1b(self, ief, ik, ib, der):
        if self.null:
            return 0.
        if der == -1:
            return 1 - self.weight_1k1b(ief, ik, ib, der=0)
        return self.weight_1k1b_priv(self.eFermis[ief], ik, ib, der=der)

    def weight_1k1b_priv(self, eFermi, ik, ib, der):
        eCorners = self.eCorners[ik, :, ib]
        return weights_tetra(eFermi, eCorners[0], eCorners[1], eCorners[2], eCorners[3], der=der)

    def __weight_1b(self, ief, ik, ib, der):
        if ib not in self.weights[ief][der][ik]:
            self.weights[ief][der][ik][ib] = self.weight_1k1b(ief, ik, ib, der)
        #                self.eFermis[ief], self.eCenter[ik, ib], self.eCorners[ik, :, :, :, ib], der=der)
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


    def delta_weights(self, ik, pairs, omega, edges=None):
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
        np.add.at(res, np.array(group), self._delta_weights_bands(ik, np.array(band_i), np.array(band_j), omega, edges))
        nband_pairs = np.array([max((i2 - i1) * (j2 - j1), 1) for (i1, i2), (j1, j2) in pairs])  # empty pairs give 0
        return res / nband_pairs[:, None]

    def _delta_weights_bands(self, ik, band_i, band_j, omega, edges):
        dE = np.ascontiguousarray((self.eCorners[ik][:, band_i] - self.eCorners[ik][:, band_j]).T)
        if edges is None:
            return delta_values_tetra(omega, dE)
        return delta_weights_tetra(edges, dE)


class TetraWeightsParal(TetraWeights):

    def weight_1k1b_priv(self, eFermi, ik, ib, der):
        occ = np.zeros(eFermi.shape)
        eCorner = self.eCorners[ik, ..., ib]
        eCenter = self.eCenter[ik, ib]

        for iface in 0, 1:
            for Eface in eCorner[iface, :, :], eCorner[:, iface, :], eCorner[:, :, iface]:
                occ += weights_tetra(eFermi, eCenter, Eface[0, 0], Eface[0, 1], Eface[1, 1], der=der)
                occ += weights_tetra(eFermi, eCenter, Eface[0, 0], Eface[1, 0], Eface[1, 1], der=der)
        return occ / 12.

    def _delta_weights_bands(self, ik, band_i, band_j, omega, edges):
        dEcenter = self.eCenter[ik, band_i] - self.eCenter[ik, band_j]
        dEcorners = np.ascontiguousarray(np.moveaxis(self.eCorners[ik][..., band_i] - self.eCorners[ik][..., band_j], -1, 0))
        if edges is None:
            return delta_values_parallel(omega, dEcenter, dEcorners)
        return delta_weights_parallel(edges, dEcenter, dEcorners)
