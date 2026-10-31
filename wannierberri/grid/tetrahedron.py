from collections import defaultdict
import numpy as np
from numba import njit

from functools import cached_property, lru_cache

from ..utility import weight_select_bands


@lru_cache
def ones(n):
    return np.ones(n)


@njit
def weights_tetra(efall, e0, e1, e2, e3, der=0):
    """for a function linear within a tetrahedron, with the values e0, ..., e3 at its corners: the fraction of the
    tetrahedron where it is below each value of efall (der=0), and its derivatives with respect to efall (der=1 is
    the density of the values, normalised to 1; der=2, 3 its derivatives). The formulas of the linear tetrahedron
    method, see e.g. P. E. Bloechl, O. Jepsen and O. K. Andersen, Phys. Rev. B 49, 16223 (1994)."""
    e = np.array(sorted([e0, e1, e2, e3]))
    # a dirty trick to avoid divisions by zero
    diff_min = 1e-12
    for i in range(3):
        if e[i + 1] - e[i] < diff_min:
            e[i + 1] = e[i] + diff_min
    e1, e2, e3, e4 = e

    nEF = len(efall)
    occ = np.zeros(nEF)

    denom3 = 1. / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    denom1 = 1. / ((e2 - e1) * (e3 - e1) * (e4 - e1))

    if der == 0:
        denom_2324 = 1. / ((e3 - e2) * (e4 - e2))
        denom_131424 = 1. / ((e3 - e1) * (e4 - e1) * (e4 - e2))
        denom_132423 = 1. / ((e3 - e1) * (e4 - e2) * (e3 - e2))

    if der > 0:
        c12 = - (e3 - e1 + e4 - e2) / ((e3 - e2) * (e4 - e2))
        denom_3141 = 3. / ((e3 - e1) * (e4 - e1))

    if der == 0:
        for i in range(nEF):
            ef = efall[i]
            if ef >= e4:
                occ[i] = 1.
            elif ef < e1:
                continue
            elif ef >= e3:  # c3
                ef4 = ef - e4
                occ[i] = 1 + ef4 ** 3 * denom3
            elif ef >= e2:  # c2  # e2:e3
                ef1, ef2, ef3, ef4 = ef - e
                # due to e4 and e3 being negative, all terms in the denominators are positive, so we do not loose precision
                occ[i] = ef2**2 * denom_2324 - denom_131424 * ef1 * ef1 * ef4 - denom_132423 * ef1 * ef2 * ef3
            else:  # c1
                ef1 = ef - e1
                occ[i] = ef1 ** 3 * denom1
    elif der == 1:
        for i in range(nEF):
            ef = efall[i]
            if ef <= e1 or ef >= e4:
                continue
            elif ef >= e3:  # c3
                ef4 = ef - e4
                occ[i] = 3 * ef4 ** 2 * denom3
            elif ef >= e2:  # c2  # e2:e3
                x = ef - e2
                occ[i] = (e2 - e1 + 2 * x + c12 * x ** 2) * denom_3141
            else:  # c1
                ef1 = ef - e1
                occ[i] = 3 * ef1 ** 2 * denom1
    elif der == 2:
        for i in range(nEF):
            ef = efall[i]
            if ef <= e1 or ef >= e4:
                continue
            elif ef >= e3:  # c3
                occ[i] = 6 * (ef - e4) * denom3
            elif ef >= e2:  # c2  # e2:e3
                occ[i] = 2 * (1 + c12 * (ef - e2)) * denom_3141
            else:  # c1
                occ[i] = 6 * (ef - e1) * denom1
    elif der == 3:
        factor_23 = 2 * c12 * denom_3141
        factor_34 = 6 * denom3
        factor_12 = 6 * denom1
        for i in range(nEF):
            ef = efall[i]
            if ef <= e1 or ef >= e4:
                continue
            elif ef >= e3:  # c3
                occ[i] = factor_34
            elif ef >= e2:  # c2  # e2:e3
                occ[i] = factor_23
            else:  # c1
                occ[i] = factor_12
    else:
        raise NotImplementedError(f"der={der} not implemented in weights_tetra_accurate")
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
        res = 0
        for vertices in self.all_tetrahedra_vertices:
            dE = np.ascontiguousarray((vertices[ik][:, band_i] - vertices[ik][:, band_j]).T)
            res = res + (delta_values_tetra(omega, dE) if edges is None else delta_weights_tetra(edges, dE))
        return res * self.factor_num_tetra
