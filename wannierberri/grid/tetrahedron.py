# ------------------------------------------------------------#
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file 'LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
# ------------------------------------------------------------#

from collections import defaultdict
import numpy as np
from numba import njit

from functools import lru_cache


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
def get_bands_in_range(emin, emax, Eband, degen_thresh=-1, degen_Kramers=False, Ebandmin=None, Ebandmax=None):
    if Ebandmin is None:
        Ebandmin = Eband
    if Ebandmax is None:
        Ebandmax = Eband
    bands = []
    for ib1, ib2 in get_borders(Eband, degen_thresh, degen_Kramers=degen_Kramers):
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

    def weights_all_band_groups(self, eFermi, der, degen_thresh=-1, degen_Kramers=False, Emin=-np.inf, Emax=np.inf):
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
                Ebandmax=self.Emax[ik])
            weights = {
                (ib1, ib2): sum(self.__weight_1b(ief, ik, ib, der) for ib in range(ib1, ib2)) / (ib2 - ib1)
                for ib1, ib2 in bands_in_range
            }

            if der == 0:
                bandmax = get_bands_below_range(eFermi[0], self.eCenter[ik], Ebandmax=self.Emax[ik])
                bandmin = get_bands_below_range(Emin, self.eCenter[ik], Ebandmax=self.Emax[ik])
                if len(bands_in_range) > 0:
                    bandmax = min(bandmax, bands_in_range[0][0])
                if bandmax > bandmin:
                    weights[(bandmin, bandmax)] = ones(len(eFermi))

            if der == -1:
                bandmin = get_bands_above_range(eFermi[-1], self.eCenter[ik], Ebandmin=self.Emin[ik])
                bandmax = get_bands_above_range(Emax, self.eCenter[ik], Ebandmin=self.Emin[ik])
                if len(bands_in_range) > 0:
                    bandmin = max(bandmin, bands_in_range[-1][-1])
                if bandmax > bandmin:
                    weights[(bandmin, bandmax)] = ones(len(eFermi))

            res.append(weights)
        return res


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
