#------------------------------------------------------------#
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
#------------------------------------------------------------#

from collections import defaultdict
import lazy_property
import numpy as np
from numba import njit


@njit
def weights_tetra(efall, e0, e1, e2, e3, der=0):

    e = [e0, e1, e2, e3]

    #    print (e0,e1,e2,e3,der)
    e = np.array(sorted([e0, e1, e2, e3]))
    # a dirty trick to avoid divisions by zero
    for i in range(3):
        if abs(e[i + 1] - e[i]) < 1e-12:
            e[i + 1:] += 1e-10
    e1, e2, e3, e4 = e

    nEF = len(efall)
    occ = np.zeros((nEF))
    denom3 = 1. / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    denom2 = 1. / ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4 - e2))
    denom1 = 1. / ((e2 - e1) * (e3 - e1) * (e4 - e1))

    if der == 0:
        c10 = -e1**3 * denom1
        c30 = -e4**3 * denom3 + 1.
        c20 = (e1**2 * (e3 - e2) * (e4 - e2) - (e2**2 * e4) * (e1 - e3) - e1 * e2 * e3 * (e2 - e4)) * denom2
    if der <= 3:
        c13 = denom1
        c33 = denom3
        c23 = denom2 * (e1 + e2 - e3 - e4)
    if der <= 2:
        c12 = -3 * e1 * denom1
        c22 = (((e3 - e2) * (e4 - e2)) - (e1 - e3) * (2 * e2 + e4) - (e3 + e1 + e2) * (e2 - e4)) * denom2
        c32 = -3 * e4 * denom3
    if der <= 1:
        c11 = 3 * e1**2 * denom1
        c21 = (
            -2 * e1 * ((e3 - e2) * (e4 - e2)) + (2 * e2 * e4 + e2**2) * (e1 - e3) + (e1 * e2 + e2 * e3 + e1 * e3) *
            (e2 - e4)) * denom2
        c31 = 3 * e4**2 * denom3

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


#@njit
def get_borders(A, degen_thresh, degen_Kramers=False):
    borders = [0] + list(np.where((A[1:] - A[:-1]) > degen_thresh)[0] + 1) + [len(A)]
    if degen_Kramers:
        borders = [i for i in borders if i % 2 == 0]
    return [[ib1, ib2] for ib1, ib2 in zip(borders, borders[1:])]


#@njit
def get_bands_in_range(emin, emax, Eband, degen_thresh=-1, degen_Kramers=False, Ebandmin=None, Ebandmax=None):
    if Ebandmin is None:
        Ebandmin = Eband
    if Ebandmax is None:
        Ebandmax = Eband
    bands = []
    for ib1, ib2 in get_borders(Eband, degen_thresh, degen_Kramers=degen_Kramers):
        if Ebandmax[ib1:ib2].max() >= emin and Ebandmax[ib1:ib2].min() <= emax:
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


def weights_parallelepiped(efermi, Ecenter, Ecorner, der=0):
    occ = np.zeros((efermi.shape))
    Ecorner = np.reshape(Ecorner, (2, 2, 2))
    for iface in 0, 1:
        for Eface in Ecorner[iface, :, :], Ecorner[:, iface, :], Ecorner[:, :, iface]:
            occ += weights_tetra(efermi, Ecenter, Eface[0, 0], Eface[0, 1], Eface[1, 1], der=der)
            occ += weights_tetra(efermi, Ecenter, Eface[0, 0], Eface[1, 0], Eface[1, 1], der=der)
    return occ / 12.


class TetraWeights():
    """the idea is to make a lazy evaluation, i.e. the weights are evaluated only once for a particular ik,ib
       the Fermi level list remains the same throughout calculation"""

    def __init__(self, eCenter, eCorners):
        self.nk, self.nb = eCenter.shape
        assert eCorners.shape == (self.nk, 2, 2, 2, self.nb)
        self.eCenter = eCenter
        self.eCorners = eCorners
        self.eFermis = []
        self.weights = defaultdict(lambda: defaultdict(lambda: {}))
        Eall = np.concatenate((self.eCenter[:, None, :], self.eCorners.reshape(self.nk, 8, self.nb)), axis=1)
        self.Emin = Eall.min(axis=1)
        self.Emax = Eall.max(axis=1)
        self.eFermi = None

    @lazy_property.LazyProperty
    def ones(self):
        return np.ones(len(self.eFermi))

    def __weight_1b(self, ik, ib, der):
        if ib not in self.weights[der][ik]:
            self.weights[der][ik][ib] = weights_parallelepiped(
                self.eFermi, self.eCenter[ik, ib], self.eCorners[ik, :, :, :, ib], der=der)
        return self.weights[der][ik][ib]


# this is for fermiocean

    def weights_all_band_groups(self, eFermi, der, degen_thresh=-1, degen_Kramers=False):
        """
             here  the key of the return dict is a pair of integers (ib1,ib2)
        """
        if self.eFermi is None:
            self.eFermi = eFermi
        else:
            assert self.eFermi is eFermi
        res = []
        for ik in range(self.nk):
            bands_in_range = get_bands_in_range(
                self.eFermi[0],
                self.eFermi[-1],
                self.eCenter[ik],
                degen_thresh=degen_thresh,
                degen_Kramers=degen_Kramers,
                Ebandmin=self.Emin[ik],
                Ebandmax=self.Emax[ik])
            weights = {
                (ib1, ib2): sum(self.__weight_1b(ik, ib, der) for ib in range(ib1, ib2)) / (ib2 - ib1)
                for ib1, ib2 in bands_in_range
            }

            if der == 0:
                bandmax = get_bands_below_range(self.eFermi[0], self.eCenter[ik], Ebandmax=self.Emax[ik])
                if len(bands_in_range) > 0:
                    bandmax = min(bandmax, bands_in_range[0][0])
                weights[(0, bandmax)] = self.ones

            res.append(weights)
        return res
