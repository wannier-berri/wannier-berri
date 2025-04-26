#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
# ------------------------------------------------------------

import numpy as np
from time import time
import abc
from functools import cached_property
import warnings

from ..system.system import System
from ..symmetry.point_symmetry import PointGroup
from .Kpoint import KpointBZparallel
from ..utility import one2three


class GridAbstract(abc.ABC):

    @abc.abstractmethod
    def __init__(self, system, use_symmetry, FFT=(1, 1, 1)):
        if use_symmetry:
            if isinstance(system, System):
                self.pointgroup = system.pointgroup
            elif isinstance(system, PointGroup):
                self.pointgroup = system
        else:
            if isinstance(system, System):
                real_lattice = system.real_lattice
            elif isinstance(system, PointGroup):
                real_lattice = system.real_lattice
            else:
                real_lattice = system
            self.pointgroup = PointGroup(real_lattice=real_lattice)
        self.FFT = np.array(FFT)

    @abc.abstractmethod
    def get_K_list(self, use_symmetry=False):
        """ get all K-points in the grid """

    @cached_property
    def points_FFT(self):
        dkx, dky, dkz = 1. / self.FFT
        return np.array(
            [
                np.array([ix * dkx, iy * dky, iz * dkz]) for ix in range(self.FFT[0]) for iy in range(self.FFT[1])
                for iz in range(self.FFT[2])
            ])


class Grid(GridAbstract):
    """ A class containing information about the k-grid.

    Parameters
    -----------
    system : :class:`~wannierberri.system.System`
        which the calculations will be made
    length :  float
        (angstroms) -- in this case the grid is NK[i]=length*||B[i]||/2pi  B- reciprocal lattice
    length_FFT :  float
        (angstroms) -- in this case the FFT grid is NKFFT[i]=length_FFT*||B[i]||/2pi  B- reciprocal lattice
    NK : int  or list or numpy.array(3)
        number of k-points along each directions
    NKFFT : int
        number of k-points in the FFT grid along each directions
    NKdiv : int
        number of k-points in the division (K-) grid along each directions
    use_symmetry : bool
        use symmetries of the system to exclude equivalent points

    Notes
    -----
    `NK`, `NKdiv`, `NKFFT`  may be given as size-3 integer arrays or lists. Also may be just numbers -- in that case the number of kppoints is the same in all directions

    the following conbinations of (NK,NKFFT,NKdiv,length) parameters may be used:

    - `length` (preferred)
    - `NK`
    - `NK`, `NKFFT`
    - `length`, `NKFFT`
    - `NKdiv`, `NKFFT`

    The others will be evaluated automatically.

    """

    def __init__(self, system, length=None, NKdiv=None, NKFFT=None, NK=None, length_FFT=None, use_symmetry=True):

        super().__init__(system=system, use_symmetry=use_symmetry)
        NKFFT_recommended = np.array(system.NKFFT_recommended)
        self.div, self.FFT = determineNK(
            system.periodic, NKdiv, NKFFT, NK, NKFFT_recommended, self.pointgroup, length=length, length_FFT=length_FFT)

    #        self.findif = FiniteDifferences(self.recip_lattice, self.FFT)

    @property
    def str_short(self):
        return f"Grid() with NKdiv={self.div}, NKFFT={self.FFT}, NKtot={self.dense}"

    @property
    def dense(self):
        return self.div * self.FFT

    def get_K_list(self, use_symmetry=True):
        """ returns the list of Symmetry-irreducible K-points"""
        dK = 1. / self.div
        factor = 1. / np.prod(self.div)
        print("generating K_list")
        t00 = time()
        K_list = [
            [
                [
                    KpointBZparallel(
                        K=np.array([x, y, z]) * dK,
                        dK=dK,
                        NKFFT=self.FFT,
                        factor=factor,
                        pointgroup=self.pointgroup,
                        refinement_level=0) for z in range(self.div[2])
                ] for y in range(self.div[1])
            ] for x in range(self.div[0])
        ]
        print(f"Done in {time() - t00} s ")
        if use_symmetry:
            t0 = time()
            print("excluding symmetry-equivalent K-points from initial grid")
            for z in range(self.div[2]):
                for y in range(self.div[1]):
                    for x in range(self.div[0]):
                        KP = K_list[x][y][z]
                        if KP is not None:
                            star = [tuple(k) for k in np.array(np.round(KP.star * self.div), dtype=int) % self.div]
                            for k in star:
                                if k != (x, y, z):
                                    KP.absorb(K_list[k[0]][k[1]][k[2]])
                                    K_list[k[0]][k[1]][k[2]] = None
            print(f"Done in {time() - t0} s ")

        K_list = [K for Kyz in K_list for Kz in Kyz for K in Kz if K is not None]
        print(
            "K_list contains {} Irreducible points({}%) out of initial {}x{}x{}={} grid".format(
                len(K_list), round(len(K_list) / np.prod(self.div) * 100, 2), self.div[0], self.div[1], self.div[2],
                np.prod(self.div)))
        return K_list


def iterate_vector(v1, v2):
    return ((x, y, z) for x in range(v1[0], v2[0]) for y in range(v1[1], v2[1]) for z in range(v1[2], v2[2]))


def autoNK(NK, NKFFTrec, pointgroup):
    # frist determine all symmetric sets between NKFFTmin and 2*NKFFTmin
    FFT_symmetric = np.array([fft for fft in iterate_vector(NKFFTrec, NKFFTrec * 3) if pointgroup.symmetric_grid(fft)])
    NKFFTmin = FFT_symmetric[np.argmin(FFT_symmetric.prod(axis=1))]
    print("Minimal symmetric FFT grid : ", NKFFTmin)
    FFT_symmetric = np.array(
        [fft for fft in iterate_vector(NKFFTmin, NKFFTmin * 2) if pointgroup.symmetric_grid(fft)])
    NKdiv_tmp = np.array(np.round(NK[None, :] / FFT_symmetric), dtype=int)
    NKdiv_tmp[NKdiv_tmp <= 0] = 1
    NKchange = NKdiv_tmp * FFT_symmetric / NK[None, :]
    sel = (NKchange > 1)
    NKchange[sel] = 1. / NKchange[sel]
    NKchange = NKchange.min(axis=1)
    FFT = FFT_symmetric[np.argmax(NKchange)]
    NKdiv = np.array(np.round(NK / FFT), dtype=int)
    NKdiv[NKdiv <= 0] = 1
    return NKdiv, FFT


def determineNK(periodic, NKdiv, NKFFT, NK, NKFFT_recommended, pointgroup, length=None, length_FFT=None):
    # print(f"determining grids from NK={NK} ({type(NK)}), NKdiv={NKdiv} ({type(NKdiv)}), NKFFT={NKFFT} ({type(NKFFT)})")
    NKdiv = one2three(NKdiv)
    NKFFT = one2three(NKFFT)
    NK = one2three(NK)

    if length is not None:
        if NK is None:
            NK = np.array(np.round(length / (2 * np.pi) * np.linalg.norm(pointgroup.recip_lattice, axis=1)), dtype=int)
            # print(f"length={length} was converted into NK={NK}")
        else:
            warnings.warn("length is disregarded in presence of NK")

    if length_FFT is not None:
        if NKFFT is None:
            NKFFT = np.array(
                np.round(length_FFT / (2 * np.pi) * np.linalg.norm(pointgroup.recip_lattice, axis=1)), dtype=int)
            # print(f"length_FFT={length_FFT} was converted into NKFFT={NKFFT}")
        else:
            warnings.warn("length_FFT is disregarded in presence of NKFFT")

    for nkname in 'NKdiv', 'NK', 'NKFFT':
        nk = locals()[nkname]
        if nk is not None:
            assert pointgroup.symmetric_grid(nk), f" {nkname}={nk} is not consistent with the given symmetry "

    if (NKdiv is not None) and (NKFFT is not None):
        if length is not None:
            warnings.warn("length is disregarded in presence of NKdiv,NKFFT")
        elif NK is not None:
            warnings.warn("NK is disregarded in presence of NKdiv,NKFFT")
    elif NK is not None:
        if NKdiv is not None:
            warnings.warn("NKdiv is disregarded in presence of NK or length")
        if NKFFT is not None:
            NKdiv = np.array(np.round(NK / NKFFT), dtype=int)
            NKdiv[NKdiv <= 0] = 1
        else:
            NKdiv, NKFFT = autoNK(NK, NKFFT_recommended, pointgroup)
    else:
        raise ValueError("you need to specify either NK or a pair (NKdiv,NKFFT) or (NK,NKFFT)."
                         f"found NK={NK}, NKdiv={NKdiv}, NKFFT={NKFFT} ")

    if NK is not None:
        if not np.all(NK == NKFFT * NKdiv):
            warnings.warn(f" the requested k-grid {NK} was adjusted to {NKFFT * NKdiv}. ")

    notperiodic = np.logical_not(periodic)
    NKdiv[notperiodic] = 1
    NKFFT[notperiodic] = 1
    # print(f"The grids were set to NKdiv={NKdiv}, NKFFT={NKFFT}, NKtot={NKdiv * NKFFT}")
    return NKdiv, NKFFT
