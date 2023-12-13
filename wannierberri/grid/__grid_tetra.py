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
from .__Kpoint_tetra import KpointBZtetra
from .__grid import GridAbstract
from ..__utility import angle_vectors_deg


class GridTetra(GridAbstract):
    """ A class containing information about the k-grid.konsisting of tetrahedra

    Parameters
    -----------
    system : :class:`~wannierberri.system.System`
        which the calculations will be made
    length :  float
        (angstroms) -- in this case the grid is NK[i]=length*||B[i]||/2pi  B- reciprocal lattice
    length_FFT :  float
        (angstroms) -- in this case the FFT grid is NKFFT[i]=length_FFT*||B[i]||/2pi  B- reciprocal lattice
    NKFFT : int
        number of k-points in the FFT grid along each directions
    IBZ_tetra : list
        list of tetrahedra describing the irreducible wedge of the Brillouin zone. By default, the stace is just divided into 5 tetrahedra
    Notes
    -----
     `NKFFT`  may be given as size-3 integer arrays or lists. Also may be just numbers -- in that case the number of kppoints is the same in all directions

    either lewngth_FFT of NKFFT should be provided

    """

    def __init__(self, system, length, NKFFT=None, IBZ_tetra=None, weights=None,
            refine_by_volume=True,
            refine_by_size=True,
            length_size=None
                ):

        if NKFFT is None:
            self.FFT = system.NKFFT_recommended
        elif isinstance(NKFFT, int):
            self.FFT = np.array([NKFFT] * 3)
        else:
            self.FFT = np.array(NKFFT)

        self.recip_lattice_reduced = system.recip_lattice / self.FFT[:, None]
        print("reduced reciprocal lattice : \n", self.recip_lattice_reduced)
        if IBZ_tetra is None:   # divide the full reciprocal unit cell into 5 tetrahedra -
            print("WARNING : irreducible wedge not provided, no use of symmetries")
            tetrahedra = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   [[1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1]],
                                   [[1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
                                   [[0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1]],
                                   [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
                                   ]) - np.array([0.5, 0.5, 0.5])[None, None, :]
        else:
            tetrahedra = np.array(IBZ_tetra)
        print("using starting tetrahedra with vertices \n", tetrahedra)
        _weights = np.array([tetra_volume(t) for t in tetrahedra])
        print(f"volumes of tetrahedra are {_weights}, total = {sum(_weights)} ")
        if weights is None:
            weights = _weights / sum(_weights)
        else:
            weights = np.array(weights) * _weights
        print(f"weights of tetrahedra are {weights}, total = {sum(weights)} ")
        self.K_list = []
        print("generating starting K_list")
        for tetr, w in zip(tetrahedra, weights):
            K = KpointBZtetra(vertices=tetr, K=0, NKFFT=self.FFT, factor=w, basis=self.recip_lattice_reduced, refinement_level=0, split_level=0)
            print(K)
            print(K.size)
            self.K_list.append(K)

        if refine_by_volume:
            dkmax = 2 * np.pi / length
            vmax = dkmax**3 / np.linalg.det(self.recip_lattice_reduced)
            self.split_tetra_volume(vmax)
            print("refinement by volume done")
        if refine_by_size:
            if length_size is None:
                length_size = 0.5 * length
            dkmax = (2 * np.pi / length_size) * np.sqrt(2)
            self.split_tetra_size(dkmax)
            print("refinement by size done")


    def split_tetra_size(self, dkmax):
        """split tetrahedra that have at lkeast one edge larger than dkmax"""
        while True:
            print(f"maximal tetrahedron size for now is {self.size_max} ({len(self.K_list)}), we need to refine down to size {dkmax}")
            volumes = [tetra_volume(K.vertices) for K in self.K_list]
            print("the volume is ", sum(volumes), min(volumes), max(volumes), np.mean(volumes))
            # print ("sizes now are ",self.sizes)
            if self.size_max < dkmax:
                break
            klist = []
            for K in self.K_list:
                if K.size > dkmax:
                    klist += K.divide(ndiv=2, refine=False)
                else:
                    klist.append(K)
            self.K_list = klist


    def split_tetra_volume(self, vmax):
        """split tetrahedra that have at least one edge larger than dkmax"""
        while True:
            volumes = [tetra_volume(K.vertices) for K in self.K_list]
            print(f"maximal tetrahedron size for now is {max(volumes)} ({len(self.K_list)}), we need to refine down to size {vmax}")
            print("the volume is ", sum(volumes), min(volumes), max(volumes), np.mean(volumes))
            if max(volumes) < vmax:
                break
            klist = []
            for K, v in zip(self.K_list, volumes):
                if v > vmax:
                    klist += K.divide(ndiv=2, refine=False)
                else:
                    klist.append(K)
            self.K_list = klist

    @property
    def size_max(self):
        return self.sizes.max()

    @property
    def sizes(self):
        return np.array([K.size for K in self.K_list])

    @property
    def str_short(self):
        return "GridTetra() with {} tetrahedrons, NKFFT={}, NKtot={}".format(len(self.K_list), self.FFT, np.prod(self.FFT) * len(self.K_list))

    def get_K_list(self, use_symmetry=True):
        """ returns the list of Symmetry-irreducible K-points"""
        return [K.copy() for K in self.K_list]


def tetra_volume(vortices):
    return abs(np.linalg.det(vortices[1:] - vortices[0][None, :])) / 6.


class GridTrigonal(GridTetra):
    """ good choice for Tellurium"""

    def __init__(self, system, length, **kwargs):

        # these ones are for the case when the reciprocal lattice vectors form a 120deg angle
        IBZ_tetra = np.array([
                    [[0, 0, 0], [1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.0], [1 / 3, 2 / 3, 0.5]],
                    [[0, 0, 0], [2 / 3, 1 / 3, 0.5], [2 / 3, 1 / 3, 0.0], [1 / 3, 2 / 3, 0.5]],
                    [[0, 0, 0], [2 / 3, 1 / 3, 0.5], [0, 0, 0.5], [1 / 3, 2 / 3, 0.5]],
                              ])

        b1, b2, b3 = system.recip_lattice
        assert angle_vectors_deg(b1, b3) == 90
        assert angle_vectors_deg(b2, b3) == 90
        assert angle_vectors_deg(b1, b2) in (60, 120)
        if angle_vectors_deg(b1, b2) == 60:
            IBZ_tetra[:, :, 0] = IBZ_tetra[:, :, 0] - IBZ_tetra[:, :, 1]

        super().__init__(system, length, IBZ_tetra=IBZ_tetra, **kwargs)


class GridTrigonalH(GridTetra):
    """ good choice for Tellurium conduction/valence band, only a small part near the H-point is considered, use NKFFT=1"""

    def __init__(self, system, length, x=0.5, NKFFT=1, **kwargs):

        # these ones are for the case when the reciprocal lattice vectors form a 120deg angle
        H = np.array([2 / 3, 1 / 3, 1 / 2])
        K = np.array([2 / 3, 1 / 3, 0])
        H1 = np.array([1 / 3, -1 / 3, 1 / 2])
        A = (H - K)
        IBZ_tetra = np.array([[H, H + x * (K - H), H + x * (A - H), H + x * (H1 - H)],
                              [H, H - x * (K - H), H + x * (A - H), H + x * (H1 - H)]])

        b1, b2, b3 = system.recip_lattice
        assert angle_vectors_deg(b1, b3) == 90
        assert angle_vectors_deg(b2, b3) == 90
        assert angle_vectors_deg(b1, b2) in (60, 120)
        if angle_vectors_deg(b1, b2) == 60:
            IBZ_tetra[:, :, 0] = IBZ_tetra[:, :, 0] - IBZ_tetra[:, :, 1]

        super().__init__(system, length, IBZ_tetra=IBZ_tetra, weights=[12, 12], NKFFT=1, **kwargs)
