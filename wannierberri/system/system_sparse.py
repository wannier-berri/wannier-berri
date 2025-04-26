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
import warnings
import numpy as np

from ..fourier.rvectors import Rvectors
from .system_R import System_R


class SystemSparse(System_R):
    """"""

    def __init__(self, real_lattice,
                 wannier_centers_red=None,
                 wannier_centers_cart=None,
                 matrices=None,
                 num_wann=None,
                 symmetrize_info=None,
                 **parameters):

        parameters = {}

        if matrices is None:
            matrices = {}
        super().__init__(**parameters)
        self.real_lattice = real_lattice
        if num_wann is None:
            self.num_wann = max(getnband(m) for m in matrices.values())
        self.set_wannier_centers(wannier_centers_cart=wannier_centers_cart,
                                 wannier_centers_red=wannier_centers_red)
        self.num_wann = self.wannier_centers_cart.shape[0]

        assert 'Ham' in matrices, "Hamiltonian ('Ham') should be provided in matrices"
        irvec_set = set()
        irvec_set.add((0, 0, 0))
        for m in matrices.values():
            irvec_set.update(set(list(m.keys())))
        iRvec = np.array(list(irvec_set))
        self.rvec = Rvectors(
            lattice=self.real_lattice,
            iRvec=iRvec,
            shifts_left_red=self.wannier_centers_red,
        )

        for k, v in matrices.items():
            shape = getshape(v)
            if shape is not None:
                X = np.zeros((self.rvec.nRvec, self.num_wann, self.num_wann) + shape, dtype=complex)
                for R, v1 in v.items():
                    iR = self.rvec.iR(R)
                    for j, h in v1.items():
                        X[iR, j[0], j[1]] = h
                self.set_R_mat(k, X)
            else:
                warnings.warn(f"{k} is empty")

        self.do_at_end_of_init()
        if symmetrize_info is not None:
            self.symmetrize(**symmetrize_info)



def getshape(dic):
    for k, v in dic.items():
        for k1, v1 in v.items():
            return np.shape(v1)
    return None


def getnband(dic):
    nband = -1
    for m in dic.values():
        for ib in m.keys():
            nband = max(nband, ib[0], ib[1])
    return nband + 1
