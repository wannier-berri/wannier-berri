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

from .system import System
from ..__utility import real_recip_lattice


class SystemSparse(System):
    """"""

    def __init__(self, real_lattice, matrices={}, symmetrize_info=None, **parameters):

        self.real_lattice, self.recip_lattice = real_recip_lattice(real_lattice=real_lattice)
        self.set_parameters(**parameters)
        if not hasattr(self, 'num_wann') or self.num_wann is None:
            self.num_wann = max(getnband(m) for m in matrices.values())

        assert 'Ham' in matrices, "Hamiltonian ('Ham') should be provided in matrices"
        irvec_set = set()
        irvec_set.add((0, 0, 0))
        for m in matrices.values():
            irvec_set.update(set(list(m.keys())))
        self.iRvec = np.array(list(irvec_set))

        for k, v in matrices.items():
            shape = getshape(v)
            if shape is not None:
                print((self.num_wann, self.num_wann, self.nRvec) + shape)
                X = np.zeros((self.num_wann, self.num_wann, self.nRvec) + shape, dtype=complex)
                for R, v1 in v.items():
                    iR = self.iRvec.tolist().index(list(R))
                    for j, h in v1.items():
                        X[j[0], j[1], iR] = h
                self.set_R_mat(k, X)
            else:
                print(f"WARNING: {k} is empty")

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
