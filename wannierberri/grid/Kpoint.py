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
# This is an auxilary class for the __evaluate.py  module

import numpy as np
from functools import cached_property
from ..symmetry.point_symmetry import SYMMETRY_PRECISION


class KpointBZ():

    def __init__(self, K=np.zeros(3), dK=np.ones(3), NKFFT=np.ones(3), factor=1., pointgroup=None, refinement_level=-1):
        self.K = np.copy(K)
        self.dK = np.copy(dK)
        self.factor = factor
        self.res = None
        self.NKFFT = np.copy(NKFFT)
        self.pointgroup = pointgroup
        self.refinement_level = refinement_level

    def set_res(self, res):
        self.res = res

    @cached_property
    def Kp_fullBZ(self):
        return self.K / self.NKFFT

    def __str__(self):
        return (
            "coord in rec.lattice = [ " + " , ".join(f"{x:10.6f}" for x in self.K) +
            f" ], refinement level:{self.refinement_level}, factor = {self.factor}"
        )

    @cached_property
    def _max(self):
        return self.res.max  # np.max(self.res_smooth)

    @property
    def evaluated(self):
        return not (self.res is None)

    @property
    def check_evaluated(self):
        if not self.evaluated:
            raise RuntimeError("result for a K-point is called, which is not evaluated")

    @property
    def max(self):
        self.check_evaluated
        return self._max * self.factor

    @property
    def norm(self):
        self.check_evaluated
        return self._norm * self.factor

    @property
    def normder(self):
        self.check_evaluated
        return self._normder * self.factor

    @property
    def get_res(self):
        self.check_evaluated
        return self.res * self.factor


class KpointBZpath(KpointBZ):

    def __init__(self, K=np.zeros(3), pointgroup=None):
        super().__init__(K=np.copy(K), pointgroup=pointgroup)

    def __str__(self):
        return "coord in rec.lattice = [ " + " , ".join(f"{x:10.6f}" for x in self.K) + " ]"


class KpointBZparallel(KpointBZ):
    "describes a Kpoint and the surrounding parallelagramm of size dK x dK x dK"

    @cached_property
    def dK_fullBZ(self):
        return self.dK / self.NKFFT

    @cached_property
    def dK_fullBZ_cart(self):
        return self.dK_fullBZ[:, None] * self.pointgroup.recip_lattice

    @cached_property
    def star(self):
        if self.pointgroup is None:
            return [self.K]
        else:
            return self.pointgroup.star(self.K)

    def __str__(self):
        return super().__str__() + f"dK={self.dK} "

    def absorb(self, other):
        if other is None:
            return
        self.factor += other.factor
        if other.res is not None:
            if self.res is not None:
                raise RuntimeError(
                    f"combining two K-points :\n {self} \n and\n  {other}\n  with calculated result should not happen")
            self.res = other.res

    def equiv(self, other):
        if self.refinement_level != other.refinement_level:
            return False
        dif = self.star[:, None, :] - other.star[None, :, :]
        res = False
        if np.linalg.norm((dif - np.round(dif)), axis=2).min() < SYMMETRY_PRECISION:
            res = True
        return res

    def divide(self, ndiv, periodic, use_symmetry=True):
        assert (ndiv.shape == (3,))
        assert (np.all(ndiv > 0))
        ndiv[np.logical_not(periodic)] = 1  # divide only along periodic directions
        include_original = np.all(ndiv % 2 == 1)

        K0 = self.K
        dK_adpt = self.dK / ndiv
        adpt_shift = (-self.dK + dK_adpt) / 2.
        newfac = self.factor / np.prod(ndiv)
        K_list_add = [
            KpointBZparallel(
                K=K0 + adpt_shift + dK_adpt * np.array([x, y, z]),
                dK=dK_adpt,
                NKFFT=self.NKFFT,
                factor=newfac,
                pointgroup=self.pointgroup,
                refinement_level=self.refinement_level + 1) for x in range(ndiv[0]) for y in range(ndiv[1])
            for z in range(ndiv[2]) if not (include_original and np.all(np.array([x, y, z]) * 2 + 1 == ndiv))
        ]

        if include_original:
            self.factor = newfac
            self.refinement_level += 1
            self.dK = dK_adpt
        else:
            self.factor = 0  # the K-point is "dead" but can be used for starting calculation on a different grid  - not implemented
        if use_symmetry and (self.pointgroup is not None):
            exclude_equiv_points(K_list_add)
        return K_list_add

    @cached_property
    def distGamma(self):
        shift_corners = np.arange(-3, 4)
        corners = np.array([[x, y, z] for x in shift_corners for y in shift_corners for z in shift_corners])
        return np.linalg.norm(((self.K % 1)[None, :] - corners).dot(self.pointgroup.recip_lattice), axis=1).min()


def exclude_equiv_points(K_list, new_points=None):
    # cnt: the number of excluded k-points
    # weight_changed_old: a dictionary that saves the "old" weights, K_list[i].factor,
    #       for k-points that are already calculated (i < n - new_points)
    #       and whose weights are changed by this function

    cnt = 0
    n = len(K_list)

    if new_points is None:
        new_points = n

    K_list_length = np.array([K.distGamma for K in K_list])
    K_list_sort = np.argsort(K_list_length)
    K_list_length = K_list_length[K_list_sort]
    wall = [0] + list(np.where(K_list_length[1:] - K_list_length[:-1] > 1e-4)[0] + 1) + [len(K_list)]

    exclude = []

    # dictionary; key: ik, value: previous factor
    weight_changed_old = {}

    for start, end in zip(wall[:-1], wall[1:]):
        for l in range(start, end):
            i = K_list_sort[l]
            if i not in exclude:
                for m in range(start, end):
                    j = K_list_sort[m]
                    if i >= j:
                        continue
                    # There are two cases:
                    # (i) if i < n - new_points <= j; or
                    # (ii) if n - new_points <= i < j
                    # In both cases, j is excluded
                    if i < n - new_points and j < n - new_points:
                        continue
                    if j not in exclude:
                        if K_list[i].equiv(K_list[j]):
                            print('exclude dbg', i, j, K_list[i].K, K_list[j].K, n, new_points)
                            exclude.append(j)
                            if i < n - new_points:
                                if i not in weight_changed_old:
                                    weight_changed_old[i] = K_list[i].factor
                            K_list[i].absorb(K_list[j])
                            cnt += 1

    for i in sorted(exclude)[-1::-1]:
        del K_list[i]
    return cnt, weight_changed_old
