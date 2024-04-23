import warnings
import numpy as np
import multiprocessing
import functools
from ..__utility import iterate3dpm


class ws_dist_map:

    def __init__(self, iRvec, wannier_centers, mp_grid, real_lattice, npar=multiprocessing.cpu_count()):
        # Find the supercell translation (i.e. the translation by a integer number of
        # supercell vectors, the supercell being defined by the mp_grid) that
        # minimizes the distance between two given Wannier functions, i and j,
        # the first in unit cell 0, the other in unit cell R.
        # I.e., we find the translation to put WF j in the Wigner-Seitz of WF i.
        # We also look for the number of equivalent translation, that happen when w_j,R
        # is on the edge of the WS of w_i,0. The results are stored
        # a dictionary shifts_iR[(iR,i,j)]
        ws_search_size = np.array([2] * 3)
        ws_distance_tol = 1e-5
        cRvec = iRvec.dot(real_lattice)
        mp_grid = np.array(mp_grid)
        shifts_int_all = np.array([ijk for ijk in iterate3dpm(ws_search_size + 1)]) * np.array(mp_grid[None, :])
        self.num_wann = wannier_centers.shape[0]
        self._iRvec_new = dict()
        param = (shifts_int_all, wannier_centers, real_lattice, ws_distance_tol, wannier_centers.shape[0])
        p = multiprocessing.Pool(npar)
        irvec_new_all = p.starmap(functools.partial(ws_dist_stars, param=param), zip(iRvec, cRvec))
        p.close()
        p.join()
        print('irvec_new_all shape', np.shape(irvec_new_all))
        for ir, iR in enumerate(iRvec):
            for ijw, irvec_new in irvec_new_all[ir].items():
                self._add_star(ir, irvec_new, ijw[0], ijw[1])
        self._iRvec_ordered = sorted(self._iRvec_new)
        for ir, R in enumerate(iRvec):
            chsum = 0
            for irnew in self._iRvec_new:
                if ir in self._iRvec_new[irnew]:
                    chsum += self._iRvec_new[irnew][ir]
            chsum = np.abs(chsum - np.ones((self.num_wann, self.num_wann))).sum()
            if chsum > 1e-12:
                warnings.warn(f"Check sum for {ir} : {chsum}")

    def __call__(self, matrix):
        ndim = len(matrix.shape) - 3
        num_wann = matrix.shape[0]
        reshaper = (num_wann, num_wann) + (1,) * ndim
        matrix_new = np.array(
            [
                sum(
                    matrix[:, :, ir] * self._iRvec_new[irvecnew][ir].reshape(reshaper)
                    for ir in self._iRvec_new[irvecnew]) for irvecnew in self._iRvec_ordered
            ]).transpose((1, 2, 0) + tuple(range(3, 3 + ndim)))
        assert (np.abs(matrix_new.sum(axis=2) - matrix.sum(axis=2)).max() < 1e-12)
        return matrix_new

    def _add_star(self, ir, irvec_new, iw, jw):
        weight = 1. / irvec_new.shape[0]
        for irv in irvec_new:
            self._add(ir, irv, iw, jw, weight)

    def _add(self, ir, irvec_new, iw, jw, weight):
        irvec_new = tuple(irvec_new)
        if irvec_new not in self._iRvec_new:
            self._iRvec_new[irvec_new] = dict()
        if ir not in self._iRvec_new[irvec_new]:
            self._iRvec_new[irvec_new][ir] = np.zeros((self.num_wann, self.num_wann), dtype=float)
        self._iRvec_new[irvec_new][ir][iw, jw] += weight


def ws_dist_stars(iRvec, cRvec, param):
    shifts_int_all, wannier_centers, real_lattice, ws_distance_tol, num_wann = param
    irvec_new = {}
    for jw in range(num_wann):
        for iw in range(num_wann):
            # function JW translated in the Wigner-Seitz around function IW
            # and also find its degeneracy, and the integer shifts needed
            # to identify it
            R_in = -wannier_centers[iw] + cRvec + wannier_centers[jw]
            dist = np.linalg.norm(R_in[None, :] + shifts_int_all.dot(real_lattice), axis=1)
            irvec_new[(iw, jw)] = iRvec + shifts_int_all[dist - dist.min() < ws_distance_tol].copy()
    return irvec_new


def wigner_seitz(real_lattice, mp_grid):
    ws_search_size = np.array([1] * 3)
    dist_dim = np.prod((ws_search_size + 1) * 2 + 1)
    origin = divmod((dist_dim + 1), 2)[0] - 1
    real_metric = real_lattice.dot(real_lattice.T)
    mp_grid = np.array(mp_grid)
    irvec = []
    ndegen = []
    for n in iterate3dpm(mp_grid * ws_search_size):
        dist = []
        for i in iterate3dpm((1, 1, 1) + ws_search_size):
            ndiff = n - i * mp_grid
            dist.append(ndiff.dot(real_metric.dot(ndiff)))
        dist = np.array(dist)
        dist_min = np.min(dist)
        if abs(dist[origin] - dist_min) < 1.e-7:
            irvec.append(n)
            ndegen.append(np.sum(abs(dist - dist_min) < 1.e-7))

    return np.array(irvec), np.array(ndegen)
