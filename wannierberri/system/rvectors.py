
from functools import cached_property
import functools
from typing import Iterable
import warnings

import numpy as np
from ..__utility import FFT_R_to_k, clear_cached, fourier_q_to_R, iterate3dpm, iterate_nd


class Rvectors:
    """
    Class to represent a vector in real space.

    """

    def __init__(self, lattice, shifts_left_red=None, shifts_right_red=None, iRvec=None, dim=3):
        """
        Initialize the Rvec object.

        Parameters
        ----------
        shift_left : int
            The left shift value for the Rvector (in reduced coordinates).
        shift_right : int
            The right shift value for the Rvector (in reduced coordinates).
        """
        self.lattice = np.array(lattice)

        if shifts_left_red is None:
            self.shifts_left_red = np.zeros((1, 3), dtype=float)
        else:
            self.shifts_left_red = np.array(shifts_left_red)

        if shifts_right_red is None:
            self.shifts_right_red = self.shifts_left_red
        else:
            self.shifts_right_red = np.array(shifts_right_red)

        self._NKFFTrec = None
        if iRvec is not None:
            self.iRvec = np.array(iRvec)

        self.dim = dim

        # print(f"created {self.nRvec} Rvectors with shilts left \n reduced coordinates {self.shifts_left_red} \n and right shifts \n reduced coordinates \n{self.shifts_right_red}\n cartesian coordinates \n{self.shifts_left_cart} \n and \n{self.shifts_right_cart}\n")

    def set_Rvec(self, mp_grid, ws_tolerance=1e-3):
        """
        set the Rvectors for all the shifts (MDRS method)

        Parameters
        ----------
        mp_grid : list
            The Monkhorst-Pack grid
        ws_tolerance : float
            The tolerance for the Wigner-Seitz search

        Note:
            if ws_toleranece is negative, the absolute value is used and all shifts are considered different with tolerance 1e-8 (This is mainly to comply with legacy tests)
            TODO: this should be removed in the future, and test data should be updated
        """
        print("setting Rvec")
        assert len(mp_grid) == 3, "NK should be a list of 3 integers"
        self.mp_grid = mp_grid
        self._NKFFTrec = mp_grid

        # TODO so far set high precision. Later increase,
        # but it may require modification of dest data
        # self.all_shifts_red = UniqueList(tolerance=1e-2
        # but for low-precision data, higher tolerance may be beneficial
        if ws_tolerance > 0:
            num_digits_tol = int(np.ceil(-np.log10(ws_tolerance))) + 1
        else:
            ws_tolerance = abs(ws_tolerance)
            num_digits_tol = 8
        self.all_shifts_red, self.shift_index = np.unique(
            np.round(-self.shifts_left_red[:, None] + self.shifts_right_red[None, :], num_digits_tol).reshape(-1, 3),
            axis=0, return_inverse=True)
        self.shift_index = self.shift_index.reshape(self.nshifts_left, self.nshifts_right)
        self.iRvec_list = []
        self.Ndegen_list = []
        self.iRvec_mod_list = []
        wigner = WignerSeitz(self.lattice, mp_grid=mp_grid, tolerance=ws_tolerance)
        self.iRvec_index_list = []  # indices of every shift in the
        for shift in self.all_shifts_red:
            iRvec, Ndegen, iRvec_mod = wigner(shift_reduced=shift)
            self.iRvec_list.append(iRvec)
            self.Ndegen_list.append(Ndegen)
            self.iRvec_mod_list.append(iRvec_mod)
        self.iRvec = np.array(list(set(tuple(a) for a in np.concatenate(self.iRvec_list))))
        self.clear_cached()
        for i, iRvec in enumerate(self.iRvec_list):
            self.iRvec_index_list.append(np.array([self.iR(R) for R in iRvec]))



    def exclude_zeros(self, XX_R_dic={}, tolerance=1e-8):
        """
        Exclude the zero R-vectors from the list of R-vectors.
        """
        if len(XX_R_dic) == 0:
            return XX_R_dic, self
        include_R = np.ones(self.nRvec, dtype=bool)
        for iR in range(self.nRvec):
            include_R[iR] = any(np.any(np.abs(XX_R[:, :, iR]) > tolerance) for XX_R in XX_R_dic.values())
        XX_R_dic = {key: XX_R[:, :, include_R] for key, XX_R in XX_R_dic.items()}
        rvec_new = Rvectors(lattice=self.lattice, shifts_left_red=self.shifts_left_red,
                            shifts_right_red=self.shifts_right_red, iRvec=self.iRvec[include_R])
        return XX_R_dic, rvec_new


    def remap_XX_R(self, XX_R, iRvec_old):
        """
        remap an old matrix XX_R, from old Rvec, to the current ones

        XX_R should have dimensions (num_wann, num_wann, len(iRvec_old), ....)
        """
        print(f"remapping {XX_R.shape} ")
        assert (XX_R.shape[0] == self.nshifts_left) or (self.nshifts_left == 1)
        assert (XX_R.shape[1] == self.nshifts_right) or (self.nshifts_right == 1)
        XX_R_sum_R_old = XX_R.sum(axis=2)
        XX_R_tmp = np.zeros(tuple(self.mp_grid) + XX_R.shape[:2] + XX_R.shape[3:], dtype=XX_R.dtype)
        for i, iR in enumerate(iRvec_old % self.mp_grid):
            XX_R_tmp[tuple(iR)] += XX_R[:, :, i]
        XX_R_sum_T_tmp = XX_R_tmp.sum(axis=(0, 1, 2))
        assert np.allclose(XX_R_sum_T_tmp, XX_R_sum_R_old), f"XX_R_sum_T_tmp {XX_R_sum_T_tmp} != XX_R_sum_R_old {XX_R_sum_R_old}"
        shape_new = list(XX_R.shape)
        shape_new[2] = self.nRvec
        XX_R_final = np.zeros(shape_new, dtype=XX_R.dtype)
        for a in range(XX_R.shape[0]):
            ia = 1 if self.nshifts_left == 1 else a
            for b in range(XX_R.shape[1]):
                ib = 1 if self.nshifts_right == 1 else b
                ishift = self.shift_index[ia, ib]
                for iRi, iRm, nd in zip(self.iRvec_index_list[ishift],
                                        self.iRvec_mod_list[ishift],
                                        self.Ndegen_list[ishift]):
                    XX_R_final[a, b, iRi] += XX_R_tmp[tuple(iRm) + (a, b)] / nd
        XX_R_sum_R_new = XX_R_final.sum(axis=2)
        assert np.allclose(XX_R_sum_R_new, XX_R_sum_T_tmp), f"XX_R_sum_R_new {XX_R_sum_R_new} != XX_R_sum_T_tmp {XX_R_sum_T_tmp}"
        return XX_R_final


    def NKFFT_recommended(self):
        if self._NKFFTrec is not None:
            return self._NKFFTrec
        NKFFTrec = np.ones(3, dtype=int)
        for i in range(3):
            R = self.iRvec[:, i]
            if len(R[R > 0]) > 0:
                NKFFTrec[i] += R.max()
            if len(R[R < 0]) > 0:
                NKFFTrec[i] -= R.min()
        # check if FFT is enough to fit all R-vectors
        assert np.unique(self.iRvec % NKFFTrec, axis=0).shape[0] == self.iRvec.shape[0]
        return NKFFTrec

    def reorder(self, order_left=None, order_right=None):
        """
        Reorder the Rvectors according to the given order.

        Parameters
        ----------
        order : list
            The new order of the Rvectors.
        """
        if order_left is None and order_right is None:
            order = np.arange(self.nRvec)
            order_left = order
            order_right = order
        if order_right is None:
            order_right = order_left
        if order_left is None:
            order_left = order_right
        self.shifts_left_red = self.shifts_left_red[order_left]
        self.shifts_right_red = self.shifts_right_red[order_right]
        self.clear_cached()


    @property
    def __len__(self):
        """
        Return the number of Rvectors.
        """
        return self.nRvec

    @property
    def nshifts_left(self):
        """
        Return the number of left shifts.
        """
        return self.shifts_left_red.shape[0]

    @property
    def nshifts_right(self):
        """
        Return the number of right shifts.
        """
        return self.shifts_right_red.shape[0]

    @cached_property
    def cRvec(self):
        return self.iRvec.dot(self.lattice)

    @cached_property
    def cRvec_shifted(self):
        """
        R+tj-ti.
        """
        return self.cRvec[None, None, :, :] + self.shifts_diff_cart[:, :, None, :]


    @cached_property
    def shifts_diff_red(self):
        return -self.shifts_left_red[:, np.newaxis] + self.shifts_right_red[np.newaxis, :]

    @cached_property
    def shifts_diff_cart(self):
        return -self.shifts_left_cart[:, np.newaxis] + self.shifts_right_cart[np.newaxis, :]

    def clear_cached(self):
        clear_cached(self, ['diff_wcc_cart', 'cRvec_p_wcc', 'diff_wcc_red',
                            "wannier_centers_reduced", 'cRvec', 'cRvec_p_wcc', 'iR0',
                            'reverseR', 'index_R', 'shifts_diff_red', 'shifts_diff_cart',
                            'shifts_left_cart', 'shifts_right_cart', 'cRvec_shifted'])

    @cached_property
    def shifts_left_cart(self):
        return self.shifts_left_red.dot(self.lattice)

    @cached_property
    def shifts_right_cart(self):
        return self.shifts_right_red.dot(self.lattice)

    @cached_property
    def iR0(self):
        return self.iRvec.tolist().index([0, 0, 0])

    @cached_property
    def index_R(self):
        return {tuple(R): i for i, R in enumerate(self.iRvec)}

    def iR(self, R):
        R = np.array(np.round(R), dtype=int).tolist()
        return self.iRvec.tolist().index(R)

    @cached_property
    def reverseR(self):
        """indices of R vectors that has -R in irvec, and the indices of the corresponding -R vectors."""
        mapping = np.all(self.iRvec[:, None, :] + self.iRvec[None, :, :] == 0, axis=2)
        # check if some R-vectors do not have partners
        notfound = np.where(np.logical_not(mapping.any(axis=1)))[0]
        if len(notfound) > 0 and not self.ignore_mR_not_found:
            for ir in notfound:
                warnings.warn(f"R[{ir}] = {self.iRvec[ir]} does not have a -R partner")
        # check if some R-vectors have more then 1 partner
        morefound = np.where(np.sum(mapping, axis=1) > 1)[0]
        if len(morefound > 0):
            raise RuntimeError(
                f"R vectors number {morefound} have more then one negative partner : "
                f"\n{self.iRvec[morefound]} \n{np.sum(mapping, axis=1)}")
        lst_R, lst_mR = [], []
        for ir1 in range(self.nRvec):
            ir2 = np.where(mapping[ir1])[0]
            if len(ir2) == 1:
                lst_R.append(ir1)
                lst_mR.append(ir2[0])
        lst_R = np.array(lst_R)
        lst_mR = np.array(lst_mR)
        # Check whether the result is correct
        assert np.all(self.iRvec[lst_R] + self.iRvec[lst_mR] == 0)
        return lst_R, lst_mR

    def conj_XX_R(self, XX_R: np.ndarray, ignore_mR_not_found=False):
        """ reverses the R-vector and takes the hermitian conjugate """
        XX_R_new = np.zeros(XX_R.shape, dtype=complex)
        self.ignore_mR_not_found = ignore_mR_not_found
        lst_R, lst_mR = self.reverseR
        XX_R_new[:, :, lst_R] = XX_R[:, :, lst_mR]
        return XX_R_new.swapaxes(0, 1).conj()

    @property
    def nRvec(self):
        return self.iRvec.shape[0]

    def check_hermitian(self, key):
        if key in self._XX_R.keys():
            _X = self.get_R_mat(key).copy()
            assert (np.max(abs(_X - self.conj_XX_R(key=key))) < 1e-8), f"{key} should obey X(-R) = X(R)^+"
        else:
            self.logfile.write(f"{key} is missing, nothing to check\n")

    def set_fft_R_to_k(self, NK, num_wann, numthreads, fftlib='pyfftw'):
        self.fft_R_to_k = FFT_R_to_k(
            self.iRvec,
            NK,
            num_wann,
            numthreads,
            fftlib=fftlib)

    def R_to_k(self, XX_R, der=0, hermitian=True):
        """ converts from real-space matrix elements in Wannier gauge to
            k-space quantities in k-space.
            der [=0] - defines the order of comma-derivative
            hermitian [=True] - consider the matrix hermitian
            WARNING: the input matrix is destroyed, use np.copy to preserve it"""

        for i in range(der):
            shape_cR = np.shape(self.cRvec_shifted)
            XX_R = 1j * XX_R.reshape((XX_R.shape) + (1,)) * self.cRvec_shifted.reshape(
                (shape_cR[0], shape_cR[1], self.nRvec) + (1,) * len(XX_R.shape[3:]) + (3,))
        return self.fft_R_to_k(XX_R, hermitian=hermitian)

    def set_fft_q_to_R(self, mp_grid, kpt_mp_grid, numthreads, fftlib='pyfftw',
                       Ndegen=None):
        if Ndegen is None:
            Ndegen = np.ones(self.nRvec, dtype=int)
        self.fft_q_to_R = functools.partial(
            fourier_q_to_R,
            mp_grid=mp_grid,
            kpt_mp_grid=kpt_mp_grid,
            iRvec=self.iRvec,
            ndegen=Ndegen,
            numthreads=numthreads,
            fftlib=fftlib)

    def q_to_R(self, XXq):
        return self.fft_q_to_R(XXq)


class WignerSeitz:

    def __init__(self, real_lattice, mp_grid, ws_search_size=3, tolerance=1e-5):
        if not isinstance(ws_search_size, Iterable):
            ws_search_size = [ws_search_size] * 3
        self.mp_grid = np.array(mp_grid)
        self.tolerance = tolerance
        self.real_lattice = np.array(real_lattice)
        self.iRvec0 = iterate_nd(self.mp_grid)
        # print (f"iRvec0 : \n{repr(self.iRvec0)}")
        self.cRvec0 = self.iRvec0.dot(self.real_lattice)
        # superlattice = self.real_lattice * np.array(ws_search_size)[:, None]
        super_vectors_i = np.array([ijk for ijk in iterate3dpm(ws_search_size)]) * self.mp_grid[None, :]
        # print (f"super_vectors_i : \n{repr(super_vectors_i)}")
        super_vectors_c = super_vectors_i.dot(self.real_lattice)
        self.iRvec_search = np.array([self.iRvec0 + ijk[None, :]
                                 for ijk in super_vectors_i]).swapaxes(0, 1)
        self.cRvec_search = np.array([self.cRvec0 + ijk[None, :]
                                 for ijk in super_vectors_c]).swapaxes(0, 1)


    def __call__(self, shift_reduced):
        shift_cartesian = shift_reduced.dot(self.real_lattice)
        dist = np.linalg.norm(self.cRvec_search + shift_cartesian, axis=2)
        # print (f"distances : \n{repr(dist)}")
        Ndegen = []
        iRvec = []
        for i, iRs in enumerate(self.iRvec_search):
            # print (f"{i} : {repr(iRs)}  : {dist[i]}")
            dist_min = dist[i].min()
            select = np.where(abs(dist[i] - dist_min) < self.tolerance)[0]
            # print (f"selecting {select} with distance {dist[i][select]} and min {dist_min}")
            ndeg = len(select)
            for j in select:
                # print (f"    {iRs[j]} : {dist[i][j]} {ndeg}")
                iRvec.append(iRs[j])
                Ndegen.append(ndeg)
        iRvec = np.array(iRvec)
        Ndegen = np.array(Ndegen)
        return iRvec, Ndegen, iRvec % self.mp_grid
