
from functools import cached_property
from typing import Iterable
import warnings

import numpy as np
from ..utility import clear_cached, iterate3dpm, iterate_nd
from .fft import FFT_R_to_k, execute_fft


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
        self.fft_R2k_set = False
        self.fft_q2R_set = False

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


    def copy(self):
        """
        Return a copy of the Rvector object.
        """
        return Rvectors(lattice=np.copy(self.lattice),
                        shifts_left_red=np.copy(self.shifts_left_red),
                        shifts_right_red=np.copy(self.shifts_right_red),
                        iRvec=np.copy(self.iRvec))

    def exclude_zeros(self, XX_R_dic={}, tolerance=1e-8):
        """
        Exclude the zero R-vectors from the list of R-vectors.
        """
        if len(XX_R_dic) == 0:
            return XX_R_dic, self
        include_R = np.ones(self.nRvec, dtype=bool)
        for iR in range(self.nRvec):
            include_R[iR] = any(np.any(np.abs(XX_R[iR]) > tolerance) for XX_R in XX_R_dic.values())
        XX_R_dic = {key: XX_R[include_R] for key, XX_R in XX_R_dic.items()}
        rvec_new = Rvectors(lattice=self.lattice, shifts_left_red=self.shifts_left_red,
                            shifts_right_red=self.shifts_right_red, iRvec=self.iRvec[include_R])
        return XX_R_dic, rvec_new


    def remap_XX_R(self, XX_R, iRvec_old):
        """
        remap an old matrix XX_R, from old Rvec, to the current ones

        XX_R should have dimensions (num_wann, num_wann, len(iRvec_old), ....)
        """
        print(f"remapping {XX_R.shape} ")
        assert (XX_R.shape[1] == self.nshifts_left) or (self.nshifts_left == 1)
        assert (XX_R.shape[2] == self.nshifts_right) or (self.nshifts_right == 1)
        XX_R_sum_old = XX_R.sum(axis=0)
        XX_R_tmp = np.zeros(tuple(self.mp_grid) + XX_R.shape[1:], dtype=XX_R.dtype)
        for i, iR in enumerate(iRvec_old % self.mp_grid):
            XX_R_tmp[tuple(iR)] += XX_R[i]
        XX_R_sum_tmp = XX_R_tmp.sum(axis=(0, 1, 2))
        assert np.allclose(XX_R_sum_tmp, XX_R_sum_old), f"XX_R_sum_T_tmp {XX_R_sum_tmp} != XX_R_sum_R_old {XX_R_sum_old}"
        return self.remap_XX_from_grid_to_list_R(XX_R_tmp)

    def remap_XX_from_grid_to_list_R(self, XX_R_grid):
        """
        remap the matrix from the grid to the list of R-vectors,
        taking into account the wannier centers

        Parameters
        ----------
        XX_R_grid : np.ndarray(shape=(mp_grid[0], mp_grid[1], mp_grid[2], num_wann_l, num_wann_r, ...))
            The matrix in the grid representation.

        Returns
        -------
        XX_R_new : np.ndarray(shape=(num_wann_l, num_wann_r, nRvec, ...))
            The matrix in the list of R-vectors representation.
        """
        assert XX_R_grid.shape[0:3] == tuple(self.mp_grid), f"XX_R_grid {XX_R_grid.shape} should be {self.mp_grid}"
        nl = self.nshifts_left
        nr = self.nshifts_right
        assert (nl == 1) or (XX_R_grid.shape[3] == nl), f"XX_R_grid {XX_R_grid.shape} should have {nl} lWFs"
        assert (nr == 1) or (XX_R_grid.shape[4] == nr), f"XX_R_grid {XX_R_grid.shape} should have {nr} rWFs"
        XX_R_sum_grid = XX_R_grid.sum(axis=(0, 1, 2))
        shape_new = (self.nRvec,) + XX_R_grid.shape[3:]
        num_wann_l = XX_R_grid.shape[3]
        num_wann_r = XX_R_grid.shape[4]
        XX_R_new = np.zeros(shape_new, dtype=XX_R_grid.dtype)
        for a in range(num_wann_l):
            ia = 0 if self.nshifts_left == 1 else a
            for b in range(num_wann_r):
                ib = 0 if self.nshifts_right == 1 else b
                ishift = self.shift_index[ia, ib]
                for iRi, iRm, nd in zip(self.iRvec_index_list[ishift],
                                        self.iRvec_mod_list[ishift],
                                        self.Ndegen_list[ishift]):
                    XX_R_new[iRi, a, b] += XX_R_grid[tuple(iRm) + (a, b)] / nd
        XX_R_sum_new = XX_R_new.sum(axis=0)
        assert np.allclose(XX_R_sum_new, XX_R_sum_grid), f"XX_R_sum_R_new {XX_R_sum_new} != XX_R_sum_T_tmp {XX_R_sum_grid}"
        return XX_R_new


    def remap_XX_from_grid_to_list_RR(self, XX_RR_grid):
        """
        remap the matrix from the double grid to the double list of R-vectors,
        taking into account the wannier centers (the "left" are used as the central, and the "right" are used asboth left and right, if you understand what I mean)

        Parameters
        ----------
        XX_RR_grid : np.ndarray(shape=(mp_grid[0], mp_grid[1], mp_grid[2], mp_grid[0], mp_grid[1], mp_grid[2], num_wann_r, num_wann_l, num_wann_r, ...))
            The matrix in the grid representation.

        Returns
        -------
        XX_R_new : np.ndarray(shape=(num_wann_r, num_wann_l, num_wann_r, nRvec, nRvec, ...))
            The matrix in the list of R-vectors representation.
        """
        XX_RR_sum_grid = XX_RR_grid.sum(axis=(0, 1, 2, 3, 4, 5))
        num_wann_r = XX_RR_grid.shape[6]
        num_wann_l = XX_RR_grid.shape[8]
        print(f"remapping {XX_RR_grid.shape} num_wann_r={num_wann_r}, num_wann_l={num_wann_l}")
        nl = self.nshifts_left
        nr = self.nshifts_right
        assert (nr == 1) or (XX_RR_grid.shape[6] == nr), f"XX_RR_grid {XX_RR_grid.shape} should have {nr} WFs"
        assert (nr == 1) or (XX_RR_grid.shape[7] == nr), f"XX_RR_grid {XX_RR_grid.shape} should have {nr} WFs"
        assert (nl == 1) or (XX_RR_grid.shape[8] == nl), f"XX_RR_grid {XX_RR_grid.shape} should have {nl} right shifts"

        shape_new = XX_RR_grid.shape[6:9] + (self.nRvec,) * 2 + XX_RR_grid.shape[9:]
        print(f"shape_new {shape_new}")
        XX_RR_new = np.zeros(shape_new, dtype=XX_RR_grid.dtype)
        for a in range(num_wann_r):
            ia = 0 if self.nshifts_right == 1 else a
            for b in range(num_wann_r):
                ib = 0 if self.nshifts_right == 1 else b
                for c in range(num_wann_l):
                    ic = 0 if self.nshifts_left == 1 else c
                    ishift1 = self.shift_index[ic, ia]
                    ishift2 = self.shift_index[ic, ib]
                    print(f"a,b,c = {a},{b},{c} : {ishift1}, {ishift2}")
                    for iRi1, iRm1, nd1 in zip(self.iRvec_index_list[ishift1],
                                            self.iRvec_mod_list[ishift1],
                                            self.Ndegen_list[ishift1]):
                        for iRi2, iRm2, nd2 in zip(self.iRvec_index_list[ishift2],
                                                self.iRvec_mod_list[ishift2],
                                                self.Ndegen_list[ishift2]):
                            XX_RR_new[a, b, c, iRi1, iRi2] += XX_RR_grid[tuple(iRm1) + tuple(iRm2) + (a, b, c)] / (nd1 * nd2)
        XX_R_sum_new = XX_RR_new.sum(axis=(3, 4))
        assert np.allclose(XX_R_sum_new, XX_RR_sum_grid), f"XX_R_sum_R_new {XX_R_sum_new} != XX_R_sum_T_tmp {XX_RR_sum_grid}"
        return XX_RR_new


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
        return self.cRvec[:, None, None, :] + self.shifts_diff_cart[None, :, :, :]


    @cached_property
    def shifts_diff_red(self):
        return -self.shifts_left_red[:, np.newaxis] + self.shifts_right_red[np.newaxis, :]

    @cached_property
    def shifts_diff_cart(self):
        return -self.shifts_left_cart[:, np.newaxis] + self.shifts_right_cart[np.newaxis, :]

    def clear_cached(self):
        clear_cached(self, ['diff_wcc_cart', 'cRvec_p_wcc', 'diff_wcc_red',
                            "wannier_centers_red", 'cRvec', 'cRvec_p_wcc', 'iR0',
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
        XX_R_new[lst_R] = XX_R[lst_mR]
        return XX_R_new.swapaxes(1, 2).conj()

    @property
    def nRvec(self):
        return self.iRvec.shape[0]

    def check_hermitian(self, key):
        if key in self._XX_R.keys():
            _X = self.get_R_mat(key).copy()
            assert (np.max(abs(_X - self.conj_XX_R(key=key))) < 1e-8), f"{key} should obey X(-R) = X(R)^+"
        else:
            self.logfile.write(f"{key} is missing, nothing to check\n")

    def set_fft_R_to_k(self, NK, num_wann, numthreads, fftlib='fftw', dK=(0, 0, 0)):
        """
        set the FFT for the R to k conversion

        Parameters
        ----------
        NK : tuple of 3 integers
            The number of k-points in the Monkhorst-Pack grid
        num_wann : int
            The number of Wannier functions
        numthreads : int
            The number of threads to paralize the FFT (if possible)
        fftlib : str
            The FFT library to use ('fftw' or 'numpy' or 'slow')
        dK : tuple of 3 floats in range [0,1)
            the shift of the grid in coordinates of the reciprocal lattice divided by the grid
        """
        self.dK = np.array(dK)
        self. expdK = np.exp(2j * np.pi * self.iRvec.dot(self.dK))

        self.fft_R_to_k = FFT_R_to_k(
            self.iRvec,
            NK,
            num_wann,
            numthreads,
            fftlib=fftlib)
        self.fft_R2k_set = True

    def apply_expdK(self, XX_R):
        """ apply the exp(2 pi i dK R) to the matrix elements in real space
            XX_R should be of shape (num_wann, num_wann, nRvec, ...)
            """
        assert XX_R.shape[0] == self.nRvec, f"XX_R {XX_R.shape} should have {self.nRvec} R-vectors"
        shape = [self.expdK.shape[0]] + [1] * (XX_R.ndim - 1)
        return XX_R * self.expdK.reshape(shape)

    def derivative(self, XX_R):
        """ apply the derivative to the matrix elements in real space
            XX_R should be of shape (num_wann, num_wann, nRvec, ...)
            the returned array has one more *last) dimension, with size 3
            """
        shape_cR = np.shape(self.cRvec_shifted)
        return 1j * XX_R.reshape((XX_R.shape) + (1,)) * self.cRvec_shifted.reshape(
            (self.nRvec, shape_cR[1], shape_cR[2]) + (1,) * len(XX_R.shape[3:]) + (3,))

    def R_to_k(self, XX_R, der=0, hermitian=True):
        """ converts from real-space matrix elements in Wannier gauge to
            k-space quantities in k-space.
            der [=0] - defines the order of comma-derivative
            hermitian [=True] - consider the matrix hermitian
            WARNING: the input matrix is destroyed, use np.copy to preserve it"""

        assert self.fft_R2k_set, "FFT_R_to_k is not set, please set it first"
        for i in range(der):
            XX_R = self.derivative(XX_R)
        return self.fft_R_to_k(XX_R, hermitian=hermitian)

    def set_fft_q_to_R(self, kpt_red, numthreads=1, fftlib='fftw'):
        """
        set the FFT for the q to R conversion

        Parameters
        ----------
        kpt_red : list
            The k-point of Monkhorst-Pack grid in reduced coordinates
        numthreads : int
            The number of threads for the FFT
        fftlib : str
            The FFT library to use ('fftw' or 'numpy' or 'slow')
        """
        kpt_red = np.array(kpt_red)
        kpt_red_mp = kpt_red * self.mp_grid[None, :]
        kpt_red_mp_int = np.round(kpt_red_mp).astype(int)
        assert kpt_red.shape == (np.prod(self.mp_grid), 3), f"kpt_red {kpt_red} should be an array of shape NK_mp x 3 (NK_mp={np.prod(self.mp_grid)})"
        assert np.allclose(kpt_red_mp_int, kpt_red_mp), f"kpt_red {kpt_red} should be a uniform grid of  {self.mp_grid} kpoints"
        self.kpt_mp_grid = [tuple(k) for k in kpt_red_mp_int % self.mp_grid]
        if (0, 0, 0) not in self.kpt_mp_grid:
            raise ValueError(
                "the grid of k-points read from .chk file is not Gamma-centered. Please, use Gamma-centered grids in the ab initio calculation"
            )
        assert len(self.kpt_mp_grid) == np.prod(self.mp_grid), f"the grid of k-points read from .chk file is not {self.mp_grid} kpoints"
        assert len(self.kpt_mp_grid) == len(set(self.kpt_mp_grid)), "the grid of k-points read from .chk file has duplicates"
        self.fft_num_threads = numthreads
        self.fftlib_q2R = fftlib
        self.fft_q2R_set = True


    def q_to_R(self, AA_q):
        assert self.fft_q2R_set, "FFT_q_to_R is not set, please set it first using set_fft_q_to_R"
        shapeA = AA_q.shape[1:]  # remember the shapes after q
        AA_q_mp = np.zeros(tuple(self.mp_grid) + shapeA, dtype=complex)
        for i, k in enumerate(self.kpt_mp_grid):
            AA_q_mp[k] = AA_q[i]
        AA_q_mp = execute_fft(AA_q_mp, axes=(0, 1, 2), numthreads=self.fft_num_threads, fftlib=self.fftlib_q2R, destroy=False) / np.prod(self.mp_grid)
        return self.remap_XX_from_grid_to_list_R(AA_q_mp)

    def qq_to_RR(self, AA_qq):
        assert self.fft_q2R_set, "FFT_q_to_R is not set, please set it first using set_fft_q_to_R"
        shapeA = AA_qq.shape[2:]  # remember the shapes after q
        AA_qq_mp = np.zeros(tuple(self.mp_grid) * 2 + shapeA, dtype=complex)
        for i, k1 in enumerate(self.kpt_mp_grid):
            for j, k2 in enumerate(self.kpt_mp_grid):
                AA_qq_mp[k1 + k2] = AA_qq[i, j]
        AA_qq_mp = execute_fft(AA_qq_mp, axes=(0, 1, 2), numthreads=self.fft_num_threads, fftlib=self.fftlib_q2R, destroy=False, inverse=True) / np.prod(self.mp_grid)
        AA_qq_mp = execute_fft(AA_qq_mp, axes=(3, 4, 5), numthreads=self.fft_num_threads, fftlib=self.fftlib_q2R, destroy=False, inverse=False) / np.prod(self.mp_grid)
        return self.remap_XX_from_grid_to_list_RR(AA_qq_mp)




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
