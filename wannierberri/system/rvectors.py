
from functools import cached_property
import functools
import warnings

import numpy as np
from ..__utility import FFT_R_to_k, clear_cached, fourier_q_to_R


class Rvectors:
    """
    Class to represent a vector in real space.

    """

    def __init__(self, lattice, shifts_left_red=None, shifts_right_red=None, NK=None, iRvec=None, dim=3):
        """
        Initialize the Rvec object.

        Parameters
        ----------
        NK : int
            The number of k-points in the system.
        shift_left : int
            The left shift value for the Rvector (in reduced coordinates).
        shift_right : int
            The right shift value for the Rvector (in reduced coordinates).
        """
        assert (NK is not None) or (iRvec is not None), "Either NK or iRvec must be provided"
        if iRvec is not None:
            self.iRvec = np.array(iRvec)
        self.NK = NK
        self.lattice = np.array(lattice)

        if shifts_left_red is None:
            self.shifts_left_red = np.zeros((1, 3), dtype=float)
        else:
            self.shifts_left_red = np.array(shifts_left_red)

        if shifts_right_red is None:
            self.shifts_right_red = self.shifts_left_red
        else:
            self.shifts_right_red = np.array(shifts_right_red)

        self.dim = dim

    def NKFFT_recommended(self):
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
                            "wannier_centers_reduced", 'cRvec', 'cRvec_p_wcc',
                            'reverseR', 'index_R', 'shifts_diff_red', 'shifts_diff_cart',
                            'shifts_left_cart', 'shifts_right_cart', 'cRvec_shifted'])

    @cached_property
    def shifts_left_cart(self):
        return self.shifts_left_red.dot(self.lattice)

    @cached_property
    def shifts_right_cart(self):
        return self.shifts_right_red.dot(self.lattice)

    @property
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
