from functools import cached_property
from time import time
import warnings

import numpy as np

try:
    import pyfftw
    PYFFTW_IMPORTED = True
except Exception as err:
    PYFFTW_IMPORTED = False
    warnings.warn(f"error importing  `pyfftw` : {err} \n will use numpy instead \n")


def fft_W(inp, axes, inverse=False, destroy=True, numthreads=1):
    try:
        assert inp.dtype == complex
        fft_in = pyfftw.empty_aligned(inp.shape, dtype='complex128')
        fft_out = pyfftw.empty_aligned(inp.shape, dtype='complex128')
        fft_object = pyfftw.FFTW(
            fft_in,
            fft_out,
            axes=axes,
            flags=('FFTW_ESTIMATE',) + (('FFTW_DESTROY_INPUT',) if destroy else ()),
            direction='FFTW_BACKWARD' if inverse else 'FFTW_FORWARD',
            threads=numthreads)
        fft_object(inp)
        return fft_out
    except RuntimeError as err:
        if "This is a bug" in str(err):
            raise RuntimeError(f"{err}\n Probably this can be fixed by importing wannierberri prior to numpy."
                               " See https://docs.wannier-berri.org/en/master/install.html#known-bug-with-pyfftw")
        else:
            raise err


def fft_np(inp, axes, inverse=False):
    assert inp.dtype == complex
    if inverse:
        return np.fft.ifftn(inp, axes=axes)
    else:
        return np.fft.fftn(inp, axes=axes)


def execute_fft(inp, axes, inverse=False, destroy=True, numthreads=1, fftlib='fftw'):
    fftlib = fftlib.lower()
    if fftlib == 'fftw' and not PYFFTW_IMPORTED:
        fftlib = 'numpy'
    if fftlib == 'fftw':
        return fft_W(inp, axes, inverse=inverse, destroy=destroy, numthreads=numthreads)
    elif fftlib == 'numpy':
        return fft_np(inp, axes, inverse=inverse)
    else:
        raise ValueError(f"unknown type of fftlib : {fftlib}")


class FFT_R_to_k:

    def __init__(self, iRvec, NKFFT, num_wann, numthreads=1, fftlib='fftw', name=None):
        t0 = time()
        self.NKFFT = tuple(NKFFT)
        self.num_wann = num_wann
        self.name = name
        fftlib = fftlib.lower()
        assert fftlib in ('fftw', 'numpy', 'slow'), f"fftlib '{fftlib}' is unknown/not supported"
        if fftlib == 'fftw' and not PYFFTW_IMPORTED:
            fftlib = 'numpy'
        self.lib = fftlib
        if fftlib == 'fftw':
            shape = self.NKFFT + (self.num_wann, self.num_wann)
            fft_in = pyfftw.empty_aligned(shape, dtype='complex128')
            fft_out = pyfftw.empty_aligned(shape, dtype='complex128')
            self.fft_plan = pyfftw.FFTW(
                fft_in,
                fft_out,
                axes=(0, 1, 2),
                flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'),
                direction='FFTW_BACKWARD',
                threads=numthreads)
        self.iRvec = iRvec % self.NKFFT
        self.nRvec = iRvec.shape[0]
        self.time_init = time() - t0
        self.time_call = 0
        self.n_call = 0

    def execute_fft(self, A):
        return self.fft_plan(A)

    def transform(self, AAA_K):
        assert np.iscomplexobj(AAA_K)
        if self.lib == 'numpy':
            AAA_K[...] = np.fft.ifftn(AAA_K, axes=(0, 1, 2))
        elif self.lib == 'fftw':
            # do recursion if array has cartesian indices. The recursion should not be very deep
            if AAA_K.ndim > 5:
                for i in range(AAA_K.shape[-1]):
                    AAA_K[..., i] = self.transform(AAA_K[..., i])
            else:
                AAA_K[...] = self.execute_fft(AAA_K[...])
            return AAA_K
        elif self.lib == 'slow':
            raise RuntimeError("FFT.transform should not be called for slow FT")

    @cached_property
    def exponent(self):
        """
        exponent for Fourier transform exp(1j*k*R)
        """
        return [np.exp(2j * np.pi / self.NKFFT[i]) ** np.arange(self.NKFFT[i]) for i in range(3)]

    def __call__(self, AAA_R, hermitian=False, antihermitean=False, reshapeKline=True):
        """
        perform the FFT from R vectors to k vectors

        Parameters
        ----------
        AAA_R : np.ndarray(complex, shape=(nRvec, num_wann, num_wann, ...))
            the matrix to be transformed
        hermitian : bool
            if True, the matrix is forced to be hermitian in the 3rd and 4th dimensions
        antihermitean : bool
            if True, the matrix is forced to be anti-hermitian in the 3rd and 4th dimensions
        reshapeKline : bool
            if True, the output is reshaped to (NKFFT[0]*NKFFT[1]*NKFFT[2], num_wann, num_wann, ...)

        Returns
        -------
        AAA_K : np.ndarray(complex, shape=(NKFFT[0], NKFFT[1], NKFFT[2], num_wann, num_wann, ...))
            the transformed matrix
        """
        t0 = time()
        # AAA_R is an array of dimension (  num_wann x num_wann x nRpts X... ) (any further dimensions allowed)
        if hermitian and antihermitean:
            raise ValueError("A matrix cannot be both hermitian and anti-hermitian, unless it is zero")
        shapeA = AAA_R.shape
        if self.lib == 'slow':
            k = np.zeros(3, dtype=int)
            AAA_K = np.array(
                [
                    [
                        [
                            sum(
                                np.prod([self.exponent[i][(k[i] * R[i]) % self.NKFFT[i]] for i in range(3)]) * A
                                for R, A in zip(self.iRvec, AAA_R)) for k[2] in range(self.NKFFT[2])
                        ] for k[1] in range(self.NKFFT[1])
                    ] for k[0] in range(self.NKFFT[0])
                ])
        else:
            assert self.nRvec == shapeA[0]
            assert self.num_wann == shapeA[1] == shapeA[2]
            AAA_K = np.zeros(self.NKFFT + shapeA[1:], dtype=complex)
            # TODO : place AAA_R to FFT grid from beginning, even before multiplying by exp(dkR)
            for ir, irvec in enumerate(self.iRvec):
                AAA_K[tuple(irvec)] += AAA_R[ir]
            self.transform(AAA_K)
            AAA_K *= np.prod(self.NKFFT)

        # TODO - think if fftlib transform of half of matrix makes sense
        if hermitian:
            AAA_K = 0.5 * (AAA_K + AAA_K.swapaxes(3, 4).conj())
        elif antihermitean:
            AAA_K = 0.5 * (AAA_K - AAA_K.swapaxes(3, 4).conj())

        if reshapeKline:
            AAA_K = AAA_K.reshape((np.prod(self.NKFFT),) + shapeA[1:])
        self.time_call += time() - t0
        self.n_call += 1
        return AAA_K
