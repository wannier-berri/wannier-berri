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
#------------------------------------------------------------

__debug = False

import inspect
import numpy as np

from lazy_property import LazyProperty as Lazy
from time import time
from termcolor import cprint
import fortio, scipy.io

try:
    import pyfftw
    PYFFTW_IMPORTED = True
except Exception as err:
    PYFFTW_IMPORTED = False
    print("WARNING : error importing  `pyfftw` : {} \n will use numpy instead \n".format(err))


# inheriting just in order to have posibility to change default values, without changing the rest of the code
class FortranFileR(fortio.FortranFile):

    def __init__(self, filename):
        print("using fortio to read")
        try:
            super().__init__(filename, mode='r', header_dtype='uint32', auto_endian=True, check_file=True)
        except ValueError:
            print("File '{}' contains subrecords - using header_dtype='int32'".format(filename))
            super().__init__(filename, mode='r', header_dtype='int32', auto_endian=True, check_file=True)


class FortranFileW(scipy.io.FortranFile):

    def __init__(self, filename):
        print("using scipy.io to write")
        super().__init__(filename, mode='w')


alpha_A = np.array([1, 2, 0])
beta_A = np.array([2, 0, 1])


def print_my_name_start():
    if __debug:
        print("DEBUG: Running {} ..".format(inspect.stack()[1][3]))


def print_my_name_end():
    if __debug:
        print("DEBUG: Running {} - done ".format(inspect.stack()[1][3]))


def conjugate_basis(basis):
    return 2 * np.pi * np.linalg.inv(basis).T


def warning(message, color="yellow"):
    cprint("\n WARNING!!!!! {} \n".format(message), color)


def real_recip_lattice(real_lattice=None, recip_lattice=None):
    if recip_lattice is None:
        if real_lattice is None:
            cprint(
                "\n WARNING!!!!! usually need to provide either with real or reciprocal lattice. If you only want to generate a random symmetric tensor - that it fine \n",
                "yellow")
            return None, None
        recip_lattice = conjugate_basis(real_lattice)
    else:
        if real_lattice is not None:
            assert np.linalg.norm(
                real_lattice.dot(recip_lattice.T) / (2 * np.pi)
                - np.eye(3)) <= 1e-8, "real and reciprocal lattice do not match"
        else:
            real_lattice = conjugate_basis(recip_lattice)
    return real_lattice, recip_lattice




def str2bool(v):
    if v[0] in "fF":
        return False
    elif v[0] in "tT":
        return True
    else:
        raise RuntimeError(" unrecognized value of bool parameter :{0}".format(v))


def fft_W(inp, axes, inverse=False, destroy=True, numthreads=1):
    assert inp.dtype == complex
    # t0=time()
    fft_in = pyfftw.empty_aligned(inp.shape, dtype='complex128')
    fft_out = pyfftw.empty_aligned(inp.shape, dtype='complex128')
    # t01=time()
    fft_object = pyfftw.FFTW(
        fft_in,
        fft_out,
        axes=axes,
        flags=('FFTW_ESTIMATE', ) + (('FFTW_DESTROY_INPUT', ) if destroy else ()),
        direction='FFTW_BACKWARD' if inverse else 'FFTW_FORWARD',
        threads=numthreads)
    # t1=time()
    fft_object(inp)
    # t2=time()
    return fft_out

    def getHead(n):
        if n <= 0:
            return ['  ']
        else:
            return [a + b for a in 'xyz' for b in getHead(n - 1)]


def fft_np(inp, axes, inverse=False):
    assert inp.dtype == complex
    if inverse:
        return np.fft.ifftn(inp, axes=axes)
    else:
        return np.fft.fftn(inp, axes=axes)


def FFT(inp, axes, inverse=False, destroy=True, numthreads=1, fft='fftw'):
    if fft == 'fftw' and not PYFFTW_IMPORTED:
        fft = 'numpy'
    if fft == 'fftw':
        return fft_W(inp, axes, inverse=inverse, destroy=destroy, numthreads=numthreads)
    elif fft == 'numpy':
        return fft_np(inp, axes, inverse=inverse)
    else:
        raise ValueError("unknown type of fft : {}".format(fft))


def fourier_q_to_R(AA_q, mp_grid, kpt_mp_grid, iRvec, ndegen, numthreads=1, fft='fftw'):
    print_my_name_start()
    mp_grid = tuple(mp_grid)
    shapeA = AA_q.shape[1:]  # remember the shapes after q
    AA_q_mp = np.zeros(tuple(mp_grid) + shapeA, dtype=complex)
    for i, k in enumerate(kpt_mp_grid):
        AA_q_mp[k] = AA_q[i]
    AA_q_mp = FFT(AA_q_mp, axes=(0, 1, 2), numthreads=numthreads, fft=fft, destroy=False)
    AA_R = np.array([AA_q_mp[tuple(iR % mp_grid)] / nd for iR, nd in zip(iRvec, ndegen)]) / np.prod(mp_grid)
    AA_R = AA_R.transpose((1, 2, 0) + tuple(range(3, AA_R.ndim)))
    print_my_name_end()
    return AA_R


class FFT_R_to_k():

    def __init__(self, iRvec, NKFFT, num_wann, numthreads=1, lib='fftw', name=None):
        t0 = time()
        print_my_name_start()
        self.NKFFT = tuple(NKFFT)
        self.num_wann = num_wann
        assert lib in ('fftw', 'numpy', 'slow'), "fft lib '{}' is not known/supported".format(lib)
        if lib == 'fftw' and not PYFFTW_IMPORTED:
            lib = 'numpy'
        self.lib = lib
        if lib == 'fftw':
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
        else:
            raise ValueError(f"Unknown type of Fourier transform :'{self.lib}'")

    @Lazy
    def exponent(self):
        '''
        exponent for Fourier transform exp(1j*k*R)
        '''
        return [np.exp(2j * np.pi / self.NKFFT[i])**np.arange(self.NKFFT[i]) for i in range(3)]

    def __call__(self, AAA_R, hermitean=False, antihermitean=False, reshapeKline=True):
        t0 = time()
        # AAA_R is an array of dimension (  num_wann x num_wann x nRpts X... ) (any further dimensions allowed)
        if hermitean and antihermitean:
            raise ValueError("A matrix cannot be both hermitean and anti-hermitean, unless it is zero")
        AAA_R = AAA_R.transpose((2, 0, 1) + tuple(range(3, AAA_R.ndim)))
        shapeA = AAA_R.shape
        if self.lib == 'slow':
            t0 = time()
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
            t0 = time()
            assert self.nRvec == shapeA[0]
            assert self.num_wann == shapeA[1] == shapeA[2]
            AAA_K = np.zeros(self.NKFFT + shapeA[1:], dtype=complex)
            # TODO : place AAA_R to FFT grid from beginning, even before multiplying by exp(dkR)
            for ir, irvec in enumerate(self.iRvec):
                AAA_K[tuple(irvec)] += AAA_R[ir]
            self.transform(AAA_K)
            AAA_K *= np.prod(self.NKFFT)

        # TODO - think if fft transform of half of matrix makes sense
        if hermitean:
            AAA_K = 0.5 * (AAA_K + AAA_K.transpose((0, 1, 2, 4, 3) + tuple(range(5, AAA_K.ndim))).conj())
        elif antihermitean:
            AAA_K = 0.5 * (AAA_K - AAA_K.transpose((0, 1, 2, 4, 3) + tuple(range(5, AAA_K.ndim))).conj())

        if reshapeKline:
            AAA_K = AAA_K.reshape((np.prod(self.NKFFT), ) + shapeA[1:])
        self.time_call += time() - t0
        self.n_call += 1
        return AAA_K


def iterate3dpm(size):
    return (
        np.array([i, j, k]) for i in range(-size[0], size[0] + 1) for j in range(-size[1], size[1] + 1)
        for k in range(-size[2], size[2] + 1))


def iterate3d(size):
    return (np.array([i, j, k]) for i in range(0, size[0]) for j in range(0, size[1]) for k in range(0, size[2]))


def find_degen(arr, degen_thresh):
    """ finds shells of 'almost same' values in array arr, and returns a list o[(b1,b2),...]"""
    A = np.where(arr[1:] - arr[:-1] > degen_thresh)[0] + 1
    A = [
        0,
    ] + list(A) + [len(arr)]
    return [(ib1, ib2) for ib1, ib2 in zip(A, A[1:])]


def is_round(A, prec=1e-14):
    # returns true if all values in A are integers, at least within machine precision
    return (np.linalg.norm(A - np.round(A)) < prec)


def get_angle(sina, cosa):
    '''Get angle in radian from sin and cos.'''
    if abs(cosa) > 1.0:
        cosa = np.round(cosa, decimals=1)
    alpha = np.arccos(cosa)
    if sina < 0.0:
        alpha = 2.0 * np.pi - alpha
    return alpha

# smearing functions
def Lorentzian(x, width):
    return 1.0 / (np.pi * width) * width**2 / (x**2 + width**2)


def Gaussian(x, width, adpt_smr):
    '''
    Compute 1 / (np.sqrt(pi) * width) * exp(-(x / width) ** 2)
    If the exponent is less than -200, return 0.
    An unoptimized version is the following.
        def Gaussian(x, width, adpt_smr):
            return 1 / (np.sqrt(pi) * width) * np.exp(-np.minimum(200.0, (x / width) ** 2))
    '''
    inds = abs(x) < width * np.sqrt(200.0)
    output = np.zeros_like(x)
    if adpt_smr:
        # width is array
        width_tile = np.tile(width, (x.shape[0], 1, 1))
        output[inds] = 1.0 / (np.sqrt(np.pi) * width_tile[inds]) * np.exp(-(x[inds] / width_tile[inds])**2)
    else:
        # width is number
        output[inds] = 1.0 / (np.sqrt(np.pi) * width) * np.exp(-(x[inds] / width)**2)
    return output
