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
import abc
from functools import cached_property
from ..parallel import pool
from ..system.system import System
from ..grid import TetraWeights, TetraWeightsParal, get_bands_in_range, get_bands_below_range
from .. import formula
from ..grid import KpointBZparallel, KpointBZtetra
from ..symmetry.point_symmetry import transform_ident, transform_odd
from .sdct_K import SDCT_K


def _rotate_matrix(X):
    return X[1].T.conj().dot(X[0]).dot(X[1])


def get_transform_Inv(name, der=0):
    """returns the transformation of the quantity  under inversion
    raises for unknown quantities"""
    ###########
    # Oscar ###
    ###########################################################################
    if name in ['Ham', 'CC', 'FF', 'OO', 'GG', 'SS']:  # even before derivative
        p = 0
    ###########################################################################
    elif name in ['D', 'AA', 'BB', 'CCab']:
        return None
    else:
        raise ValueError(f"parity under inversion unknown for {name}")
    if (p + der) % 2 == 1:
        return transform_odd
    else:
        return transform_ident


def get_transform_TR(name, der=0):
    """returns transformation of quantity is under TR, (after a real trace is taken, if appropriate)
    False otherwise
    raises ValueError for unknown quantities"""
    if name in ['Ham']:  # even before derivative
        p = 0
    #########
    # Oscar #
    ###########################################################################
    elif name in ['CC', 'FF', 'OO', 'GG', 'SS']:  # odd before derivative
        p = 1
    ###########################################################################
    elif name in ['D', 'AA', 'BB', 'CCab']:
        return None
    else:
        raise ValueError(f"parity under TR unknown for {name}")
    if (p + der) % 2 == 1:
        return transform_odd
    else:
        return transform_ident


class Data_K(System, abc.ABC):
    """
    class to store many data calculated on a specific FFT grid.
    The stored data can be used to evaluate many quantities.
    Is destroyed after  everything is evaluated for the FFT grid

    Parameters
    -----------
    random_gauge : bool
        applies random unitary rotations to degenerate states. Needed only for testing, to make sure that gauge
        covariance is preserved.
    degen_thresh_random_gauge : float
        threshold to consider bands as degenerate for random_gauge
    fftlib :  str
        library used to perform fftlib : 'fftw' (defgault) or 'numpy' or 'slow'
    """

    # Those are not used at the moment , but will be restored (TODO):
    #    frozen_max : float
    #        position of the upper edge of the frozen window. Used in the evaluation of orbital moment. But not necessary.
    #        If not specified, attempts to read this value from system. Othewise set to
    #    delta_fz:float
    #        size of smearing for B matrix with frozen window, from frozen_max-delta_fz to frozen_max.

    def __init__(self, system, dK, grid, Kpoint=None,
                 # Those are not used at the moment, but will be restored (TODO):
                 # frozen_max = -np.inf,
                 # delta_fz = 0.1,
                 Emin=-np.inf,
                 Emax=np.inf,
                 fftlib='fftw',
                 npar_k=1,
                 random_gauge=False,
                 degen_thresh_random_gauge=1e-4
                 ):
        self.system = system
        self.Emin = Emin
        self.Emax = Emax
        self.fftlib = fftlib
        self.npar_k = npar_k
        self.random_gauge = random_gauge
        self.degen_threshold_random_gauge = degen_thresh_random_gauge
        self.force_internal_terms_only = system.force_internal_terms_only
        self.grid = grid
        self.NKFFT = grid.FFT
        self.select_K = np.ones(self.nk, dtype=bool)
        #   self.findif = grid.findif
        self.real_lattice = system.real_lattice
        self.num_wann = self.system.num_wann
        self.Kpoint = Kpoint
        self.nkptot = self.NKFFT[0] * self.NKFFT[1] * self.NKFFT[2]

        self.poolmap = pool(self.npar_k)[0]

        self.dK = dK
        self._bar_quantities = {}
        self._covariant_quantities = {}

    ###########################################
    #   Now the **_R objects are evaluated only on demand
    # - as cached_property (if used more than once)
    #   as property   - iif used only once
    #   let's write them explicitly, for better code readability
    ###########################

    @property
    def is_phonon(self):
        return self.system.is_phonon

    ###############################################################

    ###########
    #  TOOLS  #
    ###########

    def _rotate(self, mat):
        assert mat.ndim > 2
        if mat.ndim == 3:
            return np.array(self.poolmap(_rotate_matrix, zip(mat, self.UU_K)))
        else:
            for i in range(mat.shape[-1]):
                mat[..., i] = self._rotate(mat[..., i])
            return mat

    #####################
    #  Basic variables  #
    #####################

    @cached_property
    def nbands(self):
        return self.num_wann

    @cached_property
    def kpoints_all(self):
        return (self.grid.points_FFT + self.dK[None]) % 1

    @cached_property
    def nk(self):
        return np.prod(self.NKFFT)

    @cached_property
    def tetraWeights(self):
        if isinstance(self.Kpoint, KpointBZparallel):
            return TetraWeightsParal(eCenter=self.E_K, eCorners=self.E_K_corners_parallel())
        elif isinstance(self.Kpoint, KpointBZtetra):
            return TetraWeights(eCenter=self.E_K, eCorners=self.E_K_corners_tetra())
        else:
            raise RuntimeError()

    def get_bands_in_range_groups_ik(self, ik, emin, emax, degen_thresh=-1, degen_Kramers=False, sea=False,
                                     Emin=-np.inf, Emax=np.inf):
        bands_in_range = get_bands_in_range(
            emin, emax, self.E_K[ik], degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)
        weights = {(ib1, ib2): self.E_K[ik, ib1:ib2].mean() for ib1, ib2 in bands_in_range}
        if sea:
            bandmax = get_bands_below_range(emin, self.E_K[ik])
            if len(bands_in_range) > 0:
                bandmax = min(bandmax, bands_in_range[0][0])
            if bandmax > 0:
                weights[(0, bandmax)] = -np.inf
        return weights

    def get_bands_in_range_groups(self, emin, emax, degen_thresh=-1, degen_Kramers=False, sea=False, Emin=-np.inf,
                                  Emax=np.inf):
        res = []
        for ik in range(self.nk):
            res.append(self.get_bands_in_range_groups_ik(ik, emin, emax, degen_thresh, degen_Kramers, sea, Emin=Emin,
                                                         Emax=Emax))
        return res

    ###################################################
    #  Basic variables and their standard derivatives #
    ###################################################

    def select_bands(self, energies):
        if hasattr(self, 'bands_selected'):
            return
        energies = energies.reshape((energies.shape[0], -1, energies.shape[-1]))
        select = np.any(energies > self.Emin, axis=1) * np.any(energies < self.Emax, axis=1)
        self.select_K = np.any(select, axis=1)
        self.select_B = np.any(select, axis=0)
        self.nk_selected = self.select_K.sum()
        self.nb_selected = self.select_B.sum()
        self.bands_selected = True

    @cached_property
    def E_K(self):
        EUU = self.poolmap(np.linalg.eigh, self.HH_K)
        E_K = self.phonon_freq_from_square(np.array([euu[0] for euu in EUU]))
        #        print ("E_K = ",E_K.min(), E_K.max(), E_K.mean())
        self.select_bands(E_K)
        self._UU = np.array([euu[1] for euu in EUU])[self.select_K, :][:, self.select_B]
        return E_K[self.select_K, :][:, self.select_B]

    # evaluate the energies in the corners of the parallelepiped, in order to use tetrahedron method

    def phonon_freq_from_square(self, E):
        """takes  sqrt(|E|)*sign(E) for phonons, returns input for electrons"""
        if self.is_phonon:
            e = np.sqrt(np.abs(E))
            e[E < 0] = -e[E < 0]
            return e
        else:
            return E

    @property
    @abc.abstractmethod
    def HH_K(self):
        """returns Wannier Hamiltonian for all points of the FFT grid"""

    @cached_property
    def delE_K(self):
        delE_K = np.einsum("klla->kla", self.Xbar('Ham', 1))
        check = np.abs(delE_K).imag.max()
        if check > 1e-10:
            raise RuntimeError(f"The band derivatives have considerable imaginary part: {check}")
        return delE_K.real

    def covariant(self, name, commader=0, gender=0, save=True):
        assert commader * gender == 0, "cannot mix comm and generalized derivatives"
        key = (name, commader, gender)
        if key not in self._covariant_quantities:
            if gender == 0:
                res = formula.Matrix_ln(
                    self.Xbar(name, commader),
                    transformTR=get_transform_TR(name, commader),
                    transformInv=get_transform_Inv(name, commader),
                )
            elif gender == 1:
                if name == 'Ham':
                    res = self.V_covariant
                else:
                    res = formula.Matrix_GenDer_ln(
                        self.covariant(name),
                        self.covariant(name, commader=1),
                        self.Dcov,
                        transformTR=get_transform_TR(name, gender),
                        transformInv=get_transform_Inv(name, gender)
                    )
            else:
                raise NotImplementedError()
            if not save:
                return res
            else:
                self._covariant_quantities[key] = res
        return self._covariant_quantities[key]

    @property
    def V_covariant(self):

        class V(formula.Matrix_ln):

            def __init__(self, matrix):
                super().__init__(matrix, transformTR=transform_odd, transformInv=transform_odd)

            def ln(self, ik, inn, out):
                return np.zeros((len(out), len(inn), 3), dtype=complex)

        return V(self.Xbar('Ham', der=1))

    @cached_property
    def Dcov(self):
        return formula.covariant.Dcov(self)

    @cached_property
    def dEig_inv(self):
        dEig_threshold = 1e-7
        dEig = self.E_K[:, :, None] - self.E_K[:, None, :]
        select = abs(dEig) < dEig_threshold
        dEig[select] = dEig_threshold
        dEig = 1. / dEig
        dEig[select] = 0.
        return dEig

    #    defining sets of degenerate states - needed only for testing with random_gauge

    @cached_property
    def degen(self):
        A = [np.where(E[1:] - E[:-1] > self.degen_thresh_random_gauge)[0] + 1 for E in self.E_K]
        A = [[
            0,
        ] + list(a) + [len(E)] for a, E in zip(A, self.E_K)]
        return [[(ib1, ib2) for ib1, ib2 in zip(a, a[1:]) if ib2 - ib1 > 1] for a in A]

    @cached_property
    def UU_K(self):
        self.E_K
        # the following is needed only for testing :
        if self.random_gauge:
            from scipy.stats import unitary_group
            cnt = 0
            s = 0
            for ik, deg in enumerate(self.true):
                for ib1, ib2 in deg:
                    self._UU[ik, :, ib1:ib2] = self._UU[ik, :, ib1:ib2].dot(unitary_group.rvs(ib2 - ib1))
                    cnt += 1
                    s += ib2 - ib1
        return self._UU

    @cached_property
    def D_H(self):
        return -self.Xbar('Ham', 1) * self.dEig_inv[:, :, :, None]

    @cached_property
    def A_H(self):
        '''Generalized Berry connection matrix, A^(H) as defined in eqn. (25) of 10.1103/PhysRevB.74.195118.'''
        return self.Xbar('AA') + 1j * self.D_H

    @property
    def A_H_internal(self):
        '''Generalized Berry connection matrix, A^(H) as defined in eqn. (25) of 10.1103/PhysRevB.74.195118. only internal term'''
        return 1j * self.D_H

    @cached_property
    def SDCT(self):
        """returns the SDC term"""
        return SDCT_K(self)


#########################################################################################################################################
