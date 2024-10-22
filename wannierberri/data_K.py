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
from .parallel import pool
from .system.system import System
from .system.system_R import System_R
from .system.system_kp import SystemKP
from .__utility import FFT_R_to_k, alpha_A, beta_A
from .grid import TetraWeights, TetraWeightsParal, get_bands_in_range, get_bands_below_range
from . import formula
from .grid import KpointBZparallel, KpointBZtetra
from .point_symmetry import transform_ident, transform_odd


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
    elif name in ['T_wcc']:  # odd before derivative
        p = 1
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
    if name in ['Ham', 'T_wcc']:  # even before derivative
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


class _Data_K(System, abc.ABC):
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
                 # frozen_max = -np.Inf,
                 # delta_fz = 0.1,
                 Emin=-np.Inf,
                 Emax=np.Inf,
                 use_wcc_phase=False,
                 fftlib='fftw',
                 npar_k=1,
                 random_gauge=False,
                 degen_thresh_random_gauge=1e-4
                 ):
        self.system = system
        self.Emin = Emin
        self.Emax = Emax
        self.use_wcc_phase = use_wcc_phase
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
        #########
        # Oscar #
        #######################################################################
        self.dEnm_threshold = 1e-3
        #######################################################################

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
                                     Emin=-np.Inf, Emax=np.Inf):
        bands_in_range = get_bands_in_range(
            emin, emax, self.E_K[ik], degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)
        weights = {(ib1, ib2): self.E_K[ik, ib1:ib2].mean() for ib1, ib2 in bands_in_range}
        if sea:
            bandmax = get_bands_below_range(emin, self.E_K[ik])
            if len(bands_in_range) > 0:
                bandmax = min(bandmax, bands_in_range[0][0])
            if bandmax > 0:
                weights[(0, bandmax)] = -np.Inf
        return weights

    def get_bands_in_range_groups(self, emin, emax, degen_thresh=-1, degen_Kramers=False, sea=False, Emin=-np.Inf,
                                  Emax=np.Inf):
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


    #########
    # Oscar #
    ###########################################################################


    @cached_property
    def kron(self):
        En = self.E_K
        kron = np.array(abs(En[:, :, None] - En[:, None, :]) < self.dEnm_threshold, dtype=int)

        return kron

    @cached_property
    def E1(self):
        ''' Electric dipole moment '''
        # Basic covariant matrices in the Hamiltonian gauge
        A = self.Xbar('AA')

        # Other matrices
        D = self.D_H
        kron = self.kron

        # _____ 1. Internal terms _____ #
        A_int = 1.j * D
        Aa_int = kron[:, :, :, None] * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        # _____ 2. External terms _____ #
        Aa_ext = kron[:, :, :, None] * A  # Energy diagonal piece
        A_ext = A - Aa_ext           # Energy non-diagonal piece

        # Final formula
        A_H = A_int + A_ext
        return -1 * A_H

    @cached_property
    def E1_internal(self):
        ''' Electric dipole moment (only internal terms) '''
        # Other matrices
        D = self.D_H
        kron = self.kron

        # _____ 1. Internal terms _____ #
        A_int = 1.j * D
        Aa_int = kron[:, :, :, None] * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        # Final formula
        A_H = A_int
        return -1 * A_H

    @cached_property
    def M1(self):
        ''' Magnetic dipole moment '''
        # Basic covariant matrices in the Hamiltonian gauge
        H = self.Xbar('Ham')
        A = self.Xbar('AA')
        B = self.Xbar('BB')
        C = self.Xbar('CC')
        O = self.Xbar('OO')

        # Other matrices
        D = self.D_H
        En = self.E_K
        kron = self.kron
        Eln_plus = 0.5 * (En[:, :, None] + En[:, None, :])

        # _____ 1. Internal terms _____ #
        A_int = 1.j * D
        Aa_int = kron[:, :, :, None] * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        Cbc_int = 1.j * np.einsum('klpa,kpm,kmnb->klnab', A_int, H, A_int)
        C_int = Cbc_int[:, :, :, alpha_A, beta_A] - Cbc_int[:, :, :, beta_A, alpha_A]

        Obc_int = 1.j * np.einsum('klpa,kpnb->klnab', A_int, A_int)
        O_int = Obc_int[:, :, :, alpha_A, beta_A] - Obc_int[:, :, :, beta_A, alpha_A]

        # _____ 2. External terms _____ #
        Aa_ext = kron[:, :, :, None] * A  # Energy diagonal piece
        A_ext = A - Aa_ext           # Energy non-diagonal piece

        Cbc_ext = -1.j * Eln_plus[:, :, :, None, None] * np.einsum('klpa,kpnb->klnab', Aa_ext, Aa_ext)
        Cbc_ext += -1.j * np.einsum('kl,klpa,kpnb->klnab', En, Aa_ext, A_ext)
        Cbc_ext += -1.j * np.einsum('kn,klpa,kpnb->klnab', En, A_ext, Aa_ext)
        C_ext = C + Cbc_ext[:, :, :, alpha_A, beta_A] - Cbc_ext[:, :, :, beta_A, alpha_A]

        Obc_ext = -1.j * np.einsum('klpa,kpnb->klnab', Aa_ext, Aa_ext)
        Obc_ext += -1.j * np.einsum('klpa,kpnb->klnab', A_ext, Aa_ext)
        Obc_ext += -1.j * np.einsum('klpa,kpnb->klnab', Aa_ext, A_ext)
        O_ext = O + Obc_ext[:, :, :, alpha_A, beta_A] - Obc_ext[:, :, :, beta_A, alpha_A]

        # _____ 3. Cross terms _____ #
        Cbc_cross = np.einsum('klpa,kpnb->klnab', A_int, B)
        Cbc_cross = 1.j * (Cbc_cross - Cbc_cross.swapaxes(1, 2).conj())
        Cbc_cross += -1.j * np.einsum('kl,klpa,kpnb->klnab', En, Aa_ext, A_int)
        Cbc_cross += -1.j * np.einsum('kn,klpa,kpnb->klnab', En, A_int, Aa_ext)
        C_cross = Cbc_cross[:, :, :, alpha_A, beta_A] - Cbc_cross[:, :, :, beta_A, alpha_A]

        Obc_cross = 1.j * np.einsum('klpa,kpnb->klnab', A_ext, A_int)
        Obc_cross += 1.j * np.einsum('klpa,kpnb->klnab', A_int, A_ext)
        O_cross = Obc_cross[:, :, :, alpha_A, beta_A] - Obc_cross[:, :, :, beta_A, alpha_A]


        # Final formula
        C_H = C_int + C_ext + C_cross
        O_H = O_int + O_ext + O_cross
        return -0.5 * (C_H - Eln_plus[:, :, :, None] * O_H)

    @cached_property
    def M1_internal(self):
        ''' Magnetic dipole moment (only internal terms) '''
        # Basic covariant matrices in the Hamiltonian gauge
        H = self.Xbar('Ham')

        # Other matrices
        D = self.D_H
        En = self.E_K
        kron = self.kron
        Eln_plus = 0.5 * (En[:, :, None] + En[:, None, :])

        # _____ 1. Internal terms _____ #
        A_int = 1.j * D
        Aa_int = kron[:, :, :, None] * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        Cbc_int = 1.j * np.einsum('klpa,kpm,kmnb->klnab', A_int, H, A_int)
        C_int = Cbc_int[:, :, :, alpha_A, beta_A] - Cbc_int[:, :, :, beta_A, alpha_A]

        Obc_int = 1.j * np.einsum('klpa,kpnb->klnab', A_int, A_int)
        O_int = Obc_int[:, :, :, alpha_A, beta_A] - Obc_int[:, :, :, beta_A, alpha_A]

        # Final formula
        C_H = C_int
        O_H = O_int
        return -0.5 * (C_H - Eln_plus[:, :, :, None] * O_H)

    @cached_property
    def E2(self):
        ''' Electric quadrupole moment '''
        # Basic covariant matrices in the Hamiltonian gauge
        A = self.Xbar('AA')
        G = self.Xbar('GG')

        # Other matrices
        D = self.D_H
        kron = self.kron

        # _____ 1. Internal terms _____ #

        A_int = 1.j * D
        Aa_int = kron[:, :, :, None] * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        Gbc_int = np.einsum('klpa,kpnb->klnab', A_int, A_int)
        G_int = 0.5 * (Gbc_int + Gbc_int.swapaxes(3, 4))

        # _____ 2. External terms _____ #

        Aa_ext = kron[:, :, :, None] * A  # Energy diagonal piece
        A_ext = A - Aa_ext           # Energy non-diagonal piece

        Gbc_ext = -np.einsum('klpa,kpnb->klnab', Aa_ext, Aa_ext)
        Gbc_ext += -np.einsum('klpa,kpnb->klnab', A_ext, Aa_ext)
        Gbc_ext += -np.einsum('klpa,kpnb->klnab', Aa_ext, A_ext)
        G_ext = G + 0.5 * (Gbc_ext + Gbc_ext.swapaxes(3, 4))

        # _____ 3. Cross terms _____ #

        Gbc_cross = np.einsum('klpa,kpnb->klnab', A_ext, A_int)
        Gbc_cross += np.einsum('klpa,kpnb->klnab', A_int, A_ext)
        G_cross = 0.5 * (Gbc_cross + Gbc_cross.swapaxes(3, 4))

        # Final formula
        G_H = G_int + G_ext + G_cross
        return -1. * G_H

    @cached_property
    def E2_internal(self):
        ''' Electric quadrupole moment (only internal terms)'''
        # Other matrices
        D = self.D_H
        kron = self.kron

        # _____ 1. Internal terms _____ #

        A_int = 1.j * D
        Aa_int = kron[:, :, :, None] * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        Gbc_int = np.einsum('klpa,kpnb->klnab', A_int, A_int)
        G_int = 0.5 * (Gbc_int + Gbc_int.swapaxes(3, 4))

        # Final formula
        G_H = G_int
        return -1. * G_H

    @cached_property
    def Bln(self):
        m = self.M1
        q = self.E2
        En = self.E_K
        Enm = En[:, :, None] - En[:, None, :]

        B_q = -0.5j * Enm[:, :, :, None, None] * q
        B_m = np.zeros((self.nk, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        B_m[:, :, :, alpha_A, beta_A] += m
        B_m[:, :, :, beta_A, alpha_A] -= m
        B = B_m + B_q
        return B_m, B_q, B

    @cached_property
    def Bln_internal(self):
        m = self.M1_internal
        q = self.E2_internal
        En = self.E_K
        Enm = En[:, :, None] - En[:, None, :]

        B_q = -0.5j * Enm[:, :, :, None, None] * q
        B_m = np.zeros((self.nk, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        B_m[:, :, :, alpha_A, beta_A] += m
        B_m[:, :, :, beta_A, alpha_A] -= m
        B = B_m + B_q
        return B_m, B_q, B

    @cached_property
    def Vn(self):
        ''' Band velocity '''
        V_H = self.Xbar('Ham', 1)  # (k, m, n, a)
        return np.diagonal(V_H, axis1=1, axis2=2).transpose(0, 2, 1)  # (k, m, a)

    ###########################################################################


#########################################################################################################################################


class Data_K_R(_Data_K, System_R):
    """ The Data_K class for systems defined by R-space matrix elements (Wannier/TB)"""

    def __init__(self, system, dK, grid,
                 _FF_antisym=False,
                 _CCab_antisym=False,
                 **parameters):
        super().__init__(system, dK, grid, **parameters)
        self._FF_antisym = _FF_antisym
        self._CCab_antisym = _CCab_antisym

        self.cRvec_wcc = self.system.cRvec_p_wcc

        self.fft_R_to_k = FFT_R_to_k(
            self.system.iRvec,
            self.NKFFT,
            self.num_wann,
            numthreads=self.npar_k if self.npar_k > 0 else 1,
            fftlib=self.fftlib)

        self.expdK = np.exp(2j * np.pi * self.system.iRvec.dot(dK))
        self.dK = dK
        self._bar_quantities = {}
        self._covariant_quantities = {}
        self._XX_R = {}

    @property
    def HH_K(self):
        return self.fft_R_to_k(self.Ham_R, hermitian=True)

    #########
    # Oscar #
    ###########################################################################
    def get_R_mat(self, key):
        memoize_R = ['Ham', 'AA', 'OO', 'BB', 'CC', 'CCab', 'GG']
        try:
            return self._XX_R[key]
        except KeyError:
            if key == 'OO':
                res = self._OO_R()
            elif key == 'CCab':
                res = self._CCab_R()
            elif key == 'FF':
                res = self._FF_R()
            elif key == 'T_wcc':
                res = self._T_wcc_R()
            else:
                X_R = self.system.get_R_mat(key)
                shape = [1] * X_R.ndim
                shape[2] = self.expdK.shape[0]
                res = X_R * self.expdK.reshape(shape)
            if key in memoize_R:
                self.set_R_mat(key, res)
        return res
    ###########################################################################


    #  this is a bit ovberhead, but to maintain uniformity of the code let's use this


    def _T_wcc_R(self):
        nw = self.num_wann
        res = np.zeros((nw, nw, self.system.nRvec, 3), dtype=complex)
        res[np.arange(nw), np.arange(nw), self.system.iR0, :] = self.system.wannier_centers_cart_wcc_phase
        return res

    def _OO_R(self):
        # We do not multiply by expdK, because it is already accounted in AA_R
        return 1j * (
            self.cRvec_wcc[:, :, :, alpha_A] * self.get_R_mat('AA')[:, :, :, beta_A] -
            self.cRvec_wcc[:, :, :, beta_A] * self.get_R_mat('AA')[:, :, :, alpha_A])

    def _CCab_R(self):
        if self._CCab_antisym:
            CCab = np.zeros((self.num_wann, self.num_wann, self.system.nRvec, 3, 3), dtype=complex)
            CCab[:, :, :, alpha_A, beta_A] = -0.5j * self.get_R_mat('CC')
            CCab[:, :, :, beta_A, alpha_A] = 0.5j * self.get_R_mat('CC')
            return CCab
        else:
            return self.system.get_R_mat('CCab') * self.expdK[None, None, :, None, None]

    def _FF_R(self):
        if self._FF_antisym:
            return self.cRvec_wcc[:, :, :, :, None] * self.get_R_mat('AA')[:, :, :, None, :]
        else:
            return self.system.get_R_mat('FF') * self.expdK[None, None, :, None, None]

    def Xbar(self, name, der=0):
        key = (name, der)
        if key not in self._bar_quantities:
            self._bar_quantities[key] = self._R_to_k_H(
                self.get_R_mat(name).copy(), der=der, hermitian=(name in ['AA', 'SS', 'OO']))
        return self._bar_quantities[key]

    def _R_to_k_H(self, XX_R, der=0, hermitian=True):
        """ converts from real-space matrix elements in Wannier gauge to
            k-space quantities in k-space.
            der [=0] - defines the order of comma-derivative
            hermitian [=True] - consider the matrix hermitian
            WARNING: the input matrix is destroyed, use np.copy to preserve it"""

        for i in range(der):
            shape_cR = np.shape(self.cRvec_wcc)
            XX_R = 1j * XX_R.reshape((XX_R.shape) + (1,)) * self.cRvec_wcc.reshape(
                (shape_cR[0], shape_cR[1], self.system.nRvec) + (1,) * len(XX_R.shape[3:]) + (3,))
        return self._rotate((self.fft_R_to_k(XX_R, hermitian=hermitian))[self.select_K])

    def E_K_corners_tetra(self):
        vertices = self.Kpoint.vertices_fullBZ
        expdK = np.exp(2j * np.pi * self.system.iRvec.dot(
            vertices.T)).T  # we omit the wcc phases here, because they do not affect the energies
        _Ecorners = np.zeros((self.nk, 4, self.num_wann), dtype=float)
        for iv, _exp in enumerate(expdK):
            _Ham_R = self.Ham_R[:, :, :] * _exp[None, None, :]
            _HH_K = self.fft_R_to_k(_Ham_R, hermitian=True)
            _Ecorners[:, iv, :] = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
        self.select_bands(_Ecorners)
        Ecorners = np.zeros((self.nk_selected, 4, self.nb_selected), dtype=float)
        for iv, _exp in enumerate(expdK):
            Ecorners[:, iv, :] = _Ecorners[:, iv, :][self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners

    def E_K_corners_parallel(self):
        dK2 = self.Kpoint.dK_fullBZ / 2
        expdK = np.exp(
            2j * np.pi * self.system.iRvec *
            dK2[None, :])  # we omit the wcc phases here, because they do not affect the energies
        expdK = np.array([1. / expdK, expdK])
        Ecorners = np.zeros((self.nk_selected, 2, 2, 2, self.nb_selected), dtype=float)
        for ix in 0, 1:
            for iy in 0, 1:
                for iz in 0, 1:
                    _expdK = expdK[ix, :, 0] * expdK[iy, :, 1] * expdK[iz, :, 2]
                    _Ham_R = self.Ham_R[:, :, :] * _expdK[None, None, :]
                    _HH_K = self.fft_R_to_k(_Ham_R, hermitian=True)
                    E = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
                    Ecorners[:, ix, iy, iz, :] = E[self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners


class Data_K_k(_Data_K):
    """ The Data_K class for systems defined by k-dependent Hamiltonians  (kp)"""

    @property
    def HH_K(self):
        return np.array([self.system.Ham(k) for k in self.kpoints_all])

    def Xbar(self, name, der=0):
        key = (name, der)
        if name != 'Ham':
            raise ValueError(f'quantity {name} is not defined for a kp model')
        if key not in self._bar_quantities:
            if der == 0:
                raise RuntimeError("Why is `Ham` called through Xbar, are you sure?")
                X = self.HH_K
            else:
                if der == 1:
                    fun = self.system.derHam
                elif der == 2:
                    fun = self.system.der2Ham
                elif der == 3:
                    fun = self.system.der3Ham
                X = np.array([fun(k) for k in self.kpoints_all])
            self._bar_quantities[key] = self._rotate(X)[self.select_K]
        return self._bar_quantities[key]

    def E_K_corners_tetra(self):
        vertices = self.Kpoint.vertices_fullBZ
        _Ecorners = np.zeros((self.nk, 4, self.num_wann), dtype=float)
        for iv, v in enumerate(vertices):
            _HH_K = np.array([self.system.Ham(k + v) for k in self.kpoints_all])
            _Ecorners[:, iv, :] = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
        self.select_bands(_Ecorners)
        Ecorners = np.zeros((self.nk_selected, 4, self.nb_selected), dtype=float)
        for iv, v in enumerate(vertices):
            Ecorners[:, iv, :] = _Ecorners[:, iv, :][self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners

    def E_K_corners_parallel(self):
        dK = self.Kpoint.dK_fullBZ
        Ecorners = np.zeros((self.nk_selected, 2, 2, 2, self.nb_selected), dtype=float)
        for ix in 0, 1:
            for iy in 0, 1:
                for iz in 0, 1:
                    v = (np.array([ix, iy, iz]) - 0.5) * dK
                    _HH_K = np.array([self.system.Ham(k + v) for k in self.kpoints_all])
                    E = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
                    Ecorners[:, ix, iy, iz, :] = E[self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners


def get_data_k(system, dK, grid, **parameters):
    if isinstance(system, SystemKP):
        return Data_K_k(system, dK, grid, **parameters)
    else:
        return Data_K_R(system, dK, grid, **parameters)
