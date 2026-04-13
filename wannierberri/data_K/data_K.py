
import numpy as np
import abc
from functools import cached_property, lru_cache

from ..utility import alpha_A, beta_A, cached_einsum
from ..system.system import System
from .. import formula
from ..grid import KpointBZparallel, KpointBZtetra
from ..symmetry.point_symmetry import transform_ident, transform_odd
from ..factors import m_spin_prefactor


def get_transform_Inv(name, der=0):
    """returns the transformation of the quantity  under inversion
    raises for unknown quantities"""
    if name in ['Ham', 'CC', 'FF', 'OO', 'GG', 'SS', 'rotAA', 'rotAAab', 'CCab_antisym']:  # even before derivative
        p = 0
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
    elif name in ['CC', 'FF', 'OO', 'GG', 'SS', 'rotAA', 'rotAAab', 'CCab_antisym']:  # odd before derivative
        p = 1
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
                 random_gauge=False,
                 degen_thresh_random_gauge=1e-4
                 ):
        self.system = system
        self.Emin = Emin
        self.Emax = Emax
        self.fftlib = fftlib
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
        return cached_einsum('kba,kbc...,kcd->kad...', self.UU_K.conj(), mat, self.UU_K)

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
            from ..grid.tetrahedron import TetraWeightsParal
            return TetraWeightsParal(eCenter=self.E_K, eCorners=self.E_K_corners_parallel())
        elif isinstance(self.Kpoint, KpointBZtetra):
            from ..grid.tetrahedron import TetraWeights

            return TetraWeights(eCenter=self.E_K, eCorners=self.E_K_corners_tetra())
        else:
            raise RuntimeError()

    def get_bands_in_range_groups_ik(self, ik, emin, emax, degen_thresh=-1, degen_Kramers=False, sea=False,
                                     Emin=-np.inf, Emax=np.inf):
        from ..grid.tetrahedron import get_bands_in_range, get_bands_below_range
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
        E, UU = np.linalg.eigh(self.HH_K)
        E_K = self.phonon_freq_from_square(E)
        #        print ("E_K = ",E_K.min(), E_K.max(), E_K.mean())
        self.select_bands(E_K)
        self._UU = UU[self.select_K, :][:, self.select_B]
        return E_K[self.select_K, :][:, self.select_B]

    # evaluate the energies in the corners of the parallelepiped, in order to use tetrahedron method

    def phonon_freq_from_square(self, E):
        r"""For phonons return :math:`\sqrt{|E|}\operatorname{sign}(E)`; for electrons return ``E`` unchanged."""
        if self.is_phonon:
            e = np.sqrt(np.abs(E))
            e[E < 0] = -e[E < 0]
            return e
        else:
            return E

    @cached_property
    @abc.abstractmethod
    def HH_K(self):
        """returns Wannier Hamiltonian for all points of the FFT grid"""

    @cached_property
    def delE_K(self):
        delE_K = cached_einsum("klla->kla", self.Xbar('Ham', 1))
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

    @lru_cache(maxsize=2)
    def get_A_H(self, external_terms=True):
        '''Generalized Berry connection matrix, A^(H) as defined in eqn. (25) of 10.1103/PhysRevB.74.195118.'''
        A_H = 1j * self.D_H
        if external_terms:
            A_H += self.Xbar('AA')
        return A_H

    def E_K_corners_parallel_test(self):
        """returns the energies in the corners of the parallelepiped around each k-point of the FFT grid"""
        cls = self.__class__
        dK = self.Kpoint.dK_fullBZ
        E_corners = np.zeros((self.nk, 2, 2, 2, self.nbands), dtype=float)
        for ix in 0, 1:
            for iy in 0, 1:
                for iz in 0, 1:
                    dk = (np.array([ix, iy, iz]) - 1 / 2) * dK
                    _data_K = cls(system=self.system, dK=self.dK + dk, grid=self.grid, Kpoint=self.Kpoint,)
                    E_corners[:, ix, iy, iz, :] = _data_K.E_K
        self.select_bands(E_corners)
        return E_corners[self.select_K, :, :, :, :][:, :, :, :, self.select_B]

    def E_K_corners_tetra_test(self):
        """returns the energies in the corners of the tetrahedra around each k-point of the FFT grid"""
        _ = self.E_K  # to ensure that the bands are selected
        cls = self.__class__
        vertices = self.Kpoint.vertices_fullBZ
        E_corners = np.zeros((self.nk, 4, self.nbands), dtype=float)
        for iv, v in enumerate(vertices):
            _data_K = cls(system=self.system, dK=self.dK + v, grid=self.grid, Kpoint=self.Kpoint,)
            E_corners[:, iv, :] = _data_K.E_K
        self.select_bands(E_corners)
        return E_corners[self.select_K, :, :][:, :, self.select_B]

#########################################################################################################################################
### SDCT


    @lru_cache
    def sdct_is_degen(self, degen_thresh=1e-3):
        En = self.E_K
        return np.abs(En[:, :, None] - En[:, None, :]) < degen_thresh

    @lru_cache
    def sdct_kron(self, degen_thresh=1e-3):
        return np.array(self.sdct_is_degen(degen_thresh), dtype=int)[:, :, :, None]

    @lru_cache
    def get_E1(self, external_terms=True, degen_thresh=1e-3):
        ''' Electric dipole moment '''
        A_H = self.get_A_H(external_terms=external_terms)
        A_H[self.sdct_is_degen(degen_thresh)] = 0.  # set degenerate terms to zero, they will be treated separately below
        return A_H


    @lru_cache
    def get_M1(self, external_terms=True,
               V_term=True,
               key_OO='rotAA', degen_thresh=1e-3,
               AH_term=False):
        ''' Magnetic dipole moment '''
        # Basic covariant matrices in the Hamiltonian gauge
        H = self.Xbar('Ham')

        # Other matrices
        En = self.E_K
        Eln_plus = 0.5 * (En[:, :, None] + En[:, None, :])

        # _____ 1. Internal terms _____ #
        A_int = self.get_E1(external_terms=False, degen_thresh=degen_thresh)

        Cbc_int = 1.j * cached_einsum('klpa,kpm,kmnb->klnab', A_int, H, A_int)
        C_H = Cbc_int[:, :, :, alpha_A, beta_A] - Cbc_int[:, :, :, beta_A, alpha_A]

        Obc_int = 1.j * cached_einsum('klpa,kpnb->klnab', A_int, A_int)
        O_H = Obc_int[:, :, :, alpha_A, beta_A] - Obc_int[:, :, :, beta_A, alpha_A]


        if external_terms:

            # Basic covariant matrices in the Hamiltonian gauge
            A = self.Xbar('AA')
            B = self.Xbar('BB')
            C = self.Xbar('CC')
            O = self.Xbar(key_OO)

            # _____ 2. External terms _____ #
            Aa_ext = self.sdct_kron(degen_thresh) * A  # Energy diagonal piece
            A_ext = A - Aa_ext      # Energy non-diagonal piece

            Cbc_ext = -1.j * Eln_plus[:, :, :, None, None] * cached_einsum('klpa,kpnb->klnab', Aa_ext, Aa_ext)
            Cbc_ext += -1.j * cached_einsum('kl,klpa,kpnb->klnab', En, Aa_ext, A_ext)
            Cbc_ext += -1.j * cached_einsum('kn,klpa,kpnb->klnab', En, A_ext, Aa_ext)
            C_ext = C + Cbc_ext[:, :, :, alpha_A, beta_A] - Cbc_ext[:, :, :, beta_A, alpha_A]

            Obc_ext = -1.j * cached_einsum('klpa,kpnb->klnab', Aa_ext, Aa_ext)
            Obc_ext += -1.j * cached_einsum('klpa,kpnb->klnab', A_ext, Aa_ext)
            Obc_ext += -1.j * cached_einsum('klpa,kpnb->klnab', Aa_ext, A_ext)
            O_ext = O + Obc_ext[:, :, :, alpha_A, beta_A] - Obc_ext[:, :, :, beta_A, alpha_A]

            # _____ 3. Cross terms _____ #
            Cbc_cross = cached_einsum('klpa,kpnb->klnab', A_int, B)
            Cbc_cross = 1.j * (Cbc_cross - Cbc_cross.swapaxes(1, 2).conj())
            Cbc_cross += -1.j * cached_einsum('kl,klpa,kpnb->klnab', En, Aa_ext, A_int)
            Cbc_cross += -1.j * cached_einsum('kn,klpa,kpnb->klnab', En, A_int, Aa_ext)
            C_cross = Cbc_cross[:, :, :, alpha_A, beta_A] - Cbc_cross[:, :, :, beta_A, alpha_A]

            Obc_cross = 1.j * cached_einsum('klpa,kpnb->klnab', A_ext, A_int)
            Obc_cross += 1.j * cached_einsum('klpa,kpnb->klnab', A_int, A_ext)
            O_cross = Obc_cross[:, :, :, alpha_A, beta_A] - Obc_cross[:, :, :, beta_A, alpha_A]

            # Final formula
            C_H += C_ext + C_cross
            O_H += O_ext + O_cross
        M = -0.5 * (C_H - Eln_plus[:, :, :, None] * O_H)
        if V_term:
            Vn = self.delE_K
            Vnm_plus = (Vn[:, :, None, :] + Vn[:, None, :, :])
            A = self.get_E1(external_terms=external_terms, degen_thresh=degen_thresh)
            M += 0.5 * (Vnm_plus[:, :, :, alpha_A] * A[:, :, :, beta_A] -
                    Vnm_plus[:, :, :, beta_A] * A[:, :, :, alpha_A])
        if AH_term:
            Eln_minus = (En[:, :, None] - En[:, None, :])
            M += 0.25 * Eln_minus[:, :, :, None, None] * O_H
        return M


    @lru_cache
    def get_E2(self, external_terms=True, degen_thresh=1e-3):
        ''' Electric quadrupole moment '''
        # _____ 1. Internal terms _____ #
        A_int = self.get_E1(external_terms=False, degen_thresh=degen_thresh)
        Gbc_int = cached_einsum('klpa,kpnb->klnab', A_int, A_int)
        G_int = 0.5 * (Gbc_int + Gbc_int.swapaxes(3, 4))

        G_H = G_int

        if external_terms:
            A = self.Xbar('AA')
            G = self.Xbar('GG')


            # _____ 2. External terms _____ #

            Aa_ext = self.sdct_kron(degen_thresh) * A  # Energy diagonal piece
            A_ext = A - Aa_ext           # Energy non-diagonal piece

            Gbc_ext = -cached_einsum('klpa,kpnb->klnab', Aa_ext, Aa_ext)
            Gbc_ext += -cached_einsum('klpa,kpnb->klnab', A_ext, Aa_ext)
            Gbc_ext += -cached_einsum('klpa,kpnb->klnab', Aa_ext, A_ext)
            G_ext = G + 0.5 * (Gbc_ext + Gbc_ext.swapaxes(3, 4))

            # _____ 3. Cross terms _____ #

            Gbc_cross = cached_einsum('klpa,kpnb->klnab', A_ext, A_int)
            Gbc_cross += cached_einsum('klpa,kpnb->klnab', A_int, A_ext)
            G_cross = 0.5 * (Gbc_cross + Gbc_cross.swapaxes(3, 4))

            # Final formula
            G_H += G_ext + G_cross
        return -1. * G_H



    @lru_cache
    def get_Bln(self, external_terms=True,
                spin=False, orb=True, V=True, Q=True,
                key_OO='rotAA', degen_thresh=1e-3):
        B = np.zeros((self.nk, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        if orb or spin:
            m = np.zeros((self.nk, self.num_wann, self.num_wann, 3), dtype=complex)
            if orb:
                m += self.get_M1(external_terms=external_terms, key_OO=key_OO, degen_thresh=degen_thresh,
                                V_term=False)
            if spin:
                m += m_spin_prefactor * self.Xbar('SS')
            B[:, :, :, alpha_A, beta_A] += m
            B[:, :, :, beta_A, alpha_A] -= m
        if V:
            Vn = self.delE_K
            Vnm_plus = 0.5 * (Vn[:, :, None, :] + Vn[:, None, :, :])
            A = self.get_E1(external_terms=external_terms, degen_thresh=degen_thresh)
            B += Vnm_plus[:, :, :, :, None] * A[:, :, :, None, :]
        if Q:
            q = self.get_E2(external_terms=external_terms, degen_thresh=degen_thresh)
            En = self.E_K
            Enm = En[:, :, None] - En[:, None, :]
            B += -0.5j * Enm[:, :, :, None, None] * q
        return B
