
#########
# Oscar #
###########################################################################




from functools import cached_property, lru_cache

import numpy as np
from ..utility import alpha_A, beta_A, cached_einsum

from scipy.constants import hbar, electron_mass, physical_constants
electron_g_factor = physical_constants['electron g factor'][0]
m_spin_prefactor = -0.5 * electron_g_factor * hbar / electron_mass


class SDCT_K:

    def __init__(self, data_K):
        self.data_K = data_K
        self.dEnm_threshold = 1e-3
        En = self.data_K.E_K
        self.is_degen = np.abs(En[:, :, None] - En[:, None, :]) < self.dEnm_threshold
        self.kron = np.array(self.is_degen, dtype=int)[:, :, :, None]

    @lru_cache
    def get_E1(self, external_terms=True):
        ''' Electric dipole moment '''
        A_H = 1j * self.data_K.D_H
        if external_terms:
            A_H += self.data_K.Xbar('AA')
        A_H[self.is_degen] = 0.  # set degenerate terms to zero, they will be treated separately below
        return A_H


    @lru_cache
    def get_M1(self, external_terms=True):
        ''' Magnetic dipole moment '''
        # Basic covariant matrices in the Hamiltonian gauge
        H = self.data_K.Xbar('Ham')

        # Other matrices
        En = self.data_K.E_K
        Eln_plus = 0.5 * (En[:, :, None] + En[:, None, :])

        # _____ 1. Internal terms _____ #
        A_int = self.get_E1(external_terms=False)

        Cbc_int = 1.j * cached_einsum('klpa,kpm,kmnb->klnab', A_int, H, A_int)
        C_H = Cbc_int[:, :, :, alpha_A, beta_A] - Cbc_int[:, :, :, beta_A, alpha_A]

        Obc_int = 1.j * cached_einsum('klpa,kpnb->klnab', A_int, A_int)
        O_H = Obc_int[:, :, :, alpha_A, beta_A] - Obc_int[:, :, :, beta_A, alpha_A]


        if external_terms:

            # Basic covariant matrices in the Hamiltonian gauge
            A = self.data_K.Xbar('AA')
            B = self.data_K.Xbar('BB')
            C = self.data_K.Xbar('CC')
            O = self.data_K.Xbar('OO')

            # _____ 2. External terms _____ #
            Aa_ext = self.kron * A  # Energy diagonal piece
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
        return -0.5 * (C_H - Eln_plus[:, :, :, None] * O_H)


    @lru_cache(maxsize=2)
    def get_E2(self, external_terms=True):
        ''' Electric quadrupole moment '''
        # _____ 1. Internal terms _____ #
        A_int = self.get_E1(external_terms=False)
        Gbc_int = cached_einsum('klpa,kpnb->klnab', A_int, A_int)
        G_int = 0.5 * (Gbc_int + Gbc_int.swapaxes(3, 4))

        G_H = G_int

        if external_terms:
            A = self.data_K.Xbar('AA')
            G = self.data_K.Xbar('GG')


            # _____ 2. External terms _____ #

            Aa_ext = self.kron * A  # Energy diagonal piece
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
    def get_Bln_q(self, external_terms=True):
        q = self.get_E2(external_terms=external_terms)
        En = self.data_K.E_K
        Enm = En[:, :, None] - En[:, None, :]
        return -0.5j * Enm[:, :, :, None, None] * q


    @lru_cache
    def get_Bln_m(self, external_terms=True, spin=False, orb=True):
        m = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3), dtype=complex)
        if orb:
            m = self.get_M1(external_terms=external_terms)
        if spin:
            m += m_spin_prefactor * self.data_K.Xbar('SS')
        B_m = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        B_m[:, :, :, alpha_A, beta_A] += m
        B_m[:, :, :, beta_A, alpha_A] -= m
        return B_m


    @cached_property
    def Vn(self):
        ''' Band velocity '''
        V_H = self.data_K.Xbar('Ham', 1)  # (k, m, n, a)
        return np.diagonal(V_H, axis1=1, axis2=2).transpose(0, 2, 1).real  # (k, m, a)

    ###########################################################################
