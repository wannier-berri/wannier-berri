
#########
# Oscar #
###########################################################################




from functools import cached_property

import numpy as np
from ..utility import alpha_A, beta_A


class SDCT_K:

    def __init__(self, data_K):
        self.data_K = data_K
        self.dEnm_threshold = 1e-3
        En = self.data_K.E_K
        self.kron = np.array(abs(En[:, :, None] - En[:, None, :]) < self.dEnm_threshold, dtype=int)[:, :, :, None]

    @cached_property
    def E1(self):
        ''' Electric dipole moment '''
        # Basic covariant matrices in the Hamiltonian gauge
        A = self.data_K.Xbar('AA')

        # Other matrices
        D = self.data_K.D_H

        # _____ 1. Internal terms _____ #
        A_int = 1.j * D
        Aa_int = self.kron * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        # _____ 2. External terms _____ #
        Aa_ext = self.kron * A  # Energy diagonal piece
        A_ext = A - Aa_ext           # Energy non-diagonal piece

        # Final formula
        A_H = A_int + A_ext
        return -1 * A_H

    @cached_property
    def E1_internal(self):
        ''' Electric dipole moment (only internal terms) '''
        # Other matrices
        D = self.data_K.D_H

        # _____ 1. Internal terms _____ #
        A_int = 1.j * D
        Aa_int = self.kron * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        # Final formula
        A_H = A_int
        return -1 * A_H

    @cached_property
    def M1(self):
        ''' Magnetic dipole moment '''
        # Basic covariant matrices in the Hamiltonian gauge
        H = self.data_K.Xbar('Ham')
        A = self.data_K.Xbar('AA')
        B = self.data_K.Xbar('BB')
        C = self.data_K.Xbar('CC')
        O = self.data_K.Xbar('OO')

        # Other matrices
        D = self.data_K.D_H
        En = self.data_K.E_K
        Eln_plus = 0.5 * (En[:, :, None] + En[:, None, :])

        # _____ 1. Internal terms _____ #
        A_int = 1.j * D
        Aa_int = self.kron * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        Cbc_int = 1.j * np.einsum('klpa,kpm,kmnb->klnab', A_int, H, A_int)
        C_int = Cbc_int[:, :, :, alpha_A, beta_A] - Cbc_int[:, :, :, beta_A, alpha_A]

        Obc_int = 1.j * np.einsum('klpa,kpnb->klnab', A_int, A_int)
        O_int = Obc_int[:, :, :, alpha_A, beta_A] - Obc_int[:, :, :, beta_A, alpha_A]

        # _____ 2. External terms _____ #
        Aa_ext = self.kron * A  # Energy diagonal piece
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
        H = self.data_K.Xbar('Ham')

        # Other matrices
        En = self.data_K.E_K
        Eln_plus = 0.5 * (En[:, :, None] + En[:, None, :])

        # _____ 1. Internal terms _____ #
        A_int = 1.j * self.data_K.D_H
        Aa_int = self.kron * A_int  # Energy diagonal piece
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
        A = self.data_K.Xbar('AA')
        G = self.data_K.Xbar('GG')

        # _____ 1. Internal terms _____ #

        A_int = 1.j * self.data_K.D_H
        Aa_int = self.kron * A_int  # Energy diagonal piece
        A_int = A_int - Aa_int           # Energy non-diagonal piece

        Gbc_int = np.einsum('klpa,kpnb->klnab', A_int, A_int)
        G_int = 0.5 * (Gbc_int + Gbc_int.swapaxes(3, 4))

        # _____ 2. External terms _____ #

        Aa_ext = self.kron * A  # Energy diagonal piece
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
        # _____ 1. Internal terms _____ #

        A_int = 1.j * self.data_K.D_H
        Aa_int = self.kron * A_int  # Energy diagonal piece
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
        En = self.data_K.E_K
        Enm = En[:, :, None] - En[:, None, :]

        B_q = -0.5j * Enm[:, :, :, None, None] * q
        B_m = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        B_m[:, :, :, alpha_A, beta_A] += m
        B_m[:, :, :, beta_A, alpha_A] -= m
        B = B_m + B_q
        return B_m, B_q, B

    @cached_property
    def Bln_internal(self):
        En = self.data_K.E_K
        Enm = En[:, :, None] - En[:, None, :]

        B_q = -0.5j * Enm[:, :, :, None, None] * self.E2_internal
        B_m = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        B_m[:, :, :, alpha_A, beta_A] += self.M1_internal
        B_m[:, :, :, beta_A, alpha_A] -= self.M1_internal
        B = B_m + B_q
        return B_m, B_q, B

    @cached_property
    def Vn(self):
        ''' Band velocity '''
        V_H = self.data_K.Xbar('Ham', 1)  # (k, m, n, a)
        return np.diagonal(V_H, axis1=1, axis2=2).transpose(0, 2, 1)  # (k, m, a)

    ###########################################################################
