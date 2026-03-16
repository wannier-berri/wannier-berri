import numpy as np
from ..utility import alpha_A, beta_A
from ..formula import Formula
from ..symmetry.point_symmetry import transform_ident, transform_odd

from scipy.constants import hbar, electron_mass, physical_constants

electron_g_factor = physical_constants['electron g factor'][0]
m_spin_prefactor = electron_g_factor * hbar / electron_mass


############################ Sea I terms ############################

class Formula_SDCT(Formula):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ndim = 3
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Formula_SDCT_sea_I(Formula_SDCT):

    def __init__(self, data_K, sign_V_term=1,
                 M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
        super().__init__(data_K, **parameters)
        # Intrinsic multipole moments
        if self.external_terms:
            A = -1. * data_K.SDCT.E1
            B_M1, B_E2, B = data_K.SDCT.Bln
            if spin:
                S = data_K.Xbar('SS')
                B_M1[:, :, :, alpha_A, beta_A] += -0.5 * m_spin_prefactor * S
                B_M1[:, :, :, beta_A, alpha_A] -= -0.5 * m_spin_prefactor * S
        else:
            A = -1. * data_K.SDCT.E1_internal
            B_M1, B_E2, B = data_K.SDCT.Bln_internal

        # Other quantities
        Vn = data_K.SDCT.Vn
        assert np.max(np.abs(Vn.imag)) < 1e-8, f"Vn should be real but has imaginary part of {np.max(np.abs(Vn.imag))}"
        Vnm_plus = 0.5 * (Vn[:, :, None, :] + Vn[:, None, :, :])

        # --- Formula --- #
        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)

        if M1_terms:
            summ += A[:, :, :, :, None, None] * B_M1.swapaxes(1, 2)[:, :, :, None, :, :]

        if E2_terms:
            summ += A[:, :, :, :, None, None] * B_E2.swapaxes(1, 2)[:, :, :, None, :, :]

        if V_terms:
            # This is weird, why we have to put a minus sign here to get the correct term for the symmetric part of the SDCT?
            summ += sign_V_term * (-Vnm_plus[:, :, :, :, None, None] * A[:, :, :, None, :, None] * A.swapaxes(1, 2)[:, :, :, None, None, :])

        self.summ = summ


class Formula_SDCT_asym_sea_I(Formula_SDCT_sea_I):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summ = -np.imag(self.summ - self.summ.swapaxes(3, 4))
        self.transformTR = transform_ident


class Formula_SDCT_sym_sea_I(Formula_SDCT_sea_I):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, sign_V_term=-1, **kwargs)
        self.summ = np.real(self.summ + self.summ.swapaxes(3, 4))
        self.transformTR = transform_odd


############################# Sea II terms ############################

class Formula_SDCT_sea_II(Formula_SDCT):

    def __init__(self, data_K, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
        super().__init__(data_K, **parameters)
        # Intrinsic multipole moments
        if self.external_terms:
            A = -1. * data_K.SDCT.E1
        else:
            A = -1. * data_K.SDCT.E1_internal

        # Other quantities
        Vn = data_K.SDCT.Vn
        Vnm_plus = 0.5 * (Vn[:, :, None, :] + Vn[:, None, :, :])

        # --- Formula --- #
        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)

        if V_terms:
            summ -= A[:, :, :, :, None, None] * A.swapaxes(1, 2)[:, :, :, None, :, None] * Vnm_plus[:, :, :, None, None, :]

        self.summ = summ


class Formula_SDCT_asym_sea_II(Formula_SDCT_sea_II):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summ = -np.imag(self.summ)
        self.transformTR = transform_ident


class Formula_SDCT_sym_sea_II(Formula_SDCT_sea_II):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summ = 2 * np.real(self.summ)
        self.transformTR = transform_odd

############################## Surface I terms ############################


class Formula_SDCT_surf_I(Formula_SDCT):

    def __init__(self, data_K, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
        super().__init__(data_K, **parameters)
        # Intrinsic multipole moments
        if self.external_terms:
            A = -1. * data_K.SDCT.E1
        else:
            A = -1. * data_K.SDCT.E1_internal

        # Other quantities
        Vn = data_K.SDCT.Vn

        # --- Formula --- #
        self.summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)

        if V_terms:
            self.summ += (A[:, :, :, :, None, None] * A.swapaxes(1, 2)[:, :, :, None, :, None]) * Vn[:, :, None, None, None, :]


    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Formula_SDCT_asym_surf_I(Formula_SDCT_surf_I):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summ = -np.imag(self.summ)
        self.transformTR = transform_ident


class Formula_SDCT_sym_surf_I(Formula_SDCT_surf_I):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summ = np.real(self.summ)
        self.transformTR = transform_odd


###################################### Surface II terms ############################

class Formula_SDCT_surf_II(Formula_SDCT):

    def trace_ln(self, ik, inn1, inn2):  # There is no sum over l
        return self.summ[ik, inn1].sum(axis=0)


class Formula_SDCT_asym_surf_II(Formula_SDCT_surf_II):

    def __init__(self, data_K, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
        super().__init__(data_K, **parameters)
        # Intrinsic multipole moments
        if self.external_terms:
            B_M1, B_E2, B = data_K.SDCT.Bln
            if spin:
                S = data_K.Xbar('SS')
                B_M1[:, :, :, alpha_A, beta_A] += -0.5 * m_spin_prefactor * S
                B_M1[:, :, :, beta_A, alpha_A] -= -0.5 * m_spin_prefactor * S
        else:
            B_M1, B_E2, B = data_K.SDCT.Bln_internal
        Bn_M1 = np.diagonal(B_M1, axis1=1, axis2=2).transpose(0, 3, 1, 2)

        # Other quantities
        Vn = data_K.SDCT.Vn

        # --- Formula --- #
        self.summ = np.zeros((data_K.nk, data_K.num_wann, 3, 3, 3), dtype=complex)

        if M1_terms:
            self.summ += Vn[:, :, :, None, None] * Bn_M1[:, :, None, :, :]

        self.summ = self.summ - self.summ.swapaxes(3, 4)
        self.transformTR = transform_ident


class Formula_SDCT_sym_surf_II(Formula_SDCT_surf_II):

    def __init__(self, data_K, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
        super().__init__(data_K, **parameters)
        Vn = data_K.SDCT.Vn

        # Formula
        self.summ = np.zeros((data_K.nk, data_K.num_wann, 3, 3, 3), dtype=complex)

        if V_terms:
            self.summ += Vn[:, :, :, None, None] * Vn[:, :, None, :, None] * Vn[:, :, None, None, :]
        self.transformTR = transform_odd
