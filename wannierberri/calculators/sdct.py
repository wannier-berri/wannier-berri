import numpy as np
from ..utility import alpha_A, beta_A
from ..formula import Formula
from ..symmetry.point_symmetry import transform_ident, transform_odd
from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom
from .. import factors as factors
from .calculator import MultitermCalculator
from .dynamic import DynamicCalculator

bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom

####################################################
#    Spatially-dispersive conductivity tensor      #
####################################################

# To keep e^2/hbar as the global factor of SDCT

electron_g_factor = physical_constants['electron g factor'][0]
m_spin_prefactor = electron_g_factor * hbar / electron_mass

# _____ Antisymmetric (time-even) spatially-dispersive conductivity tensor _____ #


class SDCT_asym(MultitermCalculator):

    def __init__(self, fermi_sea=True, fermi_surf=True,
                 M1_terms=True, E2_terms=True, V_terms=True, spin=False,
                 **kwargs):
        super().__init__(**kwargs)
        params_terms = dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin)
        # Fermi sea terms
        if fermi_sea:
            self.terms.extend([SDCT_asym_sea_I(**params_terms, **kwargs), SDCT_asym_sea_II(**params_terms, **kwargs)])
        # Fermi surface terms
        if fermi_surf:
            self.terms.extend([SDCT_asym_surf_I(**params_terms, **kwargs), SDCT_asym_surf_II(**params_terms, **kwargs)])


class Formula_SDCT_asym_sea_I(Formula):

    def __init__(self, data_K, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
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
        Vnm_plus = 0.5 * (Vn[:, :, None, :] + Vn[:, None, :, :])

        # --- Formula --- #
        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)

        if M1_terms:
            summ += -np.imag(A[:, :, :, :, None, None] * B_M1.swapaxes(1, 2)[:, :, :, None, :, :])

        if E2_terms:
            summ += -np.imag(A[:, :, :, :, None, None] * B_E2.swapaxes(1, 2)[:, :, :, None, :, :])

        if V_terms:
            summ += Vnm_plus[:, :, :, :, None, None] * np.imag(A[:, :, :, None, :, None] * A.swapaxes(1, 2)[:, :, :, None, None, :])

        summ = summ - summ.swapaxes(3, 4)

        self.summ = summ
        self.ndim = 3
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class SDCT_asym_sea_I(DynamicCalculator):

    def __init__(self, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin))
        self.Formula = Formula_SDCT_asym_sea_I
        self.constant_factor = factors.factor_SDCT

    def factor_omega(self, E1, E2):
        omega = self.omega + 1.j * self.smr_fixed_width
        Z_arg_12 = (E2 - E1)**2 - omega**2  # argument of Z_ln function [iw, n, m]
        Zfac = 1. / Z_arg_12
        return omega * Zfac


class Formula_SDCT_asym_sea_II(Formula):

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
            summ += np.imag(A[:, :, :, :, None, None] * A.swapaxes(1, 2)[:, :, :, None, :, None]) * Vnm_plus[:, :, :, None, None, :]

        self.summ = summ
        self.ndim = 3
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class SDCT_asym_sea_II(DynamicCalculator):

    def __init__(self, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin))
        self.Formula = Formula_SDCT_asym_sea_II
        self.constant_factor = factors.factor_SDCT

    def factor_omega(self, E1, E2):
        omega = self.omega + 1.j * self.smr_fixed_width
        Z_arg_12 = (E2 - E1)**2 - omega**2  # argument of Z_ln function [iw, n, m]
        Zfac = 1. / Z_arg_12
        return omega * (3.0 * (E2 - E1)**2 - omega**2) * Zfac**2


class Formula_SDCT_asym_surf_I(Formula):

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
        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)

        if V_terms:
            summ += -np.imag(A[:, :, :, :, None, None] * A.swapaxes(1, 2)[:, :, :, None, :, None]) * Vn[:, :, None, None, None, :]

        self.summ = summ
        self.ndim = 3
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class SDCT_asym_surf_I(DynamicCalculator):

    def __init__(self, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin))
        self.Formula = Formula_SDCT_asym_surf_I
        self.constant_factor = factors.factor_SDCT

    def factor_omega(self, E1, E2):
        omega = self.omega + 1.j * self.smr_fixed_width
        Z_arg_12 = (E2 - E1)**2 - omega**2  # argument of Z_ln function [iw, n, m]
        Zfac = 1. / Z_arg_12
        return omega * (E2 - E1) * Zfac

    def factor_Efermi(self, E1, E2):
        return -self.FermiDirac(E1)**2 * np.exp((E1 - self.Efermi) / self.kBT) / self.kBT


class Formula_SDCT_asym_surf_II(Formula):

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
        summ = np.zeros((data_K.nk, data_K.num_wann, 3, 3, 3), dtype=complex)

        if M1_terms:
            summ += Vn[:, :, :, None, None] * Bn_M1[:, :, None, :, :]

        summ = summ - summ.swapaxes(3, 4)

        self.summ = summ
        self.ndim = 3
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):  # There is no sum over l
        return self.summ[ik, inn1].sum(axis=0)


class SDCT_asym_surf_II(DynamicCalculator):

    def __init__(self, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin))
        self.Formula = Formula_SDCT_asym_surf_II
        self.constant_factor = factors.factor_SDCT

    def factor_omega(self, E1, E2):
        omega = self.omega + 1.j * self.smr_fixed_width
        return 1. / omega

    def factor_Efermi(self, E1, E2):
        return -self.FermiDirac(E1)**2 * np.exp((E1 - self.Efermi) / self.kBT) / self.kBT

# _____ Symmetric (time-odd) spatially-dispersive conductivity tensor _____ #


class SDCT_sym(MultitermCalculator):
    def __init__(self, fermi_sea=True, fermi_surf=True, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        param_terms = dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin)
        # Fermi sea terms
        if fermi_sea:
            self.terms.extend([SDCT_sym_sea_I(**param_terms, **kwargs), SDCT_sym_sea_II(**param_terms, **kwargs)])
        # Fermi surface terms
        if fermi_surf:
            self.terms.extend([SDCT_sym_surf_I(**param_terms, **kwargs), SDCT_sym_surf_II(**param_terms, **kwargs)])


class Formula_SDCT_sym_sea_I(Formula):

    def __init__(self, data_K, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
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
        Vnm_plus = 0.5 * (Vn[:, :, None, :] + Vn[:, None, :, :])

        # Formula
        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)

        if M1_terms:
            summ += np.real(A[:, :, :, :, None, None] * B_M1.swapaxes(1, 2)[:, :, :, None, :, :])

        if E2_terms:
            summ += np.real(A[:, :, :, :, None, None] * B_E2.swapaxes(1, 2)[:, :, :, None, :, :])

        if V_terms:
            summ += Vnm_plus[:, :, :, :, None, None] * np.real(A[:, :, :, None, :, None] * A.swapaxes(1, 2)[:, :, :, None, None, :])

        summ = summ + summ.swapaxes(3, 4)

        self.summ = summ
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class SDCT_sym_sea_I(DynamicCalculator):

    def __init__(self, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin))
        self.Formula = Formula_SDCT_sym_sea_I
        self.constant_factor = factors.factor_SDCT

    def factor_omega(self, E1, E2):
        omega = self.omega + 1.j * self.smr_fixed_width
        Z_arg_12 = (E2 - E1)**2 - omega**2  # argument of Z_ln function [iw, n, m]
        Zfac = 1. / Z_arg_12
        return 1.j * (E2 - E1) * Zfac


class Formula_SDCT_sym_sea_II(Formula):

    def __init__(self, data_K, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
        super().__init__(data_K, **parameters)
        # Intrinsic multipole moments
        if self.external_terms:
            A = -1. * data_K.SDCT.E1
        else:
            A = -1. * data_K.SDCT.E1_internal

        # Other quantities
        Vn = data_K.SDCT.Vn
        Vnm_plus = Vn[:, :, None, :] + Vn[:, None, :, :]

        # Formula
        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)

        if V_terms:
            summ -= np.real(A[:, :, :, :, None, None] * A.swapaxes(1, 2)[:, :, :, None, :, None]) * Vnm_plus[:, :, :, None, None, :]

        self.summ = summ
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class SDCT_sym_sea_II(DynamicCalculator):

    def __init__(self, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin))
        self.Formula = Formula_SDCT_sym_sea_II
        self.constant_factor = factors.factor_SDCT

    def factor_omega(self, E1, E2):
        omega = self.omega + 1.j * self.smr_fixed_width
        Z_arg_12 = (E2 - E1)**2 - omega**2  # argument of Z_ln function [iw, n, m]
        Zfac = 1. / Z_arg_12
        return 1.j * (E2 - E1)**3 * Zfac**2


class Formula_SDCT_sym_surf_I(Formula):

    def __init__(self, data_K, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
        super().__init__(data_K, **parameters)
        # Intrinsic multipole moments
        if self.external_terms:
            A = -1. * data_K.SDCT.E1
        else:
            A = -1. * data_K.SDCT.E1_internal

        # Other quantities
        Vn = data_K.SDCT.Vn

        # Formula
        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)

        if V_terms:
            summ += np.real(A[:, :, :, :, None, None] * A.swapaxes(1, 2)[:, :, :, None, :, None]) * Vn[:, :, None, None, None, :]

        self.summ = summ
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class SDCT_sym_surf_I(DynamicCalculator):

    def __init__(self, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin))
        self.Formula = Formula_SDCT_sym_surf_I
        self.constant_factor = factors.factor_SDCT

    def factor_omega(self, E1, E2):
        omega = self.omega + 1.j * self.smr_fixed_width
        Z_arg_12 = (E2 - E1)**2 - omega**2  # argument of Z_ln function [iw, n, m]
        Zfac = 1. / Z_arg_12
        return 1.j * (E2 - E1)**2 * Zfac

    def factor_Efermi(self, E1, E2):
        return -self.FermiDirac(E1)**2 * np.exp((E1 - self.Efermi) / self.kBT) / self.kBT


class Formula_SDCT_sym_surf_II(Formula):

    def __init__(self, data_K, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **parameters):
        super().__init__(data_K, **parameters)
        Vn = data_K.SDCT.Vn

        # Formula
        summ = np.zeros((data_K.nk, data_K.num_wann, 3, 3, 3), dtype=complex)

        if V_terms:
            summ = Vn[:, :, :, None, None] * Vn[:, :, None, :, None] * Vn[:, :, None, None, :]

        self.summ = summ
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):  # There is no sum over l
        return self.summ[ik, inn1].sum(axis=0)


class SDCT_sym_surf_II(DynamicCalculator):

    def __init__(self, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin))
        self.Formula = Formula_SDCT_sym_surf_II
        self.constant_factor = factors.factor_SDCT

    def factor_omega(self, E1, E2):
        omega = self.omega + 1.j * self.smr_fixed_width
        return -1.j / omega**2

    def factor_Efermi(self, E1, E2):
        return -self.FermiDirac(E1)**2 * np.exp((E1 - self.Efermi) / self.kBT) / self.kBT

###############################################################################
