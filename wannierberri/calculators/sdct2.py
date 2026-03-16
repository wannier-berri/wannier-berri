import numpy as np
from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom
from .. import factors as factors
from .calculator import MultitermCalculator
from .dynamic import DynamicCalculator
from ..formula.sdct import Formula_SDCT_asym_sea_I, Formula_SDCT_asym_sea_II, Formula_SDCT_sym_sea_I, Formula_SDCT_sym_sea_II, Formula_SDCT_asym_surf_I, Formula_SDCT_asym_surf_II, Formula_SDCT_sym_surf_I, Formula_SDCT_sym_surf_II

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


class SDCT(MultitermCalculator):

    def __init__(self, sym=True, asym=True,
                 fermi_sea=True, fermi_surf=True,
                 M1_terms=True, E2_terms=True, V_terms=True, spin=False,
                 **kwargs):
        super().__init__(**kwargs)
        params_terms = dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin)
        # Fermi sea terms
        if fermi_sea:
            if asym:
                self.terms.extend([SDCT_asym_sea_I(**params_terms, **kwargs), SDCT_asym_sea_II(**params_terms, **kwargs)])
            if sym:
                self.terms.extend([SDCT_sym_sea_I(**params_terms, **kwargs), SDCT_sym_sea_II(**params_terms, **kwargs)])
        # Fermi surface terms
        if fermi_surf:
            if asym:
                self.terms.extend([SDCT_asym_surf_I(**params_terms, **kwargs), SDCT_asym_surf_II(**params_terms, **kwargs)])
            if sym:
                self.terms.extend([SDCT_sym_surf_I(**params_terms, **kwargs), SDCT_sym_surf_II(**params_terms, **kwargs)])
        assert len(self.terms) > 0, "At least one term must be included in the SDCT calculation (set fermi_sea and/or fermi_surf to True)."


class SDCT_asym(SDCT):

    def __init__(self, **kwargs):
        super().__init__(asym=True, sym=False, **kwargs)


class SDCT_sym(SDCT):

    def __init__(self, **kwargs):
        super().__init__(asym=False, sym=True, **kwargs)


class _SDCT_term(DynamicCalculator):

    def __init__(self, formula, M1_terms=True, E2_terms=True, V_terms=True, spin=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(M1_terms=M1_terms, E2_terms=E2_terms, V_terms=V_terms, spin=spin))
        self.Formula = formula
        self.constant_factor = factors.factor_SDCT

    def _factor_omega(self, E1, E2):
        omega = self.omega + 1.j * self.smr_fixed_width
        Z_arg_12 = (E2 - E1)**2 - omega**2  # argument of Z_ln function [iw, n, m]
        Zfac = 1. / Z_arg_12
        return omega, Z_arg_12, Zfac


class SDCT_sym_sea_I(_SDCT_term):

    def __init__(self, **kwargs):
        super().__init__(formula=Formula_SDCT_sym_sea_I, **kwargs)

    def factor_omega(self, E1, E2):
        omega, Z_arg_12, Zfac = self._factor_omega(E1, E2)
        return 1.j * (E2 - E1) * Zfac


class SDCT_asym_sea_I(_SDCT_term):

    def __init__(self, **kwargs):
        super().__init__(formula=Formula_SDCT_asym_sea_I, **kwargs)

    def factor_omega(self, E1, E2):
        omega, Z_arg_12, Zfac = self._factor_omega(E1, E2)
        return omega * Zfac


class SDCT_sym_sea_II(_SDCT_term):

    def __init__(self, **kwargs):
        super().__init__(formula=Formula_SDCT_sym_sea_II, **kwargs)

    def factor_omega(self, E1, E2):
        omega, Z_arg_12, Zfac = self._factor_omega(E1, E2)
        return 1.j * (E2 - E1)**3 * Zfac**2


class SDCT_asym_sea_II(_SDCT_term):

    def __init__(self, **kwargs):
        super().__init__(formula=Formula_SDCT_asym_sea_II, **kwargs)

    def factor_omega(self, E1, E2):
        omega, Z_arg_12, Zfac = self._factor_omega(E1, E2)
        return omega * (3.0 * (E2 - E1)**2 - omega**2) * Zfac**2


class _SDCT_surf_term(_SDCT_term):

    def factor_Efermi(self, E1, E2):
        return -self.FermiDirac(E1)**2 * np.exp((E1 - self.Efermi) / self.kBT) / self.kBT


class SDCT_sym_surf_I(_SDCT_surf_term):

    def __init__(self, **kwargs):
        super().__init__(formula=Formula_SDCT_sym_surf_I, **kwargs)

    def factor_omega(self, E1, E2):
        omega, Z_arg_12, Zfac = self._factor_omega(E1, E2)
        return 1.j * (E2 - E1)**2 * Zfac


class SDCT_asym_surf_I(_SDCT_surf_term):

    def __init__(self, **kwargs):
        super().__init__(formula=Formula_SDCT_asym_surf_I, **kwargs)

    def factor_omega(self, E1, E2):
        omega, Z_arg_12, Zfac = self._factor_omega(E1, E2)
        return omega * (E2 - E1) * Zfac


class SDCT_asym_surf_II(_SDCT_surf_term):

    def __init__(self, **kwargs):
        super().__init__(formula=Formula_SDCT_asym_surf_II, **kwargs)

    def factor_omega(self, E1, E2):
        omega, Z_arg_12, Zfac = self._factor_omega(E1, E2)
        return 1. / omega


class SDCT_sym_surf_II(_SDCT_surf_term):

    def __init__(self, **kwargs):
        super().__init__(formula=Formula_SDCT_sym_surf_II, **kwargs)

    def factor_omega(self, E1, E2):
        omega, Z_arg_12, Zfac = self._factor_omega(E1, E2)
        return -1.j / omega**2
