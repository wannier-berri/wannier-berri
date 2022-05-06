from .classes import StaticCalculator
from wannierberri import covariant_formulak as frml
from wannierberri import covariant_formulak_basic as frml_basic
from ..formula import FormulaProduct, FormulaSum
#from wannierberri import fermiocean

#######################################
#                                     #
#         integration (Efermi-only)   #
#                                     #
#######################################

#  TODO: Ideally, a docstring of every calculator should contain the equation that it implements
#        and references (with urls) to the relevant papers
#  XxLiu: Add references url later

######################
# physical constants #
######################

from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom  #, Boltzmann
from ..__utility import TAU_UNIT
#bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom

###########
# factors #
###########

fac_morb_Z = elementary_charge/2/hbar*Ang_SI**2 # change unit of m_orb*B to (eV).
fac_spin_Z = elementary_charge * hbar / (2 * electron_mass) # change unit of m_spin*B to (eV).

#gme
factor_t0_0_1 = -(elementary_charge / Ang_SI**2
                * elementary_charge / hbar) # change velocity unit (red)
# Anomalous Hall conductivity
factor_t0_1_0 = -(elementary_charge**2 / hbar / Ang_SI) * 100
# Ohmic conductivity
factor_t1_1_0 = (elementary_charge**2 / hbar / Ang_SI * TAU_UNIT *100
                * elementary_charge / hbar) # change velocity unit (red)
# Linear magnetoresistance
factor_t1_1_1 = (elementary_charge**3 /hbar**2 * Ang_SI * TAU_UNIT
                * elementary_charge / hbar) # change velocity unit (red)
# Quaduratic magnetoresistance
factor_t1_1_2 = (elementary_charge**4 /hbar**3 * Ang_SI**3 * TAU_UNIT
                * elementary_charge / hbar) # change velocity unit (red)
# Nonlinear anomalous Hall conductivity
factor_t1_2_0 = elementary_charge**3 /hbar**2 * TAU_UNIT
# Nonlinear Hall conductivity
factor_t1_2_1 = (elementary_charge**4 /hbar**3 * Ang_SI**2 * TAU_UNIT)
# Classic Hall conductivity
factor_t2_1_1 = -(elementary_charge**3 /hbar**2 * Ang_SI * TAU_UNIT**2 * 100
                * elementary_charge**2 / hbar**2) # change velocity unit (red)
# Drude conductivity
factor_t2_2_0 = -(elementary_charge**3 /hbar**2 * TAU_UNIT**2
                * elementary_charge / hbar) # change velocity unit (red)
# Electical magnetochiral anistropy
factor_t2_2_1 = -(elementary_charge**4 /hbar**3 * Ang_SI**2 * TAU_UNIT**2
                * elementary_charge / hbar) # change velocity unit (red)

####################
# basic quantities #
####################

class _DOS(StaticCalculator):

    def __init__(self, fder,**kwargs):
        self.Formula = frml.Identity
        self.factor = 1
        self.fder = fder
        super().__init__(**kwargs)

    def __call__(self, data_K):
        return super().__call__(data_K) * data_K.cell_volume


class DOS(_DOS):

    def __init__(self, **kwargs):
        super().__init__(fder=1, **kwargs)


class CumDOS(_DOS):

    def __init__(self, **kwargs):
        super().__init__(fder=0, **kwargs)


class Spin(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.Spin
        self.factor = 1
        self.fder = 0
        self.comment = r"""Spin per unit cell (dimensionless)
:math: `\int [dk] s f`"""
        super().__init__(**kwargs)


class Hplus(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.Morb_Hpm
        self.factor = -eV_au / bohr**2
        self.fder = 0
        super().__init__(**kwargs)


class Omega(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.Omega
        self.factor = -eV_au / bohr**2
        self.fder = 0
        super().__init__(**kwargs)



####################
#  cunductivities  #
####################

class AHC(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.Omega
        self.factor = factor_t0_1_0
        self.fder = 0
        self.comment = r"""Anomalous Hall effect (s^3 * A^2 / (kg * m^3) = S/m)
Return:
:math: `O = - e^2/\hbar \int [dk] \Omega f`
Instruction:
:math: `j_\alpha = \sigma_{\alpha\beta} E_\beta = \epsilon_{\alpha\beta\delta} O_delta E_\beta`"""
        super().__init__(**kwargs)


class Ohmic(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.InvMass
        self.factor = factor_t1_1_0
        self.fder = 0
        self.comment  = r"""Ohmic conductivity (s^3 * A^2 / (kg * m^3) = S/m)
With Fermi surface integral.
Return:
:math: `\sigma_{\alpha\beta} = e^2/\hbar \tau \int [dk] v_{\alpha\beta} f`
Instruction:
:math: `j_\alpha = \sigma_{\alpha\beta} E_\beta`
    """
        super().__init__(**kwargs)


class BerryDipole_FermiSurf(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.VelOmega
        self.factor = 1
        self.fder = 1
        self.comment = r"""Berry curvature dipole (dimensionless)
With Fermi surface integral.
Return:
:math: `D_{\beta\delta} = -\tau \int [dk] \Omega_\delta v_\beta f'`"""
        super().__init__(**kwargs)


class BerryDipole_FermiSea(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.DerOmega
        self.factor = 1
        self.fder = 0
        self.comment = r"""Berry curvature dipole (dimensionless)
With Fermi sea integral.
Return:
:math: `D_{\beta\delta} = \tau \int [dk] \partial_beta \Omega_\delta f`"""
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


############
#  TOOLS   #
############

class staticCalculator_Morb(Calculator):

    def __init__(self, Efermi, tetra=False, kwargs_formula={}, **kwargs):


