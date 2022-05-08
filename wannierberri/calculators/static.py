from .classes import StaticCalculator
from wannierberri import covariant_formulak as frml
from wannierberri import covariant_formulak_basic as frml_basic
from termcolor import cprint
import numpy as np
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
from ..__utility import TAU_UNIT, alpha_A, beta_A
#bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom

###########
# factors #
###########

fac_morb_Z = elementary_charge/2/hbar * Ang_SI**2 # change unit of m_orb*B to (eV).
fac_spin_Z = elementary_charge * hbar / (2 * electron_mass) / Ang_SI**2# change unit of m_spin*B to (eV).

#gme
factor_t0_0_1 = -(elementary_charge / Ang_SI**2
                * elementary_charge / hbar) # change velocity unit (red)
# Anomalous Hall conductivity
factor_t0_1_0 = -(elementary_charge**2 / hbar / Ang_SI) /100.
# Ohmic conductivity
factor_t1_1_0 = (elementary_charge**2 / hbar / Ang_SI * TAU_UNIT /100.
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
factor_t2_1_1 = -(elementary_charge**3 /hbar**2 * Ang_SI * TAU_UNIT**2 /100.
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
        self.comment = r"""DOS"""
        super().__init__(fder=1, **kwargs)


class CumDOS(_DOS):

    def __init__(self, **kwargs):
        self.comment = r"""CumDOS"""
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
        self.factor = 1
        self.fder = 0
        self.comment = r""":math: `\int [dk] (G + H) f`"""
        super().__init__(**kwargs)


class Hplus_test(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml_basic.tildeHGc
        self.factor = 1
        self.fder = 0
        self.comment = r""":math: `\int [dk] (G + H) f`"""
        super().__init__(**kwargs)


class Morb():

    def __init__(self, Efermi, tetra=False, use_factor=True, print_comment=True, kwargs_formula={}, **kwargs):
        self.Efermi = Efermi
        self.tetra = tetra
        self.comment = r"""Orbital magnetic moment per unit cell (mu_B)
        :math: `M = -\int [dk] (H + G - 2Ef*\Omega) f`"""
        self.kwargs = kwargs_formula
        if use_factor:
            self.factor = -eV_au / bohr**2 
        else:
            self.factor = np.sign(self.factor)
        if print_comment:
            cprint("{}\n".format(self.comment), 'cyan', attrs=['bold'])

    def __call__(self, data_K):
        return self.factor * data_K.cell_volume * (
                Hplus(Efermi=self.Efermi, tetra=self.tetra,
                    use_factor=False, print_comment=False, kwargs_formula=self.kwargs)(data_K)
            + 2 * AHC(Efermi=self.Efermi, tetra=self.tetra, use_factor=False,
                print_comment=False, kwargs_formula=self.kwargs)(data_K).mul_array(self.Efermi))
            #with use_factor, the factor of AHC is -1.

class Morb_test():

    def __init__(self, Efermi, tetra=False, use_factor=True, print_comment=True, kwargs_formula={}, **kwargs):
        self.Efermi = Efermi
        self.tetra = tetra
        self.comment = r"""Orbital magnetic moment per unit cell (mu_B)
        :math: `M = -\int [dk] (H + G - 2Ef*\Omega) f`"""
        self.kwargs = kwargs_formula
        if use_factor:
            self.factor = -eV_au / bohr**2 
        else:
            self.factor = np.sign(self.factor)
        if print_comment:
            cprint("{}\n".format(self.comment), 'cyan', attrs=['bold'])

    def __call__(self, data_K):
        return self.factor * data_K.cell_volume * (
                Hplus_tert(Efermi=self.Efermi, tetra=self.tetra,
                    use_factor=False, print_comment=False, kwargs_formula=self.kwargs)(data_K)
            + 2 * AHC_test(Efermi=self.Efermi, tetra=self.tetra, use_factor=False,
                print_comment=False, kwargs_formula=self.kwargs)(data_K).mul_array(self.Efermi))
            #with use_factor, the factor of AHC is -1.


####################
#  cunductivities  #
####################

# E^0 B^1
class VelHplus(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.VelHplus 
        self.factor = 1
        self.fder = 1
        self.comment = r""":math: `-\tau \int [dk] v_\beta (H + G)_\delta f'`"""
        super().__init__(**kwargs)


class GME_orb_FermiSurf():

    def __init__(self, Efermi, tetra=False, use_factor=True, print_comment=True, kwargs_formula={}, **kwargs):
        self.Efermi = Efermi
        self.tetra = tetra
        self.kwargs = kwargs_formula
        self.comment = r"""Gyrotropic tensor orbital part (A/m^2/T)
        With Fermi surface integral.
        Return
        :math: `m = H + G - Ef*\Omega`
        :math: `K^{orb}_{\alpha\mu} = e \tau \int [dk] v_\alpha * m_\mu f'`
        Instruction:
        :math: `j_\alpha = K_{\alpha\mu} B_\mu"""
        if use_factor:
            self.factor =  -elementary_charge**2 / (2 * hbar) # * factor_t0_0_1
        else:
            self.factor = np.sign(self.factor)
        if print_comment:
            cprint("{}\n".format(self.comment), 'cyan', attrs=['bold'])

    def __call__(self, data_K):
        return self.factor * (
                VelHplus(Efermi=self.Efermi, tetra=self.tetra,
                    use_factor=False, print_comment=False, kwargs_formula=self.kwargs)(data_K)
            - 2 * NLAHC_FermiSurf(Efermi=self.Efermi, tetra=self.tetra, use_factor=False,
                print_comment=False, kwargs_formula=self.kwargs)(data_K).mul_array(self.Efermi))


class DerHplus(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.DerMorb 
        self.factor = 1
        self.fder = 0
        self.comment = r""":math: `\tau \int [dk] \partial_\alpha (H + G)_\mu f`"""
        super().__init__(**kwargs)
    
    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class DerHplus_test(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml_basic.tildeHGc_d
        self.factor = 1
        self.fder = 0
        self.comment = r""":math: `\tau \int [dk] \partial_\alpha (H + G)_\mu f`"""
        super().__init__(**kwargs)
    
    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class GME_orb_FermiSea():

    def __init__(self, Efermi, tetra=False, use_factor=True, print_comment=True, kwargs_formula={}, **kwargs):
        self.Efermi = Efermi
        self.tetra = tetra
        self.kwargs = kwargs_formula
        self.comment = r"""Gyrotropic tensor orbital part (A/m^2/T)
        With Fermi sea integral.
        Return
        :math: `m = H + G - Ef*\Omega`
        :math: `K^{orb}_{\alpha\mu} = -e \tau \int [dk] \partial_\alpha m_\mu f`
        Instruction:
        :math: `j_\alpha = K_{\alpha\mu} B_\mu"""
        if use_factor:
            self.factor =  -elementary_charge**2 / (2 * hbar)# * factor_t0_0_1
        else:
            self.factor = np.sign(self.factor)
        if print_comment:
            cprint("{}\n".format(self.comment), 'cyan', attrs=['bold'])

    def __call__(self, data_K):
        return self.factor * (
                DerHplus(Efermi=self.Efermi, tetra=self.tetra,
                    use_factor=False, print_comment=False, kwargs_formula=self.kwargs)(data_K)
            - 2 * NLAHC_FermiSea(Efermi=self.Efermi, tetra=self.tetra, use_factor=False,
                print_comment=False, kwargs_formula=self.kwargs)(data_K).mul_array(self.Efermi))


class GME_orb_FermiSea_test():

    def __init__(self, Efermi, tetra=False, use_factor=True, print_comment=True, kwargs_formula={}, **kwargs):
        self.Efermi = Efermi
        self.tetra = tetra
        self.kwargs = kwargs_formula
        self.comment = r"""Gyrotropic tensor orbital part (A/m^2/T)
        With Fermi sea integral.
        Return
        :math: `m = H + G - Ef*\Omega`
        :math: `K^{orb}_{\alpha\mu} = -e \tau \int [dk] \partial_\alpha m_\mu f`
        Instruction:
        :math: `j_\alpha = K_{\alpha\mu} B_\mu"""
        if use_factor:
            self.factor =  -elementary_charge**2 / (2 * hbar)# * factor_t0_0_1
        else:
            self.factor = np.sign(self.factor)
        if print_comment:
            cprint("{}\n".format(self.comment), 'cyan', attrs=['bold'])

    def __call__(self, data_K):
        return self.factor * (
                DerHplus_test(Efermi=self.Efermi, tetra=self.tetra,
                    use_factor=False, print_comment=False, kwargs_formula=self.kwargs)(data_K)
            - 2 * NLAHC_FermiSea_test(Efermi=self.Efermi, tetra=self.tetra, use_factor=False,
                print_comment=False, kwargs_formula=self.kwargs)(data_K).mul_array(self.Efermi))


class GME_spin_FermiSea(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.DerSpin
        self.factor = -bohr_magneton / Ang_SI**2 # * factor_t0_0_1
        self.fder = 0
        self.comment = r"""Gyrotropic tensor spin part (A/m^2/T)
        With Fermi sea integral.
        Return
        :math: `K^{spin}_{\alpha\mu} = -e \tau \int [dk] \partial_\alpha s_\mu f`
        Instruction:
        :math: `j_\alpha = K_{\alpha\mu} B_\mu"""
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class GME_spin_FermiSurf(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.VelSpin
        self.factor = -bohr_magneton / Ang_SI**2 # * factor_t0_0_1
        self.fder = 1
        self.comment = r"""Gyrotropic tensor spin part (A/m^2/T)
        With Fermi sea integral.
        Return
        :math: `K^{spin}_{\alpha\mu} = e \tau \int [dk] v_\alpha s_\mu f'`
        Instruction:
        :math: `j_\alpha = K_{\alpha\mu} B_\mu"""
        super().__init__(**kwargs)


# E^1 B^0
class AHC(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.Omega
        self.factor = factor_t0_1_0
        self.fder = 0
        self.comment = r"""Anomalous Hall conductivity (s^3 * A^2 / (kg * m^3) = S/m)
        Return:
        :math: `O = - e^2/\hbar \int [dk] \Omega f`
        Instruction:
        :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta = \epsilon_{\alpha\beta\delta} O_delta E_\beta`"""
        super().__init__(**kwargs)


class AHC_test(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml_basic.tildeFc
        self.factor = factor_t0_1_0
        self.fder = 0
        self.comment = r"""Anomalous Hall conductivity (s^3 * A^2 / (kg * m^3) = S/m)
        Return:
        :math: `O = - e^2/\hbar \int [dk] \Omega f`
        Instruction:
        :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta = \epsilon_{\alpha\beta\delta} O_delta E_\beta`"""
        super().__init__(**kwargs)


class Ohmic_FermiSea(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.InvMass
        self.factor = factor_t1_1_0
        self.fder = 0
        self.comment  = r"""Ohmic conductivity (s^3 * A^2 / (kg * m^3) = S/m)
        With Fermi sea integral.
        Return:
        :math: `\sigma_{\alpha\beta} = e^2/\hbar \tau \int [dk] v_{\alpha\beta} f`
        Instruction:
        :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta`
    """
        super().__init__(**kwargs)


class Ohmic_FermiSurf(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.VelVel
        self.factor = factor_t1_1_0
        self.fder = 1
        self.comment  = r"""Ohmic conductivity (s^3 * A^2 / (kg * m^3) = S/m)
        With Fermi surface integral.
        Return:
        :math: `\sigma_{\alpha\beta} = -e^2/\hbar \tau \int [dk] v_\alpha v_\beta f'`
        Instruction:
        :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta`
    """
        super().__init__(**kwargs)


# E^1 B^1
class Hall_classic_FermiSurf(StaticCalculator):
    def __init__(self, **kwargs):
        self.Formula = frml.VelMassVel
        self.factor = factor_t2_1_1
        self.fder = 1
        self.comment  = r"""Classic Hall conductivity (S/m/T)
        With Fermi surface integral.
        Return:
        :math: `\sigma_{\alpha\beta:\mu} = e^3/\hbar^2 \tau^2 \epsilon_{\gamma\mu\rho} \int [dk] v_\alpha \partial_\rho v_\beta v_\gamma f'`
        Instruction:
        :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta B_\mu`
    """
        super().__init__(**kwargs)
    
    def __call__(self, data_K):
        res = super().__call__(data_K)
        res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
        res.data = 0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
        res.rank -= 2
        return res


class Hall_classic_FermiSea(StaticCalculator):
    def __init__(self, **kwargs):
        self.Formula = frml.MassMass
        self.factor = factor_t2_1_1
        self.fder = 0
        self.comment  = r"""Classic Hall conductivity (S/m/T)
        With Fermi sea integral.
        Return:
        :math: `\sigma_{\alpha\beta:\mu} = -e^3/\hbar^2 \tau^2 \epsilon_{\gamma\mu\rho} \int [dk] \partial_\gamma v_\alpha \partial_\rho v_\beta f`
        Instruction:
        :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta B_\mu`
    """
        super().__init__(**kwargs)
    
    def __call__(self, data_K):
        res = super().__call__(data_K)
        res.data = res.data.transpose(0, 4, 1, 2, 3)
        res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
        res.data = 0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
        res.rank -= 2
        return res


# E^2 B^0
class NLAHC_FermiSurf(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.VelOmega
        self.factor = factor_t1_2_0
        self.fder = 1
        self.comment = r"""Nonlinear anomalous Hall conductivity  (S^2/A) with use_factor=True
Berry curvature dipole (dimensionless) with use_factor=False
        With Fermi surface integral.
        Return:
        :math: `D_{\beta\delta} = -e^3/\hbar^2 \tau \int [dk] v_\beta \Omega_\delta f'`
        Instruction:
        :math: `j_\alpha = \epsilon_{\alpha\delta\gamma} \D_{\beta\delta} E_\beta E\gamma`"""
        super().__init__(**kwargs)


class NLAHC_FermiSea(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.DerOmega
        self.factor = factor_t1_2_0
        self.fder = 0
        self.comment = r"""Nonlinear anomalous Hall conductivity  (S^2/A) with use_factor=True
Berry curvature dipole (dimensionless) with use_factor=False
        With Fermi sea integral.
        Return:
        :math: `D_{\beta\delta} = e^3/\hbar^2 \tau \int [dk] \partial_beta \Omega_\delta f`
        Instruction:
        :math: `j_\alpha = \epsilon_{\alpha\delta\gamma} \D_{\beta\delta} E_\beta E\gamma`"""
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class NLAHC_FermiSea_test(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml_basic.tildeFc_d
        self.factor = factor_t1_2_0
        self.fder = 0
        self.comment = r"""Nonlinear anomalous Hall conductivity  (S^2/A) with use_factor=True
Berry curvature dipole (dimensionless) with use_factor=False
        With Fermi sea integral.
        Return:
        :math: `D_{\beta\delta} = e^3/\hbar^2 \tau \int [dk] \partial_beta \Omega_\delta f`
        Instruction:
        :math: `j_\alpha = \epsilon_{\alpha\delta\gamma} \D_{\beta\delta} E_\beta E\gamma`"""
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class Drude_FermiSea(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.Der3E
        self.factor = factor_t2_2_0
        self.fder = 0
        self.comment = r"""Drude conductivity (S^2/A)
        With Fermi sea integral.
        Return:
        :math: `\sigma_{\alpha\beta\gamma} = -e^3/\hbar^2 \tau^2 \int [dk] \partial_{\beta\gamma} v_\alpha f`
        Instruction:
        :math: `j_\alpha = \sigma_{\alpha\beta\gamma} E_\beta E\gamma`"""
        super().__init__(**kwargs)


class Drude_FermiSurf(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.MassVel
        self.factor = factor_t2_2_0
        self.fder = 1
        self.comment = r"""Drude conductivity (S^2/A)
        With Fermi surface integral.
        Return:
        :math: `\sigma_{\alpha\beta\gamma} = e^3/\hbar^2 \tau^2 \int [dk] \partial_\beta v_\alpha v_\gamma f'`
        Instruction:
        :math: `j_\alpha = \sigma_{\alpha\beta\gamma} E_\beta E\gamma`"""
        super().__init__(**kwargs)


class Drude_Fermider2(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.VelVelVel
        self.factor = factor_t2_2_0
        self.fder = 2
        self.comment = r"""Drude conductivity (S^2/A)
        With Fermi surface integral.
        Return:
        :math: `\sigma_{\alpha\beta\gamma} = -e^3/\hbar^2 \tau^2 \int [dk] v_\beta v_\alpha v_\gamma f'`
        Instruction:
        :math: `j_\alpha = \sigma_{\alpha\beta\gamma} E_\beta E\gamma`"""
        super().__init__(**kwargs)


class Spin_Hall(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.SpinOmega
        self.factor = factor_t0_1_0 * -0.5
        self.fder = 0
        super().__init__(**kwargs)


class Zeeman_AHC_spin(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.SOmega
        self.factor = bohr_magneton / (elementary_charge * Ang_SI) * elementary_charge**2 / hbar / 100
        self.fder = 1
        self.comment = r"""  """
        super().__init__(**kwargs)


class OmegaOmega(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.OmegaOmega
        self.factor = 1
        self.fder = 1
        super().__init__(**kwargs)

class OmegaHplus(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.OmegaHplus
        self.factor = 1
        self.fder = 1
        super().__init__(**kwargs)


class Zeeman_AHC_orb():

    def __init__(self, Efermi, tetra=False, use_factor=True, print_comment=True, kwargs_formula={}, **kwargs):
        self.Efermi = Efermi
        self.tetra = tetra
        self.kwargs = kwargs_formula
        self.comment = r"""  """
        if use_factor:
            self.factor = Ang_SI * elementary_charge / (2 * hbar) * elementary_charge**2 / hbar / 100
        else:
            self.factor = np.sign(self.factor)
        if print_comment:
            cprint("{}\n".format(self.comment), 'cyan', attrs=['bold'])

    def __call__(self, data_K):
        return self.factor * (
                OmegaHplus(Efermi=self.Efermi, tetra=self.tetra,
                    use_factor=False, print_comment=False, kwargs_formula=self.kwargs)(data_K)
            - 2 * OmegaOmega(Efermi=self.Efermi, tetra=self.tetra, use_factor=False,
                print_comment=False, kwargs_formula=self.kwargs)(data_K).mul_array(self.Efermi))
