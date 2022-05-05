import numpy as np
from .__utility import alpha_A, beta_A, TAU_UNIT
from collections import defaultdict
from . import __result as result
from math import ceil
from . import covariant_formulak as frml
from .formula import FormulaProduct
from . import covariant_formulak_basic as frml_basic


######################
# physical constants #
######################

from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom  #, Boltzmann
bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom

###########
# factors #
###########

fac_morb_Z = elementary_charge/2/hbar*Ang_SI**2 # change unit of m.B to (eV).

#gme
factor_t0_0_1 = -(elementary_charge / Ang_SI**2
                * elementary_charge / hbar) # change velocity unit (red)
# Anomalous Hall conductivity
factor_t0_1_0 = -(elementary_charge**2 / hbar / Ang_SI)
# Ohmic conductivity
factor_t1_1_0 = (elementary_charge**2 / hbar / Ang_SI * TAU_UNIT
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
factor_t2_1_1 = -(elementary_charge**3 /hbar**2 * Ang_SI * TAU_UNIT**2
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

def cumdos(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.Identity(), data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * data_K.cell_volume


def dos(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.Identity(), data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * data_K.cell_volume


def spin(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `S = \int [dk] s f`
    """
    return FermiOcean(
        frml.Spin(data_K), data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()


def Hplus_der(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" Fermi sea
    :math: `H_{\alpha\beta} = \tau \int [dk] \partial_\alpha (H + G)_\beta f`
    """
    res = FermiOcean(
        frml.DerMorb(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def Hplus_der_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" Fermi sea
    :math: `H_{\alpha\beta} = \tau \int [dk] \partial_\alpha (H + G)_\beta f`
    """
    res = FermiOcean(
        frml_basic.tildeHGc_d(data_K, sign=+1, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def Morb(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `M = - \int [dk] (H + G - 2Ef * \Omega) f`
    Unit: :math: `\mu_B per unit cell`
    """
    fac_morb = -eV_au / bohr**2
    return (
        FermiOcean(
            frml.Morb_Hpm(data_K, sign=+1, **kwargs_formula),
            data_K,
            Efermi,
            tetra,
            fder=0,
            degen_thresh=degen_thresh,
            degen_Kramers=degen_Kramers)() - 2 * FermiOcean(
                frml.Omega(data_K, **kwargs_formula),
                data_K,
                Efermi,
                tetra,
                fder=0,
                degen_thresh=degen_thresh,
                degen_Kramers=degen_Kramers)().mul_array(Efermi)) * (data_K.cell_volume * fac_morb)


def Morb_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `M = - \int [dk] (H + G - 2Ef * \Omega) f`
    Unit: :math: `\mu_B per unit cell`
    """
    fac_morb = -eV_au / bohr**2
    return (
        FermiOcean(
            frml_basic.tildeHGc(data_K, sign=+1, **kwargs_formula),
            data_K,
            Efermi,
            tetra,
            fder=0,
            degen_thresh=degen_thresh,
            degen_Kramers=degen_Kramers)() - 2 * FermiOcean(
                frml_basic.tildeFc(data_K, **kwargs_formula),
                data_K,
                Efermi,
                tetra,
                fder=0,
                degen_thresh=degen_thresh,
                degen_Kramers=degen_Kramers)().mul_array(Efermi)) * (data_K.cell_volume * fac_morb)

####################
#  cunductivities  #
####################

#E0 B1
def gme_orb_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""
    :math: `K^{orb}_{\alpha\mu} = e \tau \int [dk] M_\mu v_\alpha f'`
    Unit: A/m^2/T 
    :math: `j_\alpha = K_{\alpha\mu} B_\mu 
    """
    formula_1 = FormulaProduct(
        [frml.Morb_Hpm(data_K, sign=+1, **kwargs_formula),
         data_K.covariant('Ham', commader=1)], name='morb_Hpm-vel')
    formula_2 = FormulaProduct(
        [frml.Omega(data_K, **kwargs_formula),
         data_K.covariant('Ham', commader=1)], name='berry-vel')
    res = FermiOcean(formula_1, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    res += -2 * FermiOcean(
        formula_2, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)().mul_array(Efermi)
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * fac_morb_Z * factor_t0_0_1
    return res


def gme_orb(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""
    :math: `K^{orb}_{\alpha\mu} = -e \tau \int [dk] \partial_\alpha M_\mu f`
    Unit: A/m^2/T 
    :math: `j_\alpha = K_{\alpha\mu} B_\mu 
    """
    Hp = Hplus_der(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    D = berry_dipole(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    tensor_K = factor_t0_0_1 * fac_morb_Z * (Hp - 2 * Efermi[:, None, None] * D)
    return result.EnergyResult(Efermi, tensor_K, TRodd=False, Iodd=True)


def gme_orb_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""
    :math: `K^{orb}_{\alpha\mu} = -e \tau \int [dk] \partial_\alpha M_\mu f`
    Unit: A/m^2/T 
    :math: `j_\alpha = K_{\alpha\mu} B_\mu 
    """
    Hp = Hplus_der_test(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    D = berry_dipole_test(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    tensor_K = factor_t0_0_1 * fac_morb_Z * (Hp - 2 * Efermi[:, None, None] * D)
    return result.EnergyResult(Efermi, tensor_K, TRodd=False, Iodd=True)


def gme_spin_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""
    :math: `K^{spin}_{\alpha,beta} = e \tau \int [dk] S_\beta v_\alpha f'`
    Unit: A/m^2/T 
    :math: `j_\alpha = K_{\alpha\mu} B_\mu 
    """
    formula = FormulaProduct([frml.Spin(data_K), data_K.covariant('Ham', commader=1)], name='spin-vel')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * factor_t0_0_1 * bohr_magneton
    return res


def gme_spin(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""
    :math: `K^{spin}_{\alpha,beta} = -e \tau \int [dk] \partial_\alpha S_\beta f`
    Unit: A/m^2/T 
    :math: `j_\alpha = K_{\alpha\mu} B_\mu 
    """
    formula = FormulaProduct([frml.DerSpin(data_K)], name='derspin')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * factor_t0_0_1 * bohr_magneton
    return res


# E1 B0
def AHC(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `O = - e^2/\hbar \int [dk] \Omega f`
    Unit: s^3 * A^2 / (kg * m^3) = S/m 
    :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta = \epsilon_{\alpha\beta\delta} O_delta E_\beta`
    """
    return FermiOcean(
        frml.Omega(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factor_t0_1_0


def AHC_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `O = - e^2/\hbar \int [dk] \Omega f`
    Unit: s^3 * A^2 / (kg * m^3) = S/m 
    :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta = \epsilon_{\alpha\beta\delta} O_delta E_\beta`
    """
    return FermiOcean(
        frml_basic.tildeFc(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factor_t0_1_0


def ohmic_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `\sigma_{\alpha\beta} = -e^2/\hbar \tau \int [dk] v_\alpha v_\beta f'`
    Unit: s^3 * A^2 / (kg * m^3) = S/m 
    :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta`
    """
    velocity = data_K.covariant('Ham', commader=1)
    formula = FormulaProduct([velocity, velocity], name='vel-vel')
    return FermiOcean(
        formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)() * factor_t1_1_0


def ohmic(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `\sigma_{\alpha\beta} = e^2/\hbar \tau \int [dk] v_{\alpha\beta} f`
    Unit: s^3 * A^2 / (kg * m^3) = S/m 
    :math: `j_\alpha = \sigma_{\alpha\beta} E_\beta`
    """
    r""" sigma10tau1"""
    formula = frml.InvMass(data_K)
    return FermiOcean(
        formula, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)() * factor_ohmic


# E2 B0
def berry_dipole_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `D_{\beta\delta} = -e^3/\hbar^2 \tau \int [dk] \Omega_\delta v_\beta f'`
    Unit: S^2/A
    :math: `j_\alpha = \epsilon_{\alpha\gamma\delta} D_{\beta\delta} E_\beta E_\gamma`
    """
    formula = FormulaProduct(
        [frml.Omega(data_K, **kwargs_formula),
         data_K.covariant('Ham', commader=1)], name='berry-vel')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * factor_t1_2_0
    return res


def berry_dipole(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `D_{\beta\delta} = e^3/\hbar^2 \tau \int [dk] \partial_beta \Omega_\delta f`
    Unit: S^2/A
    :math: `j_\alpha = \epsilon_{\alpha\gamma\delta} D_{\beta\delta} E_\beta E_\gamma`
    """
    res = FermiOcean(
        frml.DerOmega(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * factor_t1_2_0
    return res


def berry_dipole_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `D_{\beta\delta} = e^3/\hbar^2 \tau \int [dk] \partial_beta \Omega_\delta f`
    Unit: S^2/A
    :math: `j_\alpha = \epsilon_{\alpha\gamma\delta} D_{\beta\delta} E_\beta E_\gamma`
    """
    res = FermiOcean(
        frml_basic.tildeFc_d(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * factor_t1_2_0
    return res


def Der3E(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `\sigma_{\alpha\sigma\rho} = -e^3/\hbar^2 \tau^2 \int [dk] v_{\alpha\sigma\rho} f`
    Unit: S^2/A
    :math: `j_\alpha = \sigma_{\alpha\sigma\rho} E_\sigma E_\rho`
    """
    res = FermiOcean(
        frml.Der3E(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    return res * factor_t2_2_0


def Der3E_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `\sigma_{\alpha\sigma\rho} = e^3/\hbar^2 \tau^2 \int [dk] v_{\alpha\rho} v_\sigma f'`
    Unit: S^2/A
    :math: `j_\alpha = \sigma_{\alpha\sigma\rho} E_\sigma E_\rho`
    """
    formula = FormulaProduct([frml.InvMass(data_K), data_K.covariant('Ham', commader=1)], name='mass-vel')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    return res * factor_t2_2_0


def Der3E_fder2(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `\sigma_{\alpha\sigma\rho} = -e^3/\hbar^2 \tau^2 \int [dk] v_\alpha v_\rho v_\sigma f''/2`
    Unit: S^2/A
    :math: `j_\alpha = \sigma_{\alpha\sigma\rho} E_\sigma E_\rho`
    """
    formula = FormulaProduct(
        [data_K.covariant('Ham', commader=1),
         data_K.covariant('Ham', commader=1),
         data_K.covariant('Ham', commader=1)],
        name='vel-vel-vel')
    res = FermiOcean(
        formula, data_K, Efermi, tetra, fder=2, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)() * 0.5
    return res * factor_t2_2_0


# E1 B1
def Hall_classic_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `\sigma_{\alpha\sigma :\mu} = e^3/\hbar^2 \tau^2 \sigma_{\beta\mu\rho} \int [dk] v_\alpha v_\beta v_{\rho\sigma} f'`
    Unit: S/m/T
    :math: `j_\alpha = \sigma_{\alpha\sigma :\mu} E_\sigma B_\mu`
    """
    formula = FormulaProduct(
        [data_K.covariant('Ham', commader=1),
         frml.InvMass(data_K),
         data_K.covariant('Ham', commader=1)],
        name='vel-mass-vel')
    res = FermiOcean(
        formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factor_Hall_classic
    res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
    res.data = factor_t2_1_1 * 0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
    res.rank -= 2
    return res


def Hall_classic(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    :math: `\sigma_{\alpha\sigma :\mu} = -e^3/\hbar^2 \tau^2 \sigma_{\beta\mu\rho} \int [dk] v_{\alpha\beta} v_{\rho\sigma} f`
    Unit: S/m/T
    :math: `j_\alpha = \sigma_{\alpha\sigma :\mu} E_\sigma B_\mu`
    """
    formula1 = FormulaProduct([frml.InvMass(data_K), frml.InvMass(data_K)], name='mass-mass')
    #formula2 = FormulaProduct([data_K.covariant('Ham', commader=1), frml.Der3E(data_K)], name='vel-Der3E')
    res = FermiOcean(
        formula1, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factor_Hall_classic
    #term2 = FermiOcean(
    #    formula2, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh,
    #    degen_Kramers=degen_Kramers)() * factor_Hall_classic
    res.data = res.data.transpose(0, 4, 1, 2, 3)# + term2.data.transpose(0, 4, 2, 3, 1)
    res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
    res.data = factor_t2_1_1 * 0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
    res.rank -= 2
    return res


def Hall_morb_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    # first, transform to SI, not forgettint e/2hbar multilier for morb - now in A*m/J,
    # restoring the sign of spin magnetic moment
    factor = -Ang_SI * elementary_charge / (2 * hbar)
    factor *= elementary_charge**2 / hbar  # multiply by a dimensional factor - now in S/(T*m)
    factor *= -1
    #factor *= 1e-2  #  finally transform to S/(T*cm)
    formula_1 = FormulaProduct(
        [frml.Omega(data_K, **kwargs_formula),
         frml.Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='berry-morb_Hpm')
    formula_2 = FormulaProduct(
        [frml.Omega(data_K, **kwargs_formula),
         frml.Omega(data_K, **kwargs_formula)], name='berry-berry')
    res = FermiOcean(formula_1, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    res += -2 * FermiOcean(
        formula_2, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)().mul_array(Efermi)
    return res * factor


def Hall_spin_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    # first, transform to SI - now in 1/(m*T) ,restoring the sign of spin magnetic moment
    factor = -bohr_magneton / (elementary_charge * Ang_SI)
    factor *= -1
    factor *= elementary_charge**2 / hbar  # multiply by a dimensional factor - now in S/(T*m)
    #factor *= 1e-2  #  finally transform to S/(T*cm)
    formula = FormulaProduct([frml.Omega(data_K, **kwargs_formula), frml.Spin(data_K)], name='berry-spin')
    return FermiOcean(
        formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)() * factor


def spin_hall(data_K, Efermi, spin_current_type, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.SpinOmega(data_K, spin_current_type, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factor_t0_1_0 * -0.5


def spin_hall_qiao(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return spin_hall(data_K, Efermi, "qiao", tetra=tetra, **kwargs_formula)


def spin_hall_ryoo(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return spin_hall(data_K, Efermi, "ryoo", tetra=tetra, **kwargs_formula)


###############################
# The private part goes here  #
###############################


class FermiOcean():
    """ formula should have a trace(ik,inn,out) method
    fder derivative of fermi distribution . 0: fermi-sea, 1: fermi-surface 2: f''
    """

    def __init__(self, formula, data_K, Efermi, tetra, fder, degen_thresh=1e-4, degen_Kramers=False):

        ndim = formula.ndim
        self.Efermi = Efermi
        self.fder = fder
        self.tetra = tetra
        self.nk = data_K.nk
        self.NB = data_K.num_wann
        self.formula = formula
        self.final_factor = 1. / (data_K.nk * data_K.cell_volume)

        # get a list [{(ib1,ib2):W} for ik in op:ed]
        if self.tetra:
            self.weights = data_K.tetraWeights.weights_all_band_groups(
                Efermi, der=self.fder, degen_thresh=degen_thresh,
                degen_Kramers=degen_Kramers)  # here W is array of shape Efermi
        else:
            self.extraEf = 0 if fder == 0 else 1 if fder in (1, 2) else 2 if fder == 3 else None
            self.dEF = Efermi[1] - Efermi[0]
            self.EFmin = Efermi[0] - self.extraEf * self.dEF
            self.EFmax = Efermi[-1] + self.extraEf * self.dEF
            self.nEF_extra = Efermi.shape[0] + 2 * self.extraEf
            self.weights = data_K.get_bands_in_range_groups(
                self.EFmin, self.EFmax, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers,
                sea=(self.fder == 0))  # here W is energy
        self.__evaluate_traces(formula, self.weights, ndim)

    def __evaluate_traces(self, formula, bands, ndim):
        """formula  - TraceFormula to evaluate
           bands = a list of lists of k-points for every
        """
        self.shape = (3, ) * ndim
        lambdadic = lambda: np.zeros(((3, ) * ndim), dtype=float)
        self.values = [defaultdict(lambdadic) for ik in range(self.nk)]
        for ik, bnd in enumerate(bands):
            if formula.additive:
                for n in bnd:
                    inn = np.arange(n[0], n[1])
                    out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], self.NB)))
                    self.values[ik][n] = formula.trace(ik, inn, out)
            else:
                nnall = set([_ for n in bnd for _ in n])
                _values = {}
                for n in nnall:
                    inn = np.arange(0, n)
                    out = np.arange(n, self.NB)
                    _values[n] = formula.trace(ik, inn, out)
                for n in bnd:
                    self.values[ik][n] = _values[n[1]] - _values[n[0]]

    def __call__(self):
        if self.tetra:
            res = self.__call_tetra()
        else:
            res = self.__call_notetra()
        res *= self.final_factor
        return result.EnergyResult(self.Efermi, res, TRodd=self.formula.TRodd, Iodd=self.formula.Iodd)

    def __call_tetra(self):
        restot = np.zeros(self.Efermi.shape + self.shape)
        for ik, weights in enumerate(self.weights):
            values = self.values[ik]
            for n, w in weights.items():
                restot += np.einsum("e,...->e...", w, values[n])
        return restot

    def __call_notetra(self):
        restot = np.zeros((self.nEF_extra, ) + self.shape)
        for ik, weights in enumerate(self.weights):
            values = self.values[ik]
            for n, E in sorted(weights.items()):
                if E < self.EFmin:
                    restot += values[n][None]
                elif E <= self.EFmax:
                    iEf = ceil((E - self.EFmin) / self.dEF)
                    restot[iEf:] += values[n]
        if self.fder == 0:
            return restot
        if self.fder == 1:
            return (restot[2:] - restot[:-2]) / (2 * self.dEF)
        elif self.fder == 2:
            return (restot[2:] + restot[:-2] - 2 * restot[1:-1]) / (self.dEF**2)
        elif self.fder == 3:
            return (restot[4:] - restot[:-4] - 2 * (restot[3:-1] - restot[1:-3])) / (2 * self.dEF**3)
        else:
            raise NotImplementedError(f"Derivatives  d^{self.fder}f/dE^{self.fder} is not implemented")
