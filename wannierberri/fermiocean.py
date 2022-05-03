import numpy as np
from .__utility import  alpha_A,beta_A, TAU_UNIT,delta_f,Levi_Civita
from collections import defaultdict
from . import __result as result
from math import ceil
from . import covariant_formulak as frml
from .formula import FormulaProduct
from . import covariant_formulak_basic as frml_basic
from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom #, Boltzmann

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
fac_morb_Z = -elementary_charge/2/hbar*Ang_SI**2 # change unit of m.B to (eV).
#fac_ahc = -1e10 * elementary_charge ** 2 / hbar
#fac_spin_hall = fac_ahc * -0.5
#factor_ohmic=(elementary_charge/Ang_SI/hbar**2  # first, transform to SI, not forgeting hbar in velocities - now in  1/(kg*m^3)
#                 *elementary_charge**2*TAU_UNIT  # multiply by a dimensional factor - now in A^2*s^2/(kg*m^3*tau_unit) = S/(m*tau_unit)
#                ) # now in  S/(cm*tau_unit)
#factor_Hall_classic=elementary_charge**2*Ang_SI/hbar**3  # first, transform to SI, not forgeting hbar in velocities - now in  m/(J*s^3)
#factor_Hall_classic*=elementary_charge**3/hbar*TAU_UNIT**2  # multiply by a dimensional factor - now in A^3*s^5*cm/(J^2*tau_unit^2) = S/(T*m*tau_unit^2)

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

def dos(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.Identity(), data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * data_K.cell_volume


def cumdos(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.Identity(), data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * data_K.cell_volume


def spin(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.Spin(data_K), data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()


def Morb(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    fac_morb = -eV_au / bohr**2
    return (
        FermiOcean(
            frml.Morb_Hpm(
                data_K, sign=+1, **kwargs_formula),data_K,Efermi,tetra,fder=0,
                degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)() 
        - 2 * FermiOcean(
            frml.Omega(
                data_K, **kwargs_formula),data_K,Efermi,tetra,fder=0,
                degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
        ) * (data_K.cell_volume * fac_morb)


def Morb_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    fac_morb = -eV_au / bohr**2
    return (
        FermiOcean(
            frml_basic.tildeHGc(
                data_K, sign=+1, **kwargs_formula),data_K,Efermi,tetra,fder=0,
                degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)() 
        - 2 * FermiOcean(
            frml_basic.tildeFc(
                data_K, **kwargs_formula),data_K,Efermi,tetra,fder=0,
                degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
        ) * (data_K.cell_volume * fac_morb)


def berry_dipole_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    formula = FormulaProduct(
        [frml.Omega(data_K, **kwargs_formula),
         data_K.covariant('Ham', commader=1)], name='berry-vel')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def berry_dipole(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    res = FermiOcean(
        frml.DerOmega(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def berry_dipole_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    res = FermiOcean(
        frml_basic.tildeFc_d(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def Hplus_der(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
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


def gme_orb_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
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
    res.data = np.swapaxes(res.data, 1, 2) * -elementary_charge**2 / (2 * hbar)
    return res


def gme_orb(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    Hp = Hplus_der(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    D = berry_dipole(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    tensor_K = -elementary_charge**2 / (2 * hbar) * (Hp - 2 * Efermi[:, None, None] * D)
    return result.EnergyResult(Efermi, tensor_K, TRodd=False, Iodd=True)


def gme_orb_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    Hp = Hplus_der_test(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    D = berry_dipole_test(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    tensor_K = -elementary_charge**2 / (2 * hbar) * (Hp - 2 * Efermi[:, None, None] * D)
    return result.EnergyResult(Efermi, tensor_K, TRodd=False, Iodd=True)


def gme_spin_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    formula = FormulaProduct([frml.Spin(data_K), data_K.covariant('Ham', commader=1)], name='spin-vel')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * -bohr_magneton / Ang_SI**2
    return res


####################
#  cunductivities  #
####################

def Hall_morb_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    # first, transform to SI, not forgettint e/2hbar multilier for morb - now in A*m/J,
    # restoring the sign of spin magnetic moment
    factor = -Ang_SI * elementary_charge / (2 * hbar)
    factor *= elementary_charge**2 / hbar  # multiply by a dimensional factor - now in S/(T*m)
    factor *= -1
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
    formula = FormulaProduct([frml.Omega(data_K, **kwargs_formula), frml.Spin(data_K)], name='berry-spin')
    return FermiOcean(
        formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)() * factor


def spin_hall(data_K,Efermi,spin_current_type,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    return FermiOcean(
            frml.SpinOmega(data_K,spin_current_type,**kwargs_formula),data_K,Efermi,tetra,fder=0,
            degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)() * factor_t0_1_0 * -0.5


def spin_hall_qiao(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    return spin_hall(data_K,Efermi,"qiao",tetra=tetra,**kwargs_formula)


def spin_hall_ryoo(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    return spin_hall(data_K,Efermi,"ryoo",tetra=tetra,**kwargs_formula)


#Berry-boltzman terms comes below:

def AHC(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" 
    Anoumalous Hall conductivity with fermi sea integral. (s^3 * A^2 / (kg * m^3) = S/m )
    :math: `\sigma^{\tau^0;1,0}_{\alpha\beta} = -\int [dk] \Omega f_0`
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
    Anoumalous Hall conductivity with fermi sea integral. (s^3 * A^2 / (kg * m^3) = S/m)
    :math: `\sigma^{\tau^0;1,0}_{\alpha\beta} = -\int [dk] \Omega f_0`
    """
    res = FermiOcean(
        frml_basic.tildeFc(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    return res * factor_t0_1_0


def ohmic_fsurf(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""
    Ohmic conductivity with fermi surface integral. (s^3 * A^2 / (kg * m^3) = S/m) 
    :math: `\sigma^{\tau^1;1,0}_{\alpha\beta} = -\tau \int [dk] v_{\alpha} v_{\beta} f_0'` 
    """
    velocity =  data_K.covariant('Ham',commader=1)
    formula  = FormulaProduct ( [velocity,velocity], name='vel-vel')
    return FermiOcean(formula,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()*factor_t1_1_0


def ohmic(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" 
    Ohmic conductivity with fermi sea integral. (s^3 * A^2 / (kg * m^3) = S/m) 
    :math: `\sigma^{\tau^1;1,0}_{\alpha\beta} = -\tau \int [dk] v_{\alpha\beta} f_0` 
    """
    formula =  frml.InvMass(data_K)
    return FermiOcean(
            frml.InvMass(data_K),data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,
            degen_Kramers=degen_Kramers)()*factor_t1_1_0


def linear_magnetoresistance_fsurf(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma11tau1 fermi surface"""
    velocity =  data_K.covariant('Ham',commader=1)
    formula  = FormulaProduct ( [frml.VelOmega(data_K,**kwargs_formula),velocity], name='velberry-vel (aup) ([pu]abb) ([au]pbb)')
    res = FermiOcean(formula,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    term2 = np.einsum('pu,nabb->naup',delta_f,res.data)
    term3 = np.einsum('au,npbb->naup',delta_f,res.data)
    res.data = -res.data + term2 + term3 
    res.data = res.data.transpose(0,1,3,2)*factor_t1_1_1
    return res


def linear_magnetoresistance(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma11tau1 fermi sea (abu) """
    velocity =  data_K.covariant('Ham',commader=1)
    formula1  = FormulaProduct ( [velocity,frml.DerOmega(data_K,**kwargs_formula)], name='vel-derberry (aup)')
    formula2  = FormulaProduct ( [frml.InvMass(data_K),frml.Omega(data_K,**kwargs_formula)], name='mass-berry (apu)([pu]abb)([au]pbb)')
    #formula2  = FormulaProduct ( [frml.delta_f(),frml.InvMass(data_K),frml.Omega(data_K,**kwargs_formula)], name='delta-mass-berry (puabb,aupbb)')
    res = FermiOcean(formula1,data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res2 = FermiOcean(formula2,data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    term1 = res.data + res2.data.transpose(0,1,3,2)
    term2 = np.einsum('pu,nabb->naup',delta_f,res2.data)
    term3 = np.einsum('au,npbb->naup',delta_f,res2.data)
    res.data = -term1 + term2 + term3
    res.data = res.data.transpose(0,1,3,2)*factor_t1_1_1
    return res


def quadra_magnetoresistance_fsurf(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma12au1 fermi surf (apuv) """
    velocity =  data_K.covariant('Ham',commader=1)
    formula  = FormulaProduct ( [velocity,velocity,frml.OmegaOmega(data_K,**kwargs_formula)],
            name='vel-vel-berryberry (apuv)([pv]abub)([au]pbvb)([au][pv]bcbc) ')
    res = FermiOcean(formula,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    term2 = np.einsum('pv,nabub->napuv',delta_f,res.data)
    term3 = np.einsum('au,pv,nbcbc->napuv',delta_f,delta_f,res.data)
    term4 = np.einsum('au,npbvb->napuv',delta_f,res.data)
    res.data += - term2 + term3 - term4
    res.data *= factor_t1_1_2
    return res


def quadra_magnetoresistance(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma12au1 fermi sea (apuv) """
    velocity =  data_K.covariant('Ham',commader=1)
    formula1  = FormulaProduct ( [frml.InvMass(data_K),frml.OmegaOmega(data_K,**kwargs_formula)],
            name='mass-berryberry (apuv) (ab[pv]ub) (pb[au]vb) ([au][pv]bcbc)')
    formula2  = FormulaProduct ( [velocity,frml.DerOmega(data_K,**kwargs_formula),frml.Omega(data_K,**kwargs_formula)],
            name='vel-derberry-berry (aupv) (avpu) (a[pv]ubb) (p[au]vbb) ([au][pv]bbcc)')
    res = FermiOcean(formula1,data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res2 = FermiOcean(formula2,data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    term1 = res.data + res2.data.transpose(0,1,3,2,4)+res2.data.transpose(0,1,3,4,2)
    term2 = np.einsum('pv,nabub->napuv',delta_f,res.data) + np.einsum('pv,naubb->napuv',delta_f,res2.data)
    term3 = np.einsum('au,pv,nbcbc->napuv',delta_f,delta_f,res.data) + np.einsum('au,pv,nbbcc->napuv',delta_f,delta_f,res2.data)
    term4 = np.einsum('au,npbvb->napuv',delta_f,res.data) + np.einsum('au,npvbb->napuv',delta_f,res2.data)
    res.data = term1 - term2  + term3 - term4
    res.data *= factor_t1_1_2
    return res


def NLAHC_fsurf(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma20tau1 fermi surface"""
    res =  berry_dipole_fsurf(data_K,Efermi,tetra=tetra,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers,**kwargs_formula)
    res.data = np.einsum('sda,ndp->naps',Levi_Civita,res.data) * factor_t1_2_0
    res.rank +=1
    return res


def NLAHC(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma20tau1 fermi sea"""
    res =  berry_dipole(data_K,Efermi,tetra=tetra,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers,**kwargs_formula)
    res.data = np.einsum('sda,ndp->naps',Levi_Civita,res.data) * factor_t1_2_0 
    res.rank +=1
    return res


def Nonlinear_Hall_fsurf(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma21tau1 fermi surface (dpu)"""
    velocity =  data_K.covariant('Ham',commader=1)
    formula  = FormulaProduct ( [frml.OmegaOmega(data_K,**kwargs_formula),velocity],
            name='berryberry-vel ([pu]dbb) (dup)')
    res =  FermiOcean(formula,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()*factor_t1_2_1
    res.data = np.einsum('pu,ndbb->ndpu',delta_f,res.data) - res.data.transpose(0,1,3,2)#np.einsum('pb,ndub->ndpu',delta_f,res.data) 
    res.data = np.einsum('sda,ndpu->napsu',Levi_Civita,res.data) 
    res.rank +=1
    return res


def Nonlinear_Hall(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma21tau1 fermi sea (dpu)"""
    velocity =  data_K.covariant('Ham',commader=1)
    formula = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula),frml.DerOmega(data_K,**kwargs_formula)],
            name='berry-der_berry ([pu]bdb) (udp) (dup)')
    res =  FermiOcean(formula,data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    term1 = np.einsum('pu,nbdb->ndpu',delta_f,res.data)
    #term2 = np.einsum('pb,ndbu->ndpu',delta_f,res.data) + np.einsum('pb,ndub->ndpu',delta_f,res.data)
    term2 = res.data.transpose(0,2,3,1) + res.data.transpose(0,1,3,2)
    res.data = term1 - term2
    res.data = np.einsum('sda,ndpu->napsu',Levi_Civita,res.data)*factor_t1_2_1 
    res.rank +=1
    return res


def Hall_classic_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""sigma11tau2 """
    formula = FormulaProduct(
        [data_K.covariant('Ham', commader=1),
         frml.InvMass(data_K),
         data_K.covariant('Ham', commader=1)],
        name='vel-mass-vel')
    res = FermiOcean(
        formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factor_Hall_classic
    res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
    res.data = -0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
    res.rank -= 2
    return res


def Hall_classic(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""sigma11tau2 """
    formula1 = FormulaProduct([frml.InvMass(data_K), frml.InvMass(data_K)], name='mass-mass')
    formula2 = FormulaProduct([data_K.covariant('Ham', commader=1), frml.Der3E(data_K)], name='vel-Der3E')
    res = FermiOcean(
        formula1, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factor_Hall_classic
    term2 = FermiOcean(
        formula2, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factor_Hall_classic
    res.data = res.data.transpose(0, 4, 1, 2, 3) + term2.data.transpose(0, 4, 2, 3, 1)
    res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
    res.data = -0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
    res.rank -= 2
    return res


def Drude(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma20tau2 fermi sea """
    res =  FermiOcean(frml.Der3E(data_K,**kwargs_formula),data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()*factor_t2_2_0
    return res


def Drude_fsurf(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma20tau2 fermi surface """
    formula  = FormulaProduct ([frml.InvMass(data_K),data_K.covariant('Ham',commader=1)], name='mass-vel')
    res =  FermiOcean(formula,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()*factor_t2_2_0
    return res


def Drude_fder2(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma20tau2 fder=2 """
    formula  = FormulaProduct ( [data_K.covariant('Ham',commader=1),data_K.covariant('Ham',commader=1),data_K.covariant('Ham',commader=1)], name='vel-vel-vel')
    res =  FermiOcean(formula,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()*0.5*factor_t2_2_0
    return res

  
def eMChA_fder2(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma21tau2 fder2 (daups)"""
    velocity =  data_K.covariant('Ham',commader=1)
    formula1  = FormulaProduct ( [velocity,frml.Omega(data_K,**kwargs_formula),velocity,velocity],
            name='vel-berry-vel-vel (aups) ([au]bbps) ([us]abbp)')
    formula2 = FormulaProduct ( [velocity,frml.Omega(data_K,**kwargs_formula),frml.InvMass(data_K)],
            name='vel-berry-mass (aups) ([au]bbps) ([us]abbp)')
    res =  FermiOcean(formula1,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res2 =  FermiOcean(formula2,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    tmp = (res.data - res2.data)
    term1 = 2*tmp
    term2 = 2*np.einsum('us,nabbp->naups',delta_f,tmp)
    term3 = np.einsum('au,nbbps->naups',delta_f,tmp)
    formula3  = FormulaProduct ( [velocity,frml.DerOmega(data_K,**kwargs_formula),velocity],
            name='vel-derberry-vel (aups) ([us]abpb)')
    res3 =  FermiOcean(formula3,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    term4 = np.einsum('us,nabpb->naups',delta_f,res3.data)
    term5 = res3.data
    
    # cross product term [xub][pta]xtbs
    termcross = np.einsum('xub,pta,nxtbs->naups',Levi_Civita,Levi_Civita,tmp) 
    res.data = -(-term1 + term2 + term3 + term4 - term5 + termcross)*factor_t2_2_1
    return res

def eMChA_fsurf(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma21tau2 fermi surface (daups)"""
    velocity =  data_K.covariant('Ham',commader=1)
    formula1  = FormulaProduct ( [frml.InvMass(data_K),frml.Omega(data_K,**kwargs_formula),velocity],
            name='mass-berry-vel (apus)(psua) ([au]bpbs) ([us]abbp)')
    formula2  = FormulaProduct ( [velocity,frml.DerOmega(data_K,**kwargs_formula),velocity],
            name='v-derberry-vel (aups) ([au]bbps) ([us]abpb)')
    res =  FermiOcean(formula1,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res2 =  FermiOcean(formula2,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    tmp = res.data.transpose(0,1,3,2,4) + res2.data 
    term1 = 2*tmp
    term2 = 2*np.einsum('us,nabbp->naups',delta_f,res.data)
    term3 = np.einsum('au,nbbps->naups',delta_f,tmp)
    term4 = np.einsum('us,nabpb->naups',delta_f,res2.data)
    term5 = res2.data
    
    # cross product term [xub][pta]xtbs
    termcross = np.einsum('xub,pta,nxtbs->naups',Levi_Civita,Levi_Civita,tmp) 
    res.data = -(-term1 + term2 + term3 + term4 - term5 + termcross)*factor_t2_2_1
    return res


def eMChA(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma21tau2 fermi sea (daups)"""
    velocity =  data_K.covariant('Ham',commader=1)
    formula1  = FormulaProduct ( [frml.Der3E(data_K),frml.Omega(data_K,**kwargs_formula)],
            name='Der3E-berry (apsu) ([au]bpsb)')
    formula2  = FormulaProduct ( [frml.InvMass(data_K),frml.DerOmega(data_K,**kwargs_formula)],
            name='mass-derberry (apus)(asup) ([au]bpbs)')
    formula3  = FormulaProduct ( [velocity,frml.Der2Omega(data_K,**kwargs_formula)],
            name='vel-der2berry-vel (aups) ([au]bbps)')
    res =  FermiOcean(formula1,data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res2 =  FermiOcean(formula2,data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res3 =  FermiOcean(formula3,data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    tmp = (res.data.transpose(0,1,4,2,3)+res2.data.transpose(0,1,3,2,4)+res2.data.transpose(0,1,3,4,2) +res3.data) 
    term1 = 2*tmp
    term2 = 2*np.einsum('us,nabbp->naups',delta_f,res2.data + res.data.transpose(0,1,2,4,3))
    term3 = np.einsum('au,nbbps->naups',delta_f,tmp)
    term4 = np.einsum('us,nabbp->naups',delta_f,res2.data)
    term5 = res2.data.transpose(0,1,3,4,2) + res3.data 
    
    # cross product term [xub][pta]xtbs
    termcross = np.einsum('xub,pta,nxtbs->naups',Levi_Civita,Levi_Civita,tmp) 
    res.data = -(-term1 + term2 + term3 + term4 - term5 + termcross)*factor_t2_2_1
    return res

#####################
### Zeeman terms  ###
#####################

#TODO How to use one matrix for Morb. Not morb_Hpm and berry*mul_array(Efermi).(Efemri only work after FermiOcean.)
# And think how calculate less terms.
def ahc_Z(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma10Ztau0 fermi sea"""
    formula_1  = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula),frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula)], name='berry-morb_Hpm')
    formula_2  = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula),frml.Omega(data_K,**kwargs_formula)], name='berry-berry')
    res =  FermiOcean(formula_1,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res -= 2* FermiOcean(formula_2,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res.data = res.data*fac_morb_Z*factor_t0_1_0
    return res


def ahc_Z2(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma10ZZtau0 fermi sea"""
    formula_1  = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula),frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula),frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula)],
            name='berry-morb_Hpm-morb_Hpm')
    formula_2  = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula),frml.Omega(data_K,**kwargs_formula),frml.Omega(data_K,**kwargs_formula)],
            name='berry-berry-berry')
    formula_3  = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula),frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula),frml.Omega(data_K,**kwargs_formula)],
            name='berry-morb_Hpm-berry')
    res1 =  FermiOcean(formula_1,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res2 =  FermiOcean(formula_2,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi).mul_array(Efermi)
    res3 =  FermiOcean(formula_3,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res1.data = 0.5*(res1.data + 4*res2.data - 2*res3.data -2*res3.data.transpose(0,1,3,2))*fac_morb_Z*fac_morb_Z*factor_t0_1_0
    return res1


def ohmic_Z(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma10Ztau1 fermi sea+surface"""
    formula_1  = FormulaProduct ( [frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula) ,frml.InvMass(data_K)], name='morb_Hpm-mass')
    formula_2  = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula) ,frml.InvMass(data_K)], name='berry-mass')
    res =  FermiOcean(formula_1,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res -= 2* FermiOcean(formula_2,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res +=  FermiOcean(frml.Der2Morb(data_K),data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res -= 2* FermiOcean(frml.Der2Omega(data_K),data_K,Efermi,tetra,fder=0,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res.data = res.data.transpose(0,2,3,1)*fac_morb_Z*factor_t1_1_0
    return res


def ohmic_Z_V2(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma10Ztau1 fermi surface"""
    formula_1  = FormulaProduct ( [frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula) ,frml.InvMass(data_K)], name='morb_Hpm-mass')
    formula_2  = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula) ,frml.InvMass(data_K)], name='berry-mass')
    res =  FermiOcean(formula_1,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res -= 2* FermiOcean(formula_2,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    formula_3  = FormulaProduct ( [frml.DerMorb(data_K,**kwargs_formula) ,data_K.covariant('Ham',commader=1)], name='der_morb_Hpm-vel')
    formula_4  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula) ,data_K.covariant('Ham',commader=1)], name='der_berry-vel')
    res -=  FermiOcean(formula_3,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res += 2* FermiOcean(formula_4,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res.data = res.data.transpose(0,2,3,1)*fac_morb_Z*factor_t1_1_0
    return res


def ohmic_Z2(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma10ZZtau1 fermi """
    formula_1  = FormulaProduct ( [frml.HplusHplus(data_K,sign=+1,**kwargs_formula) ,frml.InvMass(data_K)], name='HplusHplus-mass')
    formula_2  = FormulaProduct ( [frml.OmegaOmega(data_K,**kwargs_formula) ,frml.InvMass(data_K)], name='berryberry-mass')
    formula_3  = FormulaProduct ( [frml.OmegaHplus(data_K,sign=+1,**kwargs_formula) ,frml.InvMass(data_K)], name='berryHplus-mass')
    res1 =  FermiOcean(formula_1,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res2 =  FermiOcean(formula_2,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi).mul_array(Efermi)
    res3 =  FermiOcean(formula_3,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res1.data = 0.5*(res1.data + 4*res2.data - 2*res3.data -2*res3.data.transpose(0,2,1,3,4)).transpose(0,3,4,1,2)
    
    formula_4  = FormulaProduct ( [frml.Der2Morb(data_K,**kwargs_formula) ,frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula)], name='Der2Hplus-Hplus')
    formula_5  = FormulaProduct ( [frml.Der2Omega(data_K,**kwargs_formula) ,frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula)], name='Der2Omega-Hplus')
    formula_6  = FormulaProduct ( [frml.Der2Morb(data_K,**kwargs_formula) ,frml.Omega(data_K,**kwargs_formula)], name='Der2Hplus-Omega')
    formula_7  = FormulaProduct ( [frml.Der2Omega(data_K,**kwargs_formula) ,frml.Omega(data_K,**kwargs_formula)], name='Der2Omega-Omega')
    res4 =  FermiOcean(formula_4,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res5 =  FermiOcean(formula_5,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res6 =  FermiOcean(formula_6,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res7 =  FermiOcean(formula_7,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi).mul_array(Efermi)
    res4.data = -(res4.data + 4*res7.data - 2*res5.data -2*res6.data).transpose(0,2,3,1,4)
    
    res1.data = (res1.data + res4.data)*fac_morb_Z*fac_morb_Z*factor_t1_1_0
    return res1


def ohmic_Z2_V2(data_K,Efermi,kpart=None,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma10ZZtau1 fermi """
    formula_1  = FormulaProduct ( [frml.HplusHplus(data_K,sign=+1,**kwargs_formula) ,frml.InvMass(data_K)], name='HplusHplus-mass')
    formula_2  = FormulaProduct ( [frml.OmegaOmega(data_K,**kwargs_formula) ,frml.InvMass(data_K)], name='berryberry-mass')
    formula_3  = FormulaProduct ( [frml.OmegaHplus(data_K,sign=+1,**kwargs_formula) ,frml.InvMass(data_K)], name='berryHplus-mass')
    res1 =  FermiOcean(formula_1,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res2 =  FermiOcean(formula_2,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi).mul_array(Efermi)
    res3 =  FermiOcean(formula_3,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res1.data = 0.5*(res1.data + 4*res2.data - 2*res3.data -2*res3.data.transpose(0,2,1,3,4)).transpose(0,3,4,1,2)
    
    formula_4  = FormulaProduct ( [frml.DerMorb(data_K,**kwargs_formula) ,frml.DerMorb(data_K,**kwargs_formula)], name='DerHplus-DerHplus')
    formula_5  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula) ,frml.DerMorb(data_K,**kwargs_formula)], name='DerOmega-DerHplus')
    formula_6  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula) ,frml.DerOmega(data_K,**kwargs_formula)], name='DerOmega-DerOmega')
    res4 =  FermiOcean(formula_4,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res5 =  FermiOcean(formula_5,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res6 =  FermiOcean(formula_6,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi).mul_array(Efermi)
    res4.data = (res4.data + 4*res6.data - 2*res5.data -2*res5.data.transpose(0,3,4,1,2)).transpose(0,2,4,1,3)
    
    formula_7  = FormulaProduct ( [frml.DerMorb(data_K,**kwargs_formula) ,frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula),data_K.covariant('Ham',commader=1)], name='DerHplus-Hplus-vel')
    formula_8  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula) ,frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula),data_K.covariant('Ham',commader=1)], name='DerOmega-Hplus-vel')
    formula_9  = FormulaProduct ( [frml.DerMorb(data_K,**kwargs_formula) ,frml.Omega(data_K,**kwargs_formula),data_K.covariant('Ham',commader=1)], name='DerHplus-Omega-vel')
    formula_10  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula) ,frml.Omega(data_K,**kwargs_formula),data_K.covariant('Ham',commader=1)], name='DerOmega-Omega-vel')
    res7 =  FermiOcean(formula_7,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res8 =  FermiOcean(formula_8,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res9 =  FermiOcean(formula_9,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res10 =  FermiOcean(formula_10,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi).mul_array(Efermi)
    res7.data = (res7.data + 4*res10.data - 2*res8.data -2*res9.data).transpose(0,2,4,1,3)
    res1.data = (res1.data + res4.data + res7.data)*fac_morb_Z*fac_morb_Z*factor_t1_1_0
    
    return res1



#def linear_magnetoresistance_Z(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
#    r""" sigma11Ztau1 fermi sea"""


def berry_dipole_Z(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma20Ztau1 fermi sea"""
    formula_1  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula), frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula)], name='der_berry_Hplus')
    formula_2  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula), frml.Omega(data_K,**kwargs_formula)], name='der_berry-berry')
    res =  FermiOcean(formula_1,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res -=  2*FermiOcean(formula_2,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res.data= np.swapaxes(res.data,1,2)*fac_morb_Z*factor_t1_2_0  # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    return res


def berry_dipole_Z2(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r""" sigma20Ztau1 fermi sea"""
    formula_1  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula), frml.HplusHplus(data_K,sign=+1,**kwargs_formula)], name='der_berry_HplusHplus')
    formula_2  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula), frml.OmegaHplus(data_K,sign=+1,**kwargs_formula)], name='der_berry-OmegaHplus')
    formula_3  = FormulaProduct ( [frml.DerOmega(data_K,**kwargs_formula), frml.OmegaOmega(data_K,**kwargs_formula)], name='der_berry-OmegaOmega')
    res =  FermiOcean(formula_1,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res += 4*FermiOcean(formula_3,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi).mul_array(Efermi)
    res1 = 2*FermiOcean(formula_2,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    res.data -= res1.data + res1.data.transpose(0,1,2,4,3)
    res.data= np.swapaxes(res.data,1,2)*fac_morb_Z*fac_morb_Z*factor_t1_2_0  # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    
    return res


def Nonlinear_Hall_Z(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma21Ztau1 fermi sea (dpuv)"""
    velocity =  data_K.covariant('Ham',commader=1)
    formula_1 = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula),frml.DerOmega(data_K,**kwargs_formula),frml.Omega(data_K,**kwargs_formula)],
            name='berry-der_berry-berry ([pu]bdbv) (udpv) (dupv)')
    formula_2 = FormulaProduct ( [frml.Omega(data_K,**kwargs_formula),frml.DerOmega(data_K,**kwargs_formula),frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula)],
            name='berry-der_berry-Hplus ([pu]bdbv) (udpv) (dupv)')
    res =  FermiOcean(formula_2,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res -= 2*FermiOcean(formula_1,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    term1 = np.einsum('pu,nbdbv->ndpuv',delta_f,res.data)
    #term2 = np.einsum('pb,ndbu->ndpu',delta_f,res.data) + np.einsum('pb,ndub->ndpu',delta_f,res.data)
    term2 = res.data.transpose(0,2,3,1,4) + res.data.transpose(0,1,3,2,4)
    res.data = term1 - term2
    res.data = np.einsum('sda,ndpuv->napsuv',Levi_Civita,res.data)*fac_morb_Z*factor_t1_2_1 
    res.rank +=1
    return res


def Drude_Z(data_K,Efermi,tetra=False,degen_thresh=1e-4,degen_Kramers=False,**kwargs_formula):
    r"""sigma20Ztau2 fermi surface """
    velocity = data_K.covariant('Ham',commader=1)
    formula_1  = FormulaProduct ([frml.Der3E(data_K,**kwargs_formula),frml.Morb_Hpm(data_K,sign=+1,**kwargs_formula) ], name='Der3E-Hplus')
    formula_2  = FormulaProduct ([frml.Der3E(data_K,**kwargs_formula),frml.Omega(data_K,**kwargs_formula) ], name='Der3E-Omega')
    res = -1*FermiOcean(formula_1,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res += 2*FermiOcean(formula_2,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    formula_3  = FormulaProduct ([frml.DerMorb(data_K,**kwargs_formula),frml.InvMass(data_K) ], name='DerHplus-mass')
    formula_4  = FormulaProduct ([frml.DerOmega(data_K,**kwargs_formula),frml.InvMass(data_K) ], name='DerOmega-mass')
    res1 = -1*FermiOcean(formula_3,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res1 += 2*FermiOcean(formula_4,data_K,Efermi,tetra,fder=1,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    formula_5  = FormulaProduct ([frml.DerMorb(data_K,**kwargs_formula),velocity,velocity], name='DerHplus-vel-vel')
    formula_6  = FormulaProduct ([frml.DerOmega(data_K,**kwargs_formula),velocity,velocity], name='DerOmega-vel-vel')
    res2 = FermiOcean(formula_5,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)()
    res2 -= 2*FermiOcean(formula_6,data_K,Efermi,tetra,fder=2,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)().mul_array(Efermi)
    
    res.data += res1.data.transpose(0,2,3,4,1) + res2.data.transpose(0,2,3,4,1)
    res.data *= fac_morb_Z*factor_t2_2_0
    return res



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
