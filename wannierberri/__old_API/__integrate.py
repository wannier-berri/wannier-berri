#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
#------------------------------------------------------------

import numpy as np
from collections import defaultdict
from copy import copy

from wannierberri.smoother import VoidSmoother
from wannierberri.__factors import TAU_UNIT
from wannierberri import result as result
from . import fermiocean as fermiocean
from . import __kubo as kubo

#If one whants to add  new quantities to tabulate, just modify the following dictionaries
#   1)  think of a name of your quantity
#   2)  if it is 'transport (depends on EFermi only) or 'optical' (depends on Efermi and Omega)
#   3)  implement the function somewhere (in one of the submodules, in another submodule,
#           or even in an external package which may be imported (in the latter case be careful
#            to keep it consistent with further versions of WannierBerri
#   4)  add the calculator to 'calculators_trans' or 'calculators_opt' dictionaries
#   5) if needed, define the additional_parameters and their descriptions (see below)
#   6) add a short description of the implemented quantity ('descriptions') which will be printed
#        by the 'print_options()'  function

# a dictionary conaining 'transport' quantities , i.e. those which are tensors
#   depending on the Fermi level, but not on the frequency
#   <quantity> : <function> , ...
# <quantity>   - name of the quantity to calculate (the same will be used in the call of 'integrate' function
# <function> - the function to be called,
#    which will receive two input parameters :
#       data   - Data_K object  (see Data_K.py)
#       Efermi - array of Fermi energies
#    and return  an object of class
#        EnergyResult or  EnergyResultDict (see __result.py)
# may have extra parameters, that should be described in the 'additional_parameters' dictionary (see below)

calculators_trans = {
    'spin': fermiocean.spin,
    'Morb': fermiocean.Morb,
    'Morb_test': fermiocean.Morb_test,
    'ahc': fermiocean.AHC,
    'ahc_test': fermiocean.AHC_test,
    'cumdos': fermiocean.cumdos,
    'dos': fermiocean.dos,
    'conductivity_ohmic': fermiocean.ohmic,
    'conductivity_ohmic_fsurf': fermiocean.ohmic_fsurf,
    'berry_dipole': fermiocean.berry_dipole,
    'berry_dipole_test': fermiocean.berry_dipole_test,
    'berry_dipole_fsurf': fermiocean.berry_dipole_fsurf,
    'gyrotropic_Korb': fermiocean.gme_orb,
    'gyrotropic_Korb_test': fermiocean.gme_orb_test,
    'gyrotropic_Korb_fsurf': fermiocean.gme_orb_fsurf,
    'gyrotropic_Kspin': fermiocean.gme_spin,
    'gyrotropic_Kspin_fsurf': fermiocean.gme_spin_fsurf,
    'Hall_classic': fermiocean.Hall_classic,
    'Hall_classic_fsurf': fermiocean.Hall_classic_fsurf,
    'Hall_morb_fsurf': fermiocean.Hall_morb_fsurf,
    'Hall_spin_fsurf': fermiocean.Hall_spin_fsurf,
    'Der3E': fermiocean.Der3E,
    'Der3E_fsurf': fermiocean.Der3E_fsurf,
    'Der3E_fder2': fermiocean.Der3E_fder2,
    'Hplus_der': fermiocean.Hplus_der,
    'Hplus_der_test': fermiocean.Hplus_der_test,
    'shc_static_qiao': fermiocean.spin_hall_qiao,
    'shc_static_ryoo': fermiocean.spin_hall_ryoo,
}

additional_parameters = defaultdict(lambda: defaultdict(lambda: None))
additional_parameters_description = defaultdict(lambda: defaultdict(lambda: "no description"))

parameters_ocean = {
    'external_terms': (True, "evaluate external terms"),
    'internal_terms': (True, "evaluate internal terms"),
    'tetra': (False, "use tetrahedron method"),
    'correction_wcc':
    (False, "include corrections to make the results coincide with and without wcc_phase for orbital moment"),
    'degen_thresh': (1e-4, "bands with energy difference smaller than this threshold will be considered as degenerate"),
    'degen_Kramers': (False, "consider bands (2i) and (2i+1) as degenerate (counting from zero)")
}

for key, val in parameters_ocean.items():
    for calc in calculators_trans:
        additional_parameters[calc][key] = val[0]
        additional_parameters_description[calc][key] = val[1]

# a dictionary containing 'optical' quantities , i.e. those which are tensors
#   depending on the Fermi level  AND on the frequency
#   <quantity> : <function> , ...
# <quantity>   - name of the quantity to calculate (the same will be used in the call of 'integrate' function
# <function> - the function to be called,
#    which will receive three input parameters :
#       data   - Data_K object  (see Data_K.py)
#       Efermi - array of Fermi energies
#       omega - array of frequencies hbar*omega (in units eV)
#    and return  an object of class
#        EnergyResult or  EnergyResultDict   (see __result.py)
# may have extra parameters, that should be described in the 'additional_parameters' dictionary (see below)

calculators_opt = {
    'opt_conductivity': kubo.opt_conductivity,
    'opt_SHCryoo': kubo.opt_SHCryoo,
    'opt_SHCqiao': kubo.opt_SHCqiao,
    'tildeD': kubo.tildeD,
    'opt_shiftcurrent': kubo.opt_shiftcurrent
}

parameters_optical = {
    'kBT': (0, "temperature in units of eV/kB"),
    'smr_fixed_width': (0.1, "fixed smearing parameter in units of eV"),
    'smr_type': ('Lorentzian', "analyitcal form of the broadened delta function"),
    'adpt_smr': (False, "use an adaptive smearing parameter"),
    'adpt_smr_fac': (np.sqrt(2), "prefactor for the adaptive smearing parameter"),
    'adpt_smr_max': (0.1, "maximal value of the adaptive smearing parameter in eV"),
    'adpt_smr_min': (1e-15, "minimal value of the adaptive smearing parameter in eV"),
    'shc_alpha': (0, "direction of spin current (1, 2, 3)"),
    'shc_beta': (0, "direction of applied electric field (1, 2, 3)"),
    'shc_gamma': (0, "direction of spin polarization (1, 2, 3)"),
    'shc_specification': (False, "calculate all 27 components of SHC if false"),
    'sc_eta': (0.04, "broadening parameter for shiftcurrent calculation, units of eV"),
    'sep_sym_asym': (False, "separate symmetric and antisymmetric parts in optical conductivity")
}

for key, val in parameters_optical.items():
    for calc in calculators_opt:
        additional_parameters[calc][key] = val[0]
        additional_parameters_description[calc][key] = val[1]

additional_parameters['Faraday']['homega'] = 0.0
additional_parameters_description['Faraday']['homega'] = "frequency of light in eV (one frequency per calculation)"

calculators = copy(calculators_trans)
calculators.update(calculators_opt)

descriptions = defaultdict(lambda: "no description")
descriptions['spin'] = "Total Spin polarization per unit cell"
descriptions['Morb'] = "Total orbital magnetization, mu_B per unit cell"
descriptions['ahc'] = "Anomalous hall conductivity (S/cm)"
descriptions['cumdos'] = "Cumulative density of states"
descriptions['dos'] = "density of states"
descriptions['conductivity_ohmic'] = "ohmic conductivity in S/cm for tau={} s . Fermi-sea formula".format(TAU_UNIT)
descriptions['conductivity_ohmic_fsurf'] = "ohmic conductivity in S/cm for tau={} s . Fermi-surface formula".format(
    TAU_UNIT)
descriptions['berry_dipole'] = "berry curvature dipole (dimensionless) - Fermi-sea formula"
descriptions['berry_dipole_fsurf'] = "berry curvature dipole (dimensionless)  - Fermi-surface formula"
descriptions['gyrotropic_Korb'] = "GME tensor, orbital part (Ampere) - Fermi-sea formula"
descriptions['gyrotropic_Korb_fsurf'] = "GME tensor, orbital part (Ampere) - Fermi-surface formula"
descriptions['gyrotropic_Kspin'] = "GME tensor, spin part (Ampere)  - Fermi-sea formula"
descriptions['gyrotropic_Kspin_fsurf'] = "GME tensor, spin part (Ampere)  - Fermi-surface formula"
descriptions[
    'Hall_classic_fsurf'] = "classical Hall coefficient, in S/(cm*T) for tau={} s. - Fermi-surface formula".format(
        TAU_UNIT)
descriptions['Hall_morb_fsurf'] = "Low field AHE, orbital part, in S/(cm*T). - Fermi-surface formula"
descriptions['Hall_spin_fsurf'] = "Low field AHE, spin    part, in S/(cm*T). - Fermi_surface formula"
descriptions['opt_conductivity'] = "Optical conductivity in S/cm"
descriptions['Faraday'] = "Tensor tildeD(omega) describing the Faraday rotation - see PRB 97, 035158 (2018)"
descriptions['opt_SHCryoo'] = "Ryoo's Optical spin Hall conductivity in hbar/e S/cm (PRB RPS19)"
descriptions['opt_SHCqiao'] = "Qiao's Optical spin Hall conductivity in hbar/e S/cm (PRB QZYZ18)"
descriptions['opt_shiftcurrent'] = "Nonlinear shiftcurrent in A/V^2 - see PRB 97, 245143 (2018)"

# omega - for optical properties of insulators
# Efrmi - for transport properties of (semi)conductors


def intProperty(
        data,
        quantities=[],
        user_quantities={},
        Efermi=None,
        omega=None,
        smootherEf=VoidSmoother(),
        smootherOmega=VoidSmoother(),
        parameters={},
        specific_parameters={}):

    def _smoother(quant):
        if quant in calculators_trans:
            return smootherEf
        elif quant in calculators_opt:
            return [smootherEf, smootherOmega]
        else:
            return VoidSmoother()

    results = {}
    for qfull in quantities:
        q = qfull.split('^')[0]
        __parameters = {}
        for param in additional_parameters[q]:
            if param in parameters:
                __parameters[param] = parameters[param]
            else:
                __parameters[param] = additional_parameters[q][param]
        if q in calculators_opt:
            __parameters['omega'] = omega
        if q == 'opt_SHCqiao' or q == 'opt_SHCryoo':
            if 'shc_alpha' in parameters and 'shc_beta' in parameters and 'shc_gamma' in parameters:
                __parameters['shc_specification'] = True
        if qfull in specific_parameters:
            __parameters.update(specific_parameters[qfull])
        results[qfull] = calculators[q](data, Efermi, **__parameters)
        results[qfull].set_smoother(_smoother(q))
    for q, func in user_quantities.items():
        if q in specific_parameters:
            __parameters = specific_parameters[q]
        else:
            __parameters = {}
        results[q] = func(data, Efermi, **__parameters)
        results[q].set_smoother(smootherEf)

    return INTresult(results=results)


class INTresult(result.__result.Result):

    def __init__(self, results={}):
        self.results = results

    def __mul__(self, other):
        return INTresult({q: v * other for q, v in self.results.items()})

    def __truediv__(self, number):
        return self * (1. / number)

    def __add__(self, other):
        if other == 0:
            return self
        results = {r: self.results[r] + other.results[r] for r in self.results if r in other.results}
        return INTresult(results=results)

    def savetxt(self, name):
        for q, r in self.results.items():
            r.savetxt(name.format(q + '{}'))

    # writing to a binary file
    def save(self, name):
        for q, r in self.results.items():
            r.save(name.format(q + '{}'))

    def transform(self, sym):
        results = {r: self.results[r].transform(sym) for r in self.results}
        return INTresult(results=results)

    @property
    def max(self):
        r = np.array([x for v in self.results.values() for x in v.max])
        #        print ("max=",r,"res=",self.results)
        return r
