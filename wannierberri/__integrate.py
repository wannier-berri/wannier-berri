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
from scipy import constants as constants
from collections import Iterable,defaultdict
from copy import copy,deepcopy

from .__utility import  print_my_name_start,print_my_name_end,voidsmoother,TAU_UNIT
from . import __result as result
from . import  __berry as berry
from . import  __fermisea2 as fermisea2
from . import  __nonabelian as nonabelian
from . import  __dos as dos
from . import  symmetry
from . import  __utility   as utility
from . import  __kubo   as kubo

#If one whants to add  new quantities to tabulate, just modify the following dictionaries

#should be functions of only one variable of class Data_K
calculators_trans={ 
         'spin'       : fermisea2.SpinTot,  
         'Morb'       : fermisea2.Morb,
         'ahc'        : fermisea2.AHC ,
         'dos'        : dos.calc_DOS ,
         'cumdos'        : dos.calc_cum_DOS ,
         'Hall_classic' : nonabelian.Hall_classic , 
         'Hall_morb' :  nonabelian.Hall_morb,
         'Hall_spin' :  nonabelian.Hall_spin,

         'conductivity_ohmic_fsurf': nonabelian.conductivity_ohmic,
         'conductivity_ohmic': fermisea2.conductivity_ohmic,

         'berry_dipole'        : fermisea2.tensor_D,
         'berry_dipole_fsurf'      : nonabelian.berry_dipole,
         'gyrotropic_Korb'  : fermisea2.tensor_K,

         'gyrotropic_Kspin'  : fermisea2.gyrotropic_Kspin,
         'gyrotropic_Korb_fsurf'   : nonabelian.gyrotropic_Korb,
         'gyrotropic_Kspin_fsurf'  : nonabelian.gyrotropic_Kspin,
         }


additional_parameters=defaultdict(lambda: defaultdict(lambda:None )   )
additional_parameters_description=defaultdict(lambda: defaultdict(lambda:"no description" )   )


calculators_opt={
    'opt_conductivity' : kubo.opt_conductivity
}

# additional parameters for optical conductivity
additional_parameters['opt_conductivity']['mu'] = 0
additional_parameters_description['opt_conductivity']['mu'] = "chemical potential in units of eV"
additional_parameters['opt_conductivity']['kBT'] = 0
additional_parameters_description['opt_conductivity']['kBT'] = "temperature in units of eV/kB"
additional_parameters['opt_conductivity']['smr_fixed_width'] = 0.1
additional_parameters_description['opt_conductivity']['smr_fixed_width'] = "fixed smearing parameter in units of eV"
additional_parameters['opt_conductivity']['smr_type'] = 'Lorentzian'
additional_parameters_description['opt_conductivity']['smr_type'] = "analyitcal form of the broadened delta function"
additional_parameters['opt_conductivity']['adpt_smr'] = False
additional_parameters_description['opt_conductivity']['adpt_smr'] = "use an adaptive smearing parameter"
additional_parameters['opt_conductivity']['adpt_smr_fac'] = np.sqrt(2)
additional_parameters_description['opt_conductivity']['adpt_smr_fac'] = "prefactor for the adaptive smearing parameter"
additional_parameters['opt_conductivity']['adpt_smr_max'] = 0.1
additional_parameters_description['opt_conductivity']['adpt_smr_max'] = "maximal value of the adaptive smearing parameter in eV"
additional_parameters['opt_conductivity']['adpt_smr_min'] = 1e-15
additional_parameters_description['opt_conductivity']['adpt_smr_min'] = "minimal value of the adaptive smearing parameter in eV"


calculators=copy(calculators_trans)
calculators.update(calculators_opt)


descriptions=defaultdict(lambda:"no description")
descriptions['ahc']="Anomalous hall conductivity (S/cm)"
descriptions['spin']="Total Spin polarization per unit cell"
descriptions['Morb']="Total orbital magnetization, mu_B per unit cell"
descriptions['cumdos']="Cumulative density of states"
descriptions['dos']="density of states"
descriptions['conductivity_ohmic']="ohmic conductivity in S/cm for tau={} s . Fermi-sea formulation".format(TAU_UNIT)
descriptions['conductivity_ohmic_fsurf']="ohmic conductivity in S/cm for tau={} s . Fermi-surface formulation".format(TAU_UNIT)
descriptions['gyrotropic_Korb']="GME tensor, orbital part (Ampere) - fermi sea formula"
descriptions['gyrotropic_Korb_fsurf']="GME tensor, orbital part (Ampere) - fermi surface formula"
descriptions['gyrotropic_Kspin']="GME tensor, spin part (Ampere)  - fermi sea formula"
descriptions['gyrotropic_Kspin_fsurf']="GME tensor, spin part (Ampere)  - fermi surface formula"
descriptions['berry_dipole']="berry curvature dipole (dimensionless) - fermi sea formula"
descriptions['berry_dipole_fsurf']="berry curvature dipole (dimensionless)  - fermi surface formula"
descriptions['Hall_classic'] =  "classical Hall coefficient, in S/(cm*T) for tau={} s".format(TAU_UNIT)
descriptions['Hall_morb'   ] = "Low field AHE, orbital part, in S/(cm*T)."
descriptions['Hall_spin'   ] = "Low field AHE, spin    part, in S/(cm*T)."
descriptions['opt_conductivity'] = "Optical conductivity in S/cm"


# omega - for optical properties of insulators
# Efrmi - for transport properties of (semi)conductors

def intProperty(data,quantities=[],Efermi=None,omega=None,smoothers={},energies={},smootherEf=utility.voidsmoother,smootherOmega=utility.voidsmoother,parameters={}):

  

    def _energy(quant):
        if quant in energies:
            return energies[quant]
        if quant in calculators_trans:
            return Efermi
        if quant in calculators_opt:
            return omega
        raise RuntimeError("quantity {} is neither optical nor transport, and energies are not defined".format(quant))

    def _smoother(quant):
        if quant in smoothers:
            return smoothers[quant]
        elif quant in calculators_trans:
            return smootherEf
        elif quant in calculators_opt:
            return smootherOmega
        else:
            return utility.voidsmoother()
    

    results={}
    for q in quantities:
        __parameters={}
        for param in additional_parameters[q]:
            if param in parameters:
                 __parameters[param]=parameters[param]
            else :
                 __parameters[param]=additional_parameters[q][param]
        results[q]=calculators[q](data,_energy(q),**__parameters)
        results[q].set_smoother(_smoother(q))

    return INTresult( results=results )



class INTresult(result.Result):

    def __init__(self,results={}):
        self.results=results
            
    def __mul__(self,other):
        return INTresult({q:v*other for q,v in self.results.items()})
    
    def __add__(self,other):
        if other == 0:
            return self
        results={r: self.results[r]+other.results[r] for r in self.results if r in other.results }
        return INTresult(results=results) 

    def write(self,name):
        for q,r in self.results.items():
            r.write(name.format(q+'{}'))

    def transform(self,sym):
        results={r:self.results[r].transform(sym)  for r in self.results}
        return INTresult(results=results)

    @property
    def max(self):
        r= np.array([x for v in self.results.values() for x in v.max])
#        print ("max=",r,"res=",self.results)
        return r


