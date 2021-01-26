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

from .__utility import  print_my_name_start,print_my_name_end,VoidSmoother,TAU_UNIT
from . import __result as result
from . import  __berry as berry
from . import  __fermisea2 as fermisea2
from . import  __nonabelian as nonabelian
from . import  __dos as dos
from . import  symmetry
from . import  __utility   as utility
from . import  __kubo   as kubo

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

calculators_trans={ 
         'spin'       : fermisea2.SpinTot,  
         'Morb'       : fermisea2.Morb,
         'ahc'        : fermisea2.AHC ,
         'ahc2'        : fermisea2.AHC2 ,
         'dos'        : dos.calc_DOS ,
         'cumdos'        : dos.calc_cum_DOS ,
         'Hall_classic' : nonabelian.Hall_classic , 
         'Hall_morb' :  nonabelian.Hall_morb,
         'Hall_spin' :  nonabelian.Hall_spin,

         'conductivity_ohmic_fsurf': nonabelian.conductivity_ohmic,
         'conductivity_ohmic'      : fermisea2.conductivity_ohmic,

         'berry_dipole'            : fermisea2.tensor_D,
         'berry_dipole_2'          : fermisea2.tensor_D_2,
         'berry_dipole_fsurf'      : nonabelian.berry_dipole,
#         'Faraday1w'                 : nonabelian.Faraday,
         'berry_dipole_findif'     : fermisea2.tensor_D_findif,
         'gyrotropic_Korb'         : fermisea2.tensor_K,
         'gyrotropic_Korb_2'       : fermisea2.tensor_K_2,

         'gyrotropic_Kspin'        : fermisea2.gyrotropic_Kspin,
         'gyrotropic_Korb_fsurf'   : nonabelian.gyrotropic_Korb,
         'gyrotropic_Kspin_fsurf'  : nonabelian.gyrotropic_Kspin,
         }


additional_parameters=defaultdict(lambda: defaultdict(lambda:None )   )
additional_parameters_description=defaultdict(lambda: defaultdict(lambda:"no description" )   )


# a dictionary conaining 'optical' quantities , i.e. those which are tensors 
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

calculators_opt={
    'opt_conductivity' : kubo.opt_conductivity,
    'opt_SHCryoo' : kubo.opt_SHCryoo,
    'opt_SHCqiao' : kubo.opt_SHCqiao,
    'tildeD'     : kubo.tildeD,
}



parameters_optical={
'kBT'             :  ( 0    ,  "temperature in units of eV/kB"          ),
'smr_fixed_width' :  ( 0.1  ,  "fixed smearing parameter in units of eV"),
'smr_type'        :  ('Lorentzian' ,  "analyitcal form of the broadened delta function" ),
'adpt_smr'        :  (  False ,  "use an adaptive smearing parameter" ),
'adpt_smr_fac'    :  ( np.sqrt(2) ,  "prefactor for the adaptive smearing parameter" ),
'adpt_smr_max'    :  (  0.1 , "maximal value of the adaptive smearing parameter in eV" ),
'adpt_smr_min'    :  ( 1e-15,  "minimal value of the adaptive smearing parameter in eV") }



for key,val in parameters_optical.items(): 
    for calc in calculators_opt: 
        additional_parameters[calc][key] = val[0]
        additional_parameters_description[calc][key] = val[1]


additional_parameters['Faraday']['homega'] = 0.0
additional_parameters_description['Faraday']['homega'] = "frequency of light in eV (one frequency per calculation)"


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
descriptions['Faraday'] = "Tensor tildeD(omega) describing the Faraday rotation - see PRB 97, 035158 (2018)"
descriptions['opt_SHCryoo'] = "Ryoo's Optical spin Hall conductivity in S/cm (PRB RPS19)"
descriptions['opt_SHCqiao'] = "Qiao's Optical spin Hall conductivity in S/cm (PRB QZYZ18)"

# omega - for optical properties of insulators
# Efrmi - for transport properties of (semi)conductors

def intProperty(data,quantities=[],Efermi=None,omega=None,smootherEf=VoidSmoother(),smootherOmega=VoidSmoother(),parameters={}):

    def _smoother(quant):
        if quant in calculators_trans:
            return smootherEf
        elif quant in calculators_opt:
            return [smootherEf,smootherOmega]
        else:
            return VoidSmoother()

    results={}
    for q in quantities:
        __parameters={}
        for param in additional_parameters[q]:
            if param in parameters:
                 __parameters[param]=parameters[param]
            else :
                 __parameters[param]=additional_parameters[q][param]
        if q in calculators_opt:
            __parameters['omega']=omega
        results[q]=calculators[q](data,Efermi,**__parameters)
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


