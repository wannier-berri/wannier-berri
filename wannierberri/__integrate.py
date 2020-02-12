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

from .__utility import  print_my_name_start,print_my_name_end,voidsmoother
from . import __result as result
from . import  __berry as berry
from . import  __gyrotropic as gyrotropic
from . import  __spin as spin
from . import  __nonabelian as nonabelian
from . import  __dos as dos
from . import  __symmetry  as symmetry
from . import  __utility   as utility

#If one whants to add  new quantities to tabulate, just modify the following dictionaries

#should be functions of only one parameter of class data_dk
calculators_trans={ 
         'spin'       : spin.calcSpinTot,  
         'Morb'       : berry.calcMorb,
         'Morb_LC'       : berry.calcMorb_LC,
         'Morb_IC'       : berry.calcMorb_IC,
#         'Morb_LC_J0'       : berry.calcMorb_LC_J0,
#         'Morb_LC_J1'       : berry.calcMorb_LC_J1,
#         'Morb_LC_J2'       : berry.calcMorb_LC_J2,
#         'Morb_IC_J0'       : berry.calcMorb_IC_J0,
#         'Morb_IC_J1'       : berry.calcMorb_IC_J1,
#         'Morb_IC_J2'       : berry.calcMorb_IC_J2,
         'ahc'        : berry.calcAHC ,
         'dos'        : dos.calc_DOS ,
         'cumdos'        : dos.calc_cum_DOS ,
#         'nonabelian_spin' : nonabelian.spin , 
         'nonabelian_ahc'     : nonabelian.ahc , 
         'nonabelian_Morb' : nonabelian.Morb , 
#         'nonabelian_spinspin' : nonabelian.spinspin , 
#         'nonabelian_velvel' : nonabelian.velvel , 
#         'nonabelian_spinvel' : nonabelian.spinvel , 
#         'nonabelian_morbvel' : nonabelian.morbvel , 
#         'nonabelian_curvvel' : nonabelian.curvvel , 
#         'nonabelian_curvmorb' : nonabelian.curvmorb , 
#         'nonabelian_curvspin' : nonabelian.curvspin , 
#         'nonabelian_curv_tot' : nonabelian.curv_tot , 
#         'nonabelian_velvel' : nonabelian.velvel , 
         'Hall_classic' : nonabelian.Hall_classic , 
         'Hall_classic_sea' : nonabelian.Hall_classic_sea , 
         'Hall_morb' :  nonabelian.Hall_morb,
         'Hall_spin' :  nonabelian.Hall_spin,
         'conductivity_ohmic': nonabelian.conductivity_ohmic,
         'conductivity_ohmic_sea': nonabelian.conductivity_ohmic_sea,
         'conductivity_ohmic_sea_1': nonabelian.conductivity_ohmic_sea_1,
         'conductivity_ohmic_sea_2': nonabelian.conductivity_ohmic_sea_2,
         'berry_dipole'      : nonabelian.berry_dipole,
         'gyrotropic_Korb'   : nonabelian.gyrotropic_Korb,
         'gyrotropic_Kspin'  : nonabelian.gyrotropic_Kspin,
         }


additional_parameters=defaultdict(lambda: defaultdict(lambda:None )   )
additional_parameters_description=defaultdict(lambda: defaultdict(lambda:"no description" )   )


calculators_opt={}




calculators=copy(calculators_trans)
calculators.update(calculators_opt)


descriptions=defaultdict(lambda:"no description")
descriptions['ahc']="Anomalous hall conductivity"
descriptions['spin']="Total Spin polarization"
descriptions['morb']="Total orbital magnetization"
descriptions['cumdos']="Cumulative density of states"
descriptions['dos']="density of states"



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
        results[q].smoother=_smoother(q)

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
            r.write(name.format(q))

    def transform(self,sym):
        results={r:self.results[r].transform(sym)  for r in self.results}
        return INTresult(results=results)

    @property
    def max(self):
        r= np.array([x for v in self.results.values() for x in v.max])
#        print ("max=",r,"res=",self.results)
        return r


