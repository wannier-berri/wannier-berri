#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                                                            #
#------------------------------------------------------------#
#                                                            #
#  written  by                                               #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#

import numpy as np
from scipy import constants as constants
from collections import Iterable,defaultdict
from copy import copy,deepcopy

from .__utility import  print_my_name_start,print_my_name_end,voidsmoother
from . import __result as result
from . import  __berry as berry
from . import  __spin as spin
from . import  __dos as dos
from . import  __symmetry  as symmetry
from . import  __utility   as utility

#If one whants to add  new quantities to tabulate, just modify the following dictionaries

#should be functions of only one parameter of class data_dk
calculators_trans={ 
         'spin'       : spin.calcSpinTot,  
         'morb'       : berry.calcMorb,
         'ahc'        : berry.calcAHC ,
         'dos'        : dos.calcDOS ,
         }



calculators_opt={}




calculators=copy(calculators_trans)
calculators.update(calculators_opt)


descriptions=defaultdict(lambda:"no description")
descriptions['ahc']="Anomalous hall conductivity"
descriptions['spin']="Total Spin polarization"
descriptions['morb']="Total orbital magnetization"
descriptions['dos']="Cumulative density of states"



# omega - for optical properties of insulators
# Efrmi - for transport properties of (semi)conductors

def intProperty(data,quantities=[],Efermi=None,omega=None,smoothers={},energies={},smootherEf=utility.voidsmoother,smootherOmega=utility.voidsmoother):

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
        if quant in calculators_trans:
            return smootherEf
        if quant in calculators_opt:
            return smootherOmega
        return utility.voidsmoother()
    

    results={}
    for q in quantities:
        results[q]=calculators[q](data,_energy(q))
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


