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
from collections import Iterable
import inspect
import sys
from .__utility import  print_my_name_start,print_my_name_end
from . import __result as result

fac_ahc  = -1.0e8*constants.elementary_charge**2/constants.hbar
bohr= constants.physical_constants['Bohr radius'][0]/constants.angstrom
eV_au=constants.physical_constants['electron volt-hartree relationship'][0] 
fac_morb =  -eV_au/bohr**2


def calcV_band(data):
    return data.delE_K

def calcV_band_kn(data):
    return result.KBandResult(data.delE_K,TRodd=True,Iodd=True)

def eval_J(dic):
    _eval={ 'i' : lambda A :  A , 
            'ii': lambda B : B[:,range(B.shape[1]),range(B.shape[1])] , 
            'oi': lambda B : B.sum(axis=(1)) - B[:,range(B.shape[1]),range(B.shape[1])] }
    return sum(_eval[k](v) for k,v in dic.items() if k in ('i','ii','oi'))

def calcImf_band(data):
    return eval_J(data.Omega)

def calcImf_band_kn(data):
    return result.KBandResult(calcImf_band(data),TRodd=True,Iodd=False)

def calcImgh_band_kn(data):
    return result.KBandResult(calcImgh_band(data),TRodd=True,Iodd=False)

def calcImgh_band(data):
    "returns g-h"
    return eval_J(data.Hminus())

def calcSpin_band(data):
    return data.SpinTot['i']

def calcSpin_band_kn(data):
    return result.KBandResult(data=calcSpin_band(data),TRodd=True,Iodd=False)

def calcHall_spin_kn(data):
    imf=calcImf_band(data)
    spn=calcSpin_band(data)
    return result.KBandResult(data=imf[:,:,:,None]*spn[:,:,None,:],TRodd=False,Iodd=False)

def calcHall_orb_kn(data):
    imf=calcImf_band(data)
    orb=calcImgh_band(data)
    return result.KBandResult(data=imf[:,:,:,None]*orb[:,:,None,:],TRodd=False,Iodd=False)


## routines that are not used now 

#def eval_Juuo(B):
#    return np.array([   sum(C.sum(axis=(0,1)) 
#                          for C in  (B[:ib,:ib,ib],B[:ib,ib+1:,ib],B[ib+1:,:ib,ib],B[ib+1:,ib+1:,ib]) )  
#                                      for ib in range(B.shape[0])])

#def eval_Juoo_deg(B,degen):
#    return np.array([   sum(C.sum(axis=(0)) 
#                          for C in ( B[:ib,ib,ib],B[ib+1:,ib,ib])  )  
#                                      for ib in range(B.shape[0])])

### routines for a band-resolved mode

#def eval_Jo_deg(A,degen):
#    return np.array([A[ib1:ib2].sum(axis=0) for ib1,ib2 in degen])

#def eval_Juo_deg(B,degen):
#    return np.array([B[:ib1,ib1:ib2].sum(axis=(0,1)) + B[ib2:,ib1:ib2].sum(axis=(0,1)) for ib1,ib2 in degen])

#def eval_Joo_deg(B,degen):
#    return np.array([B[ib1:ib2,ib1:ib2].sum(axis=(0,1))  for ib1,ib2 in degen])

