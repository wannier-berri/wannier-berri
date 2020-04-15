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

from .__utility import  print_my_name_start,print_my_name_end
from . import __result as result
#from .__berry import eval_J0,get_occ,calcImf_band,calcImgh_band



def calcSpin_band(data):
    return data.SSUU_K_rediag


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

def calcSpinTot(data,Efermi=None,occ_old=None):

    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)

    if isinstance(Efermi, Iterable):
        nFermi=len(Efermi)
        SPN=np.zeros( ( nFermi,3) ,dtype=float )
        for iFermi in range(nFermi):
            SPN[iFermi]=calcSpinTot(data,Efermi=Efermi[iFermi],occ_old=occ_old)
        return result.EnergyResultAxialV(Efermi,np.cumsum(SPN,axis=0))

    occ_new=get_occ(data.E_K,Efermi)
    selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
    occ_old_selk=occ_old[selectK]
    occ_new_selk=occ_new[selectK]
    delocc=occ_new_selk!=occ_old_selk
    occ_old[:,:]=occ_new[:,:]
    return eval_J0(calcSpin_band(data)[selectK], delocc)/data.NKFFT_tot


