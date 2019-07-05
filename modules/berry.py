#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# this file initially was  an adapted translation of         #
# the corresponding Fortran90 code from  Wannier 90 project  #
#                                                            #
# with significant modifications for better performance      #
#   it is nor a lot different                                #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                                                            #
# The webpage of the Wannier90 code is www.wannier.org       #
# The Wannier90 code is hosted on GitHub:                    #
# https://github.com/wannier-developers/wannier90            #
#------------------------------------------------------------#
#                                                            #
#  Translated to python and adapted for wannier19 project by #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#

import numpy as np
from scipy import constants as constants
from collections import Iterable

alpha=np.array([1,2,0])
beta =np.array([2,0,1])


def eval_J0(A,occ):
    return A[occ].sum(axis=0)

def eval_J12(B,unoccocc):
    return -2*B[unoccocc].sum(axis=0)


def get_occ(E_K,Efermi):
    return (E_K< Efermi)
#    unocc=(E_K>Efermi)
#    unoccocc=unocc[:,:,None]*occ[:,None,:]
#    return occ,unocc,unoccocc
        
def calcAHC(data,Efermi=None, evalJ0=True,evalJ1=True,evalJ2=True):

    if isinstance(Efermi, Iterable):
        fermiscan=True
        Efermi0=Efermi[0]
    else:
        Efermi0=Efermi        
        fermiscan=False

    fac = -1.0e8*constants.elementary_charge**2/(constants.hbar*data.cell_volume)/np.prod(data.NKFFT)

#  First calculate for the 1st Fermi level
    occ=get_occ(data.E_K,Efermi0)
    unocc= np.logical_not(occ)
    unoccocc=unocc[:,:,None]*occ[:,None,:]
    
    AHC0=np.zeros((4,3))


    if evalJ0:
        AHC0[0]= eval_J0(data.OOmegaUU_K, occ)
    if evalJ1:
        AHC0[1]=eval_J12(data.delHH_dE_AA_K,unoccocc)
    if evalJ2:
        AHC0[2]=eval_J12(data.delHH_dE_SQ_K,unoccocc)
    
        
    AHC0[3]=AHC0[:3,:].sum(axis=0)

    if not fermiscan:  return AHC0*fac
    
    # now move upwards in Efermi
    
    Nfermi=len(Efermi)
    AHC=np.zeros( ( Nfermi,4,3) ,dtype=float )
    AHC[0]=AHC0
    
    occ_new=occ
    unocc_new=unocc
    for ifermi in range(1,Nfermi):
        occ_old=occ_new
        unocc_old=unocc_new
        
        occ_new=get_occ(data.E_K,Efermi[ifermi])
        unocc_new=np.logical_not(occ_new)
        selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
        occ_old_selk=occ_old[selectK]
        occ_new_selk=occ_new[selectK]
        unocc_old_selk=unocc_old[selectK]
        unocc_new_selk=unocc_new[selectK]
        delocc=occ_new_selk!=occ_old_selk

  #     unoccocc=unocc[:,:,None]*occ[:,None,:]
        unoccocc_plus=unocc_new_selk[:,:,None]*delocc[:,None,:]
        unoccocc_minus=delocc[:,:,None]*occ_old_selk[:,None,:]

        
        if evalJ0:
            AHC[ifermi,0]= eval_J0(data.OOmegaUU_K[selectK], delocc)
        if evalJ1:
            B=data.delHH_dE_AA_K[selectK]
            AHC[ifermi,1]=eval_J12(B,unoccocc_plus)-eval_J12(B,unoccocc_minus)
        if evalJ2:
            B=data.delHH_dE_SQ_K[selectK]
            AHC[ifermi,2]=eval_J12(B,unoccocc_plus)-eval_J12(B,unoccocc_minus)
        AHC[ifermi,3,:]=AHC[ifermi,:3,:].sum(axis=0)
        

#        AHC[iFermi]=calcAHC(data,Efermi=Efermi[iFermi], evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
        
    return np.cumsum(AHC,axis=0)*fac
    
    
    
    
