#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# this file represents  an adapted translation of            #
# Fortran90 code from  Wannier 90 project                    #
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
from get_data import Data_dk
from collections import Iterable

alpha=np.array([1,2,0])
beta =np.array([2,0,1])


def  calcAHC(data,Efermi=None, evalJ0=True,evalJ1=True,evalJ2=True,printJ=False):

    if isinstance(Efermi, Iterable):
        return np.array( [calcAHC(data,Efermi=Ef, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2,printJ=printJ)
                 for Ef in Efermi])

    selm=np.sum(data.E_K< Efermi,axis=1)
    seln=np.sum(data.E_K<=Efermi,axis=1)
    
    occ=(data.E_K< Efermi)
    unocc=(data.E_K>Efermi)
    unoccocc=unocc[:,:,None]*occ[:,None,:]

    AHC0=np.zeros(3)
    AHC1=np.zeros(3)
    AHC2=np.zeros(3)
    fac = -1.0e8*constants.elementary_charge**2/(constants.hbar*data.cell_volume)/np.prod(data.NKFFT)

    if evalJ0:
        AHC0= fac* np.sum(OO[:m,:].sum(axis=0) for OO,m in zip(data.OOmegaUU_K,selm) )
#        AHC0= fac*data.OOmegaUU_K[occ].sum(axis=0) #np.sum(  OO[:m,:].sum(axis=0) for OO,m in zip(data.OOmegaUU_K,selm) )
        if printJ: print ("J0 term:",AHC0)
    if evalJ1:
#        AHC1=-2*fac*sum( delhhaa[n:,:m,:].sum(axis=(0,1))
#                    for delhhaa,n,m in zip(data.delHH_dE_AA_K,seln,selm)  ) 
        AHC1=-2*fac*data.delHH_dE_AA_K[unoccocc].sum(axis=0)
#        sum( delhhaa[n:,:m,:].sum(axis=(0,1))
#                    for delhhaa,n,m in zip(data.delHH_dE_AA_K,seln,selm)  ) 
        if printJ: print ("J1 term:",AHC1)
    if evalJ2:
#        AHC2=-2*fac*sum( delhhsq[n:,:m,:].sum(axis=(0,1)) 
#           for delhhsq,n,m in zip(data.delHH_dE_SQ_K,seln,selm) ) # for a,b in zip(alpha,beta)] )
        AHC2=-2*fac*data.delHH_dE_SQ_K[unoccocc].sum(axis=0)
        if printJ: print ("J2 term:",AHC2)
    AHC=(AHC0+AHC1+AHC2)
    
    if printJ: print ("Anomalous Hall conductivity: (in S/cm ) \n",AHC)
    return np.array([AHC0,AHC1,AHC2,AHC])
