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

def calcAHC(data,Efermi=None, evalJ0=True,evalJ1=True,evalJ2=True):

    if isinstance(Efermi, Iterable):
        return np.array( [calcAHC(data,Efermi=Ef, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
                 for Ef in Efermi])

    occ=(data.E_K< Efermi)
    unocc=(data.E_K>Efermi)
    unoccocc=unocc[:,:,None]*occ[:,None,:]

    AHC=np.zeros((4,3))

    if evalJ0:
        AHC[0]= data.OOmegaUU_K[occ].sum(axis=0) 
    if evalJ1:
        AHC[1]=-2*data.delHH_dE_AA_K[unoccocc].sum(axis=0)
    if evalJ2:
        AHC[2]=-2*data.delHH_dE_SQ_K[unoccocc].sum(axis=0)

    AHC[3]=AHC[:3,:].sum(axis=0)

    fac = -1.0e8*constants.elementary_charge**2/(constants.hbar*data.cell_volume)/np.prod(data.NKFFT)
    
    return AHC*fac
