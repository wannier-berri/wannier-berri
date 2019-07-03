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


import wan_ham as wham
import numpy as np
from scipy import constants as constants
from get_data import Data_dk


#def  calcAHC_dk(dk,NK,data,Efermi=None,occ=None, evalJ0=True,evalJ1=True,evalJ2=True,printJ=False):
#    return calcAHC(Data_dk(data,dk,NK=NK),Efermi=Efermi,occ=occ, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2,printJ=printJ)


def  calcAHC(data,Efermi=None,occ=None, evalJ0=True,evalJ1=True,evalJ2=True,printJ=False):

    if evalJ0 or evalJ1 or evalJ2:
        E_K, delE_K, UU_K, HH_K, delHH_K =wham.get_eig_deleig(data.NKFFT,data.HH_R,data.iRvec,data.cRvec)
    
    if evalJ1 or evalJ2:
        JJp_list,JJm_list=wham.get_JJp_JJm_list(delHH_K, UU_K, E_K, efermi=Efermi,occ_K=occ)
    
    if evalJ0:
        f_list,g_list=wham.get_occ_mat_list(UU_K, efermi=Efermi, eig_K=E_K, occ_K=occ)

    
    AHC0=np.zeros(3)
    AHC1=np.zeros(3)
    AHC2=np.zeros(3)
    fac = -1.0e8*constants.elementary_charge**2/(constants.hbar*data.cell_volume)/np.prod(data.NKFFT)

    if evalJ0:
        AHC0= fac*np.einsum("knm,kmna->a",f_list,data.get_OOmega_K()).real
        if printJ: print ("J0 term:",AHC0)
    if evalJ1:
        AHC1=-2*fac*( np.einsum("knma,kmna ->a" , data.get_AA_K()[:,:,:,wham.alpha],JJp_list[:,:,:,wham.beta]).imag +
               np.einsum("knma,kmna ->a" , data.get_AA_K()[:,:,:,wham.beta],JJm_list[:,:,:,wham.alpha]).imag )
        if printJ: print ("J1 term:",AHC1)
    if evalJ2:
        AHC2=-2*fac*np.einsum("knma,kmna ->a" , JJm_list[:,:,:,wham.alpha],JJp_list[:,:,:,wham.beta]).imag 
        if printJ: print ("J2 term:",AHC2)
    AHC=(AHC0+AHC1+AHC2)
    
    if printJ: print ("Anomalous Hall conductivity: (in S/cm ) \n",AHC)
    return np.array([AHC0,AHC1,AHC2,AHC])
    
    
    
"""