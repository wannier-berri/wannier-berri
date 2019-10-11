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

#              -1.0e8_dp*elem_charge_SI**2/(hbar_SI*cell_volume)
fac_ahc  = -1.0e8*constants.elementary_charge**2/constants.hbar
bohr= constants.physical_constants['Bohr radius'][0]/constants.angstrom
eV_au=constants.physical_constants['electron volt-hartree relationship'][0] 
fac_morb =  -eV_au/bohr**2



def eval_J0(A,occ):
    return A[occ].sum(axis=0)

def eval_J12(B,UnoccOcc):
    return -2*B[UnoccOcc].sum(axis=0)

def eval_J3(B,UnoccUnoccOcc):
    return -2*B[UnoccUnoccOcc].sum(axis=0)

def get_occ(E_K,Efermi):
    return (E_K< Efermi)
        
def calcAHC(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):

    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)

    if isinstance(Efermi, Iterable):
        nFermi=len(Efermi)
        AHC=np.zeros( ( nFermi,4,3) ,dtype=float )
        for iFermi in range(nFermi):
            AHC[iFermi]=calcAHC(data,Efermi=Efermi[iFermi],occ_old=occ_old, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
        return np.cumsum(AHC,axis=0)
    
    # now code for a single Fermi level:
    AHC=np.zeros((4,3))

    occ_new=get_occ(data.E_K,Efermi)
    unocc_new=np.logical_not(occ_new)
    unocc_old=np.logical_not(occ_old)
    selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
    occ_old_selk=occ_old[selectK]
    occ_new_selk=occ_new[selectK]
    unocc_old_selk=unocc_old[selectK]
    unocc_new_selk=unocc_new[selectK]
    delocc=occ_new_selk!=occ_old_selk
    unoccocc_plus=unocc_new_selk[:,:,None]*delocc[:,None,:]
    unoccocc_minus=delocc[:,:,None]*occ_old_selk[:,None,:]

    if evalJ0:
        AHC[0]= eval_J0(data.OOmegaUU_K_rediag[selectK], delocc)
    if evalJ1:
        B=data.delHH_dE_AA_K[selectK]
        AHC[1]=eval_J12(B,unoccocc_plus)-eval_J12(B,unoccocc_minus)
    if evalJ2:
        B=data.delHH_dE_SQ_K[selectK]
        AHC[2]=eval_J12(B,unoccocc_plus)-eval_J12(B,unoccocc_minus)
    AHC[3,:]=AHC[:3,:].sum(axis=0)

    occ_old[:,:]=occ_new[:,:]
    return AHC*fac_ahc/(data.NKFFT_tot*data.cell_volume)




def calcAHC_uIu(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):

    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)

    if isinstance(Efermi, Iterable):
        nFermi=len(Efermi)
        AHC=np.zeros( ( nFermi,4,3) ,dtype=float )
        for iFermi in range(nFermi):
            AHC[iFermi]=calcAHC(data,Efermi=Efermi[iFermi],occ_old=occ_old, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
        return np.cumsum(AHC,axis=0)
    
    # now code for a single Fermi level:
    AHC=np.zeros(3)

    occ_new=get_occ(data.E_K,Efermi)
    selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
    occ_old_selk=occ_old[selectK]
    occ_new_selk=occ_new[selectK]
    delocc=occ_new_selk!=occ_old_selk

    AHC= eval_J0(data.FF[selectK], delocc)

    return AHC*fac_ahc/(data.NKFFT_tot*data.cell_volume)



def calcImfgh(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):

    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)

    if isinstance(Efermi, Iterable):
        nFermi=len(Efermi)
        imfgh=np.zeros( ( nFermi,3,4,3) ,dtype=float )
        for iFermi in range(nFermi):
            imfgh[iFermi]=calcImfgh(data,Efermi=Efermi[iFermi],occ_old=occ_old, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
        return np.cumsum(imfgh,axis=0)
    
    # now code for a single Fermi level:
    imfgh=np.zeros((3,4,3))

    occ_new=get_occ(data.E_K,Efermi)
    unocc_new=np.logical_not(occ_new)
    unocc_old=np.logical_not(occ_old)
    selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
    occ_old_selk=occ_old[selectK]
    occ_new_selk=occ_new[selectK]
    unocc_old_selk=unocc_old[selectK]
    unocc_new_selk=unocc_new[selectK]
    delocc=occ_new_selk!=occ_old_selk
    UnoccOcc_plus=unocc_new_selk[:,:,None]*delocc[:,None,:]
    UnoccOcc_minus=delocc[:,:,None]*occ_old_selk[:,None,:]

    UnoccUnoccOcc_new=unocc_new_selk[:,:,None,None]*unocc_new_selk[:,None,:,None]*occ_new_selk[:,None,None,:]
    UnoccUnoccOcc_old=unocc_old_selk[:,:,None,None]*unocc_old_selk[:,None,:,None]*occ_old_selk[:,None,None,:]

    UnoccOccOcc_new=unocc_new_selk[:,:,None,None]*  occ_new_selk[:,None,:,None]*occ_new_selk[:,None,None,:]
    UnoccOccOcc_old=unocc_old_selk[:,:,None,None]*  occ_old_selk[:,None,:,None]*occ_old_selk[:,None,None,:]
    
    UnoccUnoccOcc_plus =UnoccUnoccOcc_new*np.logical_not(UnoccUnoccOcc_old)
    UnoccUnoccOcc_minus=UnoccUnoccOcc_old*np.logical_not(UnoccUnoccOcc_new)

    UnoccOccOcc_plus =UnoccOccOcc_new*np.logical_not(UnoccOccOcc_old)
    UnoccOccOcc_minus=UnoccOccOcc_old*np.logical_not(UnoccOccOcc_new)

    OccOcc_new=occ_new_selk[:,:,None]*occ_new_selk[:,None,:]
    OccOcc_old=occ_old_selk[:,:,None]*occ_old_selk[:,None,:]
    OccOcc_plus = OccOcc_new * np.logical_not(OccOcc_old)
    
    if evalJ0:
        imfgh[0,0]= eval_J0(data.OOmegaUU_K_rediag[selectK], delocc)
        s=-eval_J12(data.HHAAAAUU_K[selectK],OccOcc_plus)
        imfgh[1,0]= eval_J0(data.CCUU_K_rediag[selectK], delocc)-s
        imfgh[2,0]= eval_J0(data.HHOOmegaUU_K[selectK] , delocc)+s
    if evalJ1:
        B=data.delHH_dE_AA_K[selectK]
        C=data.delHH_dE_BB_K[selectK]
        D=data.delHH_dE_HH_AA_K[selectK]
        imfgh[0,1]=eval_J12(B,UnoccOcc_plus)-eval_J12(B,UnoccOcc_minus)
        imfgh[1,1]=eval_J12(C,UnoccOcc_plus)-eval_J12(C,UnoccOcc_minus)
        imfgh[2,1]=eval_J12(D,UnoccOcc_plus)-eval_J12(D,UnoccOcc_minus)
    if evalJ2:
        B=data.delHH_dE_SQ_K[selectK]
        C,D=data.delHH_dE_SQ_HH_K
        C=C[selectK]
        D=D[selectK]
        imfgh[0,2]=eval_J12(B,UnoccOcc_plus)-eval_J12(B,UnoccOcc_minus)
        imfgh[1,2]=eval_J3(C,UnoccUnoccOcc_plus)-eval_J3(C,UnoccUnoccOcc_minus)
        imfgh[2,2]=eval_J3(D,UnoccOccOcc_plus)-eval_J3(D,UnoccOccOcc_minus)

    imfgh[:,3,:]=imfgh[:,:3,:].sum(axis=1)

    occ_old[:,:]=occ_new[:,:]
    return imfgh/(data.NKFFT_tot)


def calcMorb(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):
    if not isinstance(Efermi, Iterable):
        Efermi=np.array([Efermi])
    imfgh=calcImfgh(data,Efermi=Efermi,occ_old=occ_old, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
    imf=imfgh[:,0,:,:]
    img=imfgh[:,1,:,:]
    imh=imfgh[:,2,:,:]
    LCtil=fac_morb*(img-Efermi[:,None,None]*imf)
    ICtil=fac_morb*(imh-Efermi[:,None,None]*imf)
    Morb = LCtil + ICtil
    return np.array([Morb,LCtil,ICtil])