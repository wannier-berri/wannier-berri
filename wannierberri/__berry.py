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
#                                                            #
# this file initially was  an adapted translation of         #
# the corresponding Fortran90 code from  Wannier 90 project  #
#                                                            #
# with significant modifications for better performance      #
#   it is now a lot different                                #
#                                                            #
# The webpage of the Wannier90 code is www.wannier.org       #
# The Wannier90 code is hosted on GitHub:                    #
# https://github.com/wannier-developers/wannier90            #
#------------------------------------------------------------#

import numpy as np
from scipy import constants as constants
from collections import Iterable

from .__utility import  print_my_name_start,print_my_name_end
from . import __result as result




alpha=np.array([1,2,0])
beta =np.array([2,0,1])

#              -1.0e8_dp*elem_charge_SI**2/(hbar_SI*cell_volume)
fac_ahc  = -1.0e8*constants.elementary_charge**2/constants.hbar
bohr= constants.physical_constants['Bohr radius'][0]/constants.angstrom
eV_au=constants.physical_constants['electron volt-hartree relationship'][0] 
fac_morb =  -eV_au/bohr**2
print ("fac_morb=",fac_morb,1/fac_morb)

def calcV_band(data):
    return data.delE_K

def calcV_band_kn(data):
    return result.KBandResult(data.delE_K,TRodd=True,Iodd=True)



def eval_J0(A,occ):
    return A[occ].sum(axis=0)

def eval_J12(B,UnoccOcc):
    return -2*B[UnoccOcc].sum(axis=0)

#def eval_J3(B,UnoccUnoccOcc):
#    return -2*B[UnoccUnoccOcc].sum(axis=0)

def get_occ(E_K,Efermi):
    return (E_K< Efermi)





def calcAHC(data,Efermi=None,occ_old=None):
    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)


    if isinstance(Efermi, Iterable):
#        print ("iterating over Fermi levels")
        nFermi=len(Efermi)
        AHC=np.zeros( ( nFermi,3) ,dtype=float )
        for iFermi in range(nFermi):
#            print ("iFermi={}".format(iFermi))
            AHC[iFermi]=calcAHC(data,Efermi=Efermi[iFermi],occ_old=occ_old)
        return result.EnergyResultAxialV(Efermi,np.cumsum(AHC,axis=0))
    
    # now code for a single Fermi level:
    AHC=np.zeros(3)

#    print ("  calculating occ matrices")
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
#    print ("  calculating occ matrices - done")

#    print ("evaluating J0")
    AHC=eval_J0(data.Omega_Hbar_diag[selectK], delocc)
#    print ("evaluating B")
    B=(data.D_A+data.D_H_sq)[selectK]
#    print ("evaluating J12")
    AHC+=eval_J12(B,unoccocc_plus)-eval_J12(B,unoccocc_minus)
#    print ("evaluating J12-done")
    occ_old[:,:]=occ_new[:,:]
    return AHC*fac_ahc/(data.NKFFT_tot*data.cell_volume)


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
    return result.EnergyResultAxialV(Efermi,Morb[:,3,:])


def calcMorb_term(data,Efermi=None,occ_old=None, J=3,LC=True,IC=True,EF=True,gh=True):
    if not isinstance(Efermi, Iterable):
        Efermi=np.array([Efermi])
    imfgh=calcImfgh(data,Efermi=Efermi,occ_old=occ_old, evalJ0=(J in (0,3)),evalJ1=(J in (1,3)),evalJ2=(J in (2,3)))
    imf=imfgh[:,0,J,:]
    img=imfgh[:,1,J,:]
    imh=imfgh[:,2,J,:]
    LCtil=fac_morb*(img*(1 if gh else 0)-Efermi[:,None]*imf*(1 if EF else 0)  )
    ICtil=fac_morb*(imh*(1 if gh else 0)-Efermi[:,None]*imf*(1 if EF else 0)  )
    Morb = LCtil*(1 if LC else 0) + ICtil*(1 if IC else 0)
    return result.EnergyResultAxialV(Efermi,Morb[:,:])

def calcMorb_LC(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=3,LC=True,IC=False)
def calcMorb_IC(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=3,LC=False,IC=True)

def calcMorb_f(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=3,LC=False,IC=True,gh=False)

def calcMorb_h(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=3,LC=False,IC=True,gh=True,EF=False)

def calcMorb_LC_J0(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=0,LC=True,IC=False)
def calcMorb_IC_J0(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=0,LC=False,IC=True)

def calcMorb_LC_J1(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=1,LC=True,IC=False)
def calcMorb_IC_J1(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=1,LC=False,IC=True)

def calcMorb_LC_J2(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=2,LC=True,IC=False)
def calcMorb_IC_J2(data,Efermi=None):
   return calcMorb_term(data,Efermi, J=2,LC=False,IC=True)


#    return np.array([Morb,LCtil,ICtil])


def calcMorb_intr(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):
    if not isinstance(Efermi, Iterable):
        Efermi=np.array([Efermi])
    imgh=calcImgh_band(data)
    EK=data.E_K
    res=np.zeros( (len(Efermi),3),dtype=float)
    res[0,:]=imgh[EK<=Efermi[0]].sum(axis=0)
    for i in range(1,len(Efermi)):
        res[i]=res[i-1]+imgh[(EK<=Efermi[i])*(EK>Efermi[i-1])].sum(axis=0)
    return result.EnergyResultAxialV(Efermi,fac_morb*res/data.NKFFT_tot)


def calcMorb2(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):
    imgh=calcImgh_band(data)
    imf=calcImf_band(data)
    dE=Efermi[1]-Efermi[0]
    EK=data.E_K
    imfE=imf*EK[:,:,None]
    res=np.zeros( (len(Efermi),3),dtype=float)
    res[0,:]=imgh[EK<=Efermi[0]].sum(axis=0)
    for i,Ef in enumerate(Efermi):
        res[i]= (imgh+2*(imfE-imf*Ef))[EK<=Ef].sum(axis=0)
    return result.EnergyResultAxialV(Efermi,fac_morb*res/data.NKFFT_tot)



### routines for a band-resolved mode

def eval_Jo_deg(A,degen):
    return np.array([A[ib1:ib2].sum(axis=0) for ib1,ib2 in degen])

def eval_Juo_deg(B,degen):
    return np.array([B[:ib1,ib1:ib2].sum(axis=(0,1)) + B[ib2:,ib1:ib2].sum(axis=(0,1)) for ib1,ib2 in degen])

def eval_Joo_deg(B,degen):
    return np.array([B[ib1:ib2,ib1:ib2].sum(axis=(0,1))  for ib1,ib2 in degen])

#def eval_Juuo_deg(B,degen):
#    return np.array([   sum(C.sum(axis=(0,1,2)) 
#                          for C in  (B[:ib1,:ib1,ib1:ib2],B[:ib1,ib2:,ib1:ib2],B[ib2:,:ib1,ib1:ib2],B[ib2:,ib2:,ib1:ib2]) )  
#                                      for ib1,ib2 in degen])

#def eval_Juoo_deg(B,degen):
#    return np.array([   sum(C.sum(axis=(0,1,2)) 
#                          for C in ( B[:ib1,ib1:ib2,ib1:ib2],B[ib2:,ib1:ib2,ib1:ib2])  )  
#                                      for ib1,ib2 in degen])



def eval_Jo(A):
    return A 

def eval_Juo(B):
    return np.array([B[:ib,ib].sum(axis=(0)) + B[ib+1:,ib].sum(axis=(0)) for ib in range(B.shape[0])])

def eval_Joo(B):
    return np.array([B[i,i] for i in range(B.shape[0])])

#def eval_Juuo(B):
#    return np.array([   sum(C.sum(axis=(0,1)) 
#                          for C in  (B[:ib,:ib,ib],B[:ib,ib+1:,ib],B[ib+1:,:ib,ib],B[ib+1:,ib+1:,ib]) )  
#                                      for ib in range(B.shape[0])])

#def eval_Juoo_deg(B,degen):
#    return np.array([   sum(C.sum(axis=(0)) 
#                          for C in ( B[:ib,ib,ib],B[ib+1:,ib,ib])  )  
#                                      for ib in range(B.shape[0])])


def calcImf_band(data):
    AA=data.Omega_Hbar_diag
    BB=data.D_A+data.D_H_sq
    return np.array([eval_Jo(A)-2*eval_Juo(B)  for A,B in zip (AA,BB) ] )


def calcImf_band_kn(data):
    return result.KBandResult(calcImf_band(data),TRodd=True,Iodd=False)

def calcImgh_band_kn(data):
    return result.KBandResult(calcImhg_band(data),TRodd=True,Iodd=False)

#returns g-h
def calcImgh_band(data):
    
    AA=data.A_E_A
    BB=data.Morb_Hbar_diag-data.OmegaHbar
    imgh=np.array([eval_Jo(B)-2*eval_Joo(A)  for A,B in zip (AA,BB) ] )
    
    AA=data.D_B-data.D_E_A
    imgh+=-2*np.array([eval_Juo(A) for A in AA])

    C,D=data.D_E_D
    AA=C-D
    imgh+=-2*np.array([eval_Juo(A) for A in AA])
    return imgh





def calcImg_band(data):
    
    AA=data.A_E_A
    BB=data.Morb_Hbar_diag-data.Omega_Hbar
    imgh=np.array([eval_Jo(B)-2*eval_Joo(A)  for A,B in zip (AA,BB) ] )
    
    AA=data.D_B-data.D_E_A
    imgh+=-2*np.array([eval_Juo(A) for A in AA])

    C,D=data.D_E_D
    AA=C-D
    imgh+=-2*np.array([eval_Juo(A) for A in AA])
    return imgh



def calcImfgh_K(data,degen,ik,J=3):
     
    imf=np.zeros( (data.NKFFT_tot,data.num_wann,3))


    imf= calcImf_K(data,degen,ik)

    s=2*eval_Joo_deg(data.A_E_A[ik],degen)   
    img=eval_Jo_deg(data.Morb_Hbar_diag[ik],degen)-s
    imh=eval_Jo_deg(data.Omega_Hbar_E[ik],degen)+s


    C=data.D_B[ik]
    D=data.D_E_A[ik]
    img+=-2*eval_Juo_deg(C,degen)
    imh+=-2*eval_Juo_deg(D,degen)

    C,D=data.D_E_D
    img+=-2*eval_Juo_deg(C[ik],degen) 
    imh+=-2*eval_Juo_deg(D[ik],degen)

    return imf,img,imh



def calcImf(data,degen_bands=None):
    AA=data.Omega_Hbar_diag
    BB=data.D_A+data.D_H_sq
    if degen_bands is None:
        degen_bands=[(b,b+1) for b in range(data.nbands)]
    return np.array([eval_Jo_deg(A,degen_bands)-2*eval_Juo_deg(B,degen_bands)  for A,B in zip (AA,BB) ] )


def calcImf_K(data,degen_bands,ik):
    A=data.Omega_Hbar_diag[ik]
    B=(data.D_A+data.D_H_sq)[ik]
    return eval_Jo_deg(A,degen_bands)-2*eval_Juo_deg(B,degen_bands) 


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

    OccOcc_new=occ_new_selk[:,:,None]*occ_new_selk[:,None,:]
    OccOcc_old=occ_old_selk[:,:,None]*occ_old_selk[:,None,:]
    OccOcc_plus = OccOcc_new * np.logical_not(OccOcc_old)
    
    if evalJ0:
        imfgh[0,0]= eval_J0(data.Omega_Hbar_diag[selectK], delocc)
        s=-eval_J12(data.A_E_A[selectK],OccOcc_plus)
        imfgh[1,0]= eval_J0(data.Morb_Hbar_diag[selectK], delocc)-s
        imfgh[2,0]= eval_J0(data.Omega_Hbar_E[selectK] , delocc)+s
    if evalJ1:
        B=data.D_A[selectK]
        C=data.D_B[selectK]
        D=data.D_E_A[selectK]
        imfgh[0,1]=eval_J12(B,UnoccOcc_plus)-eval_J12(B,UnoccOcc_minus)
        imfgh[1,1]=eval_J12(C,UnoccOcc_plus)-eval_J12(C,UnoccOcc_minus)
        imfgh[2,1]=eval_J12(D,UnoccOcc_plus)-eval_J12(D,UnoccOcc_minus)
    if evalJ2:
        B=data.D_H_sq[selectK]
        C,D=data.D_E_D
        C=C[selectK]
        D=D[selectK]
        imfgh[0,2]=eval_J12(B,UnoccOcc_plus)-eval_J12(B,UnoccOcc_minus)
        imfgh[1,2]=eval_J12(C,UnoccOcc_plus)-eval_J12(C,UnoccOcc_minus)
        imfgh[2,2]=eval_J12(D,UnoccOcc_plus)-eval_J12(D,UnoccOcc_minus)
#        imfgh[1,2]=eval_J3(C,UnoccUnoccOcc_plus)-eval_J3(C,UnoccUnoccOcc_minus)
#        imfgh[2,2]=eval_J3(D,UnoccOccOcc_plus)-eval_J3(D,UnoccOccOcc_minus)

    imfgh[:,3,:]=imfgh[:,:3,:].sum(axis=1)

    occ_old[:,:]=occ_new[:,:]
    return imfgh/(data.NKFFT_tot)










def calcAHC_uIu(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):

    raise NotImplementedError()
    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)

    if isinstance(Efermi, Iterable):
        nFermi=len(Efermi)
        AHC=np.zeros( ( nFermi,4,3) ,dtype=float )
        for iFermi in range(nFermi):
            AHC[iFermi]=calcAHC_uIu(data,Efermi=Efermi[iFermi],occ_old=occ_old, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
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

