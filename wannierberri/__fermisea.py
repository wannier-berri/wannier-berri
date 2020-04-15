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
# some parts of this file were inspired by                    #
# the corresponding Fortran90 code from  Wannier90 project  #
#                                                            ##                                                            #
# The webpage of the Wannier90 code is www.wannier.org       #
# The Wannier90 code is hosted on GitHub:                    #
# https://github.com/wannier-developers/wannier90            #
#------------------------------------------------------------#

import numpy as np
from numpy import set_printoptions
from scipy import constants as constants
from collections import Iterable
import inspect
import sys
from .__utility import  print_my_name_start,print_my_name_end,TAU_UNIT,alpha_A,beta_A
from . import __result as result

from scipy.constants import Boltzmann,elementary_charge,hbar,electron_mass,physical_constants,angstrom
bohr_magneton=elementary_charge*hbar/(2*electron_mass)


fac_ahc  = -1.0e8*elementary_charge**2/hbar
bohr= physical_constants['Bohr radius'][0]/angstrom
eV_au=physical_constants['electron volt-hartree relationship'][0] 
Ang_SI=angstrom
fac_morb =  -eV_au/bohr**2
set_printoptions(edgeitems=100)

class OccDelta():
 
    def __init__(self,occ_old,data,Efermi,triple=False):
#       print ("EF={}, number of occ old:{}".format(Efermi,occ_old.sum()/data.NKFFT_tot))
        Ek=data.E_K
        occ_new=(data.E_K<Efermi)
        unocc_new=np.logical_not(occ_new)
        unocc_old=np.logical_not(occ_old)
        selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
        occ_old_selk=occ_old[selectK]
        occ_new_selk=occ_new[selectK]
        unocc_old_selk=unocc_old[selectK]
        unocc_new_selk=unocc_new[selectK]
        self.delocc=occ_new_selk!=occ_old_selk
        self.selectK=selectK
        self.UnoccOcc_plus=unocc_new_selk[:,:,None]*self.delocc[:,None,:]
        self.UnoccOcc_minus=self.delocc[:,:,None]*occ_old_selk[:,None,:]
    
        OccOcc_new=occ_new_selk[:,:,None]*occ_new_selk[:,None,:]
        OccOcc_old=occ_old_selk[:,:,None]*occ_old_selk[:,None,:]
        self.OccOcc_plus = OccOcc_new * np.logical_not(OccOcc_old)
        '''
        print('Ek={}'.format(Ek[0,:]),Ek.shape)
        print('occ_new={}'.format(occ_new[0,:]),occ_new.shape)
        print('unocc_new={}'.format(unocc_new[0,:]),unocc_new.shape)
        print('self.delocc={}'.format(self.delocc[0,:]),self.delocc.shape)
        '''

        if triple:
            UnoccUnocc_new=unocc_new_selk[:,:,None]*unocc_new_selk[:,None,:]
            UnoccUnocc_old=unocc_old_selk[:,:,None]*unocc_old_selk[:,None,:]
            UnoccUnocc_minus = UnoccUnocc_old * np.logical_not(UnoccUnocc_new)
    
            self.UnoccOccOcc_plus=unocc_new_selk[:,:,None,None]*self.OccOcc_plus[:,None,:,:]
            self.UnoccOccOcc_minus=self.delocc[:,:,None,None]*OccOcc_old[:,None,:,:]
        
            self.UnoccUnoccOcc_plus=UnoccUnocc_new[:,:,:,None]*self.delocc[:,None,None,:]
            self.UnoccUnoccOcc_minus=UnoccUnocc_minus[:,:,:,None]*occ_old_selk[:,None,None,:]
        occ_old[:,:]=occ_new[:,:]

    def eval_O(self,A):
        return A[self.selectK][self.delocc].sum(axis=0)

    def eval_UO(self,B):
        B1=B[self.selectK]
        return B1[self.UnoccOcc_plus].sum(axis=0)-B1[self.UnoccOcc_minus].sum(axis=0)

    def eval_OO(self,B):
        return B[self.selectK][self.OccOcc_plus].sum(axis=0)

    def eval_UUO(self,B):
        B1=B[self.selectK]
        return B1[self.UnoccUnoccOcc_plus].sum(axis=0)-B1[self.UnoccUnoccOcc_minus].sum(axis=0)

    def eval_UOO(self,B):
        B1=B[self.selectK]
        return B1[self.UnoccOccOcc_plus].sum(axis=0)-B1[self.UnoccOccOcc_minus].sum(axis=0)


def IterateEf(data,Efermi,TRodd,Iodd,rank=None,kwargs={}):
    occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)
    funname=inspect.stack()[1][3]
#    print ("iterating function '{}' for Efermi={}".format(funname,Efermi))
    RES=[] 
    fun=vars(sys.modules[__name__])[funname]
    for iFermi,Ef in enumerate(Efermi):
#        print ("iFermi[]={}".format(iFermi,Ef))
        RES.append(fun(data,Efermi=Ef,occ_old=occ_old,**kwargs))
    return result.EnergyResult(Efermi,np.cumsum(RES,axis=0)/(data.NKFFT_tot*data.cell_volume),TRodd=TRodd,Iodd=Iodd,rank=rank)
    


def calcAHC(data,Efermi,occ_old=None):

    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=True,Iodd=False)
    OCC=OccDelta(occ_old,data,Efermi)

    AHC=OCC.eval_O(data.Omega_Hbar_diag) - 2* OCC.eval_UO( data.D_A+data.D_H_sq )

    return AHC*fac_ahc


def calc_dipole_D(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=True)
    A,B,C=data.D_gdD
    dipole= -2*( OCC.eval_UO(A) 
            + OCC.eval_UOO(B)
            + OCC.eval_UUO(C)  )
    return dipole

def calc_grd_orbital_mag(data,Efermi,occ_old=None):
    return ( calc_grd_orbital_mag_1(data,Efermi) + calc_grd_orbital_mag_2(data,Efermi)
                                                       )



def calc_grd_orbital_mag_2(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=True)
    B,C=data.VDD_DVD
    dipole= -(OCC.eval_UOO(B) + OCC.eval_UUO(C))
    return dipole

def calc_grd_orbital_mag_1(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=True)
    A,B,C=data.E_D_gdD
    dipole= 2*( OCC.eval_UO(A)
            + OCC.eval_UOO(B)
            + OCC.eval_UUO(C)  )
   # print ("A=: ".format(A),A.shape)

    return dipole

def calc_dipole_D_old(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=True)
    A,B,C=data.D_gdD_old
    dipole= -2*( OCC.eval_UO(A) 
            + OCC.eval_UOO(B)
            + OCC.eval_UUO(C)  )
    return dipole




def calc_dipole_ext_1(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi)
    A=data.Omega_bar_der_rediag
    B=data.Omega_bar_D_re
    dipole= OCC.eval_O(A)+2*OCC.eval_UO(B)
    return dipole


def calc_dipole_ext_1_debug(data,Efermi):
    return ( 
              calc_dipole_ext_1_1(data,Efermi) + 
              calc_dipole_ext_1_2(data,Efermi) 
           )



def calc_dipole_ext_2_debug(data,Efermi):
    return ( 
              calc_dipole_ext_2_3(data,Efermi) + 
              calc_dipole_ext_2_4(data,Efermi) + 
              calc_dipole_ext_2_5(data,Efermi) + 
              calc_dipole_ext_2_6(data,Efermi) 
           )



def calc_dipole_ext_1_1(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi)
    A=data.Omega_bar_der_rediag
#    B=data.Omega_bar_D_re
    dipole= OCC.eval_O(A)
    return dipole

def calc_dipole_ext_1_2(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi)
#    A=data.Omega_bar_der_rediag
    B=data.Omega_bar_D_re
    dipole= 2*OCC.eval_UO(B)
    return dipole



def calc_dipole_ext_2(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=True)
    A,B,C=data.A_gdD
    A1,B1,C1=data.DdA_DAD_DDA
#    print ("shapes for Ef={0}: ".format(Efermi),A.shape,B.shape,C.shape)
#    print ("shapes for Ef={0}: ".format(Efermi),A.shape,B.shape,C.shape,A1.shape,B1.shape,C1.shape)
    return 2*( ( OCC.eval_UO(A)+OCC.eval_UOO(B)+OCC.eval_UUO(C) ) + ( OCC.eval_UO(A1)+OCC.eval_UOO(B1)+OCC.eval_UUO(C1) )  )
#    return 2*( OCC.eval_UO(A+A1)+OCC.eval_UOO(B+B1)+OCC.eval_UUO(C+C1) )


def calc_dipole_ext_2_3(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=True)
    A,B,C=data.A_gdD
#    print ("shapes for Ef={0}: ".format(Efermi),A.shape,B.shape,C.shape)
    return 2*( OCC.eval_UO(A)+OCC.eval_UOO(B)+OCC.eval_UUO(C) )

def calc_dipole_ext_2_4(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=True)
    A,B,C=data.DdA_DAD_DDA
#    print ("shapes for Ef={0}: ".format(Efermi),A.shape,B.shape,C.shape)
    return 2*( OCC.eval_UO(A) ) 

def calc_dipole_ext_2_5(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=True)
    A,B,C=data.DdA_DAD_DDA
#    print ("shapes for Ef={0}: ".format(Efermi),A.shape,B.shape,C.shape)
    return 2*( OCC.eval_UUO(C) ) 

def calc_dipole_ext_2_6(data,Efermi,occ_old=None):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=True)
    A,B,C=data.DdA_DAD_DDA
#    print ("shapes for Ef={0}: ".format(Efermi),A.shape,B.shape,C.shape)
    return 2*( OCC.eval_UOO(B) ) 



def calc_dipole(data,Efermi):
    return calc_dipole_D(data,Efermi)+calc_dipole_ext1(data,Efermi)+calc_dipole_ext2(data,Efermi)




# an equivalent fermi-sea formulation - to test the effective mass tensor

factor_ohmic=(elementary_charge/Ang_SI/hbar**2  # first, transform to SI, not forgeting hbar in velocities - now in  1/(kg*m^3)
                 *elementary_charge**2*TAU_UNIT  # multiply by a dimensional factor - now in A^2*s^2/(kg*m^3*tau_unit) = S/(m*tau_unit)
                   * 1e-2  ) # now in  S/(cm*tau_unit)

def conductivity_ohmic_sea(data,Efermi,occ_old=None):
    # _general yields integral(V*V*f0') in units eV/Ang
    # we want in S/(cm)/tau_unit
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=False)

    OCC=OccDelta(occ_old,data,Efermi,triple=False)
#    print ("shapes:",data.del2E_H_diag.shape,data.Db_Va_re.shape,OCC.eval_O(data.del2E_H_diag).shape,OCC.eval_UO(data.Db_Va_re).shape)
   
    return (OCC.eval_O(data.del2E_H_diag)+OCC.eval_UO(data.Db_Va_re))*factor_ohmic

factor_Kspin=-bohr_magneton/Ang_SI**2   ## that's it!

def gyrotropic_Kspin_sea(data,Efermi,occ_old=None):
    # _general yields integral(V*V*f0') in units eV/Ang
    # we want in S/(cm)/tau_unit
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=False,Iodd=True)
    OCC=OccDelta(occ_old,data,Efermi,triple=False)
#    print ("shapes:",data.del2E_H_diag.shape,data.Db_Va_re.shape,OCC.eval_O(data.del2E_H_diag).shape,OCC.eval_UO(data.Db_Va_re).shape)
    return (OCC.eval_O(data.delS_H_diag)+OCC.eval_UO(data.Db_Sa_re))*factor_Kspin



def calcMorb(data,Efermi,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True,evalLC=True,evalIC=True):

    imfgh=calcImfgh(data,Efermi,occ_old=None, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2,
              evalF=True,evalG=True,evalH=True).data
    imf=imfgh[:,0]
    img=imfgh[:,1]
    imh=imfgh[:,2]

    LCtil=fac_morb*(img-Efermi[:,None]*imf)
    ICtil=fac_morb*(imh-Efermi[:,None]*imf)
    Morb =(LCtil*(1. if evalLC else 0.) + ICtil*(1. if evalIC else 0.))*data.cell_volume
    return result.EnergyResult(Efermi,Morb,TRodd=True,Iodd=False)


def calcImfgh(data,Efermi,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True,evalF=True,evalG=True,evalH=True):
    if isinstance(Efermi, Iterable):
        return IterateEf(data,Efermi,TRodd=True,Iodd=False,rank=1,
               kwargs={'evalJ0':evalJ0,'evalJ1':evalJ1,'evalJ2':evalJ2})

    OCC=OccDelta(occ_old,data,Efermi,triple=False)

    imfgh=np.zeros( (3,3) )
    if evalJ0:
        if evalG or evalH: s=2*OCC.eval_OO(data.A_E_A)
        if evalF: imfgh[0]+= OCC.eval_O(data.Omega_Hbar_diag)
        if evalG: imfgh[1]+= OCC.eval_O(data.Morb_Hbar_diag )-s
        if evalH: imfgh[2]+= OCC.eval_O(data.Omega_Hbar_E   )+s
    if evalJ1:
        if evalF: B=data.D_A
        if evalG: C=data.D_B
        if evalH: D=data.D_E_A
        if evalF: imfgh[0]+=-2*OCC.eval_UO(B)
        if evalG: imfgh[1]+=-2*OCC.eval_UO(C)
        if evalH: imfgh[2]+=-2*OCC.eval_UO(D)
    if evalJ2:
        if evalF: B=data.D_H_sq
        if evalG or evalH :C,D=data.D_E_D
        if evalF: imfgh[0]+=-2*OCC.eval_UO(B)
        if evalG: imfgh[1]+=-2*OCC.eval_UO(C)
        if evalH: imfgh[2]+=-2*OCC.eval_UO(D)

    return  imfgh
