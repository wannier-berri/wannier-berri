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



# this is a generalized 

import numpy as np
from scipy import constants as constants
from collections import Iterable
import inspect
import sys
from .__utility import  print_my_name_start,print_my_name_end,TAU_UNIT,alpha_A,beta_A
from . import __result as result

from scipy.constants import Boltzmann,elementary_charge,hbar,electron_mass,physical_constants,angstrom
bohr_magneton=elementary_charge*hbar/(2*electron_mass)


bohr= physical_constants['Bohr radius'][0]/angstrom
eV_au=physical_constants['electron volt-hartree relationship'][0] 
Ang_SI=angstrom



def AHC(data,Efermi):
    fac_ahc  = -1.0e8*elementary_charge**2/hbar
    return Omega_tot(data,Efermi)*fac_ahc

def Omega_tot(data,Efermi):
    return IterateEf(data.Omega,data,Efermi,TRodd=True,Iodd=False)

def SpinTot(data,Efermi):
    return IterateEf(data.SpinTot,data,Efermi,TRodd=True,Iodd=False)*data.cell_volume

factor_ohmic=(elementary_charge/Ang_SI/hbar**2  # first, transform to SI, not forgeting hbar in velocities - now in  1/(kg*m^3)
                 *elementary_charge**2*TAU_UNIT  # multiply by a dimensional factor - now in A^2*s^2/(kg*m^3*tau_unit) = S/(m*tau_unit)
                   * 1e-2  ) # now in  S/(cm*tau_unit)
def conductivity_ohmic(data,Efermi):
    return IterateEf(data.Ohmic,data,Efermi,TRodd=False,Iodd=False)*factor_ohmic

def gyrotropic_Kspin(data,Efermi):
    factor_Kspin=-bohr_magneton/Ang_SI**2   ## that's it!
    return IterateEf(data.gyroKspin,data,Efermi,TRodd=False,Iodd=True)*factor_Kspin


def Morb(data,Efermi, evalJ0=True,evalJ1=True,evalJ2=True):
    fac_morb =  -eV_au/bohr**2
#    return Omega_tot(data,Efermi).mul_array(Efermi)
    return  fac_morb*( 
               IterateEf(data.Hplus(),data,Efermi,TRodd=True,Iodd=False) 
                           -2*Omega_tot(data,Efermi).mul_array(Efermi) )*data.cell_volume


def tensor_D(data,Efermi):
    return IterateEf(data.derOmegaTr,data,Efermi,TRodd=False,Iodd=True)

def Hplus(data,Efermi):
    return IterateEf(data.derHplusTr,data,Efermi,TRodd=False,Iodd=True)

def tensor_K(data,Efermi):
    Hp = Hplus(data,Efermi).data
    D = tensor_D(data,Efermi).data
    tensor_K = - elementary_charge**2/(2*hbar)*(Hp - 2*Efermi[:,None,None]*D  )
    return result.EnergyResult(Efermi,tensor_K,TRodd=False,Iodd=True)

#########################
####  Private part ######
#########################


def IterateEf(dataIO,data,Efermi,TRodd,Iodd,rank=None,kwargs={}):
    """ this is a general function which accepts dataIO  -- a dictionary like {'i':i , 'io':io, ...}
and sums for a series of Fermi levels"""
#    funname=inspect.stack()[1][3]
#    print ("iterating function '{}' for Efermi={}".format(funname,Efermi))
    OCC=OccDelta(data.E_K,dataIO)
    RES=[OCC.evaluate(Ef) for Ef in  Efermi ]
    return result.EnergyResult(Efermi,np.cumsum(RES,axis=0)/(data.NKFFT_tot*data.cell_volume),TRodd=TRodd,Iodd=Iodd,rank=rank)



# an auxillary class for iteration 
class OccDelta():
 
    def __init__(self,E_K,dataIO):
        self.occ_old=np.zeros(E_K.shape,dtype=bool)
        self.E_K=E_K
        self.dataIO=dataIO
        self.Efermi=-np.Inf
        self.keys=list(dataIO.keys())
        self.shape=None
        try:
          for k,v in dataIO.items():
            if k not in ['i','ii','oi','ooi','oii']:
               raise ValueError("Unknown type of fermi-sea summation : <{}>".format(k))
            assert v.shape[0]==E_K.shape[0], "number of kpoints should match : {} and {}".format(v.shape[0],E_K.shape[0])
            assert np.all( np.array(v.shape[1:len(k)+1])==E_K.shape[1]), "number of bands should match : {} and {}".format(
                              v.shape[1:len(k)+1],E_K.shape[1])
            vshape=v.shape[len(k)+1:]
            if self.shape is None:
                self.shape = vshape
            else : 
                assert self.shape == vshape
        except AssertionError as err:
            print (err) 
            raise ValueError("shapes for fermi-sea summation do not match : EK:{} , ( {} )".format(E_K.shape,
                      " , ".join("{}:{}".format(k,v.shape)  for  k,v in dataIO.items()  )  ) )


    def evaluate(self,Efermi):
#        print ("EF={}, number of occ old:{}".format(Efermi,occ_old.sum()/data.NKFFT_tot))
        assert  Efermi>=self.Efermi, "Fermi levels should be in ascending order"
        self.Efermi=Efermi
        occ_new=(self.E_K<Efermi)
        selectK=np.where(np.any(self.occ_old!=occ_new,axis=1))[0]
        if len(selectK)==0:
            return np.zeros(self.shape)

        occ_old=self.occ_old[selectK]
        self.occ_old=occ_new
        occ_new=occ_new[selectK]
        #  now work only on selected k-points

        unocc_new=np.logical_not(occ_new)
        unocc_old=np.logical_not(occ_old)
        delocc=occ_new!=occ_old

        if 'oi' in self.keys:
            UnoccOcc_plus=unocc_new[:,:,None]*delocc[:,None,:]
            UnoccOcc_minus=delocc[:,:,None]*occ_old[:,None,:]

        if 'ii' in self.keys or 'oii' in self.keys:
            OccOcc_new=occ_new[:,:,None]*occ_new[:,None,:]
            OccOcc_old=occ_old[:,:,None]*occ_old[:,None,:]
            OccOcc_plus = OccOcc_new * np.logical_not(OccOcc_old)
      
        result=0
        for k,V in self.dataIO.items():
            if k=='i'  : 
                result += V[selectK][delocc].sum(axis=0)
            if k=='oi' :
                tmp=V[selectK]
                result += tmp[UnoccOcc_plus].sum(axis=0) - tmp[UnoccOcc_minus].sum(axis=0)
            if k=='ii' : 
                result += V[selectK][OccOcc_plus].sum(axis=0)
            if k=='ooi':
                UnoccUnocc_new=unocc_new[:,:,None]*unocc_new[:,None,:]
                UnoccUnocc_old=unocc_old[:,:,None]*unocc_old[:,None,:]
                UnoccUnocc_minus = UnoccUnocc_old * np.logical_not(UnoccUnocc_new)
                UnoccUnoccOcc_plus=UnoccUnocc_new[:,:,:,None]*delocc[:,None,None,:]
                UnoccUnoccOcc_minus=UnoccUnocc_minus[:,:,:,None]*occ_old[:,None,None,:]
                tmp=V[selectK]
                result+= tmp[UnoccUnoccOcc_plus].sum(axis=0)-tmp[UnoccUnoccOcc_minus].sum(axis=0)
            if k=='oii':
                UnoccOccOcc_plus=unocc_new[:,:,None,None]*OccOcc_plus[:,None,:,:]
                UnoccOccOcc_minus=delocc[:,:,None,None]*OccOcc_old[:,None,:,:]
                tmp=V[selectK]
                result+= tmp[UnoccOccOcc_plus].sum(axis=0)-tmp[UnoccOccOcc_minus].sum(axis=0)

#        if result==0:
#            raise RuntimeError("Nothing was evaluated for the Fermi sea")

        return result
            






