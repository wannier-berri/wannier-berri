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
import functools
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

def AHC2(data,Efermi):
    fac_ahc  = -1.0e8*elementary_charge**2/hbar
    return Omega_tot2(data,Efermi)*fac_ahc


def Omega_tot(data,Efermi):
    return IterateEf(data.Omega,data,Efermi,TRodd=True,Iodd=False)

def Omega_tot2(data,Efermi):
    return IterateEf(data.Omega2,data,Efermi,TRodd=True,Iodd=False)

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
    return  fac_morb*(
                    IterateEf(data.Hplus(),data,Efermi,TRodd=True,Iodd=False)
                            -2*Omega_tot(data,Efermi).mul_array(Efermi) )*data.cell_volume

def HplusTr_2(data,Efermi):
    return IterateEf(data.derHplusTr2,data,Efermi,sep=False,TRodd=False,Iodd=True)

def HplusTr(data,Efermi):
    return IterateEf(data.derHplusTr,data,Efermi,sep=True,TRodd=False,Iodd=True)

def tensor_K(data,Efermi):
    Hp = HplusTr(data,Efermi).data
    D = tensor_D(data,Efermi).data
    tensor_K = - elementary_charge**2/(2*hbar)*(Hp - 2*Efermi[:,None,None]*D  )
    return result.EnergyResult(Efermi,tensor_K,TRodd=False,Iodd=True)

def tensor_K_2(data,Efermi):
    Hp = HplusTr_2(data,Efermi).data
    D = tensor_D_2(data,Efermi).data
    tensor_K = - elementary_charge**2/(2*hbar)*(Hp - 2*Efermi[:,None,None]*D  )
    return result.EnergyResult(Efermi,tensor_K,TRodd=False,Iodd=True)

def tensor_D(data,Efermi):
    return IterateEf(data.derOmegaTr,data,Efermi,sep=True,TRodd=False,Iodd=True)
    #return data.derOmegaTr

def tensor_D_2(data,Efermi):
        return IterateEf(data.derOmegaTr2,data,Efermi,sep=False,TRodd=False,Iodd=True)

def tensor_D_findif(data,Efermi):
        return IterateEf(data.berry_dipole_findif,data,Efermi,sep=False,TRodd=False,Iodd=True)

def tensor_Dw(data,Efermi,omega0=0):
        return IterateEf(partial(data.derOmegaWTr,omega=omega0),data,Efermi,sep=True,TRodd=False,Iodd=True)

#########################
####  Private part ######
#########################


def IterateEf(dataIO,data,Efermi,TRodd,Iodd,sep=False,rank=None,kwargs={}):
    """ this is a general function which accepts dataIO  -- a dictionary like {'i':i , 'io':io, ...}
     and sums for a series of Fermi levels
     parameter dataIO can be a dictionary or a funciton. 
     If needed use callable(dataIO) for judgment and run 
     OCC=OccDelta(data.E_K,dataIO(op,ed),op,ed) or OCC=OccDelta(data.E_K(op,ed),dataIO(op,ed),op,ed)"""
#    funname=inspect.stack()[1][3]
#    print ("iterating function '{}' for Efermi={}".format(funname,Efermi))
# try to make sum better
    if sep:
        res = 0.0
        for op in range(0,data.nkptot,data.ksep):
            ed=min(op+data.ksep,data.nkptot)
            OCC=OccDelta(dataIO(op,ed))
            RES=[OCC.evaluate(Ef) for Ef in  Efermi ]
            res+=np.cumsum(RES,axis=0)/(data.NKFFT_tot*data.cell_volume)
        return result.EnergyResult(Efermi,res,TRodd=TRodd,Iodd=Iodd)
    elif 'sea' in dataIO:  # !!! This is the preferred option for now 
        if 'EFmin' in dataIO and 'EFmax' in dataIO:
            A,EFmin,EFmax=dataIO['sea'],dataIO['EFmin'],dataIO['EFmax']
#            for ik in range(10):
#                print (np.vstack((E[ik],EFmin[ik],EFmax[ik])).T)
            RES=np.array([A[(EFmin<=Ef)*(EFmax>Ef)].sum(axis=(0))  for Ef in Efermi ])
        else:
            RES=np.array([sum(maxocc(E,Ef,A) for A,E in zip(dataIO['sea'],dataIO['E'])) for Ef in Efermi ])
        return result.EnergyResult(Efermi,RES/(data.NKFFT_tot*data.cell_volume),TRodd=TRodd,Iodd=Iodd,rank=rank)
    else:
        OCC=OccDelta(dataIO)
        RES=[OCC.evaluate(Ef) for Ef in  Efermi ]
        return result.EnergyResult(Efermi,np.cumsum(RES,axis=0)/(data.NKFFT_tot*data.cell_volume),TRodd=TRodd,Iodd=Iodd,rank=rank)


def maxocc(E,Ef,A):
    occ=(E<=Ef)
    if True not in occ : 
#        print ("no occ states for Ef={} and E={}".format(Ef,E)) 
        return np.zeros(A.shape[1:])
    else:
#        i=
#        print ("summing upto band {} for Ef={} E={}".format (i,Ef,E))
        return A[max(np.where(occ)[0])]


class DataIO(dict):
    """ a class to store data, which are to be summed over inner 'i' or  outer 'o' states, 
        or over fermi sea 'sea'
        the IO in the name doed NOT stand for input/output methods. 
    """

    @property
    def E(self):
        return self['E']

    @property
    def nk(self):
        return self.E.shape[0]

    @property
    def nb(self):
        return self.E.shape[1]

    def to_sea(self,degen_thresh=0):
        """replaces all keys, like 'ooi' , 'oi', etc by a single key 'sea' - the corresponding sum when
           the n-th band is the highest occupied
        """
## TODO : Check if this routine takes muchtime for large systems. then optimize it with taking differences 
##    between (n+1)th and n-th  step like in the OCC class
        sea=0
        for key,val in self.items():
            if key=='E':
                continue
            elif key=='i':
                sea=sea+ np.array([val[:,:n].sum(axis=1) for n in range(1,self.nb+1)])
            elif key=='ii':
                sea=sea+ np.array([val[:,:n,:n].sum(axis=(1,2)) for n in range(1,self.nb+1)])
            elif key=='oi':
                sea=sea+ np.array([val[:,n:,:n].sum(axis=(1,2)) for n in range(1,self.nb+1)])
            elif key=='ooi':
                sea=sea+ np.array([val[:,n:,n:,:n].sum(axis=(1,2,3)) for n in range(1,self.nb+1)])
            elif key=='oii':
                sea=sea+ np.array([val[:,n:,:n,:n].sum(axis=(1,2,3)) for n in range(1,self.nb+1)])
            else :
                raise RuntimeError("Unknown key in dataIO : '{}' ".formta(key))
        sea=sea.transpose([1,0]+list(range(2,sea.ndim)))
        EFmin_list=[]
        EFmax_list=[]
        sea_list=[]
        for ik in range(self.nk):
            select=np.hstack( ( (self.E[ik,1:]-self.E[ik,:-1])>degen_thresh,[True]) )
            E=self.E[ik,select]
            sea_list.append(sea[ik,select])
            EFmin_list.append(E)
            EFmax_list.append(E[1:])
            EFmax_list.append([np.Inf])
        res= DataIO({'sea':np.vstack(sea_list),
                        'EFmin':np.hstack(EFmin_list), 
                        'EFmax':np.hstack(EFmax_list) })
#        print ("shapes of DataIO : ",res['sea'].shape,res['EFmin'].shape,res['EFmax'].shape)
        return res



def _stack(lst):
    if lst[0].ndim>1:
        return np.vstack(lst)
    else:
        return np.hstack(lst)


def mergeDataIO(data_list):
    keys=set([key for data in data_list for key in data.keys()])
    for key in keys:
        arrays=[data[key] for data in data_list]
#        print ("key={}, shapes={}".format(key,[a.shape for a in arrays]))
    res=DataIO({ key:_stack([data[key] for data in data_list ] ) for key in  
                  set([key for data in data_list for key in data.keys()])  })
#    for key in keys:
#        print ("key={}, res_shape={}".format(key,res[key].shape))
    return res



# an auxillary class for iteration 
class OccDelta():
 
    def __init__(self,dataIO):
        self.E_K=dataIO['E']
        self.occ_old=np.zeros(self.E_K.shape,dtype=bool)
        self.dataIO=dataIO
        self.Efermi=-np.Inf
        self.keys=list(dataIO.keys())
        self.shape=None
        try:
          for k,v in dataIO.items():
            if k=='E': 
                continue
            if k not in ['i','ii','oi','ooi','oii','E','sea']:
                raise ValueError("Unknown type of fermi-sea summation : <{}>".format(k))
            assert v.shape[0]==self.E_K.shape[0], "number of kpoints should match : {} and {}".format(v.shape[0],self.E_K.shape[0])
            assert np.all( np.array(v.shape[1:len(k)+1])==self.E_K.shape[1]), "number of bands should match : {} and {}".format(
                              v.shape[1:len(k)+1],E_K.shape[1])
            vshape=v.shape[len(k)+1:]
            if self.shape is None:
                self.shape = vshape
            else : 
                assert self.shape == vshape
        except AssertionError as err:
            print (err) 
            raise ValueError("shapes for fermi-sea summation do not match : EK:{} , ( {} )".format(self.E_K.shape,
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
            






