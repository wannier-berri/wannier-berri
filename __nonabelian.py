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
import sys
from . import __berry
from collections import Iterable, defaultdict
from . import __result as result
from time import time
from .__utility import alpha_A,beta_A,TAU_UNIT,TAU_UNIT_TXT

from scipy.constants import Boltzmann,elementary_charge,hbar,electron_mass
bohr_magneton=elementary_charge*hbar/(2*electron_mass)

Ang_SI=1e-10

def __spin(data):
    return data.spin_nonabelian

def __vel(data):
    return data.vel_nonabelian

def __mass(data):
    return data.mass_nonabelian

def __curvE(data):
    return [[o*e for o,e in zip(O,E)]
        for O,E in zip(data.Berry_nonabelian,data.E_K_degen)]

def __curv(data):
    return data.Berry_nonabelian

def __curvD(data):
    return data.Berry_nonabelian_D

def __curvExt1(data):
    return data.Berry_nonabelian_ext1

def __curvExt2(data):
    return data.Berry_nonabelian_ext2


def __morb(data):
    return data.Morb_nonabelian

__dimensions=defaultdict(lambda : 1)
__dimensions['mass']=2

#quantities that should be odd under TRS and inversion
TRodd  = set(['spin','morb','vel','curv','curvE','morbg','morb2','curvD','curvExt1','curvExt2'])
INVodd = set(['vel'])


def spin(data,Efermi):
    return nonabelian_general(data,Efermi,['spin'])

def spinvel(data,Efermi):
    return nonabelian_general(data,Efermi,['spin','vel'])

def curvvel(data,Efermi):
    return nonabelian_general(data,Efermi,['curv','vel'])

def curvmorb(data,Efermi):
    return nonabelian_general(data,Efermi,['curv','morb'])

def curvspin(data,Efermi):
    return nonabelian_general(data,Efermi,['curv','spin'])


def velvel(data,Efermi):
    return nonabelian_general(data,Efermi,['vel','vel'])


def morbvel(data,Efermi):
    return nonabelian_general(data,Efermi,['morb','vel'])


def spinspin(data,Efermi):
    return nonabelian_general(data,Efermi,['spin','spin'])


def curv_tot(data,Efermi):
    return nonabelian_general(data,Efermi,['curv'],mode='fermi-sea')


def ahc(data,Efermi):
    return nonabelian_general(data,Efermi,['curv'],mode='fermi-sea',factor=__berry.fac_ahc)

def berry_dipole(data,Efermi):
    # _general yields integral(omega*v*(-fo')), which is dimensionlesss - what we want 
    return nonabelian_general(data,Efermi,['curv','vel'],mode='fermi-surface',factor=1)


def berry_dipole_D(data,Efermi):
    # _general yields integral(omega*v*(-fo')), which is dimensionlesss - what we want 
    return nonabelian_general(data,Efermi,['curvD','vel'],mode='fermi-surface',factor=1)

def berry_dipole_ext1(data,Efermi):
    # _general yields integral(omega*v*(-fo')), which is dimensionlesss - what we want 
    return nonabelian_general(data,Efermi,['curvExt1','vel'],mode='fermi-surface',factor=1)

def berry_dipole_ext2(data,Efermi):
    # _general yields integral(omega*v*(-fo')), which is dimensionlesss - what we want 
    return nonabelian_general(data,Efermi,['curvExt2','vel'],mode='fermi-surface',factor=1)


def gyrotropic_Kspin(data,Efermi):
    # _general yields integral(spin*v*(-fo')), which is in Ang^-2
    # we want in Ampere
    factor=-bohr_magneton/Ang_SI**2   ## that's it!
    return nonabelian_general(data,Efermi,['vel','spin'],mode='fermi-surface',factor=factor)


def gyrotropic_Korb(data,Efermi):
    # _general yields integral(morb*v*(-fo')), which is in eV
    # we want in Ampere
    factor=-elementary_charge**2/(2*hbar)   ## that's it!
    return nonabelian_general(data,Efermi,['vel','morb'],mode='fermi-surface',factor=factor)
    


def Morb(data,Efermi):
    r1=nonabelian_general(data,Efermi,['morb' ],mode='fermi-sea')
    r2=nonabelian_general(data,Efermi,['curvE'],mode='fermi-sea')
    r3=nonabelian_general(data,Efermi,['curv' ],mode='fermi-sea')
    r3.data[:,:]=r3.data[:,:]*Efermi[:,None]
    return (r1+2*r2-2*r3)*__berry.fac_morb*data.cell_volume

def Hall_morb(data,Efermi):
    # _general yields integral(omega*morb*(-fo'_) in units Ang
    # we want in S/(cm*T)
    # S/T=A^3*s^5/(kg^2*m^2))
    factor=-Ang_SI*elementary_charge/(2*hbar) # first, transform to SI, not forgettint e/2hbar multilier for morb - now in A*m/J ,restoring the sign of spin magnetic moment
    factor*=elementary_charge**2/hbar  # multiply by a dimensional factor - now in S/(T*m)
    factor*=-1 
    factor*=1e-2   #  finally transform to S/(T*cm)
    return nonabelian_general(data,Efermi,['curv','morb'],mode='fermi-surface',factor=factor)

def Hall_spin(data,Efermi):
    # _general yields integral(Omrga*s*(-fo')) in units 1/(eV*Ang)
    # we want in S/(cm*T)
    # S/T=A^3*s^5/(kg^2*m^2))
    factor=-bohr_magneton/(elementary_charge*Ang_SI) # first, transform to SI - now in 1/(m*T) ,restoring the sign of spin magnetic moment
    factor*=-1 
    factor*=elementary_charge**2/hbar  # multiply by a dimensional factor - now in S/(T*m)
    factor*=1e-2   #  finally transform to S/(T*cm)
    return nonabelian_general(data,Efermi,['curv','spin'],mode='fermi-surface',factor=factor)


factor=elementary_charge**2*Ang_SI/hbar**3  # first, transform to SI, not forgeting hbar in velocities - now in  m/(J*s^3)
factor*=elementary_charge**3/hbar*TAU_UNIT**2  # multiply by a dimensional factor - now in A^3*s^5*cm/(J^2*tau_unit^2) = S/(T*m*tau_unit^2)
factor*=1e-2   #  finally transform to S/(T*cm*tau_unit^2)
#print ("factor_Hall_classic = {} ".format(factor))

def Hall_classic(data,Efermi):
    # _general yields integral(V*V*V'*(-f0')) in units eV^2*Ang
    # we want in S/(cm*T)/tau_unit^2
    # S/T=A^3*s^5/(kg^2*m^2))
    factor=elementary_charge**2*Ang_SI/hbar**3  # first, transform to SI, not forgeting hbar in velocities - now in  m/(J*s^3)
    factor*=elementary_charge**3/hbar*TAU_UNIT**2  # multiply by a dimensional factor - now in A^3*s^5*cm/(J^2*tau_unit^2) = S/(T*m*tau_unit^2)
    factor*=1e-2   #  finally transform to S/(T*cm*tau_unit^2)
    r1= nonabelian_general(data,Efermi,['vel','mass','vel'],mode='fermi-surface',factor=factor)
    print ("r1 - shape",r1.data.shape)
    print (alpha_A,beta_A)
    res=r1.data[:,:,:,beta_A,alpha_A]-r1.data[:,:,:,alpha_A,beta_A]
    res=-0.5*(res[:,alpha_A,beta_A,:]-res[:,beta_A,alpha_A,:])
#    print ("res - shape",res.shape)
    return result.EnergyResult(Efermi, res  ,TRodd=False,Iodd=False)


# this formulation is not correct in general
def Hall_classic_sea(data,Efermi):
    # _general yields integral(W*W*f0) in units eV^2*Ang
    # we want in S/(cm*T)/tau_unit^2
    # S/T=A^3*s^5/(kg^2*m^2))
    factor=elementary_charge**2*Ang_SI/hbar**3  # first, transform to SI, not forgeting hbar in velocities - now in  m/(J*s^3)
    factor*=elementary_charge**3/hbar*TAU_UNIT**2  # multiply by a dimensional factor - now in A^3*s^5*cm/(J^2*tau_unit^2) = S/(T*m*tau_unit^2)
    factor*=1e-2   #  finally transform to S/(T*cm*tau_unit^2)
    r1= nonabelian_general(data,Efermi,['mass','mass'],mode='fermi-sea',factor=factor)
#    print ("r1 - shape",r1.data.shape)
#    print (alpha_A,beta_A)
    res=r1.data.transpose((0,1,3,2,4))
    res=res[:,:,:,alpha_A,beta_A]-res[:,:,:,beta_A,alpha_A]
    res=-0.5*(res[:,alpha_A,beta_A,:]-res[:,beta_A,alpha_A,:])
#    print ("res - shape",res.shape)
    return result.EnergyResult(Efermi, res  ,TRodd=False,Iodd=False)

def conductivity_ohmic(data,Efermi):
    # _general yields integral(V*V*f0') in units eV/Ang
    # we want in S/(cm)/tau_unit
    factor=elementary_charge/Ang_SI/hbar**2  # first, transform to SI, not forgeting hbar in velocities - now in  1/(kg*m^3)
    factor*=elementary_charge**2*TAU_UNIT  # multiply by a dimensional factor - now in A^2*s^2/(kg*m^3*tau_unit) = S/(m*tau_unit)
    factor*=1e-2 # now in  S/(cm*tau_unit)
    res=nonabelian_general(data,Efermi,['vel','vel'],mode='fermi-surface',factor=factor)
#    print ("factor=",factor)
#    print ("res=",res.data.sum(axis=0))
    return res

# an equivalent fermi-sea formulation - to test the effective mass tensor
def conductivity_ohmic_sea(data,Efermi):
    # _general yields integral(V*V*f0') in units eV/Ang
    # we want in S/(cm)/tau_unit
    factor=elementary_charge/Ang_SI/hbar**2  # first, transform to SI, not forgeting hbar in velocities - now in  1/(kg*m^3)
    factor*=elementary_charge**2*TAU_UNIT  # multiply by a dimensional factor - now in A^2*s^2/(kg*m^3*tau_unit) = S/(m*tau_unit)
    factor*=1e-2 # now in  S/(cm*tau_unit)
    return nonabelian_general(data,Efermi,['mass'],mode='fermi-sea',factor=factor)



def nonabelian_general(data,Efermi,quantities,subscripts=None,mode='fermi-surface',factor=1):
    E_K=data.E_K

    dE=Efermi[1]-Efermi[0]
    Emax=Efermi[-1]+dE/2

    variables=vars(sys.modules[__name__])
    M=[variables["__"+Q](data) for Q in quantities]


    if subscripts is None:
        ind_cart="abcdefghijk"
        left=[]
        right=""
        for Q in quantities:
            d=__dimensions[Q]
            left.append(ind_cart[:d])
            right+=ind_cart[:d]
            ind_cart=ind_cart[d:]
    else:
        left,right=subscripts.split("->")
        left=left.split(",")
        for Q,l,q in zip(quantities,left,quantities):
            d=__dimensions[Q]
            if d!=len(left):
                raise RuntimeError("The number of subscripts in '{}' does not correspond to dimention '{}' of quantity '{}' ".format(l,d,q))


    ind_bands="lmnopqrstuvwxyz"[:len(quantities)]
    ind_bands+=ind_bands[0]
    einleft=[]
    for l in left:
        einleft.append(ind_bands[:2]+l)
        ind_bands=ind_bands[1:]

    einline=",".join(einleft)+"->"+right
#    print ("using line '{}' for einsum".format(einline))


    res=np.zeros(  (len(Efermi),)+(3,)*len(right)  )

    if mode=='fermi-surface':
      for ik in range(data.NKFFT_tot):
        indE=np.array(np.round( (data.E_K_degen[ik]-Efermi[0])/dE ),dtype=int )
        indEtrue= (0<=indE)*(indE<len(Efermi))
        for ib,ie,it  in zip(range(len(indE)),indE,indEtrue):
            if it:
#                print ("quantities: ",quantities,einline,[m[ik][ib] for m in M])
                res[ie]+=np.einsum(einline,*(m[ik][ib] for m in M)).real
      res=res/dE
    elif mode=='fermi-sea':
      for ik in range(data.NKFFT_tot):
        indE=np.array(np.round( (data.E_K_degen[ik]-Efermi[0])/dE ),dtype=int )
        indEtrue= (0<=indE)*(indE<len(Efermi))
        for ib,eav  in zip(range(len(indE)),data.E_K_degen[ik]):
            if eav<Emax:
                 res[eav<Efermi]+=np.einsum(einline,*(m[ik][ib] for m in M)).real
    else:
      raise ValueError('unknown mode in non-abelian: <{}>'.format(mode))
    res*=(factor/(data.NKFFT_tot*data.cell_volume))
#    print ("res=",res.sum(axis=0))
#    print ("data.cell_volume",data.cell_volume)
    return result.EnergyResult(Efermi,res,TRodd=odd_prod_TR(quantities),Iodd=odd_prod_INV(quantities))



def odd_prod_TR(quant):
   return odd_prod(quant,TRodd)

def odd_prod_INV(quant):
   return odd_prod(quant,INVodd)


def odd_prod(quant,odd):
    return  bool(sum( (q in odd) for q in quant )%2)
