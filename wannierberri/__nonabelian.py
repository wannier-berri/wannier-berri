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
from .__utility import alpha_A,beta_A

def __spin(data):
    return data.spin_nonabelian

def __vel(data):
    return data.vel_nonabelian


def __curvE(data):
    return [[o*e for o,e in zip(O,E)]
        for O,E in zip(data.Berry_nonabelian,data.E_K_degen)]

def __curv(data):
    return data.Berry_nonabelian


def __morb(data):
    return data.Morb_nonabelian

def __morbg(data):
    return data.Morb_nonabelian_g

def __morb2(data):
    return data.Morb_nonabelian_2

__dimensions=defaultdict(lambda : 1)

#quantities that should be odd under TRS and inversion
TRodd  = set(['spin','morb','vel','curv','curvE','morbg','morb2'])
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



def Morb_loc(data,Efermi):
    return  (  nonabelian_general(data,Efermi,['morb' ],mode='fermi-sea') 
                *__berry.fac_morb*data.cell_volume)

def Morb_loc2(data,Efermi):
    return  (  nonabelian_general(data,Efermi,['morb2' ],mode='fermi-sea') 
                *__berry.fac_morb*data.cell_volume)


def Morb(data,Efermi):
    r1=nonabelian_general(data,Efermi,['morbg' ],mode='fermi-sea')
    r2=nonabelian_general(data,Efermi,['curvE'],mode='fermi-sea')
    r3=nonabelian_general(data,Efermi,['curv' ],mode='fermi-sea')
    r3.data[:,:]=r3.data[:,:]*Efermi[:,None]
    return (r1+r2-2*r3)*__berry.fac_morb*data.cell_volume

def Morb2(data,Efermi):
    r1=nonabelian_general(data,Efermi,['morb' ],mode='fermi-sea')
    r2=nonabelian_general(data,Efermi,['curvE'],mode='fermi-sea')
    r3=nonabelian_general(data,Efermi,['curv' ],mode='fermi-sea')
    r3.data[:,:]=r3.data[:,:]*Efermi[:,None]
    return (r1+2*r2-2*r3)*__berry.fac_morb*data.cell_volume


def Morb_IC(data,Efermi):
    r2=nonabelian_general(data,Efermi,['curvE'],mode='fermi-sea')
    r3=nonabelian_general(data,Efermi,['curv' ],mode='fermi-sea')
    print (r3.data)
    r3.data[:,:]=-r3.data[:,:]*Efermi[:,None]
    print (r3.data)
    return (r2+r3)*__berry.fac_morb*data.cell_volume

def Morb_h(data,Efermi):
    r2=nonabelian_general(data,Efermi,['curvE'],mode='fermi-sea')
#    r3=nonabelian_general(data,Efermi,['curv' ],mode='fermi-sea')
#    r3.data[:,:]=-r3.data[:,:]*Efermi[:,None]
    return r2*__berry.fac_morb*data.cell_volume

def Morb_f(data,Efermi):
#    r2=nonabelian_general(data,Efermi,['curvE'],mode='fermi-sea')
    r3=nonabelian_general(data,Efermi,['curv' ],mode='fermi-sea')
    r3.data[:,:]=-r3.data[:,:]*Efermi[:,None]
    return r3*__berry.fac_morb*data.cell_volume



def Morb_LC(data,Efermi):
    r1=nonabelian_general(data,Efermi,['morbg'],mode='fermi-sea')
#    r2=nonabelian_general(data,Efermi,['curvE'],mode='fermi-sea')
    r3=nonabelian_general(data,Efermi,['curv'],mode='fermi-sea')
    r3.data[:,:]=-r3.data[:,:]*Efermi[:,None]
    return (r1+r3)*__berry.fac_morb*data.cell_volume


             

    res*=__berry.fac_morb*data.cell_volume

    return result.EnergyResultAxialV(Efermi,res/(data.NKFFT_tot*data.cell_volume))
    


#def Morb_IC(data,Efermi):
#    r1=nonabelian_general(data,Efermi,['morb'],mode='fermi-sea',factor=__berry.fac_morb*data.cell_volume)
#    return r1

def nonabelian_general(data,Efermi,quantities,subscripts=None,mode='fermi-surface',factor=1):
    E_K=data.E_K

    dE=Efermi[1]-Efermi[0]
#    Emin=Efermi[0]-dE/2
    Emax=Efermi[-1]+dE/2
#    include_lower=(mode=='fermi-sea')
#    data.set_degen(Emin=Emin,Emax=Emax)
#    data.set_degen(degen_thresh)

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

    return result.EnergyResult(Efermi,res*(factor/(data.NKFFT_tot*data.cell_volume)),TRodd=odd_prod_TR(quantities),Iodd=odd_prod_INV(quantities))



def odd_prod_TR(quant):
   return odd_prod(quant,TRodd)

def odd_prod_INV(quant):
   return odd_prod(quant,INVodd)


def odd_prod(quant,odd):
    return  bool(sum( (q in odd) for q in quant )%2)
