#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                     written by                             #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#


import numpy as np
import sys
from collections import Iterable
from . import __result as result
from time import time


def __spin(data):
    return data.SSUU_K

def __vel(data):
    return data.delHHUU_K


#quantities that should be odd under TRS and inversion
TRodd  = set(['spin','vel'])
INVodd = set(['vel'])


def spin(data,Efermi,degen_thresh):
    return calc_nonabelian(data,Efermi,['spin'],degen_thresh=degen_thresh)

def spinvel(data,Efermi,degen_thresh):
    return calc_nonabelian(data,Efermi,['spin','vel'],degen_thresh=degen_thresh)


def spinspin(data,Efermi,degen_thresh):
    return calc_nonabelian(data,Efermi,['spin','spin'],degen_thresh=degen_thresh)

def calc_nonabelian(data,Efermi,quantities,subscripts=None,degen_thresh=1e-5):
    t0=time()
    E_K=data.E_K
    variables=vars(sys.modules[__name__])
    M0=[variables["__"+Q](data) for Q in quantities]
    t1=time()

    A=[ [0,] +list(np.where(E[1:]-E[:1]>degen_thresh)[0]+1)+ [E.shape[0],]  for E in E_K ]
    t1a=time()
    deg= [[(ib1,ib2) for ib1,ib2 in zip(a,a[1:])] for a in A]
    t1b=time()
    Eav= [ np.array( [E[b1:b2].mean() for b1,b2 in deg  ]) for E,deg in zip(E_K,deg)]
    t1c=time()
    M=[ [ np.array( [M2[b1:b2,b1:b2] for b1,b2 in deg  ]) for M2,deg in zip(M1,deg)] for M1 in M0]
    t2=time()


    if subscripts is None:
        ind_cart="abcdefghijk"
        left=[]
        right=""
        for m in M:
            d=len(m[0][0].shape)-2
            left.append(ind_cart[:d])
            right+=ind_cart[:d]
            ind_cart=ind_cart[d:]
    else:
        left,right=subscripts.split("->")
        left=left.split(",")
        for m,l,q in zip(M,left,quantities):
            d=len(m[0][0].shape())-2
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

    dE=Efermi[1]-Efermi[0]

    res=np.zeros(  (len(Efermi),)+(3,)*len(right)  )

    for ik in range(data.NKFFT_tot):
#        print ("shapes are:",(m[ik].shape for m in M)
        indE=np.array(np.round( (Eav[ik]-Efermi[0])/dE ),dtype=int )
        indEtrue= (0<=indE)*(indE<len(Efermi))
        for ib,ie,it  in zip(range(len(indE)),indE,indEtrue):
            if it:
                res[ie]+=np.einsum(einline,*(m[ik][ib] for m in M)).real
    t3=time()
    print ("times in  calc_nonabelian ",t1-t0,t1a-t1,t1b-t1a,t1c-t1b,t2-t1c,t3-t2," tot: ",t3-t0)
    return result.EnergyResult(Efermi,res/data.NKFFT_tot,TRodd=odd_prod_TR(quantities),Iodd=odd_prod_INV(quantities))



def odd_prod_TR(quant):
   return odd_prod(quant,TRodd)

def odd_prod_INV(quant):
   return odd_prod(quant,INVodd)


def odd_prod(quant,odd):
    return  bool(sum( (q in odd) for q in quant )%2)
