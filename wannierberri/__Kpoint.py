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
# This is an auxilary class for the __evaluate.py  module

from time import time
import numpy as np
import lazy_property
from copy import copy,deepcopy
from .symmetry import SYMMETRY_PRECISION

class  KpointBZ():

    def __init__(self,K=np.zeros(3),dK=np.ones(3),NKFFT=np.ones(3),factor=1.,symgroup=None,refinement_level=-1):
        self.K=np.copy(K)
        self.dK=np.copy(dK)    
        self.factor=factor
        self.res=None
        self.NKFFT=np.copy(NKFFT)
        self.symgroup=symgroup
        self.refinement_level=refinement_level

    def set_res(self,res):
        self.res=res 
        
    @lazy_property.LazyProperty
    def Kp_fullBZ(self):
        return self.K/self.NKFFT

    @lazy_property.LazyProperty
    def dK_fullBZ(self):
        return self.dK/self.NKFFT

    @lazy_property.LazyProperty
    def star(self):
        if self.symgroup is None:
            return [self.K]
        else:
            return self.symgroup.star(self.K)

    def __str__(self):
        k_cart=self.K.dot(self.symgroup.recip_lattice)
        return  ( "coord in rec.lattice = [ {0:10.6f}  , {1:10.6f} ,  {2:10.6f} ], refinement level:{3}, dK={4} ".format(self.K[0],self.K[1],self.K[2],self.refinement_level,self.dK) )  
#                +   "         coord in cartesian = [ {0:10.6f}  , {1:10.6f} ,  {2:10.6f} ]".format(k_cart[0],k_cart[1],k_cart[2]) + "\n star = "+"\n      ".join(str(s) for s in self.star) )

    @property
    def _max(self):
        return self.res.max #np.max(self.res_smooth)
    
    @property
    def evaluated(self):
        return not (self.res is None)
        
    @property
    def check_evaluated(self):
        if not self.evaluated:
            raise RuntimeError("result for a K-point is called, which is not evaluated")
        
    @property
    def max(self):
        self.check_evaluated
        return self._max*self.factor

    @property
    def norm(self):
        self.check_evaluated
        return self._norm*self.factor

    @property
    def normder(self):
        self.check_evaluated
        return self._normder*self.factor

    @property
    def get_res(self):
        self.check_evaluated
        return self.res*self.factor
#        return np.hstack((self.res,self.res_smooth))*self.factor


    def absorb(self,other):
        self.factor+=other.factor
        if other.res is not None:
            if self.res is not None:
                raise RuntimeError("combining two K-points :\n {} \n and\n  {}\n  with calculated result should not happen".format(self,other))
            self.res=other.res

    def equiv(self,other):
        if self.refinement_level!=other.refinement_level: 
            return False
        dif=self.star[:,None,:]-other.star[None,:,:]
        res=False
        if np.linalg.norm((dif-np.round(dif)),axis=2).min() < SYMMETRY_PRECISION :
            res=True
#        print(str(self))
#        print(str(other))
#        print ("dif=\n {} \n equiv={}".format(dif,res))
        return res


        
    def divide(self,ndiv):
        assert (ndiv.shape==(3,))
        assert (np.all(ndiv>0))
        include_original= np.all( ndiv%2==1)
        
        K0=self.K
        dK_adpt=self.dK/ndiv
        adpt_shift=(-self.dK+dK_adpt)/2.
        newfac=self.factor/np.prod(ndiv)
        K_list_add=[KpointBZ(K=K0+adpt_shift+dK_adpt*np.array([x,y,z]),dK=dK_adpt,NKFFT=self.NKFFT,factor=newfac,symgroup=self.symgroup,refinement_level=self.refinement_level+1)
                                 for x in range(ndiv[0]) 
                                  for y in range(ndiv[1]) 
                                   for z in range(ndiv[2])
                            if not (include_original and np.all(np.array([x,y,z])*2+1==ndiv)) ]
#        print ("ndiv={}, include_original={} ".format(ndiv,include_original))

        if include_original:
            self.factor=newfac
            self.refinement_level+=1
            self.dK=dK_adpt

#            K_list_add.append(self.fraction(ndiv))
        else:
            self.factor=0  # the K-point is "dead" but can be used for starting calculation on a different grid  - not implemented

#        print ("ndiv={}, include_original={} ".format(ndiv,include_original))

        n=len(K_list_add)

        if not (self.symgroup is None):
            exclude_equiv_points(K_list_add)

#        print ("dividing {} into : \n".format(self)+"\n".join(str(K) for K in K_list_add))

        return K_list_add


    @lazy_property.LazyProperty
    def distGamma(self):
        shift_corners=np.arange(-3,4)
        corners=np.array([[x,y,z] for x in shift_corners for y in shift_corners for z in shift_corners])
        return np.linalg.norm(((self.K%1)[None,:]-corners).dot(self.symgroup.recip_lattice),axis=1).min() 


def exclude_equiv_points(K_list,new_points=None):
    print ("Excluding symmetry-equivalent K-points")
    t0=time()
    cnt=0
    n=len(K_list)
    
    K_list_length=np.array([ K.distGamma  for K in K_list])
    K_list_sort=np.argsort(K_list_length)
    K_list_length=K_list_length[K_list_sort]
    wall=[0]+list(np.where(K_list_length[1:]-K_list_length[:-1]>1e-4)[0]+1)+[len(K_list)]

    exclude=[]

    for start,end in zip(wall[:-1],wall[1:]):
        for l in range(start,end):
            i=K_list_sort[l]
            if i not in exclude:
                for m in range(l+1,end):
                    j=K_list_sort[m]
                    if new_points is not None:
                          if i<n-new_points and  j<n-new_points:
                              continue 
                    if not j in exclude:
                        if K_list[i].equiv(K_list[j]):
                             exclude.append(j)
                             K_list[i].absorb(K_list[j])
    for i in sorted(exclude)[-1::-1]:
        del K_list[i]
    print ("Done. Excluded  {} K-points in {} sec".format(len(exclude),time()-t0))
    return cnt
