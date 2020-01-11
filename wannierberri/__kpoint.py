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
# This is an auxilary class for the __evaluate.py  module

from time import time
import numpy as np
import lazy_property
from copy import copy

class  KpointBZ():

    def __init__(self,k=np.zeros(3),dk=np.ones(3),NKFFT=np.ones(3),factor=1,symgroup=None,refinement_level=0):
        self.k=np.copy(k)
        self.dk=np.copy(dk)    
        self.factor=factor
        self.res=None
        self.NKFFT=np.copy(NKFFT)
        self.symgroup=symgroup
        self.refinement_level=refinement_level

    def set_res(self,res):
        self.res=res # sum(sym.transform_data(res) for sym in self.symmetries)*np.prod(self.dk)
#        self.res_smooth=smooth(self.res)
        
    @lazy_property.LazyProperty
    def kp_fullBZ(self):
        return self.k/self.NKFFT


    @lazy_property.LazyProperty
    def star(self):
        if self.symgroup is None:
            return [self.k]
        else:
            return self.symgroup.star(self.k)

    def __str__(self):
        res="coord in rec.lattice = [ {0:10.6f}  , {1:10.6f} ,  {2:10.6f} ] ".format(self.k[0],self.k[1],self.k[2])
        
#        if not (self.star is None):
#            res+="star : \n"+"".join(" [ {0:10.6f}  , {1:10.6f} ,  {2:10.6f} ] \n".format(s[0],s[1],s[2]) for s in self.star)
        return res

    @property
    def _max(self):
        return self.res.max #np.max(self.res_smooth)
    
    @property
    def evaluated(self):
        return not (self.res is None)
        
    @property
    def check_evaluated(self):
        if not self.evaluated:
            raise RuntimeError("result for a k-point is called, which is not evaluated")
        
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
        if not (other.res is None):
            if not (self.res is None):
                raise RuntimeError("combining two k-points with calculated result should not happen")
            self.res=other.res

    def equiv(self,other):
        if self.refinement_level!=other.refinement_level: 
            return False
        dif=self.star[:,None,:]-other.star[None,:,:]
        if np.linalg.norm((dif-np.round(dif)),axis=2).min() < 1e-10:
            return True
        return False


    def fraction(self,ndiv):
        assert (ndiv.shape==(3,))
        kp=copy(self)
        kp.dk=self.dk/ndiv
        kp.factor=self.factor/np.prod(ndiv)
        kp.refinement_level=self.refinement_level+1
        return kp
        
    def divide(self,ndiv):
        assert (ndiv.shape==(3,))
        assert (np.all(ndiv>0))
        include_original= np.all( ndiv%2==1)
        
        k0=self.k
        dk_adpt=self.dk/ndiv
        adpt_shift=(-self.dk+dk_adpt)/2.
        newfac=self.factor/np.prod(ndiv)
        k_list_add=[KpointBZ(k=k0+adpt_shift+dk_adpt*np.array([x,y,z]),dk=dk_adpt,NKFFT=self.NKFFT,factor=newfac,symgroup=self.symgroup,refinement_level=self.refinement_level+1)
                                 for x in range(ndiv[0]) 
                                  for y in range(ndiv[1]) 
                                   for z in range(ndiv[2])
                            if not (include_original and np.all(np.array([x,y,z])*2+1==ndiv)) ]
        if include_original:
            self.fractor=newfac
            self.refinement_level+=1
            k_list_add.append(self.fraction(ndiv))
        else:
            self.factor=0  # the k-point is "dead" but can be used is starting calculation on a different grid grid"
            
            
        n=len(k_list_add)

        if not (self.symgroup is None):
            exclude_equiv_points(k_list_add)

        return k_list_add



def exclude_equiv_points(k_list,new_points=None):
    print ("Excluding symmetry-equivalent points")
    t0=time()
    cnt=0
    n=len(k_list)
#    print (n,new_points)
#    print (-1 if new_points is None else max(-1,n-1-new_points))
    for i in range(n-1,-1 if new_points is None else max(-1,n-1-new_points),-1):
#        print (i)
        for j in range(i-1,-1,-1):
            ki=k_list[i]
            kj=k_list[j]
            if ki.equiv(kj):
                if ki.equiv(kj):
                    kj.absorb(ki)
                    cnt+=1
                    del k_list[i]
                    break
    print ("Done. Excluded  {} points in {} sec".format(cnt,time()-t0))
    return cnt
