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

    def set_res(self,res,smooth):
        self.res=res # sum(sym.transform_data(res) for sym in self.symmetries)*np.prod(self.dk)
        self.res_smooth=smooth(self.res)
        
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
        res="coord in rec.lattice = [ {0:10.6f}  , {1:10.6f} ,  {2:10.6f} ] \n".format(self.k[0],self.k[1],self.k[2])
        
        if not (self.star is None):
            res+="star : \n"+"".join(" [ {0:10.6f}  , {1:10.6f} ,  {2:10.6f} ] \n".format(s[0],s[1],s[2]) for s in self.star)
        return res

    @lazy_property.LazyProperty
    def _max(self):
        return np.max(self.res_smooth)

    @lazy_property.LazyProperty
    def _norm(self):
        return np.linalg.norm(self.res_smooth)

    @lazy_property.LazyProperty
    def _normder(self):
        return np.linalg.norm(self.res_smooth[1:]-self.res_smooth[:-1])
    
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
        return np.hstack((self.res,self.res_smooth))*self.factor


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
            k_list_add.append(self.fraction(ndiv))
        n=len(k_list_add)

        if not (self.symgroup is None):
            exclude_equiv_points(k_list_add)

        return k_list_add



def exclude_equiv_points(k_list):
    cnt=0
    n=len(k_list)
    for i in range(n-1,-1,-1):
        for j in range(i-1,-1,-1):
            ki=k_list[i]
            kj=k_list[j]
            if ki.equiv(kj):
                if ki.equiv(kj):
                    kj.absorb(ki)
                    cnt+=1
                    del k_list[i]
                    break
    return cnt
