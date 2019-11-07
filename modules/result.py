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
#
#  The purpose of this module is to provide Result classes for  
#  different types of  calculations. 
#  child classes can be defined specifically in each module

import numpy as np


## A class to contain results or a calculation:
## For any calculation there should be a class with the samemethods implemented

class Result():

    def __init__(self):
        raise NotImprementedError()

#  multiplication by a number 
    def __mul__(self,other):
        raise NotImprementedError()

# +
    def __add__(self,other):
        raise NotImprementedError()

# writing to a file
    def write(self,name):
        raise NotImprementedError()

#  how result transforms under symmetry operations
    def transform(self,sym):
        raise NotImprementedError()

# a list of numbers, by each of those the refinement points will be selected
    def max(self):
        raise NotImprementedError()


### these methods do no need re-implementation: 
    def __rmul__(self,other):
        return self*other

    def __radd__(self,other):
        return self+other
        
    def __truediv__(self,number):
        return self*(1./number)


# a class for data defined for a set of Fermi levels
#Data is stored in an array data, where first dimension indexes the Fermi level

class EfermiResult(Result):

    def __init__(self,Efermi,data,dataSmooth=None,smoother=None):
        assert (Efermi.shape[0]==data.shape[0])
        self.Efermi=Efermi
        self.data=data
        if not (dataSmooth is None):
            self.dataSmooth=dataSmooth
        elif not (smoother is None):
            self.dataSmooth=smoother(self.data)
        else:
             raise ValueError("either  dataSmooth or smoother should be defined")

    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float) :
            return EfermiResult(self.Efermi,self.data*other,self.dataSmooth*other)
        else:
            raise TypeError("result can only be multilied by a number")

    def __add__(self,other):
        if other == 0:
            return self
        if np.linalg.norm(self.Efermi-other.Efermi)>1e-8:
            raise RuntimeError ("Adding results with different Fermi energies - not allowed")
        return EfermiResult(self.Efermi,self.data+other.data,self.dataSmooth+other.dataSmooth)

    def write(self,name):
        # assule, that the dimensions starting from first - are cartesian coordinates       
        def getHead(n):
           if n<=0:
              return ['  ']
           else:
              return [a+b for a in 'xyz' for b in getHead(n-1)]
        rank=len(self.data.shape[1:])

        open(name,"w").write(
           "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [b for b in getHead(rank)*2])+"\n"+
          "\n".join(
           "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for x in data.reshape(-1)]+[x for x in datasm.reshape(-1)]) 
                      for ef,data,datasm in zip (self.Efermi,self.data,self.dataSmooth)  )
               +"\n") 


    def _maxval(self):
        return self.dataSmooth.max() 

    def _norm(self):
        return np.linalg.norm(self.dataSmooth)

    def _normder(self):
        return np.linalg.norm(self.dataSmooth[1:]-self.dataSmooth[:-1])

    def max(self):
        return np.array([self._maxval(),self._norm(),self._normder()])



class ScalarResult(EfermiResult):
    def transform(self,sym):
        return self 

class AxialVectorResult(EfermiResult):
    def transform(self,sym):
        return AxialVectorResult(self.Efermi,sym.transform_axial_vector(self.data),sym.transform_axial_vector(self.dataSmooth) )

class PolarVectorResult(EfermiResult):
    def transform(self,sym):
        return PolarVectorResult(self.Efermi,sym.transform_polar_vector(self.data),sym.transform_polar_vector(self.dataSmooth) )


#a more general class. Scalar,polar and axial vectors may be derived as particular cases of the tensor class
class TensorResult(EfermiResult):

    def __init__(self,Efermi,data,dataSmooth=None,smoother=None,trueVector=[]):
        shape=data.shape[1:]
        assert  len(shape)==len(trueVector)
        assert np.all(np.array(shape)==3)
        super(TensorResult,self).__init__(Efermi,data,dataSmooth=dataSmooth,smoother=smoother)
        self.trueVector=np.copy(trueVector)
 
    def __transform(self,data,sym):
        res=np.copy(data)
        rank=len(self.trueVector)
        for i,TV in enumerate(self.trueVector):
            trans=sym.transform_polar_vector if TV else sym.transform_axial_vector
            res=trans(res.transpose( (0,)+tuple(range(1,i+1))+tuple(range(i+2,rank+1))+(i+1,) ) ).transpose( 
                    (0,)+tuple(range(1,i+1))+(rank,)+tuple(range(i+1,rank))  )
        return res

    def transform(self,sym):
        return TensorResult(self.Efermi,self.__transform(self.data,sym),self.__transform(self.dataSmooth,sym),trueVector=self.trueVector)


    def __mul__(self,other):
        res=super(TensorResult,self).__mul__(other)
        return TensorResult(res.Efermi,res.data, res.dataSmooth ,trueVector=self.trueVector)
        res.trueVector=np.copy(sef.trueVector)
        return res

    def __add__(self,other):
        assert np.all(self.trueVector==other.trueVector)
        res=super(TensorResult,self).__add__(other)
        return TensorResult(res.Efermi,res.data, res.dataSmooth ,trueVector=self.trueVector)
