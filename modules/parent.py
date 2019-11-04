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
#  The purpose of this module is to provide some Parent classes 
#  common for the calculations. 
#  child classes will be defined specifically in each module

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



class AxialVectorResult(Result):

    def __init__(self,Efermi,AHC,AHCsmooth=None,smoother=None):
        assert (Efermi.shape[0]==AHC.shape[0])
        assert (AHC.shape[-1]==3)
        self.Efermi=Efermi
        self.AHC=AHC
        if not (AHCsmooth is None):
            self.AHCsmooth=AHCsmooth
        elif not (smoother is None):
            self.AHCsmooth=smoother(self.AHC)
        else:
             raise ValueError("either  AHCsmooth or smoother should be defined")

    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float) :
            return AxialVectorResult(self.Efermi,self.AHC*other,self.AHCsmooth*other)
        else:
            raise TypeError("result can only be multilied by a number")

    def __add__(self,other):
        if other == 0:
            return self
        if np.linalg.norm(self.Efermi-other.Efermi)>1e-8:
            raise RuntimeError ("Adding results with different Fermi energies - not allowed")
        return AxialVectorResult(self.Efermi,self.AHC+other.AHC,self.AHCsmooth+other.AHCsmooth)

    def write(self,name):
        open(name,"w").write(
           "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [b for b in ("x","y","z")*2])+"\n"+
          "\n".join(
           "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for x in ahc]) 
                      for ef,ahc in zip (self.Efermi,np.hstack( (self.AHC,self.AHCsmooth) ) ))
               +"\n") 

    def transform(self,sym):
        return AxialVectorResult(self.Efermi,sym.transform_axial_vector(self.AHC),sym.transform_axial_vector(self.AHCsmooth) )

    def _maxval(self):
        return self.AHCsmooth.max() 

    def _norm(self):
        return np.linalg.norm(self.AHCsmooth)

    def _normder(self):
        return np.linalg.norm(self.AHCsmooth[1:]-self.AHCsmooth[:-1])

    def max(self):
        return np.array([self._maxval(),self._norm(),self._normder()])
