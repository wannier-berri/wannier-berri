#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                                                            #
#------------------------------------------------------------#
#                                                            #
#  written  by                                               #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#

import numpy as np
from scipy import constants as constants
from collections import Iterable

from utility import  print_my_name_start,print_my_name_end,voidsmoother
import parent
from copy import deepcopy


class TABresult(parent.Result):

    def __init__(self,kpoints,basis,Enk=None,berry=None):
        self.nband=None

        self.basis=basis
        self.kpoints=list(deepcopy(kpoints))
        if berry is not None:
            assert len(kpoints)==berry.shape[0]
            self.berry=berry
            self.nband=berry.shape[1]
        else: 
            self.berry=None

        if Enk is not None:
            assert len(kpoints)==Enk.shape[0]
            self.Enk=Enk
            if self.nband is not None:
                assert(self.nband==Enk.shape[1])
            else:
                self.nband=Enk.shape[1]
        else: 
            self.Enk=None

        
    def __mul__(self,other):
        return self
    
    def __add__(self,other):
        if other == 0:
            return self
        if self.nband!=other.nband:
            raise RuntimeError ("Adding results with different number of bands {} and {} - not allowed".format(
                self.nband,other.nband) )

        if not ((self.berry is None) or (other.berry is None)):
            Berry=np.vstack( (self.berry,other.berry) )
        else:
            Berry=None

        if not ((self.Enk is None) or (other.Enk is None)):
            Enk=np.vstack( (self.Enk,other.Enk) )
        else:
            Enk=None

        return TABresult(self.kpoints+other.kpoints, basis=self.basis,berry=Berry,Enk=Enk) 


    def write(self,name):
        return   # do nothing so far
        raise NotImplementedError()
        open(name,"w").write(
           "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [b for b in ("x","y","z")*2])+"\n"+
          "\n".join(
           "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for x in ahc]) 
                      for ef,ahc in zip (self.Efermi,np.hstack( (self.AHC,self.AHCsmooth) ) ))
               +"\n") 

    def transform(self,sym):
        kpoints=[sym.transform_k_vector(k,self.basis) for k in self.kpoints]
        berry=None if self.berry is None else sym.transform_axial_vector(self.berry)
        return TABresult(kpoints=kpoints,basis=self.basis,berry=berry,Enk=self.Enk)


    def max(self):
        if self.berry is not None:
            return [self.berry.max()]
        else:
            return [0]



def tabEnk(data):

    Enk=data.E_K_only
    dkx,dky,dkz=1./data.NKFFT
    kpoints=[data.dk+np.array([ix*dkx,iy*dky,iz*dkz]) 
          for ix in range(data.NKFFT[0])
              for iy in range(data.NKFFT[1])
                  for  iz in range(data.NKFFT[2])]

    return TABresult(kpoints=kpoints,Enk=Enk,basis=data.recip_lattice )






