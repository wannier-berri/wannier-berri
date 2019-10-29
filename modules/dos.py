#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# this file initially was  an adapted translation of         #
# the corresponding Fortran90 code from  Wannier 90 project  #
#                                                            #
# with significant modifications for better performance      #
#   it is now a lot different                                #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                                                            #
# The webpage of the Wannier90 code is www.wannier.org       #
# The Wannier90 code is hosted on GitHub:                    #
# https://github.com/wannier-developers/wannier90            #
#------------------------------------------------------------#
#                                                            #
#                   written  by                              #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#

import numpy as np
from scipy import constants as constants
from collections import Iterable

from utility import  print_my_name_start,print_my_name_end,voidsmoother
import parent


class DOSresult(parent.Result):

    def __init__(self,Efermi,cumDOS,cumDOSsmooth=None,smoother=None):
        assert (Efermi.shape[0]==cumDOS.shape[0])
        self.Efermi=Efermi
        self.cumDOS=cumDOS
        if not (cumDOSsmooth is None):
            self.cumDOSsmooth=cumDOSsmooth
        elif not (smoother is None):
            self.cumDOSsmooth=smoother(self.cumDOS)
        else:
             raise ValueError("either  DOSsmooth or smoother should be defined")

    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float) :
            return DOSresult(self.Efermi,self.cumDOS*other,self.cumDOSsmooth*other)
        else:
            raise TypeError("result can only be multilied by a number")
    
    def __add__(self,other):
        if other == 0:
            return self
        if np.linalg.norm(self.Efermi-other.Efermi)>1e-8:
            raise RuntimeError ("Adding results with different Fermi energies - not allowed")
        return DOSresult(self.Efermi,self.cumDOS+other.cumDOS,self.cumDOSsmooth+other.cumDOSsmooth)

    def write(self,name):
        DOS=np.zeros(self.cumDOSsmooth.shape)
        dE=self.Efermi[1]-self.Efermi[0]
        DOS[1:-1]=(self.cumDOSsmooth[2:]-self.cumDOSsmooth[:-2])/(2*dE)
        DOS[0]=(self.cumDOSsmooth[1]-self.cumDOSsmooth[0])/dE
        DOS[-1]=(self.cumDOSsmooth[-1]-self.cumDOSsmooth[-2])/dE
        open(name,"w").write(
           "    ".join("{0:^15s}".format(s) for s in ["EF","cumDOS","cumDOSsmooth","DOS"])+"\n"+
          "\n".join(
           "    ".join("{0:15.8f}".format(x) for x in line )
                      for line  in np.array([self.Efermi,self.cumDOS,self.cumDOSsmooth,DOS] ).T )
               +"\n")

    def transform(self,sym=None):
        return self

    def _maxval(self):
        return self.cumDOSsmooth.max()

    def _norm(self):
        return np.linalg.norm(self.cumDOSsmooth)

    def _normder(self):
        return np.linalg.norm(self.cumDOSsmooth[1:]-self.cumDOSsmooth[:-1])

    def max(self):
        return np.array([self._maxval(),self._norm(),self._normder()])


bohr= constants.physical_constants['Bohr radius'][0]/constants.angstrom
eV_au=constants.physical_constants['electron volt-hartree relationship'][0] 




def calcDOS(data,Efermi=None,smoother=voidsmoother):

    cumDOS=np.zeros(Efermi.shape,dtype=int)

    for e in data.E_K_only.reshape(-1):
        cumDOS[e<=Efermi]+=1

    cumDOS=np.array(cumDOS,dtype=float)/(data.NKFFT_tot)
    
    return DOSresult(Efermi,cumDOS,smoother=smoother )


