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

from .__utility import  print_my_name_start,print_my_name_end,voidsmoother
from . import __result as result



bohr= constants.physical_constants['Bohr radius'][0]/constants.angstrom
eV_au=constants.physical_constants['electron volt-hartree relationship'][0] 




def calc_cum_DOS(data,Efermi=None,smoother=voidsmoother):

    cumDOS=np.zeros(Efermi.shape,dtype=int)

    for e in data.E_K.reshape(-1):
        cumDOS[e<=Efermi]+=1

    cumDOS=np.array(cumDOS,dtype=float)/(data.NKFFT_tot)
    
    return result.EnergyResultScalar(Efermi,cumDOS,smoother=smoother )


def calc_DOS(data,Efermi=None,smoother=voidsmoother):

    DOS=np.zeros(Efermi.shape,dtype=int)
    E=data.E_K.reshape(-1)
    dE=Efermi[1]-Efermi[0]
    indE=np.array(np.round( (E-Efermi[0])/dE ),dtype=int )
    for i in indE[ (0<=indE)*(indE<len(Efermi)) ]:
        DOS[i]+=1

    DOS=DOS/(dE*data.NKFFT_tot)

    return result.EnergyResultScalar(Efermi,DOS,smoother=smoother )


