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
from scipy import constants as constants
from collections import Iterable

from .__utility import  print_my_name_start,print_my_name_end
from . import __result as result
from .__berry import calcImf_band,calcImgh_band


def calcSpin_band(data):
    return data.SSUU_K_rediag


def calcSpin_band_kn(data):
    return result.KBandResult(data=calcSpin_band(data),TRodd=True,Iodd=False)

def calcHall_spin_kn(data):
    imf=calcImf_band(data)
    spn=calcSpin_band(data)
    return result.KBandResult(data=imf[:,:,:,None]*spn[:,:,None,:],TRodd=False,Iodd=False)

def calcHall_orb_kn(data):
    imf=calcImf_band(data)
    orb=calcImgh_band(data)
    return result.KBandResult(data=imf[:,:,:,None]*orb[:,:,None,:],TRodd=False,Iodd=False)
