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
#
#  The purpose of this module is to provide Result classes for
#  different types of  calculations.
#  child classes can be defined specifically in each module

import numpy as np
from lazy_property import LazyProperty as Lazy


# A class to contain results or a calculation:
# For any calculation there should be a class with the samemethods implemented

from .__dict import  ResultDict
from .__energyresult import EnergyResult
from .__tabresult import TABResult
from .__kbandresult import KBandResult
