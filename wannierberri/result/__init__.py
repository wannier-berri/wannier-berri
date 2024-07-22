#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file 'LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
# ------------------------------------------------------------
#
#  The purpose of this module is to provide Result classes for
#  different types of  calculations.
#  child classes can be defined specifically in each module


from .result import Result
from .resultdict import ResultDict
from .energyresult import EnergyResult
from .tabresult import TABresult
from .kbandresult import KBandResult, K__Result
