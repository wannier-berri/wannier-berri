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
# ------------------------------------------------------------
"""
wannierberri - a module for Wannier interpolation
"""

__version__ = "0.15.0"

try:
    import pyfftw
    PYFFTW_IMPORTED = True
except Exception as err:
    PYFFTW_IMPORTED = False
    print("WARNING : error importing  `pyfftw` : {} \n will use numpy instead \n".format(err))


from .run import run
from . import symmetry
from . import system
from .system import System_w90, System_fplo, System_tb, System_PythTB, System_TBmodels, System_ASE, System_Phonon_QE
from .grid import Grid, Path
from . import calculators
from . import result
from .parallel import Parallel, Serial
from .smoother import get_smoother
from .evaluate_k import evaluate_k
from . import utils
from . import data_K

from termcolor import cprint


def welcome():
    # ogiginally obtained by pyfiglet, font='cosmic'
    # with small modifications
    logo = """
.::    .   .::: .:::::::.  :::.    :::.:::.    :::. :::.,::::::  :::::::..       :::::::.  .,::::::  :::::::..   :::::::..   :::
';;,  ;;  ;;;' '  ;;`;;  ` `;;;;,  `;;;`;;;;,  `;;; ;;;;;;;''''  ;;;;``;;;;       ;;;'';;' ;;;;''''  ;;;;``;;;;  ;;;;``;;;;  ;;;
 '[[, [[, [['    ,[[ '[[,    [[[[[. '[[  [[[[[. '[[ [[[ [[cccc    [[[,/[[['       [[[__[[\\. [[cccc    [[[,/[[['   [[[,/[[['  [[[
   Y$c$$$c$P    c$$$cc$$$c   $$$ "Y$c$$  $$$ "Y$c$$ $$$ $$\"\"\"\"    $$$$$$c         $$\"\"\"\"Y$$ $$\"\"\"\"    $$$$$$c     $$$$$$c    $$$
    "88"888      888   888,  888    Y88  888    Y88 888 888oo,__  888b "88bo,    _88o,,od8P 888oo,__  888b "88bo, 888b "88bo,888
     "M "M"      YMM   ""`   MMM     YM  MMM     YM MMM \"\"\"\"YUMMM MMMM   "W"     ""YUMMMP"  \"\"\"\"YUMMM MMMM   "W"  MMMM   "W" MMM
"""
    cprint(logo, 'yellow')

    cprint("""\n  The Web page is :  HTTP://WANNIER-BERRI.ORG  \n""", 'yellow')

    cprint("\nVersion: {}\n".format(__version__), 'cyan', attrs=['bold'])
