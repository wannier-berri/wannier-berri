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
"""
wannierberri - a module for Wannier interpolation
"""

import warnings
__version__ = "1.4.2"

from .run import run
from .symmetry import point_symmetry
from . import system
from . import models
from . import w90files
from .system import (System_w90, System_fplo, System_tb,
                     System_PythTB, System_TBmodels, System_ASE,
                     System_Phonon_QE, System_R
                     )
from .grid import Grid, Path
from . import calculators
from . import result
from . import wannierise
from .parallel import Parallel, Serial
from .smoother import get_smoother
from .evaluate_k import evaluate_k, evaluate_k_path
from . import utils
# from . import data_K
from .result.tabresult import npz_to_fermisurfer
from termcolor import cprint


from packaging import version
IRREP_IRREDUCIBLE_VERSION = version.parse("2.2.0")  # the version of irrep that supports irreducible band structure


def welcome():
    # originally obtained by pyfiglet, font='cosmic'
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

    cprint(f"\nVersion: {__version__}\n", 'cyan', attrs=['bold'])
