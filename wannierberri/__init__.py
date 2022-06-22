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
"""
wannierberri - a module for Wannier interpolation
"""

__version__ = "0.13.4"

from .run import run
from .__old_API.__main import integrate, tabulate #, integrate_options, tabulate_options, print_options
from . import symmetry
from . import system
from .system import System_w90, System_fplo, System_tb, System_PythTB, System_TBmodels, System_ASE
from .__grid import Grid
from .__path import Path
from . import calculators
from . import result
from .parallel import Parallel, Serial
from .smoother import get_smoother

from termcolor import cprint


def figlet(text, font='cosmike', col='red'):
    init(strip=not sys.stdout.isatty())  # strip colors if stdout is redirected
    letters = [figlet_format(X, font=font).rstrip("\n").split("\n") for X in text]
    logo = []
    for i in range(len(letters[0])):
        logo.append("".join(L[i] for L in letters))
    cprint("\n".join(logo), col, attrs=['bold'])

def welcome():
    # ogiginally obtained by
    # figlet("WANN IER BERRI",font='cosmic',col='yellow')
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
