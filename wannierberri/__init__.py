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
__version__ = "1.8.0"

from .run import run
from .symmetry import point_symmetry
from . import system
from . import models
from . import w90files
from .parallel import ray_init, ray_shutdown
from .system import (System_w90, System_fplo, System_tb,
                     System_PythTB, System_TBmodels, System_ASE,
                     System_Phonon_QE, System_R
                     )
from .grid import Grid, Path
from . import calculators
from . import result
from . import wannierise
from .evaluate_k import evaluate_k, evaluate_k_path
from .result.tabresult import npz_to_fermisurfer
from termcolor import cprint


# from packaging import version
# import irrep
# assert version.parse(irrep.__version__) >= version.parse("2.3.2"), \
#     f"irrep version >= 2.3.2 is required, found {irrep.__version__}"


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
    cprint(logo, 'green')
    cprint(f"Version: {__version__}\n", 'cyan', attrs=['bold'])
    cprint("""\n   HTTP://WANNIER-BERRI.ORG  \n""", 'yellow')

    print ("Checking dependencies …")
    not_found = []
    symmetry = "symmetry-related functionality (SAWF, symmetrization, projections, …)"
    needed_for = {
        "irrep": "symmetry related ",
        "spglib": symmetry,
        "numpy": "ESSENTIAL",
        "scipy": "ESSENTIAL",
        "spgrep": "projections searcher",
        "numba": "tetrahedron integration",
        "pyfftw": "fast Fourier transforms (optional, otherwise uses numpy's FFT)",
        "seekpath": "automatic generation of k-point paths",
        "matplotlib": "plotting",
        "sympy": symmetry,
        "fortio": "reading Fortran unformatted files (uHu, chk, spn, unk, …)",
        "gpaw"  : "interface with GPAW",
        "ase" : "interface with ASE and GPAW",
        "pythtb" : "interface with PythTB (optional)",
        "xmltodict" : "reading QuantumEspresso dynamical matrices for phonons",
    }
    for package in needed_for.keys():
        try:
            mod = __import__(package)
            cprint(f"{package} : {mod.__version__}", 'cyan')
        except ImportError:
            nfor = needed_for.get(package, "No description available")
            if nfor == "ESSENTIAL":
                cprint(f"{package} : not found. {nfor}. Please install it to use wannierberri.", 'red')
            else:
                cprint(f"{package} : not found. {nfor}", 'yellow')
            not_found.append(package)

welcome()