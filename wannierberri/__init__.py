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
from importlib.metadata import version as get_package_version, PackageNotFoundError
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
_symmetry_str = "symmetry-related functionality (SAWF, symmetrization, projections, …)"
_needed_packages = {
    "irrep": "symmetry related ",
    "spglib": _symmetry_str,
    "numpy": "ESSENTIAL",
    "scipy": "ESSENTIAL",
    "spgrep": "projections searcher",
    "numba": "tetrahedron integration",
    "pyfftw": "fast Fourier transforms (optional, otherwise uses numpy's FFT)",
    "seekpath": "automatic generation of k-point paths",
    "matplotlib": "plotting",
    "sympy": _symmetry_str,
    "fortio": "reading Fortran unformatted files (uHu, chk, spn, unk, …)",
    "gpaw": "interface with GPAW",
    "ase": "interface with ASE and GPAW",
    "pythtb": "interface with PythTB (optional)",
    "xmltodict": "reading QuantumEspresso dynamical matrices for phonons",
}


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

    print("Checking dependencies …")
    versions = {}
    for package in _needed_packages.keys():
        try:
            mod = __import__(package)
            # Try module's __version__ first, then fall back to package metadata
            version = getattr(mod, '__version__', None)
            if version is None:
                try:
                    version = get_package_version(package)
                except PackageNotFoundError:
                    version = 'unknown'
            cprint(f"{package} : {version}", 'cyan')
            versions[package] = version
        except ImportError:
            nfor = _needed_packages.get(package, "No description available")
            if nfor == "ESSENTIAL":
                cprint(f"{package} : not found. {nfor}. Please install it to use wannierberri.", 'red')
            else:
                cprint(f"{package} : not found. {nfor}", 'yellow')
            versions[package] = None
    print("#")
    return versions


# welcome()
