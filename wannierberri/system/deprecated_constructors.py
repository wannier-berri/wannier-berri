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

import warnings

def constructor_deprecation_warning(old_constructor_name, new_constructor_name):
    warnings.warn(f"DeprecationWarning: {old_constructor_name} is deprecated "
                  "and will be removed in the future. Use "
                  f"System_R.from_{new_constructor_name} instead.", 
                  DeprecationWarning)

def System_w90(*args, **kwargs):
    constructor_deprecation_warning("System_w90", "w90data")
    from .system_w90 import get_system_w90
    return get_system_w90(*args, **kwargs)


def System_fplo(*args, **kwargs):
    constructor_deprecation_warning("System_fplo", "FPLO")
    from .system_fplo import get_system_fplo
    return get_system_fplo(*args, **kwargs)


def System_tb(*args, **kwargs):
    constructor_deprecation_warning("System_tb", "tb_dat")
    from .system_tb import system_tb
    return system_tb(*args, **kwargs)


def System_TBmodels(*args, **kwargs):
    constructor_deprecation_warning("System_TBmodels", "tbmodels")
    from .system_tb_py import get_system_tbmodels
    return get_system_tbmodels(*args, **kwargs)


def System_PythTB(*args, **kwargs):
    constructor_deprecation_warning("System_PythTB", "pythtb")
    from .system_tb_py import get_system_pythtb
    return get_system_pythtb(*args, **kwargs)


def System_ASE(*args, **kwargs):
    constructor_deprecation_warning("System_ASE", "ASE")
    from .system_ase import get_system_ase
    return get_system_ase(*args, **kwargs)


def SystemSparse(*args, **kwargs):
    constructor_deprecation_warning("SystemSparse", "sparse")
    from .system_sparse import get_system_sparse
    return get_system_sparse(*args, **kwargs)


def SystemRandom(*args, **kwargs):
    constructor_deprecation_warning("SystemRandom", "random")
    from .system_random import get_system_random
    return get_system_random(*args, **kwargs)


def System_Phonon_QE(*args, **kwargs):
    constructor_deprecation_warning("System_Phonon_QE", "phonons_qe")
    from .system_phonon_qe import get_system_phonons_qe
    return get_system_phonons_qe(*args, **kwargs)
