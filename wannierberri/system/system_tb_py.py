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

import numpy as np
from termcolor import cprint

from ..fourier.rvectors import Rvectors
from .system_R import System_R
from .needed_data import NeededData
# from .Rvec import Rvec



def System_tb_py(model,
                 module,
                 spin=False,
                 **parameters):
    """This interface initializes the System class from a tight-binding
    model packed by one of the available python modules (see below)


    Parameters
    ----------
    tbmodel :
        name of the tight-binding model object.
    module : str
        name of the module 'pythtb' or 'tbmodels'
    spin : bool
        generate SS_R matrix (if PythTB model  has spin)

    Notes
    -----
    always uses use_wcc_phase=True, force_internal_terms_only=True
    see also  parameters of the :class:`~wannierberri.System`
    """

    names = {'tbmodels': 'TBmodels', 'pythtb': 'PythTB'}
    system = System_R(force_internal_terms_only=True,
                     name=f'model_{names[module]}',
                     **parameters)
    needed_data = NeededData(**parameters)  # to set needed_R_matrices
    if module == 'tbmodels':
        # Extract the parameters from the model
        real = model.uc
        system.num_wann = model.size
        if needed_data.need_any(['SS', 'SHA', 'SA', 'SH', 'SRA', 'SR']):
            raise ValueError(
                f"System_{names[module]} class cannot be used for evaluation of spin properties")
        system.spinors = False
        positions = model.pos
        iRvec = np.array([R[0] for R in model.hop.items()], dtype=int)
    elif module == 'pythtb':
        real = model._lat
        positions = model._orb
        if model._nspin == 1:
            system.spinors = False
            system.num_wann = model._norb
        elif model._nspin == 2:
            system.spinors = True
            system.num_wann = model._norb * 2
            positions = np.array(sum(([p, p] for p in positions), []))
        else:
            raise Exception("\n\nWrong value of nspin!")
        print("number of wannier functions:", system.num_wann)
        iRvec = np.array([R[-1] for R in model._hoppings], dtype=int)
    else:
        raise ValueError(f"unknown tight-binding module {module}")

    system.dimr = real.shape[1]
    system.norb = positions.shape[0]
    wannier_centers_red = np.zeros((system.norb, 3))
    wannier_centers_red[:, :system.dimr] = positions
    system.real_lattice = np.eye(3, dtype=float)
    system.real_lattice[:system.dimr, :system.dimr] = np.array(real)
    system.wannier_centers_cart = wannier_centers_red.dot(system.real_lattice)

    system.periodic[system.dimr:] = False

    if len(iRvec) == 0:
        iRvec = np.zeros((1, 3), dtype='int32')
    else:
        iRvec0 = np.array(iRvec, dtype='int32')
        iRvec = np.zeros((iRvec0.shape[0], 3), dtype='int32')
        iRvec[:, :system.dimr] = iRvec0
        iRvec = np.vstack((iRvec, np.zeros((1, 3), dtype='int32'), -iRvec))
        iRvec = [tuple(row) for row in iRvec]
        iRvec = np.unique(iRvec, axis=0).astype('int32')


    system.rvec = Rvectors(lattice=system.real_lattice,
                        iRvec=iRvec,
                        shifts_left_red=wannier_centers_red,
                        dim=system.dimr)
    index0 = system.rvec.iR0
    # Define Ham_R matrix from hoppings
    nRvec = system.rvec.nRvec
    Ham_R = np.zeros((nRvec, system.num_wann, system.num_wann), dtype=complex)
    iRvec = system.rvec.iRvec
    if module == 'tbmodels':
        for hop in model.hop.items():
            R = np.array(hop[0], dtype=int)
            hops = np.array(hop[1]).reshape((system.num_wann, system.num_wann))
            iR = int(np.argwhere(np.all((R - iRvec[:, :system.dimr]) == 0, axis=1)))
            inR = int(np.argwhere(np.all((-R - iRvec[:, :system.dimr]) == 0, axis=1)))
            Ham_R[iR] += hops
            Ham_R[inR] += np.conjugate(hops.T)
    elif module == 'pythtb':
        for nhop in model._hoppings:
            i = nhop[1]
            j = nhop[2]
            iR = np.argwhere(np.all((nhop[-1] - system.rvec.iRvec[:, :system.dimr]) == 0, axis=1))[0][0]
            inR = np.argwhere(np.all((-nhop[-1] - system.rvec.iRvec[:, :system.dimr]) == 0, axis=1))[0][0]
            if model._nspin == 1:
                Ham_R[iR, i, j] += nhop[0]
                Ham_R[inR, j, i] += np.conjugate(nhop[0])
            elif model._nspin == 2:
                print("hopping :", nhop[0].shape, Ham_R.shape, iR,
                      Ham_R[iR, 2 * i:2 * i + 2, 2 * j:2 * j + 2].shape)
                Ham_R[iR, 2 * i:2 * i + 2, 2 * j:2 * j + 2] += nhop[0]
                Ham_R[inR, 2 * j:2 * j + 2, 2 * i:2 * i + 2] += np.conjugate(nhop[0].T)

        # Set the onsite energies at H(R=[000])
        for i in range(model._norb):
            if model._nspin == 1:
                Ham_R[index0, i, i] = model._site_energies[i]
            elif model._nspin == 2:
                Ham_R[index0, 2 * i:2 * i + 2, 2 * i:2 * i + 2] = model._site_energies[i]
        if model._nspin == 2 and spin:
            system.set_spin_pairs([(i, i + 1) for i in range(0, system.num_wann, 2)])

    system.set_R_mat('Ham', Ham_R)
    print(f"shape of Ham_R = {Ham_R.shape}")

    system.do_at_end_of_init()
    cprint(f"Reading the system from {names[module]} finished successfully", 'green', attrs=['bold'])
    return system


def System_TBmodels(tbmodel, **parameters):
    """This interface initializes the System class from a tight-binding
    model created with `TBmodels. <http://z2pack.ethz.ch/tbmodels/doc/1.3/index.html>`_
    It defines the Hamiltonian matrix Ham_R (from hoppings matrix elements)
    and the AA_R  matrix (from orbital coordinates) used to calculate Berry
    related quantities.


    Parameters
    ----------
    tbmodel :
        name of the TBmodels tight-binding model object.

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System_tb_py`
    """

    return System_tb_py(tbmodel, module='tbmodels', **parameters)


def System_PythTB(ptb_model, **parameters):
    """This interface is a way to initialize the System class from a tight-binding
    model created with  `PythTB. <http://www.physics.rutgers.edu/pythtb/>`_
    It defines the Hamiltonian matrix Ham_R (from hoppings matrix elements)
    and the AA_R  matrix (from orbital coordinates) used to calculate
    Berry related quantities.

    Parameters
    ----------
    ptb_model : class
        name of the PythTB tight-binding model class.


    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System_tb_py`
    """

    return System_tb_py(ptb_model, module='pythtb', **parameters)
