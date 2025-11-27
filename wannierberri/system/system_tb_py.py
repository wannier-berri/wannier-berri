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
from packaging import version
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
        if needed_data.need_R_any(['SS', 'SHA', 'SA', 'SH', 'SRA', 'SR']):
            raise ValueError(
                f"System_{names[module]} class cannot be used for evaluation of spin properties")
        system.spinor = False
        positions = model.pos
        iRvec = np.array([R[0] for R in model.hop.items()], dtype=int)
    elif module == 'pythtb':
        import pythtb
        version1 = (version.parse(pythtb.__version__) < version.parse("2.0.0"))
        if version1:
            real = model._lat
            positions = model._orb
            norb_loc = model._norb
            hoppings = model._hoppings
        else:
            real = model.lat_vecs
            positions = model.get_orb_vecs(cartesian=False)
            norb_loc = model.norb
            hoppings = model.hoppings

        if model._nspin == 1:
            system.spinor = False
            system.num_wann = norb_loc
        elif model._nspin == 2:
            system.spinor = True
            system.num_wann = norb_loc * 2
            positions = np.array(sum(([p, p] for p in positions), []))
        else:
            raise Exception("\n\nWrong value of nspin!")
        print("number of wannier functions:", system.num_wann)
        if version1:
            iRvec = [hop[-1] for hop in hoppings]
        else:
            iRvec = []
            for hop in hoppings:
                if "lattice_vector" in hop:
                    iRvec.append(hop["lattice_vector"])
    else:
        raise ValueError(f"unknown tight-binding module {module}")
    dimr = real.shape[0]
    iRvec = np.array(iRvec, dtype=int)
    Rzero = np.zeros((1, dimr), dtype=int)
    iRvec = np.vstack((Rzero, iRvec, -iRvec))
    iRvec = np.unique(iRvec, axis=0)
    iRvec = np.hstack((iRvec, np.zeros((iRvec.shape[0], 3 - dimr))))

    real_lattice = np.eye(3)
    real_lattice[:dimr, :dimr] = real

    system.real_lattice = real_lattice
    wannier_centers_red = positions % 1.0
    wannier_centers_red = np.hstack((wannier_centers_red, np.zeros((wannier_centers_red.shape[0], 3 - dimr))))

    system.set_wannier_centers(wannier_centers_red=wannier_centers_red)
    system.periodic = np.array([True] * dimr + [False] * (3 - dimr))
    system.rvec = Rvectors(lattice=system.real_lattice,
                        iRvec=iRvec,
                        shifts_left_red=wannier_centers_red,
                        dim=dimr)
    index0 = system.rvec.iR0
    # Define Ham_R matrix from hoppings
    nRvec = system.rvec.nRvec
    Ham_R = np.zeros((nRvec, system.num_wann, system.num_wann), dtype=complex)
    iRvec = system.rvec.iRvec
    if module == 'tbmodels':
        for hop in model.hop.items():
            R = np.array(hop[0], dtype=int)
            hops = np.array(hop[1]).reshape((system.num_wann, system.num_wann))
            iR = int(np.argwhere(np.all((R - iRvec[:, :dimr]) == 0, axis=1)))
            inR = int(np.argwhere(np.all((-R - iRvec[:, :dimr]) == 0, axis=1)))
            Ham_R[iR] += hops
            Ham_R[inR] += np.conjugate(hops.T)
    elif module == 'pythtb':
        for nhop in hoppings:
            R = np.zeros(3, dtype=int)
            if version1:
                i = nhop[1]
                j = nhop[2]
                R[:dimr] = nhop[-1]
                amplitude = nhop[0]
            else:
                i = nhop["from_orbital"]
                j = nhop["to_orbital"]
                if "lattice_vector" in nhop:
                    R[: dimr] = nhop["lattice_vector"]
                amplitude = nhop["amplitude"]

            iR = system.rvec.iR(R)
            inR = system.rvec.iR(-R)
            if model._nspin == 1:
                Ham_R[iR, i, j] += amplitude
                Ham_R[inR, j, i] += np.conjugate(amplitude)
            elif model._nspin == 2:
                print("hopping :", amplitude.shape, Ham_R.shape, iR,
                      Ham_R[iR, 2 * i:2 * i + 2, 2 * j:2 * j + 2].shape)
                Ham_R[iR, 2 * i:2 * i + 2, 2 * j:2 * j + 2] += amplitude
                Ham_R[inR, 2 * j:2 * j + 2, 2 * i:2 * i + 2] += np.conjugate(amplitude.T)

        # Set the onsite energies at H(R=[000])
        for i in range(norb_loc):
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
