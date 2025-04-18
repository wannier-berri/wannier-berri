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
from .system_R import System_R


class System_tb_py(System_R):
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

    def __init__(self, model, module,
                 spin=False,
                 **parameters
                 ):
        names = {'tbmodels': 'TBmodels', 'pythtb': 'PythTB'}
        super().__init__(spin=spin,
                         force_internal_terms_only=True,
                         name=f'model_{names[module]}',
                         **parameters)

        if module == 'tbmodels':
            # Extract the parameters from the model
            real = model.uc
            self.num_wann = model.size
            if self.need_R_any(['SS', 'SHA', 'SA', 'SH', 'SRA', 'SR']):
                raise ValueError(
                    f"System_{names[module]} class cannot be used for evaluation of spin properties")
            self.spinors = False
            positions = model.pos
            Rvec = np.array([R[0] for R in model.hop.items()], dtype=int)
        elif module == 'pythtb':
            real = model._lat
            positions = model._orb
            if model._nspin == 1:
                self.spinors = False
                self.num_wann = model._norb
            elif model._nspin == 2:
                self.spinors = True
                self.num_wann = model._norb * 2
                positions = np.array(sum(([p, p] for p in positions), []))
            else:
                raise Exception("\n\nWrong value of nspin!")
            print("number of wannier functions:", self.num_wann)
            Rvec = np.array([R[-1] for R in model._hoppings], dtype=int)
        else:
            raise ValueError(f"unknown tight-binding module {module}")

        self.dimr = real.shape[1]
        self.norb = positions.shape[0]
        wannier_centers_reduced = np.zeros((self.norb, 3))
        wannier_centers_reduced[:, :self.dimr] = positions
        self.real_lattice = np.eye(3, dtype=float)
        self.real_lattice[:self.dimr, :self.dimr] = np.array(real)
        self.wannier_centers_cart = wannier_centers_reduced.dot(self.real_lattice)

        self.periodic[self.dimr:] = False
        Rvec = [tuple(row) for row in Rvec]
        Rvecs = np.unique(Rvec, axis=0).astype('int32')

        nR = Rvecs.shape[0]
        for i in range(3 - self.dimr):
            column = np.zeros(nR, dtype='int32')
            Rvecs = np.column_stack((Rvecs, column))

        Rvecsneg = np.array([-r for r in Rvecs])
        R_all = np.concatenate((Rvecs, Rvecsneg), axis=0)
        R_all = np.unique(R_all, axis=0)
        # Find the R=[000] index (used later)
        index0 = np.argwhere(np.all(([0, 0, 0] - R_all) == 0, axis=1))
        # make sure it exists; otherwise, add it manually
        # add it manually
        if index0.size == 0:
            R_all = np.column_stack((np.array([0, 0, 0]), R_all.T)).T
            index0 = 0
        elif index0.size == 1:
            print(f"R=0 found at position(s) {index0}")
            index0 = index0[0][0]
        else:
            raise RuntimeError(f"wrong value of index0={index0}, with R_all={R_all}")

        self.iRvec = R_all
        nRvec = self.iRvec.shape[0]
        self.nRvec0 = nRvec
        # Define Ham_R matrix from hoppings
        Ham_R = np.zeros((self.num_wann, self.num_wann, self.nRvec0), dtype=complex)
        if module == 'tbmodels':
            for hop in model.hop.items():
                R = np.array(hop[0], dtype=int)
                hops = np.array(hop[1]).reshape((self.num_wann, self.num_wann))
                iR = int(np.argwhere(np.all((R - R_all[:, :self.dimr]) == 0, axis=1)))
                inR = int(np.argwhere(np.all((-R - R_all[:, :self.dimr]) == 0, axis=1)))
                Ham_R[:, :, iR] += hops
                Ham_R[:, :, inR] += np.conjugate(hops.T)
        elif module == 'pythtb':
            for nhop in model._hoppings:
                i = nhop[1]
                j = nhop[2]
                iR = np.argwhere(np.all((nhop[-1] - self.iRvec[:, :self.dimr]) == 0, axis=1))[0][0]
                inR = np.argwhere(np.all((-nhop[-1] - self.iRvec[:, :self.dimr]) == 0, axis=1))[0][0]
                if model._nspin == 1:
                    Ham_R[i, j, iR] += nhop[0]
                    Ham_R[j, i, inR] += np.conjugate(nhop[0])
                elif model._nspin == 2:
                    print("hopping :", nhop[0].shape, Ham_R.shape, iR,
                          Ham_R[2 * i:2 * i + 2, 2 * j:2 * j + 2, iR].shape)
                    Ham_R[2 * i:2 * i + 2, 2 * j:2 * j + 2, iR] += nhop[0]
                    Ham_R[2 * j:2 * j + 2, 2 * i:2 * i + 2, inR] += np.conjugate(nhop[0].T)

            # Set the onsite energies at H(R=[000])
            for i in range(model._norb):
                if model._nspin == 1:
                    Ham_R[i, i, index0] = model._site_energies[i]
                elif model._nspin == 2:
                    Ham_R[2 * i:2 * i + 2, 2 * i:2 * i + 2, index0] = model._site_energies[i]
            if model._nspin == 2 and spin:
                self.set_spin_pairs([(i, i + 1) for i in range(0, self.num_wann, 2)])

        self.set_R_mat('Ham', Ham_R)

        self.do_at_end_of_init()
        cprint(f"Reading the system from {names[module]} finished successfully", 'green', attrs=['bold'])


class System_TBmodels(System_tb_py):
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

    def __init__(self, tbmodel, **parameters):
        super().__init__(tbmodel, module='tbmodels', **parameters)


class System_PythTB(System_tb_py):
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

    def __init__(self, ptb_model, **parameters):
        super().__init__(ptb_model, module='pythtb', **parameters)
