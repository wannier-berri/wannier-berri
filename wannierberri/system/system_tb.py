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
import os
from termcolor import cprint
from .system_R import System_R


class System_tb(System_R):
    """
    System initialized from the `*_tb.dat` file, which can be written either by  `Wannier90 <http://wannier.org>`__ code,
    or composed by the user based on some tight-binding model.
    See Wannier90 `code <https://github.com/wannier-developers/wannier90/blob/2f4aed6a35ab7e8b38dbe196aa4925ab3e9deb1b/src/hamiltonian.F90#L698-L799>`_
    for details of the format.

    Parameters
    ----------
    tb_file : str
        name (and path) of file to be read

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.system.System`
    """

    def __init__(self, tb_file="wannier90_tb.dat",
                 **parameters):
        if "name" not in parameters:
            parameters["name"] = os.path.splitext(os.path.split(tb_file)[-1])[0]
        super().__init__(**parameters)
        for key in self.needed_R_matrices:
            if key not in ['Ham', 'AA']:
                raise ValueError(f"System_tb class cannot be used for evaluation of {key}_R")

        self.seedname = tb_file.split("/")[-1].split("_")[0]
        f = open(tb_file, "r")
        line = f.readline().strip()
        cprint(f"reading TB file {tb_file} ( {line} )", 'green', attrs=['bold'])
        self.real_lattice = np.array([f.readline().split()[:3] for _ in range(3)], dtype=float)

        self.num_wann = int(f.readline())
        nRvec = int(f.readline())
        self.nRvec0 = nRvec
        self.Ndegen = []
        while len(self.Ndegen) < nRvec:
            self.Ndegen += f.readline().split()
        self.Ndegen = np.array(self.Ndegen, dtype=int)

        self.iRvec = []

        Ham_R = np.zeros((self.num_wann, self.num_wann, nRvec), dtype=complex)

        for ir in range(nRvec):
            f.readline()
            self.iRvec.append(f.readline().split())
            hh = np.array(
                [[f.readline().split()[2:4] for _ in range(self.num_wann)] for _ in range(self.num_wann)],
                dtype=float).transpose((1, 0, 2))
            Ham_R[:, :, ir] = (hh[:, :, 0] + 1j * hh[:, :, 1]) / self.Ndegen[ir]
        self.set_R_mat('Ham', Ham_R)

        self.iRvec = np.array(self.iRvec, dtype=int)

        if 'AA' in self.needed_R_matrices:
            AA_R = np.zeros((self.num_wann, self.num_wann, nRvec, 3), dtype=complex)
            for ir in range(nRvec):
                f.readline()
                assert np.all(np.array(f.readline().split(), dtype=int) == self.iRvec[ir])
                aa = np.array(
                    [[f.readline().split()[2:8] for _ in range(self.num_wann)] for _ in range(self.num_wann)],
                    dtype=float)
                AA_R[:, :, ir, :] = (aa[:, :, 0::2] + 1j * aa[:, :, 1::2]).transpose((1, 0, 2)) / self.Ndegen[ir]
            self.wannier_centers_cart = np.diagonal(AA_R[:, :, self.iR0, :], axis1=0, axis2=1).T
            self.set_R_mat('AA', AA_R)

        f.close()

        self.do_at_end_of_init()
        if self.use_wcc_phase:
            self.convention_II_to_I()

        cprint(f"Reading the system from {tb_file} finished successfully", 'green', attrs=['bold'])
