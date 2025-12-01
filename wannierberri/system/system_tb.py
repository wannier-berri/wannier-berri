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

from ..fourier.rvectors import Rvectors
from .needed_data import NeededData
from .system_R import System_R


def System_tb(tb_file="wannier90_tb.dat",
              convention_II_to_I=True,
              wannier_centers_cart=None,
              **parameters):
    """
    System initialized from the `*_tb.dat` file, which can be written either by  `Wannier90 <http://wannier.org>`__ code,
    or composed by the user based on some tight-binding model.
    See Wannier90 `code <https://github.com/wannier-developers/wannier90/blob/2f4aed6a35ab7e8b38dbe196aa4925ab3e9deb1b/src/hamiltonian.F90#L698-L799>`_
    for details of the format.

    Parameters
    ----------
    tb_file : str
        name (and path) of file to be read
    convention_II_to_I : bool
        By default, the tb file in wannier90 format is in the convention II, which is different from the convention I used in wannierberri.
        If the file is already in the convention I, set this parameter to False.
    wannier_centers_cart : np.ndarray(num_wann, 3)
        If provided, will override the wannier centers read from the file. (and hence they will be subtracted from the AA_R matrix if convention_II_to_I is True)

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.system.System`
    """

    if "name" not in parameters:
        parameters["name"] = os.path.splitext(os.path.split(tb_file)[-1])[0]
    parameters, param_needed_data = NeededData.get_parameters(**parameters)
    needed_data = NeededData(**param_needed_data)
    system = System_R(**parameters)
    for key in needed_data.matrices:
        if key not in ['Ham', 'AA']:
            raise ValueError(f"System_tb class cannot be used for evaluation of {key}_R")
    f = open(tb_file, "r")
    line = f.readline().strip()
    cprint(f"reading TB file {tb_file} ( {line} )", 'green', attrs=['bold'])
    system.real_lattice = np.array([f.readline().split()[:3] for _ in range(3)], dtype=float)

    system.num_wann = int(f.readline())
    nRvec = int(f.readline())
    Ndegen = []
    while len(Ndegen) < nRvec:
        Ndegen += f.readline().split()
    Ndegen = np.array(Ndegen, dtype=int)



    iRvec = []

    Ham_R = np.zeros((nRvec, system.num_wann, system.num_wann), dtype=complex)

    for ir in range(nRvec):
        f.readline()
        iRvec.append(f.readline().split())
        hh = np.array(
            [[f.readline().split()[2:4] for _ in range(system.num_wann)] for _ in range(system.num_wann)],
            dtype=float).transpose((1, 0, 2))
        Ham_R[ir] = (hh[:, :, 0] + 1j * hh[:, :, 1]) / Ndegen[ir]
    system.set_R_mat('Ham', Ham_R)
    iRvec = np.array(iRvec, dtype=int)
    iR0 = Rvectors(lattice=system.real_lattice, iRvec=iRvec).iR0


    if needed_data.need_any('AA'):
        AA_R = np.zeros((nRvec, system.num_wann, system.num_wann, 3), dtype=complex)
        for ir in range(nRvec):
            f.readline()
            assert np.all(np.array(f.readline().split(), dtype=int) == iRvec[ir])
            aa = np.array(
                [[f.readline().split()[2:8] for _ in range(system.num_wann)] for _ in range(system.num_wann)],
                dtype=float)
            AA_R[ir] = (aa[:, :, 0::2] + 1j * aa[:, :, 1::2]).transpose((1, 0, 2)) / Ndegen[ir]
        if wannier_centers_cart is None:
            wannier_centers_cart = np.diagonal(AA_R[iR0], axis1=0, axis2=1).T.copy().real
        if convention_II_to_I:
            # convert to convention I
            # print(f"convention_II_to_I = {convention_II_to_I} wannier_centers_cart = \n{wannier_centers_cart}\n num_wann = {self.num_wann}, A.shape = {AA_R.shape}")
            AA_R[iR0, np.arange(system.num_wann), np.arange(system.num_wann), :] -= wannier_centers_cart
        system.set_R_mat('AA', AA_R)
    elif wannier_centers_cart is None:
        wannier_centers_cart = np.zeros((3, system.num_wann), dtype=float)
        for ir in range(iR0):
            f.readline()
            assert np.all(np.array(f.readline().split(), dtype=int) == iRvec[ir])
            for _ in range(system.num_wann**2):
                f.readline()
        ir = iR0
        f.readline()
        assert np.all(np.array(f.readline().split(), dtype=int) == iRvec[ir])
        aa = np.array(
            [[f.readline().split()[2:8] for _ in range(system.num_wann)] for _ in range(system.num_wann)],
            dtype=float)
        aa = (aa[:, :, 0::2] + 1j * aa[:, :, 1::2]).transpose((1, 0, 2)) / Ndegen[ir]
        wannier_centers_cart = np.diagonal(aa, axis1=0, axis2=1).T.real


    system.wannier_centers_cart = wannier_centers_cart
    system.clear_cached_wcc()
    system.rvec = Rvectors(
        lattice=system.real_lattice,
        iRvec=iRvec,
        shifts_left_red=system.wannier_centers_red,
    )

    f.close()


    system.do_at_end_of_init()

    cprint(f"Reading the system from {tb_file} finished successfully", 'green', attrs=['bold'])
    return system
