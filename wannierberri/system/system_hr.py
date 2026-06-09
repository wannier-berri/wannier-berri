
import numpy as np
import os
from termcolor import cprint

from ..fourier.rvectors import Rvectors
from .needed_data import NeededData


def system_hr(hr_file,wannier_centers_cart=None,real_lattice=None,**parameters):
    """
    System initialized from the `*_hr.dat` file and wannier centres (cartesian) +lattice vectors in array format (Ang)

    Parameters
    ----------
    hr_file : str
        name (and path) of file to be read
    wannier_centers_cart : np.ndarray, shape=(n_wann, 3)
        It NEEDS to be provided, as hr_file does not contains this infromaiton
    real_lattice: np.ndarray, shape=(3, 3)
        It NEEDS to be provided
    Notes
    -----
    see also  parameters of the :class:`~wannierberri.system.System`
    """

    if "name" not in parameters:
        parameters["name"] = os.path.splitext(os.path.split(hr_file)[-1])[0]
    parameters, param_needed_data = NeededData.get_parameters(**parameters)
    needed_data = NeededData(**param_needed_data)
    from .system_R import System_R
    system = System_R(**parameters)
    for key in needed_data.matrices:
        if key not in ['Ham', 'AA']:
            raise ValueError(f"System_tb class cannot be used for evaluation of {key}_R")
    f = open(hr_file, "r")
    line = f.readline().strip()
    cprint(f"reading HR file {hr_file} ( {line} )", 'green', attrs=['bold'])
    system.real_lattice = real_lattice#np.array([f.readline().split()[:3] for _ in range(3)], dtype=float)

    system.num_wann = int(f.readline())
    nRvec = int(f.readline())
    Ndegen = []
    while len(Ndegen) < nRvec:
        Ndegen += f.readline().split()
    Ndegen = np.array(Ndegen, dtype=int)

    iRvec = []

    Ham_R = np.zeros((nRvec, system.num_wann, system.num_wann), dtype=complex)

    for ir in range(nRvec):
        pos = f.tell()
        line = f.readline()
        iRvec.append(line.split()[:3])
        f.seek(pos)

        hh = np.array(
            [[f.readline().split()[5:7] for _ in range(system.num_wann)] for _ in range(system.num_wann)],
            dtype=float).transpose((1, 0, 2))
        Ham_R[ir] = (hh[:, :, 0] + 1j * hh[:, :, 1]) / Ndegen[ir]

    system.set_R_mat('Ham', Ham_R)
    iRvec = np.array(iRvec, dtype=int)

    system.wannier_centers_cart = wannier_centers_cart
    system.clear_cached_wcc()
    system.rvec = Rvectors(
        lattice=system.real_lattice,
        iRvec=iRvec,
        shifts_left_red=system.wannier_centers_red,
    )

    f.close()

    system.do_at_end_of_init()

    cprint(f"Reading the system from {hr_file} finished successfully", 'green', attrs=['bold'])
    return system
