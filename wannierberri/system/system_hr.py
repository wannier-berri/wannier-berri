import numpy as np
from termcolor import cprint

from ..fourier.rvectors import Rvectors
from .needed_data import NeededData


def get_system_hr(seedname, wannier_centers_cart=None, real_lattice=None, **parameters):
    """
    System initialized from the `*_hr.dat` file and wannier centres (cartesian) +lattice vectors in array format (Ang)

    Parameters
    ----------
    hr_file : str
        name (and path) of file to be read
    wannier_centers_cart : np.ndarray, shape=(n_wann, 3)
        if not provided, the wannier centres will be read from the file `*_wannier_centre_WT_format.dat` (if it exists). Otherwise, it NEEDS to be provided
    real_lattice: np.ndarray, shape=(3, 3)
        It NEEDS to be provided
    Notes
    -----
    see also  parameters of the :class:`~wannierberri.system.System`
    """

    if "name" not in parameters:
        parameters["name"] = seedname
    parameters, param_needed_data = NeededData.get_parameters(**parameters)
    needed_data = NeededData(**param_needed_data)
    from .system_R import System_R
    system = System_R(**parameters)
    for key in needed_data.matrices:
        if key not in ['Ham']:
            raise ValueError(f"System_tb class cannot be used for evaluation of {key}_R")
    hr_file = seedname + "_hr.dat"
    f = open(hr_file, "r")
    line = f.readline().strip()
    cprint(f"reading HR file {hr_file} ( {line} )", 'green', attrs=['bold'])

    if real_lattice is None:
        try:
            real_lattice = read_real_lattice_win(seedname)
        except FileNotFoundError as e:
            raise ValueError(f"{e}. The real_lattice was not provided  and could not be read from the win file. ")

    system.real_lattice = real_lattice  # np.array([f.readline().split()[:3] for _ in range(3)], dtype=float)

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
    if wannier_centers_cart is None:
        wannier_centers_cart = read_WCC_WT_format(seedname)
    print(f"wannier_centers_cart = {wannier_centers_cart}")

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


def write_WCC_WT_format(seedname, wannier_centers_cart):
    r = open(seedname + "_wannier_centre_WT_format.dat", "w")
    data = wannier_centers_cart
    for i in data[::2]:
        r.write(f"{(i[0] if np.abs(i[0]) > 1e-7 else 0.0):10} {(i[1] if np.abs(i[1]) > 1e-7 else 0.0):10} {(i[2] if np.abs(i[2]) > 1e-7 else 0.0):10}\n")
    for i in data[1::2]:
        r.write(f"{(i[0] if np.abs(i[0]) > 1e-7 else 0.0):10} {(i[1] if np.abs(i[1]) > 1e-7 else 0.0):10} {(i[2] if np.abs(i[2]) > 1e-7 else 0.0):10}\n")
    r.close()


def read_WCC_WT_format(seedname):
    r = open(seedname + "_wannier_centre_WT_format.dat", "r")
    data = np.array([[float(x) for x in line.split()] for line in r.readlines()])
    data_2 = np.zeros(data.shape, dtype=float)
    data_2[::2] = data[:data.shape[0] // 2]
    data_2[1::2] = data[data.shape[0] // 2:]
    r.close()
    return data_2


def read_real_lattice_win(seedname):
    from ..w90files.win import WIN
    return WIN.from_w90_file(seedname=seedname)["unit_cell_cart"]
