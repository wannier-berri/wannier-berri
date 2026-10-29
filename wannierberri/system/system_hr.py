import numpy as np
from termcolor import cprint

from ..fourier.rvectors import Rvectors
from .needed_data import NeededData
import logging
logger = logging.getLogger(__name__)


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
    logger.info(f"wannier_centers_cart = {wannier_centers_cart}")

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

def get_system_hr_from_FHI(path="./", positions=None, real_lattice=None,
                           fft_mesh=(9, 9, 9), fermi_energy=None,
                           overlap_floor=1e-8, validate=True, **parameters):
    """Read FHI-aims H(R), S(R), Lowdin-orthogonalize on an FFT grid.

    Parameters
    ----------
    path : directory with rs_indices.out, basis-indices.out,
           rs_hamiltonian.h5, rs_overlap.h5
    positions : (natoms,3) Cartesian Angstrom positions (FHI-aims atom order)
    real_lattice : (3,3) lattice vectors as rows, Cartesian Angstrom
    fft_mesh : 3 positive integers; start with (9,9,9); test (11,11,11) and (13,13,13)
    fermi_energy : float or None; Fermi energy in eV, subtracted from H
    overlap_floor : float; reject S(k) eigenvalues below this threshold
    validate : bool; compare generalized and orthogonal eigenvalues on grid
    **parameters : WannierBerri System_R options

    Important
    ---------
    This orthogonalizes the *scalar-relativistic* Hamiltonian. It does not
    add SOC, and does not reconstruct position-operator matrix elements.
    The Fourier grid must resolve the original real-space hoppings: if two
    original R vectors alias modulo fft_mesh, this function raises an error.

    TO GENERATE THE FILES PLEASE USE THE LINE BELOW IN YOUR control.in :
    output_rs_matrices hdf5
    """
    import h5py
    from .system_R import System_R
    # NeededData and Rvectors must already be imported by system_hr.py.

    HARTREE_TO_EV = 27.211386245988
    directory = Path(path)
    if positions is None or real_lattice is None:
        raise ValueError("positions and real_lattice are required")
    lattice = np.asarray(real_lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if lattice.shape != (3, 3) or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Expected real_lattice=(3,3), positions=(natoms,3)")
    mesh = tuple(int(x) for x in fft_mesh)
    if len(mesh) != 3 or any(x < 3 or x % 2 == 0 for x in mesh):
        raise ValueError("fft_mesh must contain three odd integers >= 3")

    # REMEMBER: basis-indices.out may have its first header line removed.
    # Here we ignore header/non-numeric lines; atom index is column 3 ('at.').
    with (directory / 'rs_indices.out').open() as f:
        lines = [line.strip() for line in f]
    meta = {line.split(':', 1)[0].strip(): int(line.split(':', 1)[1])
            for line in lines[:3]}
    n = meta['n_basis']
    nc = meta['n_cells_in_hamiltonian']
    ns = meta['n_hamiltonian_matrix_size']
    keys = ('cell_index', 'index_hamiltonian(1,:,:)',
            'index_hamiltonian(2,:,:)', 'column_index_hamiltonian')
    a, b, c, d = (lines.index(x) for x in keys)
    cells = np.array([list(map(int, x.split())) for x in lines[a+1:b] if x], dtype=int)
    starts = np.array([list(map(int, x.split())) for x in lines[b+1:c] if x], dtype=int)
    ends = np.array([list(map(int, x.split())) for x in lines[c+1:d] if x], dtype=int)
    cols = np.array([int(x) for x in lines[d+1:] if x], dtype=int)
    if cells.shape != (nc, 3) or starts.shape != (nc, n) or ends.shape != (nc, n) or len(cols) != ns:
        raise ValueError('Unexpected rs_indices.out dimensions')
    valid = np.all(np.abs(cells) < 999999999, axis=1)
    if np.any(starts[~valid] > 0):
        raise ValueError('Sentinel cell contains matrix entries')
    R = cells[valid]
    lookup = {tuple(r): i for i, r in enumerate(R)}
    if len(lookup) != len(R) or any(tuple(-r) not in lookup for r in R):
        raise ValueError('Duplicate or unpaired R vectors')

    atom_indices = []
    for line in (directory / 'basis-indices.out').read_text().splitlines():
        fields = line.split()
        try:
            int(fields[0]); atom_indices.append(int(fields[2]))
        except (IndexError, ValueError):
            continue
    atom_indices = np.asarray(atom_indices, dtype=int)
    if len(atom_indices) != n or np.any(atom_indices < 1) or np.any(atom_indices > len(positions)):
        raise ValueError('basis-indices.out inconsistent with n_basis or positions')
    centers = positions[atom_indices - 1]

    def load_matrix(filename):
        with h5py.File(directory / filename, 'r') as f:
            values = np.asarray(f['sparse_matrix'][:])
        if values.size != ns:
            raise ValueError(f'{filename}: expected {ns} sparse entries')
        mat = np.zeros((len(R), n, n), dtype=np.complex128)
        for ir, original in enumerate(np.flatnonzero(valid)):
            for i in range(n):
                st, en = int(starts[original, i]), int(ends[original, i])
                if st == 0 and en == -1:
                    continue
                if not 1 <= st <= en <= ns:
                    raise ValueError(f'Bad sparse range: cell={original}, row={i}')
                js = cols[st-1:en] - 1
                if np.any(js < 0) or np.any(js > i) or len(np.unique(js)) != len(js):
                    raise ValueError(f'Bad lower-triangle columns: cell={original}, row={i}')
                mat[ir, i, js] = values[st-1:en]
        ii, jj = np.triu_indices(n, 1)
        for ir, r in enumerate(R):
            mat[ir, ii, jj] = mat[lookup[tuple(-r)], jj, ii].conj()
        err = max(np.max(np.abs(mat[ir] - mat[lookup[tuple(-r)]].conj().T)) for ir, r in enumerate(R))
        if err > 1e-7:
            raise ValueError(f'{filename}: real-space Hermiticity error {err}')
        return mat

    H_R = load_matrix('rs_hamiltonian.h5') * HARTREE_TO_EV
    S_R = load_matrix('rs_overlap.h5')
    if fermi_energy is not None:
        # In a nonorthogonal basis, subtract EF * S, not EF * I.
        H_R = H_R - float(fermi_energy) * S_R

    # A discrete Fourier grid identifies R vectors differing by mesh periods.
    # Reject aliases in original input, rather than silently folding them.
    grid_index = np.mod(R, np.array(mesh))
    if len({tuple(x) for x in grid_index}) != len(R):
        raise ValueError('fft_mesh aliases distinct input R vectors; increase mesh')
    Hgrid = np.zeros((*mesh, n, n), dtype=np.complex128)
    Sgrid = np.zeros_like(Hgrid)
    for ir, xyz in enumerate(grid_index):
        idx = tuple(xyz)
        Hgrid[idx] = H_R[ir]
        Sgrid[idx] = S_R[ir]

    # numpy.ifftn has +2pi i k.R and includes 1/N; undo normalization.
    nk = int(np.prod(mesh))
    Hk = np.fft.ifftn(Hgrid, axes=(0, 1, 2)) * nk
    Sk = np.fft.ifftn(Sgrid, axes=(0, 1, 2)) * nk
    del Hgrid, Sgrid
    max_error = 0.0
    smallest_overlap = np.inf
    for ix in np.ndindex(*mesh):
        h = (Hk[ix] + Hk[ix].conj().T) / 2
        s = (Sk[ix] + Sk[ix].conj().T) / 2
        eigenvalues, U = np.linalg.eigh(s)
        smallest_overlap = min(smallest_overlap, eigenvalues[0])
        if eigenvalues[0] <= overlap_floor:
            raise ValueError(f'S(k) not positive definite at grid index {ix}: min={eigenvalues[0]:.4g}')
        Sinvhalf = (U * eigenvalues**(-0.5)) @ U.conj().T
        horth = Sinvhalf @ h @ Sinvhalf
        horth = (horth + horth.conj().T) / 2
        if validate:
            # Generalized eigenvalues equal eigenvalues of the orthogonalized matrix.
            # The similarity is exact up to floating-point roundoff; check Hermiticity.
            max_error = max(max_error, np.max(np.abs(horth - horth.conj().T)))
        Hk[ix] = horth
    del Sk

    # Forward FFT gives H_orth(R) for H(k) = sum_R exp(+2pi i k.R) H(R).
    Horth_grid = np.fft.fftn(Hk, axes=(0, 1, 2)) / nk
    del Hk
    # Retain all FFT coefficients (including long-range induced hoppings).
    # Choose the symmetric representative of each FFT translation.
    axes_R = [np.rint(np.fft.fftfreq(m) * m).astype(int) for m in mesh]
    rout = np.array(np.meshgrid(*axes_R, indexing='ij')).reshape(3, -1).T
    Ham_out = Horth_grid.reshape(nk, n, n)
    print(f'FHI-aims Lowdin: basis={n}, mesh={mesh}, min eig(S)={smallest_overlap:.6g}, '
          f'grid Hermiticity={max_error:.2e}')
    parameters.setdefault('name', 'FHI-aims Lowdin')
    parameters, param_needed_data = NeededData.get_parameters(**parameters)
    needed_data = NeededData(**param_needed_data)
    for key in needed_data.matrices:
        if key != 'Ham':
            raise ValueError(f'Only Ham available; requested {key}')
    system = System_R(**parameters)
    system.real_lattice = lattice
    system.num_wann = n
    system.set_R_mat('Ham', Ham_out)
    system.wannier_centers_cart = centers
    system.clear_cached_wcc()
    system.rvec = Rvectors(lattice=system.real_lattice, iRvec=rout,
                           shifts_left_red=system.wannier_centers_red)
    system.do_at_end_of_init()
    return system
                               
def write_hr_file(system, seedname=None):
    """
    Write the system in the format of the wannier90_hr.dat file, see http://www.wanniertools.com/input.html#wannier90-dat
    for more informations
    """
    if seedname is None:
        seedname = system.seedname
    if seedname is None:
        seedname = "seedname"
    hr_file = seedname + "_hr.dat"
    f = open(hr_file, "w")
    f.write("written by wannier-berri from the chk file\n")
    f.write(f"{system.num_wann}\n")
    f.write(f"{system.rvec.nRvec}\n")
    Ndegen = np.ones(system.rvec.nRvec, dtype=int)
    for i in range(0, system.rvec.nRvec, 15):
        a = Ndegen[i:min(i + 15, system.rvec.nRvec)]
        f.write("  ".join(f"{x:2d}" for x in a) + "\n")
    for iR in range(system.rvec.nRvec):
        _ham = system.Ham_R[iR] * Ndegen[iR]
        for n in system.range_wann:
            for m in system.range_wann:
                f.write(
                    "{:3d} {:3d} {:3d} {:3d} {:3d} {:15.8e} {:15.8e}\n".format(*tuple(system.rvec.iRvec[iR]), m + 1, n + 1, _ham[m, n].real, _ham[m, n].imag)
                )
    f.close()
    r = open(seedname + "_wannier_centre_WT_format.dat", "w")
    data = system.wannier_centers_cart
    for i in data[::2]:
        r.write(f"{(i[0] if np.abs(i[0]) > 1e-7 else 0.0):10} {(i[1] if np.abs(i[1]) > 1e-7 else 0.0):10} {(i[2] if np.abs(i[2]) > 1e-7 else 0.0):10}\n")
    for i in data[1::2]:
        r.write(f"{(i[0] if np.abs(i[0]) > 1e-7 else 0.0):10} {(i[1] if np.abs(i[1]) > 1e-7 else 0.0):10} {(i[2] if np.abs(i[2]) > 1e-7 else 0.0):10}\n")
    r.close()
    from .system_hr import write_WCC_WT_format
    write_WCC_WT_format(seedname, system.wannier_centers_cart)


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
