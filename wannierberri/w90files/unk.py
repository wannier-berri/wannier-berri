import os

import numpy as np

from ..io import FortranFileR
from .w90file import W90_file, auto_kptirr, check_shape
from glob import glob


class UNK(W90_file):

    """
    A class that stores all UNK files

    Parameters
    ----------
    data : list of numpy.ndarray( (NB, nr1, nr2, nr3, nspinor), dtype=complex)
        the data of the UNK files, where NB is the number of bands, nr1, nr2, nr3 are the grid sizes,
        and nspinor is 2 if spinor is True, otherwise 1.
        some entries can be None 

    Attributes
    ----------
    NB : int
        number of bands
    grid_size : tuple of int
        the size of the grid (nr1, nr2, nr3)
    spinor : bool
        whether the UNK files are spinor or not
    data : list of numpy.ndarray( (NB, nr1, nr2, nr3, nspinor), dtype=complex)
        the data of the UNK files, where NB is the number of bands, nr1, nr2, nr3 are the grid sizes,
        and nspinor is 2 if spinor is True, otherwise 1.

    """

    extension = 'unk'

    def __init__(self, data, NK=None):
        super().__init__(data=data, NK=NK)
        shape = check_shape(self.data)
        self.NB, self.grid_size, nspinor = shape[0], shape[1:4], shape[4]
        self.spinor = (nspinor == 2)


    @classmethod
    def from_w90_file(cls, seedname=None,
                      path=None,
                      NK=None, NKmax=10000, spinor=False,
                      spin_channel=1,
                      reduce_grid=(1, 1, 1),
                      selected_kpoints=None):

        assert (path is None) != (seedname is None), "either path or seedname should be provided, and not both"
        if path is None:
            path = os.path.dirname(seedname)

        if spinor:
            spin_channel = 'NC'
        else:
            spin_channel = str(spin_channel)


        ik_available = [int(os.path.basename(f).split('.')[0][3:]) - 1 for f in glob(os.path.join(path, f"UNK*.{spin_channel}"))]
        assert len(ik_available) > 0, f"No UNK files found in {path} with spin channel {spin_channel}"
        if NK is None:
            NK = max(ik_available) + 1
        if selected_kpoints is None:
            selected_kpoints = ik_available
        else:
            for k in selected_kpoints:
                assert k in ik_available, f"k-point {k} not found in available k-points {ik_available}"

        data = {}

        nspinor = 2 if spinor else 1
        grid_size = None

        for ik in selected_kpoints:
            filename = os.path.join(path, f"UNK{ik + 1:05d}.{spin_channel}")
            assert os.path.exists(filename), f"UNK file {filename} does not exist"
            # print(f"reading {filename}")
            f = FortranFileR(filename)
            nr1, nr2, nr3, ikr, NB = f.read_record(dtype=np.int32)
            nr1_red, nr2_red, nr3_red = int(nr1 // reduce_grid[0]), int(nr2 // reduce_grid[1]), int(nr3 // reduce_grid[2])
            if grid_size is None:
                grid_size = (nr1, nr2, nr3)
            else:
                assert grid_size == (nr1, nr2, nr3), f"NK={ik} : grid_size={grid_size} != {(nr1, nr2, nr3)}"
            assert ikr == ik + 1, f"read ik = {ikr} from file {filename}, expected {ik + 1}"
            U = np.zeros((NB, nr1_red, nr2_red, nr3_red, nspinor), dtype=complex)
            for ib in range(NB):
                for js in range(nspinor):
                    U[ib, :, :, :, js] = f.read_record(dtype=np.complex128).reshape(
                        nr1, nr2, nr3, order='F')[::reduce_grid[0], ::reduce_grid[1], ::reduce_grid[2]]
                    print(f"norm of band {ib} (spinor {js}) = {np.linalg.norm(U[ib, :, :, :, js])}")
                print(f"norm of band {ib} = {np.linalg.norm(U[ib, :, :, :, :])}")
            data[ik] = U
        return UNK(data=data, NK=NK)

    @classmethod
    def from_bandstructure(cls, bandstructure,
                           grid_size=None,
                           normalize=False,
                           selected_kpoints=None,
                           kptirr=None,
                           NK=None
                           ):
        """
        Initialize UNK from a bandstructure object.
        This is useful for reading UNK files from a bandstructure calculation.
        """
        NK, selected_kpoints, kptirr = auto_kptirr(
            bandstructure, selected_kpoints=selected_kpoints, kptirr=kptirr, NK=NK)

        from ..import IRREP_IRREDUCIBLE_VERSION
        from packaging import version
        from irrep import __version__ as irrep__version__
        irrep_new_version = (version.parse(irrep__version__) >= IRREP_IRREDUCIBLE_VERSION)

        # NK = len(bandstructure.kpoints)
        NB = bandstructure.num_bands
        spinor = bandstructure.spinor
        nspinor = 2 if spinor else 1
        if irrep_new_version:
            ig_list = [kp.ig for kp in bandstructure.kpoints]
        else:
            ig_list = [kp.ig.T for kp in bandstructure.kpoints]
        ig_list = [ig_list[ik] for ik in selected_kpoints]
        if grid_size is None:
            igmin_k = np.array([ig[:, :3].min(axis=0) for ig in ig_list])
            igmax_k = np.array([ig[:, :3].max(axis=0) for ig in ig_list])
            igmin_glob = igmin_k.min(axis=0)
            igmax_glob = igmax_k.max(axis=0)
            ig_grid = igmax_glob - igmin_glob + 1
            grid_size = tuple(ig_grid)
            print(f"grid_size is not provided, using {grid_size} from bandstructure g-vectors")
        else:
            assert len(grid_size) == 3, "grid_size should be a tuple of 3 integers"
            grid_size = tuple(grid_size)
            print(f"using provided grid_size {grid_size}")

        data = {}

        for ikirr in kptirr:
            kp = bandstructure.kpoints[selected_kpoints[ikirr]]
            WF_grid = np.zeros((NB, *grid_size, nspinor), dtype=complex)
            g_loc = ig_list[ikirr][:, :3]
            ng = g_loc.shape[0]
            WF_loc = kp.WF if irrep_new_version else kp.WF.reshape((NB, ng, nspinor), order='F')
            if normalize:
                norm = np.linalg.norm(WF_loc, axis=(1, 2))
                WF_loc = WF_loc / norm[:, None, None]
            for ig, g in enumerate(g_loc):
                WF_grid[:, g[0], g[1], g[2]] = kp.WF[:, ig, :]
            WF_grid = np.fft.ifftn(WF_grid, axes=(1, 2, 3), norm='forward')
            data[ikirr] = WF_grid
        return UNK(data=data, NK=NK)
