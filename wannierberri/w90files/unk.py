import os

import numpy as np

from ..io import FortranFileR
from .w90file import W90_file


class UNK(W90_file):

    """
    A class that stores all UNK files

    Parameters
    ----------
    seedname : str
        the seedname of the wannier90 calculation (including the path. where only the directory is used)
        if seedname is ||path/to/wannier90 the fiiles should be in path/to/UNK00????.1
    selected_bands : list of int
        the list of bands to be read from the UNK files
    path : str
        the path to the UNK files
    NK : int
        the number of UNK files to be read
    NKmax : int
        the maximum number of UNK files to be read (if NK is not provided)
    kptirr : list of int
        the list of kpoints to be read from the UNK files (not implemented yet)
    spinor : bool
        if True, the wavefunctions are expected to be spinors
    """

    def __init__(self, seedname=None, autoread=False, **parameters):
        super().__init__(seedname=seedname, autoread=autoread, **parameters)
        # if autoread:
        #     self.from_w90_file(seedname, **parameters)

    "a class that stores all UNK files"

    def from_w90_file(self, seedname=None, selected_bands=None, path=None, NK=None, NKmax=10000, kptirr=None, spinor=False):

        assert (path is None) != (seedname is None), "either path or seedname should be provided, and not both"
        if path is None:
            path = os.path.dirname(seedname)

        if NK is None:
            print("NK is not provided, reading all UNK until one is missing")
        else:
            NKmax = NK

        self.data = []
        self.spinor = spinor

        nspinor = 2 if spinor else 1
        self.grid_size = None

        # readint = lambda: FIN.read_record('i4')
        # readfloat = lambda: FIN.read_record('f8')
        self._NB = None
        for i in range(NKmax):
            filename = os.path.join(path, f"UNK{i + 1:05d}.1")
            # print(f"trying to read {filename}")
            if (kptirr is None or i in kptirr):
                if os.path.exists(filename):
                    # print(f"reading {filename}")
                    f = FortranFileR(filename)
                    nr1, nr2, nr3, ikr, _NB = f.read_record(dtype=np.int32)
                    if self.grid_size is None:
                        self.grid_size = (nr1, nr2, nr3)
                    else:
                        assert self.grid_size == (nr1, nr2, nr3), f"NK={i} : grid_size={self.grid_size} != {(nr1, nr2, nr3)}"
                    assert ikr == i + 1, f"read ik = {ikr} from file {filename}, expected {i + 1}"
                    if selected_bands is None:
                        selected_bands = np.arange(_NB)
                    if self._NB is None:
                        self._NB = len(selected_bands)
                    else:
                        assert self._NB == len(selected_bands), f"NK={i} : _NB={self._NB} != len(selected_bands)={len(selected_bands)} (selected_bands={selected_bands})"
                    U = np.zeros((self.NB, nr1, nr2, nr3, nspinor), dtype=complex)
                    for i in range(self.NB):
                        for j in range(nspinor):
                            U[i, :, :, :, j] = f.read_record(dtype=np.complex128).reshape(nr1, nr2, nr3, order='F')
                    self.data.append(U)
                else:
                    print(f"{filename} not found, stopping reading UNK files, read {len(self.data)} files")
                    NK = i
                    break
            else:
                print(f"skipping {filename}")
                self.data.append(None)
                continue
        print(f"NK={NK}")



    def select_bands(self, selected_bands):
        if selected_bands is not None:
            self._NB = len(selected_bands)
            for i, u in enumerate(self.data):
                if u is not None:
                    self.data[i] = u[selected_bands]

    @property
    def NB(self):
        return self._NB
