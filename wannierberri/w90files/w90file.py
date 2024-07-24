import abc
import numpy as np
import os


class W90_file(abc.ABC):
    """
    Abstract class for the files of wannier90

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extensions, like `.chk`, `.mmn`, etc)
    ext : str
        the extension of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'
    tags : list(str)
        the tags to be saved in the npz file
    read_npz : bool
        if True, try to read the files converted to npz (e.g. wanier90.mmn.npz instead of wannier90.mmn)
    write_npz : bool
        if True, write the files to npz
    kwargs : dict
        the keyword arguments to be passed to the constructor of the file
        see `~wannierberri.system.w90_files.W90_file`, `~wannierberri.system.w90_files.MMN`, `~wannierberri.system.w90_files.EIG`, `~wannierberri.system.w90_files.AMN`, `~wannierberri.system.w90_files.UIU`, `~wannierberri.system.w90_files.UHU`, `~wannierberri.system.w90_files.SIU`, `~wannierberri.system.w90_files.SHU`, `~wannierberri.system.w90_files.SPN`
        for more details
    """

    def __init__(self, seedname="wannier90", ext="", tags=["data"], read_npz=True, write_npz=True, data=None, **kwargs):
        if data is not None:
            self.data = data
            return
        f_npz = f"{seedname}.{ext}.npz"
        kwargs_str = "; ".join([f"{k}={v}" for k, v in kwargs.items() if k not in ["reorder_bk", ]])
        print(f"calling w90 file with {seedname}, {ext}, tags={tags}, read_npz={read_npz}, write_npz={write_npz}, kwargs={kwargs_str}")
        if os.path.exists(f_npz) and read_npz:
            dic = np.load(f_npz)
            for k in tags:
                self.__setattr__(k, dic[k])
        else:
            self.from_w90_file(seedname, **kwargs)
            dic = {k: self.__getattribute__(k) for k in tags}
            if write_npz:
                np.savez_compressed(f_npz, **dic)

    @abc.abstractmethod
    def from_w90_file(self, **kwargs):
        """
        abstract method to read the necessary data from Wannier90 file
        """
        self.data = None

    def get_disentangled(self, v_matrix_dagger, v_matrix):
        """
        reduce number of bands

        Parameters
        ----------
        v_matrix : np.ndarray(NB,num_wann)
            the matrix of column vectors defining the Wannier gauge

        """
        data = np.einsum("klm,kmn...,kno->klo", v_matrix_dagger, self.data, v_matrix)
        return self.__class__(data=data)

    @property
    def n_neighb(self):
        """
        number of nearest neighbours indices
        """
        return 0

    @property
    def NK(self):

        return self.data.shape[0]

    @property
    def NB(self):
        return self.data.shape[1 + self.n_neighb]

    @property
    def NNB(self):
        if self.n_neighb > 0:
            return self.data.shape[1]
        else:
            return None
