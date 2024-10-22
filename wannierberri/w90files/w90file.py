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
        see :class:`~wannierberri.w90files.MMN`, :class:`~wannierberri.w90files.EIG`, :class:`~wannierberri.w90files.AMN`, 
        :class:`~wannierberri.w90files.UIU`, :class:`~wannierberri.w90files.UHU`, :class:`~wannierberri.w90files.SIU`, 
        :class:`~wannierberri.w90files.SHU`, :class:`~wannierberri.w90files.SPN`, :class:`~wannierberri.w90files.WIN`
        :class:'wannierberri.w90files.DMN'	
        for more details

    Attributes
    ----------
    npz_tags : list(str)
        the tags to be saved/loaded in the npz file
    """

    def __init__(self, seedname="wannier90", ext="", read_npz=True, write_npz=True, data=None, selected_bands=None, **kwargs):
        if not hasattr(self, "npz_tags"):
            self.npz_tags = ["data"]
        if data is not None:
            self.data = data
            return
        f_npz = f"{seedname}.{ext}.npz"
        print(f"calling w90 file with {seedname}, {ext}, tags={self.npz_tags}, read_npz={read_npz}, write_npz={write_npz}, kwargs={kwargs}")
        if os.path.exists(f_npz) and read_npz:
            self.from_npz(f_npz)
        else:
            self.from_w90_file(seedname, **kwargs)
            if write_npz:
                self.to_npz(f_npz)
        # window is applied after, so that npz contains same data as original file
        self.apply_window(selected_bands)

    def to_npz(self, f_npz):
        dic = {k: self.__getattribute__(k) for k in self.npz_tags}
        np.savez_compressed(f_npz, **dic)

    def from_npz(self, f_npz):
        dic = np.load(f_npz)
        for k in self.npz_tags:
            self.__setattr__(k, dic[k])


    @abc.abstractmethod
    def from_w90_file(self, **kwargs):
        """
        abstract method to read the necessary data from Wannier90 file
        """
        self.data = None

    @abc.abstractmethod
    def apply_window(self, selected_bands):
        pass

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
