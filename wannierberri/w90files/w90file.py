import abc
import os
from ..io import SavableNPZ


class W90_file(SavableNPZ):
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
        for more details

    Attributes
    ----------
    npz_tags : list(str)
        the tags to be saved/loaded in the npz file
    """

    def __init__(self, autoread=False, **kwargs):
        if not hasattr(self, "npz_tags"):
            self.npz_tags = ["data"]
        super().__init__()
        if autoread:
            self.init_auto(**kwargs)

    def init_auto(self, seedname="wannier90", ext="", read_npz=True, write_npz=True, data=None, selected_bands=None, **kwargs):
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
        self.select_bands(selected_bands)



    @abc.abstractmethod
    def from_w90_file(self, **kwargs):
        """
        abstract method to read the necessary data from Wannier90 file
        """
        self.data = None

    @abc.abstractmethod
    def select_bands(self, selected_bands):
        """
        abstract method to select the bands from the data

        Select the bands to be used in the calculation, the rest are excluded

        Parameters
        ----------
        selected_bands : list of int
            the list of bands to be used in the calculation
        """
        pass

    @property
    def n_neighb(self):
        """
        number of nearest neighbours (in the k grid) indices 
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
