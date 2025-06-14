import abc
import os

import numpy as np
from ..io import SavableNPZ


class W90_file(SavableNPZ):
    """
    Abstract class for the files of wannier90

    Parameters
    ----------
    data : dict
        the data of the file

    
    Attributes
    ----------
    npz_tags : list(str)
        the tags to be saved/loaded in the npz file
    """

    npz_keys_dict_int = ["data"]
    npz_tags = ["NK"]
    
    @abc.abstractmethod
    def __init__(self, data: dict |list, NK=None):
        if isinstance(data, list) or isinstance(data, np.ndarray):
            NK = len(data)
            data = {i: d for i, d in enumerate(data) if d is not None}
        assert isinstance(data, dict), "data must be a dictionary or a list of numpy arrays"
        assert NK is not None, "NK must be provided if data is a dictionary"
        self.data = data
        print (f"setting NK = {NK}")
        self.NK = NK
        
    @classmethod
    def autoread(cls, seedname="wannier90", ext="", 
                  read_npz=True, 
                  read_w90=True,
                  bandstructure=None,
                  write_npz=True, 
                  selected_bands=None, 
                  kwargs_w90=None,
                  kwargs_bandstructure=None):
        """First try to read  npz file, then read the w90 file if npz does not exist, 
        otherwise generate from bandstructure if provided.
        """
        
        f_npz = f"{seedname}.{ext}.npz"
        if os.path.exists(f_npz) and read_npz:
            obj = cls.from_npz(f_npz)
            write_npz = False  # do not write npz again if it was read
        elif read_w90 and os.path.exists(f"{seedname}.{ext}"):
            obj = cls.from_w90_file(seedname, **kwargs_w90)
        elif bandstructure is not None:
            if kwargs_bandstructure is None:
                kwargs_bandstructure = {}
            obj = cls.from_bandstructure(bandstructure, **kwargs_bandstructure)
        if write_npz:
            obj.to_npz(f_npz)
        # window is applied after, so that npz contains same data as original file
        obj.select_bands(selected_bands)
        return obj


    @abc.abstractmethod
    def from_w90_file(**kwargs):
        """
        abstract method to read the necessary data from Wannier90 file
        """
        pass
    
    @abc.abstractmethod
    def from_bandstructure(bandstructure, **kwargs):
        """
        abstract method to create the data from a band structure object

        Parameters
        ----------
        bandstructure : irrep.bandstructure.BandStructure
            the band structure object
        """
        pass

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

    def equals(self, other, tolerance=1e-8):
        """
        Compare two W90_file objects for equality.

        Parameters
        ----------
        other : W90_file
            The other W90_file object to compare with.
        tolerance : float, optional
            The tolerance for comparing floating point numbers

        Returns
        -------
        bool
            True if the two objects are equal, False otherwise.
        str
            An error message if the objects are not equal, otherwise an empty string.

        """
        if self.NK != other.NK:
            return False, f"the number of k-points is not equal: {self.NK} and {other.NK} correspondingly"
        if self.NB != other.NB:
            return False, f"the number of bands is not equal: {self.NB} and {other.NB} correspondingly"
        ik1 = set(self.data.keys())
        ik2 = set(other.data.keys())
        if ik1 != ik2 :
            return False, f"the sets of selected k_points are not equal. {ik1} and {ik2} correspondingly"
        for i in ik1:
            if not np.allclose( self.data[i], other.data[i], atol=tolerance):
                return False, f"the data at k-point {i} are not equal, the error is {np.max(np.abs(self.data[i] - other.data[i]))}"
        return True, ""
    
    def select_kpoints(self, selected_kpoints, tag=None):
        """
        Select the k-points from the data. (modify the data in place)

        Parameters
        ----------
        selected_kpoints : list of int
            the list of k-points to be used in the calculation
        tag : str, optional
            the tag of the data to be selected, if None, all data is selected.
            The tags are defined in the npz_keys_dict_int class variable.
            If tag is None, all data is selected.
        Raises
        ------
        AssertionError
            If a selected k-point is not in the data or if the tag is not in the data.
        Returns
        -------
        self : W90_file
            The modified W90_file object with only the selected k-points.
        """
        if tag is None:
            for t in self.__class__.npz_keys_dict_int:
                self.select_kpoints(selected_kpoints, t)
        else:
            dict = getattr(self, tag)
            keys_to_remove = [k for k in dict if k not in selected_kpoints]
            for k in keys_to_remove:
                del dict[k]
            for k in selected_kpoints:
                assert k in dict, f"selected k-point {k} is not in the data {tag}"
            return self
    
    @property
    def num_bands(self):
        return self.NB
    
    @property
    def num_wann(self):
        return self.NW
    
    @property
    def nspinor(self):
        return 2 if self.spinor else 1
    
    


def check_shape(data, shape = None):
    """
    Check if the data has the expected shape.
    
    Parameters
    ----------
    data : dict
        The data to check. a dictionary with keys as integers and values as np.arrays (should be the same shape)
    shape : tuple of int, optional
        The expected shape of the data. If None, taken from the first non-None element of data.
    
    Raises
    ------
    ValueError
        If the data does not have the expected shape or if all elements are None.

    Returns
    -------
    tuple of int
        The shape of the data if it is not None, otherwise None.
    """
    if data is None or len(data) == 0:
        raise ValueError("Data is None or empty")
    for i in sorted(data):
        d = data[i]
        if shape is None:
            shape = d.shape
        elif d.shape != shape:
            raise ValueError(f"Data has unexpected shape {d.shape}, expected {shape}")
    if shape is None:
        raise ValueError("all elements of data are None, cannot determine shape")
    return shape        
    