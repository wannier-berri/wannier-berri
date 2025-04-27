import functools
from typing import Iterable
import warnings
import numpy as np
from scipy.constants import physical_constants
from .utility import get_mp_grid



class WIN:
    """
    Class to read and store the wannier90.win input file

    Parameters
    ----------
    seedname : str
         the prefix of the file (including relative/absolute path, but not including the extensions, like `.win`)

    Attributes
    ----------
    name : str
        the name of the file
    parsed : dict(str, Any)
        the parsed data
    units_length : dict(str, float)
        the units of length (Angstrom or Bohr radius)  
    """

    def __init__(self, seedname='wannier90', data=None, autoread=False):
        if not autoread:
            return
        self.data = {}
        self.seedname = seedname
        self.units_length = {'ang': 1., 'bohr': physical_constants['Bohr radius'][0] * 1e10}
        self.blocks = ["unit_cell_cart", "projections", "kpoints", "kpoint_path", "atoms_frac"]
        if seedname is not None:
            name = seedname + ".win"
            self.parsed = parse_win_raw(name)
            self.data.update(self.parsed["parameters"])
            self["unit_cell_cart"] = self.get_unit_cell_cart_ang()
            self["kpoints"] = self.get_kpoints()
            self["projections"] = self.get_projections()
            self["atoms_frac"], self.data["atoms_names"] = self.get_atoms()
        if data is not None:
            for k, v in data.items():
                self.data[k.lower()] = v
            for k in ["kpoints", "unit_cell_cart"]:
                if k in data:
                    self[k] = np.array(data[k], dtype=float)
        if "kpoints" in self.data:
            if self.data["kpoints"].shape[1] > 3:
                self.data["kpoints"] = self.data["kpoints"][:, :3]
            mp_grid = get_mp_grid(self.data["kpoints"])
            if "mp_grid" in self.data:
                assert tuple(mp_grid) == tuple(self.data["mp_grid"])
            else:
                self.data["mp_grid"] = mp_grid
        for key in ["unit_cell_cart", "kpoints", "atoms_frac"]:
            if key in self.data:
                self.data[key] = np.array(self.data[key], dtype=float)



    # @functools.lru_cache()
    # def _get_param(self, param):
    #     """
    #     Get the parameter from the parsed data

    #     Parameters
    #     ----------
    #     param : str
    #         the parameter to be retrieved

    #     Returns
    #     -------
    #     Any
    #         the value of the parameter
    #     """
    #     return self.parsed['parameters'][param]

    # def __getitem__(self, key):
    #     return self.data[key]

        # self.data[key] = value


    def __getitem__(self, key):
        """
        get the value of a parameter

        Parameters:
        -----------
        key(str) : the key of the parameter

        Returns:
        --------
        Any : the value of the parameter
        """
        return self.data[key]

    def __setitem__(self, key, value):
        """
        Set the value of a parameter

        Parameters:
        -----------
        key(str) : the key of the parameter
        value(Any) : the value of the parameter
            the key of the parameter
        """
        self.data[key] = value

    def __delitem__(self, key):
        """
        Delete a parameter

        Parameters:
        -----------
        key(str) : the key of the parameter
        """
        if key in self.data:
            del self.data[key]
        else:
            warnings.warn(f"key {key} not found in the data, nothing to delete")

    def __contains__(self, key):
        """
        Check if a parameter is present

        Parameters:
        -----------
        key(str) : the key of the parameter

        Returns:
        --------
        bool : True if the parameter is present, False otherwise
        """
        return key in self.data

    def update(self, dic):
        """
        Update the parameters with a dictionary

        Parameters:
        -----------
        dic(dict) : the dictionary with the new parameters
        """
        self.data.update(dic)

    def write(self, seedname=None, comment="written by WannierBerri"):
        """
        Write the wannier90.win file

        Parameters
        ----------
        seedname : str
            the prefix of the file (including relative/absolute path, but not including the extensions, like `.win`)
            if None, the file is written to self.seedname + ".win"
        comment : str
            the comment to be written at the beginning of the file
        """
        def list2str(l):
            if isinstance(l, Iterable):
                return " ".join(str(x) for x in l)
            else:
                return str(l)
        if seedname is None:
            seedname = self.seedname
        f = open(seedname + ".win", "w")
        f.write("#" + comment + "\n")
        for k, v in self.data.items():
            if v is not None and k not in ["atoms_names"]:
                if k in self.blocks:
                    f.write(f"begin {k}\n")
                    if isinstance(v, list):
                        for l in v:
                            f.write(l + "\n")
                    elif isinstance(v, np.ndarray):
                        assert v.ndim == 2
                        assert v.dtype in [int, float]
                        if k == "unit_cell_cart":
                            f.write("ang\n")
                        if k == "atoms_frac":
                            names = self.data["atoms_names"]
                        else:
                            names = [""] * v.shape[0]
                        for l, name in zip(v, names):
                            f.write(" " * 5 + name + "   ".join([f"{x:16.12f}" for x in l]) + "\n")
                    elif isinstance(v, str):
                        f.write(v)
                        if v[-1] != "\n":
                            f.write("\n")
                    f.write(f"end {k}\n")
                else:
                    f.write(f"{k} = {list2str(v)}\n")
                f.write("\n")
        f.close()

    @functools.lru_cache()
    def get_unit_cell_cart_ang(self):
        """
        Get the unit cell in Angstrom in Cartesian coordinates

        Returns
        -------
        numpy.ndarray(float, shape=(3, 3))
            the unit cell in Angstrom in Cartesian coordinates
        """
        try:
            cell = self.parsed['unit_cell_cart']
        except KeyError:
            return None
        A = np.array([cell['a1'], cell['a2'], cell['a3']])
        units = cell['units']
        if units is None:
            return A
        else:
            return A * self.units_length[units.lower()]

    @functools.lru_cache()
    def get_kpoints(self):
        """
        Get the kpoints in reciprocal coordinates

        Returns
        -------
        numpy.ndarray(float, shape=(NK, 3))
            the kpoints in reciprocal coordinates
        """
        try:
            kpoints = self.parsed['kpoints']['kpoints']
            kpoints = np.array([k[:3] for k in kpoints])
            return kpoints
        except KeyError:
            return None

    def get_projections(self):
        """
        Get the projections

        Returns
        -------
        list(str)
            the projections in the wannier90 format
        """
        try:
            return [l.strip() for l in self.parsed['projections']['projections']]
        except KeyError:
            return None

    def get_atoms(self):
        if "atoms_frac" in self.parsed:
            atoms = self.parsed["atoms_frac"]["atoms"]
            atoms_names = [a["species"] for a in atoms]
            atoms_frac = np.array([a["basis_vector"] for a in atoms])
            return atoms_frac, atoms_names
        else:
            return None, None


def parse_win_raw(filename=None, text=None):
    """
    Parse the win file (from a file or from a string) using wannier90io

    Parameters
    ----------
    filename : str
        the name of the file to be read
    text : str
        the text to be parsed

    Returns
    -------
    dict(str, Any)
        the parsed data
    """
    import wannier90io as w90io
    if filename is not None:
        with open(filename) as f:
            return w90io.parse_win_raw(f.read())
    elif text is not None:
        return w90io.parse_win_raw(text)
