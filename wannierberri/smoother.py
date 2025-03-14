from scipy.constants import Boltzmann, elementary_charge
import abc
import numpy as np
from typing import Union

numeric = Union[int, float]


class AbstractSmoother(abc.ABC):
    """ Smoother for smoothing an array by convolution.
    This is an abstract class which cannot by instantiated. Only the specific child classes
    can be instantiated. Each child class should implement its own version of ``_params``,
    ``__init__``, and ``_broaden``.

    Parameters
    -----------
    E : 1D array
        The energies on which the data are calculated at.
    smear : float
        Smearing parameter in eV.
    maxdE : int
        Determines the width of the convoluting function as (-smear * maxdE, smear * maxdE).
    """

    @property
    @abc.abstractmethod
    def _params(self):
        """list of parameters that uniquely define the smoother."""

    @abc.abstractmethod
    def __init__(self, E: np.ndarray, smear: numeric, maxdE: numeric):
        """initialize Smoother parameters"""
        self.smear = smear
        self.E = np.copy(E)
        self.maxdE = maxdE
        self.dE = E[1] - E[0]
        self.Emin = E[0]
        self.Emax = E[-1]
        self.NE1 = int(self.maxdE * self.smear / self.dE)
        self.NE = E.shape[0]
        self.smt = self._broaden(np.arange(-self.NE1, self.NE1 + 1) * self.dE) * self.dE

    @abc.abstractmethod
    def _broaden(self, E):
        """The broadening method to be used."""

    def __str__(self):
        return f"<{type(self).__name__}>"

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            for param in self._params:
                if not np.allclose(getattr(self, param), getattr(other, param)):
                    return False
        return True

    def __call__(self, A, axis=0):
        """Apply smoother to ``A`` along the given axis"""
        assert self.E.shape[0] == A.shape[axis]
        A = A.transpose((axis,) + tuple(range(0, axis)) + tuple(range(axis + 1, A.ndim)))
        res = np.zeros(A.shape, dtype=A.dtype)
        # TODO maybe change tensordot to numba
        for i in range(self.NE):
            start = max(0, i - self.NE1)
            end = min(self.NE, i + self.NE1 + 1)
            start1 = self.NE1 - (i - start)
            end1 = self.NE1 + (end - i)
            res[i] = np.tensordot(A[start:end], self.smt[start1:end1], axes=(0, 0)) / self.smt[start1:end1].sum()
        return res.transpose(tuple(range(1, axis + 1)) + (0,) + tuple(range(axis + 1, A.ndim)))


class FermiDiracSmoother(AbstractSmoother):
    """ Smoother that uses the derivative of Fermi-Dirac function.

    Parameters
    -----------
    E : 1D array
        The energies on which the data are calculated at.
    T_Kelvin : float
        Temperature in Kelvin. Transformed into self.smear, which is in eV.
    maxdE : flaot
        Determines the width of the convoluting function as (-T * maxdE, T * maxdE)
    """
    _params = ['smear', 'E', 'maxdE', 'NE1']

    def __init__(self, E: np.ndarray, T_Kelvin: float, maxdE: numeric = 8):
        self.T_Kelvin = T_Kelvin
        smear = T_Kelvin * Boltzmann / elementary_charge  # convert K to eV
        super().__init__(E, smear, maxdE)

    def _broaden(self, E):
        return 0.25 / self.smear / np.cosh(E / (2 * self.smear)) ** 2

    def __str__(self):
        return f"<FermiDiracSmoother T={self.smear} ({self.T_Kelvin:.1f} K), NE={self.NE}, NE1={self.NE1}, E={self.Emin}..{self.Emax}, step {self.dE}>"


class GaussianSmoother(AbstractSmoother):
    """ Smoother that uses Gaussian function.

    Parameters
    -----------
    E : 1D array
        The energies on which the data are calculated at.
    smear : float
        Smearing parameter in eV.
    maxdE : int
        Determines the width of the convoluting function as (-smear * maxdE, smear * maxdE)
    """
    _params = ['smear', 'E', 'maxdE', 'NE1']

    def __init__(self, E: np.ndarray, smear: float, maxdE: numeric = 8):
        super().__init__(E, smear, maxdE)

    def _broaden(self, E):
        return np.exp(-(E / self.smear) ** 2) / self.smear / np.sqrt(np.pi)

    def __str__(self):
        return f"<GaussianSmoother smear={self.smear}, NE={self.NE}, NE1={self.NE1}, E={self.Emin}..{self.Emax}, step {self.dE}>"


class VoidSmoother(AbstractSmoother):
    """ Void smoother. When called, do nothing and return the original array."""
    _params = []

    def __init__(self):
        pass

    def _broaden(self, E):
        pass

    def __call__(self, A, axis=0):
        return A


def get_smoother(energy: np.ndarray, smear: float, mode: str = None):
    """
    Return a smoother that applies for the given energy range. The smoother can
    be used a function that applies to an array. The axis of the array to be smoothed
    can be controlled with an argument ``axis``, whose default is 0.

    If you calculated some quantity ``data`` over a range of ``efermi`` and want
    to apply the Fermi-Dirac smoother at 300 K, run the following code::

        smoother = wannierberri.get_smoother(efermi, 300, "Fermi-Dirac")
        data_smooth = smoother(data)

    Parameters
    -----------
    energy : 1D array of float
        The energies on which the data are calculated at. Must be evenly spaced.
    smear : float
        - ``mode == None``: not used.
        - ``mode == "Fermi-Dirac"``: Smearing parameter in Kelvin units.
        - ``mode == "Gaussian"``: Smearing parameter in eV units.
    mode : str or None
        Smoother mode. Default: ``None``.
        Avaliable options: ``None``, ``"Fermi-Dirac"``, and ``"Gaussian"``.
    """

    if energy is None:
        return VoidSmoother()
    if smear is None or smear <= 0:
        return VoidSmoother()
    if len(energy) <= 1:
        return VoidSmoother()
    if mode == "Fermi-Dirac":
        return FermiDiracSmoother(energy, smear)
    elif mode == "Gaussian":
        return GaussianSmoother(energy, smear)
    else:
        raise ValueError("Smoother mode not recognized.")


def ceil_int_frac(x, y):
    """return the integer part of x/y, rounded up"""
    if x%y == 0:
        return x // y
    else:
        return x//y + 1
    

class Smear:

    """ A class to provide the smeared delta function on a pre-defined grid of E-points
    
    Parameters
    -----------
    E : 1D array
        The energies on which the data are calculated at.
    smear : float
        Smearing parameter in eV.
    maxdE : int
        in units of smear, Points distant from the center of the delta function to be included (further - treated as zero).
    """

    def __init__(self, E: np.ndarray, smear: numeric, maxdE: numeric, smr_type ="gaussian", density_loc = 10):
        """initialize Smear parameters"""
        self.smear = smear
        self.E = np.copy(E)
        self.maxdE = maxdE
        self.dE = E[1] - E[0]
        dE_loc = smear/density_loc
        self.denser_E = int(np.ceil(self.dE/dE_loc))
        self.dE_loc = self.dE/self.denser_E
        self.iemax_loc = int(np.ceil(maxdE*smear/self.dE_loc))
        self.E_loc = np.arange(-self.iemax_loc,self.iemax_loc+1)*self.dE_loc
        self.nEloc = len(self.E_loc)
                        
        self.Emin = E[0]
        self.Emax = E[-1]
        self.NE = E.shape[0]
        self.NE_dense = (self.NE-1)*self.denser_E+1
        if smr_type.lower() in ["gaussian", "gauss"]:
            self.smr_loc = np.exp(-(self.E_loc / self.smear) ** 2) / self.smear / np.sqrt(np.pi)
        else:
            raise ValueError(f"Smearing type {smr_type} not recognized")
        
    def __call__(self, e, arr=None):
        """Apply smoother to ``e``
        
        Parameters
        ----------
        e : float
            the energy in eV
        arr : 1D array
            if provided, the value is added to the array. Otherwise a new array is created and returned.
    
        Returns
        -------
        1D array
            the smeared array (if arr is None) or arr (if arr is provided)
        """
        if arr is None:
            arr = np.zeros(self.NE)
        print (f"Emin = {self.Emin}, Emax = {self.Emax}, e = {e}, dE = {self.dE}, dE_loc = {self.dE_loc}, denser_E = {self.denser_E}, iemax_loc = {self.iemax_loc}, nEloc = {self.nEloc}")
        i_center = int(np.round((e-self.Emin)/self.dE_loc)) # 
        i_center_shift = i_center % self.denser_E
        # i_center_result = (i_center - i_center_shift)//self.denser_E
        print (f"i_center = {i_center}, i_center_shift = {i_center_shift}")
        i_start_loc = max((self.iemax_loc - i_center_shift )%self.denser_E, self.iemax_loc-i_center-1)
        i_end_loc = min(self.nEloc, self.NE_dense-i_center+self.iemax_loc)
        i_start_res = max(0, ceil_int_frac(i_center-self.iemax_loc,self.denser_E))
        i_end_res = min(self.NE, (i_center + self.iemax_loc)//self.denser_E+1)
        print (f"i_start_loc = {i_start_loc}, i_end_loc = {i_end_loc}, i_start_res = {i_start_res}, i_end_res = {i_end_res}")
        index_loc = np.arange(i_start_loc, i_end_loc, self.denser_E)
        index_res = np.arange(i_start_res, i_end_res)
        print ("index_loc = ", index_loc)
        print ("index_res = ", index_res)
        assert( len(index_loc) == len(index_res) ) ,  f"len(index_loc) = {len(index_loc)}, len(index_res) = {len(index_res)}"
        if len(index_loc) > 0:
            arr[index_res] += self.smr_loc[index_loc]
        return arr


def test_smear():
    E = np.linspace(-1,1,11)
    smear = 0.1
    smearer = Smear(E=E, smear=smear, maxdE=5, density_loc=100)
    for e in np.random.random((1000,))*4-2:
        print (f"e={e}")
        arr = smearer(e)
        f = np.exp(-((e-E) / smear) ** 2) / smear / np.sqrt(np.pi)
        assert np.allclose(arr, f , atol=1e-3), f"e={e}, \narr={arr} \n f={f}"
        
    