import numpy as np
import abc


class Result(abc.ABC):

    def __init__(self, save_mode="bin+txt"):
        self.save_mode = set()
        for s in "bin", "txt":
            if s in save_mode:
                self.save_mode.add(s)

    #  multiplication by a number
    @abc.abstractmethod
    def __mul__(self, other):
        raise NotImplementedError()

    # +
    @abc.abstractmethod
    def __add__(self, other):
        raise NotImplementedError()

    # -
    def __sub__(self, other):
        raise NotImplementedError()

    @abc.abstractmethod
    def as_dict(self):
        pass

    # writing to a file
    def savetxt(self, name):
        raise NotImplementedError()

    # saving as binary
    def save(self, name):
        """
        writes a dictionary-like object to file called `name`  defined in
        :func:`~wannierberri.result.EnergyResult.as_dict`
        """
        name = name.format('')
        with open(name + ".npz", "wb") as f:
            np.savez_compressed(f, **self.as_dict(), allow_pickle=True)

    @property
    def _maxval_raw(self):
        return np.abs(self.data).max()

    #  how result transforms under symmetry operations
    def transform(self, sym):
        raise NotImplementedError()

    # a list of numbers, by each of those the refinement points will be selected
    @property
    def max(self):
        raise NotImplementedError()

    def __truediv__(self, number):
        # not that result/x amd result*(1/x) is not the same thing for tabulation
        raise NotImplementedError()

    # these methods do no need re-implementation:
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
