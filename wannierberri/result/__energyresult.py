import numpy as np
from lazy_property import LazyProperty as Lazy
from wannierberri.smoother import VoidSmoother
from .__result import Result

class EnergyResult(Result):
    """A class to store data dependent on several energies, e.g. Efermi and Omega
      Energy may also be an empty list, then the quantity does not depend on any energy (does it work?)

    Parameters
    -----------
    Energies  : 1D array or list of 1D arrays
        |  The energies, on which the data depend
        |  Energy may also be an empty list, then the quantity does not depend on any energy (does it work?)
    data : array(float) or array(complex)
        | the data. The first dimensions should match the sizes of the Energies arrays. The rest should be equal to 3
    smoothers :  a list of :class:`~wannierberri.smoother.Smoother`
        | smoothers, one per each energy variable (usually do not need to be set by the calculator function.
        | but are set automaticaly for Fermi levels and Omega's , and during the further * and + operations
    TRodd : bool
        | True if the symmetric part of the result is Odd under time-reversal operation (False if it is even)
        | relevant if system has TimeReversal, either alone or in combination with spatial symmetyries
    TRtrans : bool
        | True if the result is should be transposed under time-reversal operation. (i.e. the symmetric and
        | antisymmetric parts have different parity under TR)
        | allowed only for rank-2 tensors
        | relevant if system has TimeReversal, either alone or in combination with spatial symmetyries
    Iodd : bool
        | `True` if the result is Odd under spatial inversion (`False` if it is even)
        | relevant if system has Inversion, either alone or as part of other symmetyries (e.g. Mx=C2x*I)
    rank : int
        | of the tensor, usually no need, to specify, it is set automatically to the number of dimensions
        | of the `data` array minus number of energies
    E_titles : list of str
        | titles to be printed above the energy columns
    file_npz : str
        | path to a np file (if provided, the parameters `Enegries`, `data`, `TRodd`, `Iodd`, `rank` and
        | `E_titles` are neglected)
    comment : str
        | Any line that can mark what is in the result

     """

    def __init__(
            self,
            Energies=None,
            data=None,
            smoothers=None,
            TRodd=False,
            Iodd=False,
            TRtrans=False,
            rank=None,
            E_titles=["Efermi", "Omega"],
            save_mode="txt+bin",
            file_npz=None,
            comment="undocumented"):
        if file_npz is not None:
            res = np.load(open(file_npz, "rb"))
            energ = [
                res[f'Energies_{i}'] for i, _ in enumerate(res['E_titles'])
            ]  # in binary mode energies are just two arrays
            try:
                TRtrans = res['TRtrans']
            except KeyError:
                pass
            try:
                comment = str(res['comment'])
            except KeyError:
                comment = "undocumented"

            self.__init__(
                Energies=energ,
                data=res['data'],
                smoothers=smoothers,
                TRodd=res['TRodd'],
                Iodd=res['Iodd'],
                rank=res['rank'],
                E_titles=list(res['E_titles']),
                TRtrans=TRtrans,
                comment=comment)
        else:
            if not isinstance(Energies, (list, tuple)):
                Energies = [Energies]
            if not isinstance(E_titles, (list, tuple)):
                E_titles = [E_titles]
            E_titles = list(E_titles)

            self.N_energies = len(Energies)
            if self.N_energies <= len(E_titles):
                self.E_titles = E_titles[:self.N_energies]
            else:
                self.E_titles = E_titles + ["???"] * (self.N_energies - len(E_titles))
            self.rank = data.ndim - self.N_energies if rank is None else rank
            if self.rank > 0:
                shape = data.shape[-self.rank:]
                assert np.all(np.array(shape) == 3), "data.shape={}".format(data.shape)
            for i in range(self.N_energies):
                assert (
                    Energies[i].shape[0] == data.shape[i]
                ), f"dimension of Energy[{i}] = {Energies[i].shape[0]} does not match do dimension of data {data.shape[i]}"
            self.Energies = Energies
            self.data = data
            self.set_smoother(smoothers)
            self.TRodd = TRodd
            self.TRtrans = TRtrans
            self.Iodd = Iodd
            self.set_save_mode(save_mode)
            self.comment = comment
            if self.TRtrans:
                assert self.rank == 2


    def set_smoother(self, smoothers):
        if smoothers is None:
            smoothers = (None, ) * self.N_energies
        if not isinstance(smoothers, (list, tuple)):
            smoothers = [smoothers]
        assert len(smoothers) == self.N_energies
        self.smoothers = [(VoidSmoother() if s is None else s) for s in smoothers]

    @Lazy
    def dataSmooth(self):
        data_tmp = self.data.copy()
        for i in range(self.N_energies - 1, -1, -1):
            data_tmp = self.smoothers[i](self.data, axis=i)
        return data_tmp

    def mul_array(self, other, axes=None):
        if isinstance(axes, int):
            axes = (axes, )
        if axes is None:
            axes = tuple(range(other.ndim))
        for i, d in enumerate(other.shape):
            assert d == self.data.shape[axes[i]], "shapes  {} should match the axes {} of {}".format(
                other.shape, axes, self.data.shape)
        reshape = tuple((self.data.shape[i] if i in axes else 1) for i in range(self.data.ndim))
        return EnergyResult(
            Energies=self.Energies,
            data=self.data * other.reshape(reshape),
            smoothers=self.smoothers,
            TRodd=self.TRodd,
            TRtrans=self.TRtrans,
            Iodd=self.Iodd,
            rank=self.rank,
            E_titles=self.E_titles)

    def __mul__(self, number):
        if isinstance(number, int) or isinstance(number, float):
            return EnergyResult(
                Energies=self.Energies,
                data=self.data * number,
                smoothers=self.smoothers,
                TRodd=self.TRodd,
                TRtrans=self.TRtrans,
                Iodd=self.Iodd,
                rank=self.rank,
                E_titles=self.E_titles,
                comment=self.comment)
        else:
            raise TypeError("result can only be multilied by a number")

    def __truediv__(self, number):
        return self * (1. / number)

    def __add__(self, other):
        assert self.TRodd == other.TRodd
        assert self.TRtrans == other.TRtrans
        assert self.Iodd == other.Iodd
        if other == 0:
            return self
        if len(self.comment)>len(other.comment):
            comment = self.comment
        else:
            comment = other.comment
        for i in range(self.N_energies):
            if np.linalg.norm(self.Energies[i] - other.Energies[i]) > 1e-8:
                raise RuntimeError(f"Adding results with different energies {i} ({self.E_titles[i]}) - not allowed")
            if self.smoothers[i] != other.smoothers[i]:
                raise RuntimeError(
                    f"Adding results with different smoothers [{i}]: {self.smoothers[i]} and {other.smoothers[i]}")
        return EnergyResult(
            Energies=self.Energies,
            data=self.data + other.data,
            smoothers=self.smoothers,
            TRodd=self.TRodd,
            TRtrans=self.TRtrans,
            Iodd=self.Iodd,
            rank=self.rank,
            E_titles=self.E_titles,
            comment=comment)

    def __sub__(self, other):
        return self + (-1) * other

    def __write(self, data, datasm, i):
        if i > self.N_energies:
            raise ValueError("not allowed value i={} > {}".format(i, self.N_energies))
        elif i == self.N_energies:
            data_tmp = list(data.reshape(-1)) + list(datasm.reshape(-1))
            if data.dtype == complex:
                return ["    " + "    ".join("{0:15.6e} {1:15.6e}".format(x.real, x.imag) for x in data_tmp)]
            elif data.dtype == float:
                return ["    " + "    ".join("{0:15.6e}".format(x) for x in data_tmp)]
        else:
            return [
                "{0:15.6e}    {1:s}".format(E, s) for j, E in enumerate(self.Energies[i])
                for s in self.__write(data[j], datasm[j], i + 1)
            ]

    def savetxt(self, name):
        frmt = "{0:^31s}" if self.data.dtype == complex else "{0:^15s}"

        def getHead(n):
            if n <= 0:
                return ['  ']
            else:
                return [a + b for a in 'xyz' for b in getHead(n - 1)]
        head = "".join("#### "+s+"\n" for s in self.comment.split("\n") )
        head += "#" + "    ".join("{0:^15s}".format(s) for s in self.E_titles) + " " * 8 + "    ".join(
            frmt.format(b) for b in getHead(self.rank) * 2) + "\n"
        name = name.format('')

        open(name, "w").write(head + "\n".join(self.__write(self.data, self.dataSmooth, i=0)))

    def save(self, name):
        """
        writes a dictionary-like objectto file called `name`  with the folloing keys:
        - 'E_titles' : list of str - titles of the energies on which the result depends
        - 'Energies_0', ['Energies_1', ... ] - corresponding arrays of energies
        - data : array of shape (len(Energies_0), [ len(Energies_1), ...] , [3  ,[ 3, ... ]] )
        """
        name = name.format('')
        energ = {f'Energies_{i}': E for i, E in enumerate(self.Energies)}
        with open(name + ".npz", "wb") as f:
            np.savez_compressed(
                f,
                E_titles=self.E_titles,
                data=self.data,
                rank=self.rank,
                TRodd=self.TRodd,
                TRtrans=self.TRtrans,
                Iodd=self.Iodd,
                comment=self.comment,
                **energ)

    def savedata(self, name, prefix, suffix, i_iter):
        suffix = "-" + suffix if len(suffix) > 0 else ""
        prefix = prefix + "-" if len(prefix) > 0 else ""
        filename = prefix + name + suffix + f"_iter-{i_iter:04d}"
        if "bin" in self.save_modes:
            self.save(filename)
        if "txt" in self.save_modes:
            self.savetxt(filename + ".dat")

    @property
    def _maxval(self):
        return np.abs(self.dataSmooth).max()

    @property
    def _maxval_raw(self):
        return np.abs(self.data).max()

    @property
    def _norm(self):
        return np.linalg.norm(self.dataSmooth)

    @property
    def _normder(self):
        return np.linalg.norm(self.dataSmooth[1:] - self.dataSmooth[:-1])

    @property
    def max(self):
        return np.array([self._maxval, self._norm, self._normder])

    def transform(self, sym):
        return EnergyResult(
            Energies=self.Energies,
            data=sym.transform_tensor(self.data, self.rank, TRodd=self.TRodd, Iodd=self.Iodd, TRtrans=self.TRtrans),
            smoothers=self.smoothers,
            TRodd=self.TRodd,
            TRtrans=self.TRtrans,
            Iodd=self.Iodd,
            rank=self.rank,
            E_titles=self.E_titles,
            comment=self.comment)
