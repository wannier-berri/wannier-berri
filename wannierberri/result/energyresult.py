import numpy as np
from functools import cached_property
from ..symmetry.point_symmetry import transform_from_dict
from ..smoother import VoidSmoother
from .result import Result
from ..utility import get_head


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
        | smoothers, one per each energy variable (usually do not need to be set by the calculator function).
        | but are set automatically for Fermi levels and Omega's , and during the further * and + operations
    transformTR : :class:~wannierberri.symmetry.Transform
        | How the result transforms  under time-reversal operation
        | relevant if system has TimeReversal, either alone or in combination with spatial symmetyries
    transformInv : :class:~wannierberri.symmetry.Transform
        | How the result transforms  under inversion
        | relevant if system has Inversion, either alone or as part of other symmetyries (e.g. Mx=C2x*I)
    rank : int
        | of the tensor, usually no need, to specify, it is set automatically to the number of dimensions
        | of the `data` array minus number of energies
    E_titles : list of str
        | titles to be printed above the energy columns
    file_npz : str
        | path to a np file (if provided, the parameters `Enegries`, `data`, `transformTR`, `transformInv`, `rank` and
        | `E_titles` are neglected)
    comment : str
        | Any line that can mark what is in the result

     """

    def __init__(
            self,
            Energies=None,
            data=None,
            smoothers=None,
            transformTR=None,
            transformInv=None,
            rank=None,
            E_titles=("Efermi", "Omega"),
            file_npz=None,
            comment="undocumented",
            **kwargs
    ):
        super().__init__(**kwargs)
        if file_npz is not None:
            res = np.load(open(file_npz, "rb"), allow_pickle=True)
            energ = [
                res[f'Energies_{i}'] for i, _ in enumerate(res['E_titles'])
            ]  # in binary mode energies are just two arrays
            try:
                comment = str(res['comment'])
            except KeyError:
                comment = "undocumented"
            try:
                save_mode = res['save_mode']
            except KeyError:
                save_mode = "bin+txt"

            self.__init__(
                Energies=energ,
                data=res['data'],
                smoothers=smoothers,
                # TODO : transform the old Iodd.TRodd,TRtrans into new transformators (if needeed))
                transformTR=transform_from_dict(res, 'transformTR'),
                transformInv=transform_from_dict(res, 'transformInv'),
                rank=res['rank'],
                E_titles=list(res['E_titles']),
                save_mode=save_mode,
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
                assert np.all(np.array(shape) == 3), f"data.shape={data.shape}"
            for i in range(self.N_energies):
                assert (
                    Energies[i].shape[0] == data.shape[i]
                ), f"dimension of Energy[{i}] = {Energies[i].shape[0]} does not match do dimension of data {data.shape[i]}"
            self.Energies = Energies
            self.data = data
            self.set_smoother(smoothers)
            self.transformTR = transformTR
            self.transformInv = transformInv
            self.comment = comment

    def set_smoother(self, smoothers):
        if smoothers is None:
            smoothers = (None,) * self.N_energies
        if not isinstance(smoothers, (list, tuple)):
            smoothers = [smoothers]
        assert len(smoothers) == self.N_energies
        self.smoothers = [(VoidSmoother() if s is None else s) for s in smoothers]

    @cached_property
    def dataSmooth(self):
        data_tmp = self.data.copy()
        for i in range(self.N_energies - 1, -1, -1):
            data_tmp = self.smoothers[i](self.data, axis=i)
        return data_tmp

    def mul_array(self, other, axes=None):
        if isinstance(axes, int):
            axes = (axes,)
        if axes is None:
            axes = tuple(range(other.ndim))
        for i, d in enumerate(other.shape):
            assert d == self.data.shape[axes[i]], \
                f"shapes  {other.shape} should match the axes {axes} of {self.data.shape}"
        reshape = tuple((self.data.shape[i] if i in axes else 1) for i in range(self.data.ndim))
        return self.__class__(
            Energies=self.Energies,
            data=self.data * other.reshape(reshape),
            smoothers=self.smoothers,
            transformTR=self.transformTR,
            transformInv=self.transformInv,
            rank=self.rank,
            save_mode=self.save_mode,
            E_titles=self.E_titles)

    def __mul__(self, number):
        if isinstance(number, int) or isinstance(number, float):
            return self.__class__(
                Energies=self.Energies,
                data=self.data * number,
                smoothers=self.smoothers,
                transformTR=self.transformTR,
                transformInv=self.transformInv,
                rank=self.rank,
                E_titles=self.E_titles,
                save_mode=self.save_mode,
                comment=self.comment)
        else:
            raise TypeError("result can only be multiplied by a number")

    def __truediv__(self, number):
        return self * (1. / number)

    def __add__(self, other):
        if other == 0:
            return self
        if (self.transformTR is not None) and (other.transformTR is not None):
            assert self.transformTR == other.transformTR
        if (self.transformInv is not None) and (other.transformInv is not None):
            assert self.transformInv == other.transformInv
        if len(self.comment) > len(other.comment):
            comment = self.comment
        else:
            comment = other.comment
        for i in range(self.N_energies):
            if np.linalg.norm(self.Energies[i] - other.Energies[i]) > 1e-8:
                raise RuntimeError(f"Adding results with different energies {i} ({self.E_titles[i]}) - not allowed")
            if self.smoothers[i] != other.smoothers[i]:
                raise RuntimeError(
                    f"Adding results with different smoothers [{i}]: {self.smoothers[i]} and {other.smoothers[i]}")
        return self.__class__(
            Energies=self.Energies,
            data=self.data + other.data,
            smoothers=self.smoothers,
            transformTR=self.transformTR,
            transformInv=self.transformInv,
            rank=self.rank,
            E_titles=self.E_titles,
            save_mode=self.save_mode.union(other.save_mode),
            comment=comment)

    def add(self, other):
        self.data += other.data

    def __sub__(self, other):
        return self + (-1) * other

    def __write(self, data, datasm, i):
        if i > self.N_energies:
            raise ValueError(f"not allowed value i={i} > {self.N_energies}")
        elif i == self.N_energies:
            data_tmp = list(data.reshape(-1)) + list(datasm.reshape(-1))
            if data.dtype == complex:
                return ["    " + "    ".join(f"{x.real:15.6e} {x.imag:15.6e}" for x in data_tmp)]
            elif data.dtype == float:
                return ["    " + "    ".join(f"{x:15.6e}" for x in data_tmp)]
        else:
            return [f"{E:15.6e}    {s:s}" for j, E in enumerate(self.Energies[i])
                    for s in self.__write(data[j], datasm[j], i + 1)]

    def savetxt(self, name):
        frmt = "{0:^31s}" if self.data.dtype == complex else "{0:^15s}"
        head = "".join("#### " + s + "\n" for s in self.comment.split("\n"))
        head += "#" + "    ".join(f"{s:^15s}" for s in self.E_titles) + " " * 8 + "    ".join(
            frmt.format(b) for b in get_head(self.rank) * 2) + "\n"
        name = name.format('')

        open(name, "w").write(head + "\n".join(self.__write(self.data, self.dataSmooth, i=0)))

    def as_dict(self):
        """
        returns a dictionary-like object with the folloing keys:
        - 'E_titles' : list of str - titles of the energies on which the result depends
        - 'Energies_0', ['Energies_1', ... ] - corresponding arrays of energies
        - data : array of shape (len(Energies_0), [ len(Energies_1), ...] , [3  ,[ 3, ... ]] )
        """
        energ = {f'Energies_{i}': E for i, E in enumerate(self.Energies)}
        return dict(
            E_titles=self.E_titles,
            data=self.data,
            rank=self.rank,
            transformTR=self.transformTR.as_dict(),
            transformInv=self.transformInv.as_dict(),
            comment=self.comment,
            **energ)

    def savedata(self, name, prefix, suffix, i_iter):
        suffix = "-" + suffix if len(suffix) > 0 else ""
        prefix = prefix + "-" if len(prefix) > 0 else ""
        filename = prefix + name + suffix + f"_iter-{i_iter:04d}"
        if "bin" in self.save_mode:
            self.save(filename)
        if "txt" in self.save_mode:
            self.savetxt(filename + ".dat")

    @property
    def _maxval(self):
        return np.abs(self.dataSmooth).max()

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
        return self.__class__(
            Energies=self.Energies,
            data=sym.transform_tensor(self.data, self.rank,
                                      transformTR=self.transformTR,
                                      transformInv=self.transformInv),
            smoothers=self.smoothers,
            transformTR=self.transformTR,
            transformInv=self.transformInv,
            rank=self.rank,
            E_titles=self.E_titles,
            save_mode=self.save_mode,
            comment=self.comment)
