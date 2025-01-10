from copy import copy
import numpy as np
from .result import Result


class DegenResult(Result):
    """"
    each key of dic is an integer, which is the index of the lower band of the degenerate pair(indexing from 0)
    each value of dic is a list of tuples (k, Eav, dE) where 
        k   is the k-point (in reciprocal coord), 
        Eav is the average energy of the two bands
        dE  is the energy difference between the two bands
    """

    def __init__(self, dic, save_mode="txt", recip_lattice=None):
        assert len(dic) > 0
        for k, v in dic.items():
            assert isinstance(k, int)
            assert len(v) > 0
            assert v.shape[1] == 5
        self.dic = dic
        self.recip_lattice = recip_lattice
        self.save_mode = save_mode


    def __mul__(self, other):
        # K-point factors do not play arole in tabulating quantities
        return self

    def __truediv__(self, other):
        # K-point factors do not play arole in tabulating quantities
        return self

    def __add__(self, other):
        if other == 0:
            return self
        if isinstance(other, DegenResultEmpty):
            return self
        assert self.save_mode == other.save_mode
        assert (np.allclose(self.recip_lattice, other.recip_lattice))
        dic = copy(self.dic)
        for k, v in other.dic.items():
            if k in dic:
                dic[k] = np.vstack((dic[k], v))
            else:
                dic[k] = v
        return DegenResult(dic=dic, save_mode=self.save_mode, recip_lattice=self.recip_lattice)

    def as_dict(self):
        return {f"bands_{k}-{k + 1}": v for k, v in self.dic.items()}

    def savetxt(self, name):
        f = open(name, "w")
        for k, v in self.dic.items():
            f.write(f"#degeneracies between bands {k} nad {k + 1}\n")
            f.write("k1, k2, k3, E, gap\n")
            np.savetxt(f, v)
            f.write("\n\n")

    def savedata(self, name, prefix, suffix, i_iter):
        suffix = "-" + suffix if len(suffix) > 0 else ""
        prefix = prefix + "-" if len(prefix) > 0 else ""
        filename = prefix + name + suffix + f"_iter-{i_iter:04d}"
        if "bin" in self.save_mode:
            self.save(filename)
        if "txt" in self.save_mode:
            self.savetxt(filename + ".dat")

    def transform(self, sym):
        dic = {}
        for k, v in self.dic.items():
            v1 = v.copy()
            v1[:, :3] = sym.transform_reduced_vector(v1[:, :3], self.recip_lattice)
            dic[k] = v1
        return DegenResult(dic=dic, save_mode=self.save_mode, recip_lattice=self.recip_lattice)


class DegenResultEmpty(DegenResult):

    def transform(self, sym):
        return self

    def __init__(self):
        pass

    def __add__(self, other):
        return other

    def savetxt(self, name):
        with open(name, "w") as f:
            f.write("#no degeneracies found")
