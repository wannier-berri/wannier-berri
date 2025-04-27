# inheriting just in order to have possibility to change default values, without changing the rest of the code
import abc
import numpy as np
import scipy
import scipy.io
import fortio


class FortranFileR(fortio.FortranFile):

    def __init__(self, filename):
        # print("using fortio to read")
        try:
            super().__init__(filename, mode='r', header_dtype='uint32', auto_endian=True, check_file=True)
        except ValueError:
            print(f"File '{filename}' contains sub-records - using header_dtype='int32'")
            super().__init__(filename, mode='r', header_dtype='int32', auto_endian=True, check_file=True)


class FortranFileW(scipy.io.FortranFile):

    def __init__(self, filename):
        print("using scipy.io to write")
        super().__init__(filename, mode='w')


class SavableNPZ(abc.ABC):
    """
    A class that can be saved to a npz file and loaded from it.
    """

    def __init__(self):
        if not hasattr(self, "npz_tags_optional"):
            self.npz_tags_optional = []
        if not hasattr(self, "default_tags"):
            self.default_tags = {}
        if not hasattr(self, "npz_tags"):
            self.npz_tags = []


    def to_npz(self, f_npz):
        dic = self.as_dict()
        print(f"saving to {f_npz} : ")
        np.savez_compressed(f_npz, **dic)
        return self

    def from_npz(self, f_npz):
        dic = np.load(f_npz)
        self.from_dict(dic)
        return self

    def as_dict(self):
        dic = {k: self.__getattribute__(k) for k in self.npz_tags}
        for k in self.npz_tags_optional:
            if hasattr(self, k):
                dic[k] = self.__getattribute__(k)
        return dic

    def from_dict(self, dic=None, **kwargs):
        if dic is None:
            dic = {}
        else:
            dic = dict(dic)
        
        dic.update(kwargs)
        for k in self.npz_tags:
            if k in dic:
                self.__setattr__(k, dic[k])
            else:
                self.__setattr__(k, self.default_tags[k])
        for k in self.npz_tags_optional:
            if k in dic:
                self.__setattr__(k, dic[k])
            elif k in self.default_tags:
                self.__setattr__(k, self.default_tags[k])
        return self
