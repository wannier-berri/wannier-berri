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

    npz_tags = []
    npz_tags_optional = []
    npz_keys_dict_int = []  # dictionary with int-valued keys

    @abc.abstractmethod
    def __init__(self):
        pass


    def to_npz(self, f_npz):
        dic = self.as_dict()
        print(f"saving to {f_npz} : ")
        np.savez_compressed(f_npz, **dic)
        return self

    @classmethod
    def from_npz(cls, f_npz, **kwargs):
        dic = np.load(f_npz)
        return cls.from_dict(dic, **kwargs)

    def as_dict(self):
        dic = {k: self.__getattribute__(k) for k in self.__class__.npz_tags}
        for k in self.__class__.npz_tags_optional:
            if hasattr(self, k):
                dic[k] = self.__getattribute__(k)
        for tag in self.__class__.npz_keys_dict_int:
            dic.update(dic_to_keydic(self.__getattribute__(tag), tag))
        return dic

    @classmethod
    def from_dict(cls, dic=None, return_obj=True, **kwargs):
        dic_loc = {}
        for k in cls.npz_tags:
            if k in kwargs:
                dic_loc[k] = kwargs[k]
            elif k in dic:
                dic_loc[k] = dic[k]

        for tag in cls.npz_keys_dict_int:
            dic_loc[tag] = keydic_to_dic(dic, tag)

        for k in cls.npz_tags_optional:
            if k in dic:
                dic_loc[k] = dic[k]

        if return_obj:
            return cls(**dic_loc)
        else:
            return dic_loc


def dic_to_keydic(dic, name="data"):
    """
    Converts a dictionary into a new dictionary with keys prefixed by a given name.

    Parameters:
    dic (dict): The input dictionary to be transformed.
    name (str): The prefix to be added to each key in the dictionary. Defaults to "data".

    Returns:
    dict: A new dictionary with keys prefixed by the specified name.
    """

    keydic = {}
    for k, v in dic.items():
        keydic[name + f"_{k}"] = v
    return keydic


def keydic_to_dic(keydic, name="data"):
    """
    Converts a dictionary with prefixed keys into a new dictionary without the prefix.
    the items whith keys that do not start with the prefix are ignored.

    Parameters:
    keydic (dict): The input dictionary with prefixed keys.
    name (str): The prefix to be removed from each key in the dictionary. Defaults to "data".

    Returns:
    dict: A new dictionary with keys without the specified prefix.
    """
    dic = {}
    if name in keydic:
        val = keydic[name]
        if isinstance(val, dict):
            return val
        elif isinstance(val, np.ndarray):
            return {i: v for i, v in enumerate(val)}
        else:
            raise ValueError(f"Expected a dict or an array for key '{name}', got {type(val)}")
    for k, v in keydic.items():
        if k.startswith(name + "_"):
            dic[int(k[len(name) + 1:])] = v
    return dic


def sparselist_to_dict(slist):
    """
    Converts a sparse list (list with None values) to a dictionary with non-None values.

    Parameters:
    slist (list): The input sparse list.

    Returns:
    dict: A dictionary with indices as keys and non-None values as values.
    """
    if isinstance(slist, dict):
        return slist
    return {i: v for i, v in enumerate(slist) if v is not None}
