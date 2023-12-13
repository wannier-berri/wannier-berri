import numpy as np
from .__result import Result
import itertools
from ..symmetry import transform_from_dict


class K__Result(Result):


    def __init__(self, data=None, transformTR=None, transformInv=None, file_npz=None, rank=None, other_properties={}):
        assert (data is not None) or (file_npz is not None)
        if file_npz is not None:
            res = np.load(open(file_npz, "rb"), allow_pickle=True)
            self.__init__(
                data=res['data'],
                transformTR=transform_from_dict(res, 'transformTR'),
                transformInv=transform_from_dict(res, 'transformInv'),
            )
        else:
            if data is not None:
                if isinstance(data, list):
                    self.data_list = data
                else:
                    self.data_list = [data]
            self.transformTR = transformTR
            self.transformInv = transformInv
        if rank is None:
            self.rank = self.get_rank()
        else:
            self.rank = rank
        self.other_properties = other_properties


    def fit(self, other):
        for var in ['transformTR', 'transformInv', 'rank']:
            if getattr(self, var) != getattr(other, var):
                print(f"parameters {var} are not fit : `{getattr(self, var)}` and `{getattr(other, var)}` ")
                return False
        return True

    @property
    def data(self):
        if len(self.data_list) > 1:
            self.data_list = [np.vstack(self.data_list)]
        return self.data_list[0]



    @property
    def nk(self):
        return sum(data.shape[0] for data in self.data_list)

    def __add__(self, other):
        assert self.fit(other)
        return self.__class__(data=self.data_list + other.data_list,
                            transformTR=self.transformTR,
                            transformInv=self.transformInv,
                            rank=self.rank,
                            other_properties=self.other_properties
                            )

    def add(self, other):
        self.data_list = [d1 + d2 for d1, d2 in zip(self.data_list, other.data_list)]



    def __mul__(self, number):
        return self.__class__(data=[d * number for d in self.data_list],
                            transformTR=self.transformTR,
                            transformInv=self.transformInv,
                            rank=self.rank,
                            other_properties=self.other_properties
                            )

    def mul_array(self, other, axes=None):
        if isinstance(axes, int):
            axes = (axes, )
        if axes is None:
            axes = tuple(range(other.ndim))
        axes = tuple((a + 1) for a in axes)  # because 0th dimentsion is k here
        for i, d in enumerate(other.shape):
            assert d == self.data_list[0].shape[axes[i]], "shapes  {} should match the axes {} of {}".format(
                other.shape, axes, self.data_list[0].shape)
        reshape = tuple((self.data.shape[i] if i in axes else 1) for i in range(self.data_list[0].ndim))
        other_reshape = other.reshape(reshape)
        return self.__class__(
                            data=[d * other_reshape for d in self.data_list],
                            transformTR=self.transformTR,
                            transformInv=self.transformInv,
                            rank=self.rank,
                            other_properties=self.other_properties
                             )


    def __sub__(self, other):
        if (self.transformTR is not None) and (other.transformTR is not None):
            assert self.transformTR == other.transformTR
        if (self.transformInv is not None) and (other.transformInv is not None):
            assert self.transformInv == other.transformInv
        return KBandResult(
            data=self.data - other.data,
            transformTR=self.transformTR,
            transformInv=self.transformInv,
        )

    def __truediv__(self, number):
        return self * 1  # actually a copy

    def as_dict(self):
        """
        returns a dictionary-like object with the folloing keys:
        - 'E_titles' : list of str - titles of the energies on which the result depends
        - 'Energies_0', ['Energies_1', ... ] - corresponding arrays of energies
        - data : array of shape (len(Energies_0), [ len(Energies_1), ...] , [3  ,[ 3, ... ]] )
        """
        return dict(
                data=self.data,
                transformTR=self.transformTR.as_dict(),
                transformInv=self.transformInv.as_dict()
        )

    def to_grid(self, k_map):
        dataall = self.data
        data = np.array([sum(dataall[ik] for ik in km) / len(km) for km in k_map])
        return self.__class__(data=data,
                            transformTR=self.transformTR,
                            transformInv=self.transformInv,
                            rank=self.rank,
                            other_properties=self.other_properties
                             )

    def average_deg(self, deg):
        for i, D in enumerate(deg):
            for ib1, ib2 in D:
                for j in range(len(self.data_list)):
                    self.data_list[j][i, ib1:ib2] = self.data_list[j][i, ib1:ib2].mean(axis=0)
        return self

    def transform(self, sym):
        data = [sym.transform_tensor(data, rank=self.rank,
                    transformTR=self.transformTR, transformInv=self.transformInv) for data in self.data_list]
        return self.__class__(data,
                            transformTR=self.transformTR,
                            transformInv=self.transformInv,
                            other_properties=self.other_properties,
                            rank=self.rank
                             )

    def get_component_list(self):
        dim = len(self.data.shape[2:])
        comp_list = ["".join(s) for s in itertools.product(*[("x", "y", "z")] * dim)]
        if self.ndim >= 2:
            comp_list.append("trace")
        return comp_list

    @property
    def ndim(self):
        dims = np.array(self.data.shape[2:])
        if not np.all(dims == 3):
            raise RuntimeError(f"dimensions of all components should be 3, found {dims}")
        return len(dims)

    def get_component(self, component=None):
        xyz = {"x": 0, "y": 1, "z": 2}
        ndim = self.ndim

        if component is not None:
            component = component.lower()
        if component == "":
            component = None
        if ndim == 0:
            if component is None:
                return self.data
            else:
                raise NoComponentError(component, 0)
        elif ndim == 1:
            if component in ["x", "y", "z"]:
                return self.data[..., xyz[component]]
            elif component == 'norm':
                return np.linalg.norm(self.data, axis=-1)
            elif component == 'sq':
                return np.linalg.norm(self.data, axis=-1)**2
            else:
                raise NoComponentError(component, 1)
        else:
            dims = tuple(np.arange(self.data.ndim))
            _data = self.data.transpose(dims[-ndim:] + dims[:-ndim])
            print(f"dims={dims}, data_shape={self.data.shape}, , _data_shape={_data.shape}")
            if component == "trace":
                return sum([_data[((i,) * ndim)] for i in range(3)])
            else:
                try:
                    return _data[tuple([xyz[c] for c in component])]
                except IndexError as err:
                    raise NoComponentError(component, 2, err)


class KBandResult(K__Result):

    def get_rank(self):
        return len(self.data_list[0].shape) - 2

    def fit(self, other):
        if self.nband != other.nband:
            print(f"parameter 'nband' does  not match : `{self.nband}` and `{other.nband}` ")
            return False
        return super().fit(other)

    @property
    def nband(self):
        return self.data_list[0].shape[1]

    def select_bands(self, ibands):
        return self.__class__(self.data[:, ibands],
                            transformTR=self.transformTR,
                            transformInv=self.transformInv,
                            rank=self.rank,
                            other_properties=self.other_properties
                            )


class NoComponentError(RuntimeError):

    def __init__(self, comp, dim, err=""):
        # Call the base class constructor with the parameters it needs
        super().__init__("component {} does not exist for tensor with dimension {} :\n{}".format(comp, dim, err))
