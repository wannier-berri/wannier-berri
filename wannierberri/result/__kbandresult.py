import numpy as np
from .__result import Result
import itertools



class KBandResult(Result):

    def __init__(self, data, TRodd, Iodd):
        if isinstance(data, list):
            self.data_list = data
        else:
            self.data_list = [data]
        self.TRodd = TRodd
        self.Iodd = Iodd

    def fit(self, other):
        for var in ['TRodd', 'Iodd', 'rank', 'nband']:
            if getattr(self, var) != getattr(other, var):
                return False
        return True

    @property
    def data(self):
        if len(self.data_list) > 1:
            self.data_list = [np.vstack(self.data_list)]
        return self.data_list[0]

    @property
    def rank(self):
        return len(self.data_list[0].shape) - 2

    @property
    def nband(self):
        return self.data_list[0].shape[1]

    @property
    def nk(self):
        return sum(data.shape[0] for data in self.data_list)

    def __add__(self, other):
        assert self.fit(other)
        return KBandResult(self.data_list + other.data_list, self.TRodd, self.Iodd)

    def __mul__(self, number):
        return KBandResult([d * number for d in self.data_list], self.TRodd, self.Iodd)

    def __truediv__(self, number):
        return self * 1  # actually a copy

    def to_grid(self, k_map):
        dataall = self.data
        data = np.array([sum(dataall[ik] for ik in km) / len(km) for km in k_map])
        return KBandResult(data, self.TRodd, self.Iodd)

    def select_bands(self, ibands):
        return KBandResult(self.data[:, ibands], self.TRodd, self.Iodd)

    def average_deg(self, deg):
        for i, D in enumerate(deg):
            for ib1, ib2 in D:
                for j in range(len(self.data_list)):
                    self.data_list[j][i, ib1:ib2] = self.data_list[j][i, ib1:ib2].mean(axis=0)
        return self

    def transform(self, sym):
        data = [sym.transform_tensor(data, rank=self.rank, TRodd=self.TRodd, Iodd=self.Iodd) for data in self.data_list]
        return KBandResult(data, self.TRodd, self.Iodd)

    def get_component_list(self):
        dim = len(self.data.shape[2:])
        return ["".join(s) for s in itertools.product(*[("x", "y", "z")] * dim)]

    def get_component(self, component=None):
        xyz = {"x": 0, "y": 1, "z": 2}
        dims = np.array(self.data.shape[2:])
        if not np.all(dims == 3):
            raise RuntimeError(f"dimensions of all components should be 3, found {dims}")

        ndim = len(dims)

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
                return self.data[:, :, xyz[component]]
            elif component == 'norm':
                return np.linalg.norm(self.data, axis=-1)
            elif component == 'sq':
                return np.linalg.norm(self.data, axis=-1)**2
            else:
                raise NoComponentError(component, 1)
        elif ndim == 2:
            if component == "trace":
                return sum([self.data[:, :, i, i] for i in range(3)])
            else:
                try:
                    return self.data[:, :, xyz[component[0]], xyz[component[1]]]
                except IndexError:
                    raise NoComponentError(component, 2)
        elif ndim == 3:
            if component == "trace":
                return sum([self.data[:, :, i, i, i] for i in range(3)])
            else:
                try:
                    return self.data[:, :, xyz[component[0]], xyz[component[1]], xyz[component[2]]]
                except IndexError:
                    raise NoComponentError(component, 3)
        elif ndim == 4:
            if component == "trace":
                return sum([self.data[:, :, i, i, i, i] for i in range(3)])
            else:
                try:
                    return self.data[:, :, xyz[component[0]], xyz[component[1]], xyz[component[2]], xyz[component[3]]]
                except IndexError:
                    raise NoComponentError(component, 4)
        else:
            raise NotImplementedError("writing tensors with rank >4 is not implemented. But easy to do")

class NoComponentError(RuntimeError):

    def __init__(self, comp, dim):
        # Call the base class constructor with the parameters it needs
        super().__init__("component {} does not exist for tensor with dimension {}".format(comp, dim))


