import numpy as np

## TODO : generalize to arbitrary number of band indices ?

from .__kbandresult import KBandResult

class KBandBandResult(KBandResult):

    def __init__(self, data):
        if isinstance(data, list):
            self.data_list = data
        else:
            self.data_list = [data]

    def fit(self, other):
        for var in ['rank', 'nband','nband2']:
            if getattr(self, var) != getattr(other, var):
                return False
        return True


    @property
    def rank(self):
        return len(self.data_list[0].shape) - 3

    @property
    def nband2(self):
        return self.data_list[0].shape[2]


    def __add__(self, other):
        assert self.fit(other)
        return KBandBandResult(self.data_list + other.data_list)

    def __mul__(self, number):
        return KBandBandResult([d * number for d in self.data_list])

    def to_grid(self, k_map):
        dataall = self.data
        data = np.array([sum(dataall[ik] for ik in km) / len(km) for km in k_map])
        return KBandBandResult(data)

    def select_bands2(self, ibands2):
        return KBandBandResult(self.data[:, :, ibands2])

    def average_deg(self, deg):
        pass   # do not average over degenerate
        return self

    def transform(self, sym):
        return self

    @property
    def allow_sym(self):
        return False

    @property
    def allow_frmsf(self):
        return False
