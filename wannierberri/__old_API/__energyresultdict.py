import numpy as np
from wannierberri.result import EnergyResult

class EnergyResultDict(EnergyResult):
    '''Stores a dictionary of instances of the class Result.'''

    def __init__(self, results):
        '''
        Initialize instance with a dictionary of results with string keys and values of type Result.
        '''
        self.results = results

    def set_smoother(self, smoother):
        for v in self.results.values():
            v.set_smoother(smoother)

    #  multiplication by a number
    def __mul__(self, other):
        return EnergyResultDict({k: v * other for k, v in self.results.items()})

    # +
    def __add__(self, other):
        if other == 0:
            return self
        results = {k: self.results[k] + other.results[k] for k in self.results if k in other.results}
        return EnergyResultDict(results)

    # writing to a text file
    def savedata(self, name, prefix, suffix, i_iter):
        for k, v in self.results.items():
            if not hasattr(v, 'save_modes'):
                v.set_save_modes(self.save_modes)
            v.savedata(name + "-" + k, prefix, suffix, i_iter)

    # -
    def __sub__(self, other):
        return self + (-1) * other

    # writing to a text file
    def savetxt(self, name):
        for k, v in self.results.items():
            v.savetxt(name.format('-' + k + '{}'))  # TODO: check formatting

    # writing to a binary file
    def save(self, name):
        for k, v in self.results.items():
            v.save(name.format('-' + k + '{}'))

    #  how result transforms under symmetry operations
    def transform(self, sym):
        results = {k: self.results[k].transform(sym) for k in self.results}
        return EnergyResultDict(results)

    # a list of numbers, by each of those the refinement points will be selected
    @property
    def max(self):
        return np.array([x for v in self.results.values() for x in v.max])


