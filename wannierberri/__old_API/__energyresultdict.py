from ..result import ResultDict

class EnergyResultDict(ResultDict):
    '''Stores a dictionary of instances of the class EnergyResult.'''

    def set_smoother(self, smoother):
        for v in self.results.values():
            v.set_smoother(smoother)

    def savedata(self, name, prefix, suffix, i_iter):
        for k, v in self.results.items():
            if not hasattr(v, 'save_modes'):
                v.set_save_modes(self.save_modes)
            v.savedata(name + "-" + k, prefix, suffix, i_iter)

    # writing to a text file
    def savetxt(self, name):
        for k, v in self.results.items():
            v.savetxt(name.format('-' + k + '{}'))  # TODO: check formatting

    # writing to a binary file
    def save(self, name):
        for k, v in self.results.items():
            v.save(name.format('-' + k + '{}'))




