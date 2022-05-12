class Result():

    def __init__(self):
        raise NotImplementedError()

    #  multiplication by a number
    def __mul__(self, other):
        raise NotImplementedError()

    # +
    def __add__(self, other):
        raise NotImplementedError()

    # -
    def __sub__(self, other):
        raise NotImplementedError()

    # writing to a file
    def savetxt(self, name):
        raise NotImplementedError()

    # saving as binary
    def save(self, name):
        raise NotImplementedError()

    def set_save_mode(self, set_mode):
        self.save_modes = set_mode.split('+')

    #  how result transforms under symmetry operations
    def transform(self, sym):
        raise NotImplementedError()

    # a list of numbers, by each of those the refinement points will be selected
    @property
    def max(self):
        raise NotImplementedError()

    def __truediv__(self, number):
        # not that result/x amd result*(1/x) is not the same thing for tabulation
        raise NotImplementedError()

    # these methods do no need re-implementation:
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
