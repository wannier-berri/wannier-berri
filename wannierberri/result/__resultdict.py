#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
# ------------------------------------------------------------
#
#  The purpose of this module is to provide Result classes for
#  different types of  calculations.
#  child classes can be defined specifically in each module

import numpy as np
from .__result import Result


# A class to contain results or a calculation:
# For any calculation there should be a class with the samemethods implemented


# a class for data defined for a set of Fermi levels
# Data is stored in an array data, where first dimension indexes the Fermi level


class ResultDict(Result):
    """Stores a dictionary of instances of the class Result."""

    def __init__(self, results):
        """
        Initialize instance with a dictionary of results with string keys and values of type Result.
        """
        self.results = results

    #  multiplication by a number
    def __mul__(self, number):
        return ResultDict({k: v * number for k, v in self.results.items()})

    def __truediv__(self, number):
        return ResultDict({k: v / number for k, v in self.results.items()})

    # +
    def __add__(self, other):
        if other == 0:
            return self
        results = {k: self.results[k] + other.results[k] for k in self.results if k in other.results}
        return ResultDict(results)

    # -
    def __sub__(self, other):
        return self + (-1) * other

    # writing to a text file
    def savedata(self, prefix, suffix, i_iter):
        for k, v in self.results.items():
            v.savedata(k, prefix, suffix, i_iter)

    #  how result transforms under symmetry operations
    def transform(self, sym):
        results = {k: self.results[k].transform(sym) for k in self.results}
        return ResultDict(results)

    # a list of numbers, by each of those the refinement points will be selected
    @property
    def max(self):
        return np.array([x for v in self.results.values() for x in v.max])
