import numpy as np
import sys
from typing import Any


def all_close_mod1(a, b, tol=1e-5):
    """check if two vectors are equal modulo 1"""
    if not np.shape(a) == () and not np.shape(b) == () and (np.shape(a) != np.shape(b)):
        return False
    diff = a - b
    return np.allclose(np.round(diff), diff, atol=tol)


class UniqueList(list):
    """	
    A list that only allows unique elements.
    uniqueness is determined by the == operator.
    Thus, non-hashable elements are also allowed.
    unlike set, the order of elements is preserved.
    """

    def _equal(self, a, b):
        if self.tolerance > 0:
            return np.allclose(a, b, atol=self.tolerance)
        else:
            return a == b

    def __init__(self, iterator=[], count=False, tolerance=-1):
        super().__init__()
        self.tolerance = tolerance
        self.do_count = count
        if self.do_count:
            self.counts = []
        for x in iterator:
            self.append(x)

    def append(self, item, count=1):
        """Add an item to the list, if it is not already present.
        If it is already present, increase the count of the item by 1.

        Parameters
        ----------
        item : object
            The item to add to the list.
        count : int
            The number of times to add the item to the list.
            If the item is already present, this is ignored.

        Returns
        -------
        int
            The index of the item in the list.
            If the item is not already present, it is added to the end of the list.
            If the item is already present, its index is returned. 
        """
        for j, i in enumerate(self):
            if self._equal(i, item):
                # print (f"{item} already in list")
                if self.do_count:
                    self.counts[self.index(i)] += count
                return j
        else:
            super().append(item)
            # print (f"adding {item} to list, length now {len(self)}")
            if self.do_count:
                self.counts.append(1)
            return len(self) - 1

    def index(self, value: Any, start=0, stop=sys.maxsize) -> int:
        stop = min(stop, len(self))
        for i in range(start, stop):
            if self._equal(self[i], value):
                return i
        raise ValueError(f"{value} not in list")

    def index_or_None(self, value: Any, start=0, stop=sys.maxsize) -> int:
        stop = min(stop, len(self))
        for i in range(start, stop):
            if self._equal(self[i], value):
                return i
        return None

    def __contains__(self, item):
        for i in self:
            if self._equal(i, item):
                return True
        return False

    def remove(self, value: Any, all=False) -> None:
        for i in range(len(self)):
            if self._equal(self[i], value):
                if all or not self.do_count:
                    del self[i]
                    del self.counts[i]
                else:
                    self.counts[i] -= 1
                    if self.counts[i] == 0:
                        del self[i]
                        del self.counts[i]
                return


class UniqueListMod1(UniqueList):

    def __init__(self, iterator=[], tol=1e-5):
        self.tol = tol
        self.appended_indices = []
        self.last_try_append = -1
        super().__init__(iterator)

    def append(self, item):
        self.last_try_append += 1
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                break
        else:
            list.append(self, item)
            self.appended_indices.append(self.last_try_append)

    def __contains__(self, item):
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                return True
        return False

    def index(self, value: Any, start=0, stop=sys.maxsize) -> int:
        stop = min(stop, len(self))
        for i in range(start, stop):
            if all_close_mod1(self[i], value):
                return i
        raise ValueError(f"{value} not in list")
