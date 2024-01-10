"""
The module describes calculators - objects that 
receive :calss:`~wannierberri.data_K._Data_K` objects and yield
:class:`~wannierberri.result.Result`
"""

from termcolor import cprint


class Calculator():

    def __init__(self, degen_thresh=1e-4, degen_Kramers=False, save_mode="bin+txt", print_comment=False):
        self.degen_thresh = degen_thresh
        self.degen_Kramers = degen_Kramers
        self.save_mode = save_mode
        self._set_comment(print_comment)

    @property
    def allow_path(self):
        return False    # change for those who can be calculated on a path instead of a grid

    @property
    def allow_grid(self):
        return True    # change for those who can be calculated ONLY on a path

    def _set_comment(self, print_comment=True):
        if not hasattr(self, 'comment'):
            if self.__doc__ is not None:
                self.comment = self.__doc__
            else:
                self.comment = "calculator not described"
        if print_comment:
            cprint("{}\n".format(self.comment), 'cyan', attrs=['bold'])



from . import static, dynamic, tabulate
from .tabulate import TabulatorAll