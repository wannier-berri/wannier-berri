from termcolor import cprint


class Calculator:
    """
    Parameters
    -----------
    save_mode : str
        'bin' or 'txt' or 'bin+txt' - save result in text format ('txt') or binary 'npz' files ('bin')
    print_comment : bool
        print the comment (or docstring) during initialization
    degen_Kramers : bool
        consider all bands as Kramers degenerate
    degen_thresh : float
        threshold (in eV) to consider bands as degenerate
    """

    def __init__(self, degen_thresh=1e-4, degen_Kramers=False, save_mode="bin+txt", print_comment=False,

                 ):
        self.degen_thresh = degen_thresh
        self.degen_Kramers = degen_Kramers
        self.save_mode = save_mode
        self._set_comment(print_comment)

    @property
    def allow_path(self):
        return False  # change for those who can be calculated on a path instead of a grid

    @property
    def allow_grid(self):
        return True  # change for those who can be calculated ONLY on a path

    def _set_comment(self, print_comment=True):
        if not hasattr(self, 'comment'):
            if self.__doc__ is not None:
                self.comment = self.__doc__
            else:
                self.comment = "calculator not described"
        if print_comment:
            cprint(self.comment + "\n", 'cyan', attrs=['bold'])


class MultitermCalculator(Calculator):

    def __init__(self, **kwargs):
        kwargs_new = {}
        for k, v in kwargs.items():
            if k in ['save_mode', 'print_comment', 'degen_thresh', 'degen_Kramers']:
                kwargs_new[k] = v
        super().__init__(**kwargs_new)
        self.terms = []


    def __call__(self, data_K):
        return sum(cal(data_K) for cal in self.terms)
