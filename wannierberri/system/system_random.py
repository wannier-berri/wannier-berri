import warnings
import numpy as np
from .system_R import System_R
from .system import num_cart_dim
from ..fourier.rvectors import Rvectors


class SystemRandom(System_R):
    """
    Randomly generated system. Mainly for testing.
    Further can be symmetrized to get generic system with certain symmetries.

    Parameters:
    -----------
    num_wann : int
        number of Wannier functions
    real_lattice : array( (3,3) )
        real lattice vectors. inf None - generated randomly
    nRvec : int
        number of R-space vectors
    max_R:
        maximal coordinate in the R-vectors
    """

    def __init__(self,
                 num_wann,
                 nRvec=10,
                 real_lattice=None,
                 max_R=5,
                 **parameters):
        if "name" not in parameters:
            parameters["name"] = "random"
        super().__init__(**parameters)
        if real_lattice is None:
            while True:
                real_lattice = np.random.random((3, 3))
                d = np.linalg.det(real_lattice)
                if abs(d) > 1e-2:
                    if d < 0:
                        real_lattice *= -1
                    break
        self.set_real_lattice(real_lattice)
        self.num_wann = num_wann

        assert (max_R * 2 + 1)**3 >= nRvec, "too many Rvectors or max_R too small"
        count = 0
        iRvec = {(0, 0, 0)}
        while len(iRvec) < nRvec and count < 10:
            R_try = np.random.randint(low=-max_R, high=max_R + 1, size=(nRvec, 3))
            R_try = set(tuple(R) for R in R_try)
            iRvec.update(R_try)
        if len(iRvec) < nRvec:
            warnings.warn(f"required number of R-vectors {nRvec} was not achieved. got only {len(iRvec)}")
        iRvec = np.array(list(iRvec))
        norm = np.linalg.norm(iRvec, axis=1)
        srt = np.argsort(norm)
        self.iRvec = iRvec[srt][:nRvec]
        np.random.shuffle(self.iRvec)
        self.wannier_centers_cart = np.random.random((self.num_wann, 3))
        self.rvec = Rvectors(
            lattice=self.real_lattice,
            iRvec=list(iRvec),
            shifts_left_red=self.wannier_centers_red,
        )

        for key in self.needed_R_matrices:
            shape = (self.rvec.nRvec, self.num_wann, self.num_wann) + (3,) * num_cart_dim(key)
            im, re = [np.random.random(shape) for _ in (0, 1)]
            self.set_R_mat(key, im + 1j * re)
        if self.has_R_mat('AA'):
            AA = self.get_R_mat('AA')
            AA[self.rvec.iR0, self.range_wann, self.range_wann] = 0
        self.do_at_end_of_init()
