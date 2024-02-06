import numpy as np

from .system_R import System_R, ndim_R


class SystemRandom(System_R):
    """
    Randomly generated system

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

        super().__init__(**parameters)
        if real_lattice is None:
            self.real_lattice = np.random.random((3, 3))
        else:
            self.real_lattice = real_lattice

        self.num_wann = num_wann

        assert 3 * (max_R * 2 + 1) >= nRvec
        count = 0
        iRvec = set({(0, 0, 0)})
        while len(iRvec) < nRvec:
            R_try = np.random.randint(low=-max_R, high=max_R + 1, size=(nRvec, 3))
            R_try = set(tuple(R) for R in R_try)
            iRvec.update(R_try)
            print(f"iRvec1={iRvec}")
            if count >= 10:
                break
        if len(iRvec) < nRvec:
            print(f"WARNING : required number of R-vectors {nRvec} was not achieved. got only {len(iRvec)}")
        print(f"iRvec2={iRvec}")
        self.iRvec = np.array(list(iRvec)[:nRvec])
        print(f"iRvec3={iRvec}")
        np.random.shuffle(self.iRvec)
        print(f"iRvec4={self.iRvec}")
        for key in self.needed_R_matrices:
            shape = (self.num_wann, self.num_wann, self.nRvec,) + (3,) * ndim_R(key)
            im, re = [np.random.random(shape) for _ in (0, 1)]
            self.set_R_mat(key, im + 1j * re)
        self.wannier_centers_cart = np.random.random((self.num_wann, 3))

        self.do_at_end_of_init(convert_convention=False)
