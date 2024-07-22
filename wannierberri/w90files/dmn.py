from functools import cached_property

import numpy as np
from copy import deepcopy

from .utility import writeints, readints
from .w90file import W90_file
from .amn import AMN


class DMN(W90_file):
    """
    Class to read and store the wannier90.dmn file
    the symmetry transformation of the Wannier functions and ab initio bands

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extensions, like `.dmn`)
        if None, the object is initialized with void values (zeroes)
    num_wann : int
        the number of Wannier functions (in the case of void initialization)
    num_bands : int
        the number of ab initio bands (in the case of void initialization)
    nkpt : int
        the number of kpoints (in the case of void initialization)

    Attributes
    ----------
    comment : str
        the comment at the beginning of the file
    NB : int
        the number of ab initio bands
    Nsym : int
        the number of symmetries
    NKirr : int
        the number of irreducible kpoints
    NK : int
        the number of kpoints
    num_wann : int
        the number of Wannier functions
    kptirr : numpy.ndarray(int, shape=(NKirr,))
        the list of irreducible kpoints
    kpt2kptirr : numpy.ndarray(int, shape=(NK,))
        the mapping from kpoints to irreducible kpoints (each number denotes the index of the irreducible kpoint in kptirr)
    kptirr2kpt : numpy.ndarray(int, shape=(NKirr, Nsym))
        the mapping from irreducible kpoints to all kpoints 
    kpt2kptirr_sym : numpy.ndarray(int, shape=(NK,))    
        the symmetry that brings the irreducible kpoint from self.kpt2kptirr into the reducible kpoint in question
    D_wann_dag : numpy.ndarray(complex, shape=(NKirr, Nsym, num_wann, num_wann))
        the Wannier function transformation matrix (conjugate transpose)
    d_band : list(numpy.ndarray(complex, shape=(NKirr, Nsym, NB, NB)))
        the ab initio band transformation matrices  
    """

    def __init__(self, seedname="wannier90", num_wann=None, num_bands=None, nkpt=None,
                 **kwargs):
        if seedname is None:
            self.void(num_wann, num_bands, nkpt)
            return

        alltags = ['D_wann', 'd_band', 'kpt2kptirr', 'kptirr', 'kptirr2kpt', 'kpt2kptirr_sym',
                   '_NK', '_NB', 'num_wann', 'comment', 'NKirr', 'Nsym',]
        super().__init__(seedname, "dmn", tags=alltags, **kwargs)

    @property
    def NK(self):
        return self._NK

    @property
    def NB(self):
        return self._NB

    @cached_property
    def isym_little(self):
        return [np.where(self.kptirr2kpt[ik] == self.kptirr[ik])[0] for ik in range(self.NKirr)]

    @cached_property
    def kpt2kptirr_sym(self):
        return np.array([np.where(self.kptirr2kpt[self.kpt2kptirr[ik], :] == ik)[0][0] for ik in range(self.NK)])

    def from_w90_file(self, seedname="wannier90", num_wann=0):
        fl = open(seedname + ".dmn", "r")
        self.comment = fl.readline().strip()
        self._NB, self.Nsym, self.NKirr, self._NK = readints(fl, 4)
        self.kpt2kptirr = readints(fl, self.NK) - 1
        self.kptirr = readints(fl, self.NKirr) - 1
        self.kptirr2kpt = np.array([readints(fl, self.Nsym) for _ in range(self.NKirr)]) - 1
        assert np.all(self.kptirr2kpt.flatten() >= 0), "kptirr2kpt has negative values"
        assert np.all(self.kptirr2kpt.flatten() < self.NK), "kptirr2kpt has values larger than NK"
        assert (set(self.kptirr2kpt.flatten()) == set(range(self.NK))), "kptirr2kpt does not cover all kpoints"
        print(self.kptirr2kpt.shape)
        # find an symmetry that brings the irreducible kpoint from self.kpt2kptirr into the reducible kpoint in question


        # read the rest of lines and convert to conplex array
        data = [l.strip("() \n").split(",") for l in fl.readlines()]
        data = np.array([x for x in data if len(x) == 2], dtype=float)
        data = data[:, 0] + 1j * data[:, 1]
        print(data.shape)
        num_wann = np.sqrt(data.shape[0] // self.Nsym // self.NKirr - self.NB**2)
        assert abs(num_wann - int(num_wann)) < 1e-8, f"num_wann is not an integer : {num_wann}"
        self.num_wann = int(num_wann)
        assert data.shape[0] == (self.num_wann**2 + self.NB**2) * self.Nsym * self.NKirr, \
            f"wrong number of elements in dmn file found {data.shape[0]} expected {(self.num_wann**2 + self.NB**2) * self.Nsym * self.NKirr}"
        n1 = self.num_wann**2 * self.Nsym * self.NKirr
        # in fortran the order of indices is reversed. therefor transpose
        self.D_wann = data[:n1].reshape(self.NKirr, self.Nsym, self.num_wann, self.num_wann
                                          ).transpose(0, 1, 3, 2)
        self.d_band = data[n1:].reshape(self.NKirr, self.Nsym, self.NB, self.NB).transpose(0, 1, 3, 2)

    def to_w90_file(self, seedname):
        f = open(seedname + ".dmn", "w")
        print(f"writing {seedname}.dmn:  comment = {self.comment}")
        f.write(f"{self.comment}\n")
        f.write(f"{self.NB} {self.Nsym} {self.NKirr} {self.NK}\n\n")
        f.write(writeints(self.kpt2kptirr + 1) + "\n")
        f.write(writeints(self.kptirr + 1) + "\n")
        for i in range(self.NKirr):
            f.write(writeints(self.kptirr2kpt[i] + 1) + "\n")
            # " ".join(str(x + 1) for x in self.kptirr2kpt[i]) + "\n")
        # f.write("\n".join(" ".join(str(x + 1) for x in l) for l in self.kptirr2kpt) + "\n")
        for M in self.D_wann, self.d_band:
            for m in M:  # loop over irreducible kpoints
                for s in m:  # loop over symmetries
                    f.write("\n".join("({:17.12e},{:17.12e})".format(x.real, x.imag) for x in s.flatten(order='F')) + "\n\n")

    def get_disentangled(self, v_matrix_dagger, v_matrix):
        NBnew = v_matrix.shape[2]
        d_band_new = np.zeros((self.NKirr, self.Nsym, NBnew, NBnew), dtype=complex)
        for ikirr, ik in enumerate(self.kptirr):
            for isym in range(self.Nsym):
                ik2 = self.kptirr2kpt[ikirr, isym]
                d_band_new[ikirr, isym] = v_matrix_dagger[ik2] @ self.d_band[ikirr, isym] @ v_matrix[ik]
        other = deepcopy(self)
        other._NB = d_band_new.shape[2]
        other.d_band = d_band_new
        return other

    def void(self, num_wann, num_bands, nkpt):
        self.comment = "only identity"
        self.NB, self.Nsym, self.NKirr, self.NK = num_bands, 1, nkpt, nkpt
        self.num_wann = num_wann
        self.kpt2kptirr = np.arange(self.NK)
        self.kptirr = self.kpt2kptirr
        self.kptirr2kpt = np.array([self.kptirr, self.Nsym])
        self.kpt2kptirr_sym = np.zeros(self.NK, dtype=int)
        # read the rest of lines and convert to conplex array
        self.d_band = np.ones((self.NKirr, self.Nsym), dtype=complex)[:, :, None, None] * np.eye(self.NB)[None, None, :, :]
        self.D_wann = np.ones((self.NKirr, self.Nsym), dtype=complex)[:, :, None, None] * np.eye(self.num_wann)[None, None, :, :]


    def select_bands(self, win_index_irr):
        self.d_band = [D[:, wi, :][:, :, wi] for D, wi in zip(self.d_band, win_index_irr)]

    def set_free(self, frozen_irr):
        free = np.logical_not(frozen_irr)
        self.d_band_free = [d[:, f, :][:, :, f] for d, f in zip(self.d_band, free)]

    def write(self):
        print(self.comment)
        print(self.NB, self.Nsym, self.NKirr, self.NK, self.num_wann)
        for i in range(self.NKirr):
            for j in range(self.Nsym):
                print()
                for M in self.D_band[i][j], self.d_wann[i][j]:
                    print("\n".join(" ".join(("X" if abs(x)**2 > 0.1 else ".") for x in m) for m in M) + "\n")

    def rotate_U(self, U, ikirr, isym, forward=True):
        """
        Rotates the umat matrix at the irreducible kpoint
        U = D_band^+ @ U @ D_wann
        """
        if forward:
            return self.d_band[ikirr, isym] @ U @ self.D_wann[ikirr, isym].conj().T
        else:
            return self.d_band[ikirr, isym].conj().T @ U @ self.D_wann[ikirr, isym]

    def rotate_Z(self, Z, isym, ikirr, free=None):
        """
        Rotates the zmat matrix at the irreducible kpoint
        Z = d_band^+ @ Z @ d_band
        """
        d_band = self.d_band[ikirr, isym]
        if free is not None:
            d_band = d_band[free, :][:, free]
        return d_band.conj().T @ Z @ d_band

    def check_unitary(self):
        """
        Check that the transformation matrices are unitary

        Returns
        -------
        float
            the maximum error for the bands 
        float
            the maximum error for the Wannier functions
        """
        maxerr_band = 0
        maxerr_wann = 0
        for ik in range(self.NK):
            ikirr = self.kpt2kptirr[ik]
            for isym in range(self.Nsym):
                d = self.d_band[ikirr, isym]
                w = self.D_wann[ikirr, isym]
                maxerr_band = max(maxerr_band, np.linalg.norm(d @ d.T.conj() - np.eye(self.NB)))
                maxerr_wann = max(maxerr_wann, np.linalg.norm(w @ w.T.conj() - np.eye(self.num_wann)))
        return maxerr_band, maxerr_wann

    def check_group(self, matrices="wann"):
        """
        check that D_wann_dag is a group

        Parameters
        ----------
        matrices : str
            the type of matrices to be checked, either "wann" or "band"

        Returns
        -------
        float
            the maximum error
        """
        if matrices == "wann":
            check_matrices = self.D_wann
        elif matrices == "band":
            check_matrices = self.d_band
        maxerr = 0
        for ikirr in range(self.NKirr):
            Dw = [check_matrices[ikirr, isym] for isym in self.isym_little[ikirr]]
            print(f'ikirr={ikirr} : {len(Dw)} matrices')
            for i1, d1 in enumerate(Dw):
                for i2, d2 in enumerate(Dw):
                    d = d1 @ d2
                    err = [np.linalg.norm(d - _d)**2 for _d in Dw]
                    j = np.argmin(err)
                    print(f"({i1}) * ({i2}) -> ({j})" + (f"err={err[j]}" if err[j] > 1e-10 else ""))
                    maxerr = max(maxerr, err[j])
        return maxerr



    def check_eig(self, eig):
        """
        Check the symmetry of the eigenvlues

        Parameters
        ----------
        eig : EIG object
            the eigenvalues

        Returns
        -------
        float
            the maximum error
        """
        maxerr = 0
        for ik in range(self.NK):
            ikirr = self.kpt2kptirr[ik]
            e1 = eig.data[ik]
            e2 = eig.data[self.kptirr[ikirr]]
            maxerr = max(maxerr, np.linalg.norm(e1 - e2))

        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                e1 = eig.data[self.kptirr[ikirr]]
                e2 = eig.data[self.kptirr2kpt[ikirr, isym]]
                maxerr = max(maxerr, np.linalg.norm(e1 - e2))
        return maxerr

    def check_amn(self, amn):
        """
        Check the symmetry of the amn

        Parameters
        ----------
        amn : AMN object of np.ndarray(complex, shape=(NK, NB, NW))
            the amn

        Returns
        -------
        float
            the maximum error
        """
        if isinstance(amn, AMN):
            amn = amn.data
        maxerr = 0

        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                ik = self.kptirr2kpt[ikirr, isym]
                a1 = amn[self.kptirr[ikirr]]
                a2 = amn[ik]
                diff = a2 - self.rotate_U(a1, ikirr, isym)
                maxerr = max(maxerr, np.linalg.norm(diff))
        return maxerr


    # def check_mmn(self, mmn, f1, f2):
    #     """
    #     Check the symmetry of data in the mmn file (not working)

    #     Parameters
    #     ----------
    #     mmn : MMN object
    #         the mmn file data

    #     Returns
    #     -------
    #     float
    #         the maximum error
    #     """
    #     assert mmn.NK == self.NK
    #     assert mmn.NB == self.NB

    #     maxerr = 0
    #     neighbours_irr = np.array([self.kpt2kptirr[neigh] for neigh in mmn.neighbours])
    #     for i in range(self.NKirr):
    #         ind1 = np.where(self.kpt2kptirr == i)[0]
    #         kirr1 = self.kptirr[i]
    #         neigh_irr = neighbours_irr[ind1]
    #         for j in range(self.NKirr):
    #             kirr2 = self.kptirr[j]
    #             ind2x, ind2y = np.where(neigh_irr == j)
    #             print(f"rreducible kpoints {kirr1} and {kirr2} are equivalent to {len(ind2x)} points")
    #             ref = None
    #             for x, y in zip(ind2x, ind2y):
    #                 k1 = ind1[x]
    #                 k2 = mmn.neighbours[k1][y]
    #                 isym1 = self.kpt2kptirr_sym[k1]
    #                 isym2 = self.kpt2kptirr_sym[k2]
    #                 d1 = self.d_band[i, isym1]
    #                 d2 = self.d_band[j, isym2]
    #                 assert self.kptirr2kpt[i, isym1] == k1
    #                 assert self.kptirr2kpt[j, isym2] == k2
    #                 assert self.kpt2kptirr[k1] == i
    #                 assert self.kpt2kptirr[k2] == j
    #                 ib = np.where(mmn.neighbours[k1] == k2)[0][0]
    #                 assert mmn.neighbours[k1][ib] == k2
    #                 data = mmn.data[k1, ib]
    #                 data = f1(d1) @ data @ f2(d2)
    #                 if ref is None:
    #                     ref = data
    #                     err = 0
    #                 else:
    #                     err = np.linalg.norm(data - ref)
    #                 print(f"   {k1} -> {k2} : {err}")
    #                 maxerr = max(maxerr, err)
    #     return maxerr
