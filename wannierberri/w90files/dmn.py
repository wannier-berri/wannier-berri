from functools import cached_property, lru_cache
from time import time
from irrep.utility import get_block_indices
import numpy as np
from copy import deepcopy
from ..__utility import get_inverse_block, rotate_block_matrix
from .utility import writeints, readints
from .w90file import W90_file
from .amn import AMN


class DMN(W90_file):
    """
    Class to read and store the wannier90.dmn file


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
    d_band_blocks : list(list(numpy.ndarray(complex, shape=(NB, NB))))
        the ab initio band transformation matrices in the block form (between almost degenerate bands)
    d_band_blocks_inverse : list(list(numpy.ndarray(complex, shape=(NB, NB))))
        the inverse matrices of d_band_blocks
    d_band_block_indices : list(np.array(int, shape=(Nblocks, 2)))
        the indices of the blocks in the ab initio band transformation matrices
    D_wann_blocks : list(list(numpy.ndarray(complex, shape=(num_wann, num_wann))))
        the Wannier function transformation matrices in the block form. Within the same irreducible representation of WFs
    D_wann_blocks_inverse : list(list(numpy.ndarray(complex, shape=(num_wann, num_wann))))
        the inverse matrices of D_wann_blocks
    D_wann_block_indices : np.array(int, shape=(Nblocks, 2))
        the indices of the blocks in the Wannier function transformation matrices
    time_reversals : array(bool, shape=(Nsym,))
        weather the symmetry operation includes time reversal or not
    """

    def __init__(self, seedname="wannier90", num_wann=None, num_bands=None, nkpt=None,
                 empty=False,
                 **kwargs):
        self.npz_tags = ['D_wann_block_indices', '_NB',
                    'kpt2kptirr', 'kptirr', 'kptirr2kpt', 'kpt2kptirr_sym',
                   '_NK', 'num_wann', 'comment', 'NKirr', 'Nsym', 'time_reversals']

        if empty:
            self._NB = 0
            self.num_wann = 0
            self.D_wann_block_indices = np.zeros((0, 2), dtype=int)
            self.NKirr = 0
            self.Nsym = 0
            self.kpoints = np.zeros((0, 3), dtype=float)
            self.kpt2kptirr = np.zeros(0, dtype=int)
            self.kptirr = np.zeros(0, dtype=int)
            self.kptirr2kpt = np.zeros((0, 0), dtype=int)
            return
        if seedname is None:
            self.set_identiy(num_wann, num_bands, nkpt)
            return
        super().__init__(seedname, "dmn", **kwargs)

    @cached_property
    def d_band_blocks_inverse(self):
        return get_inverse_block(self.d_band_blocks)

    @cached_property
    def D_wann_blocks_inverse(self):
        return get_inverse_block(self.D_wann_blocks)

    def as_dict(self):
        dic = super().as_dict()
        for ik in range(self.NKirr):
            dic[f'd_band_block_indices_{ik}'] = self.d_band_block_indices[ik]
            for i in range(len(self.d_band_block_indices[ik])):
                dic[f'd_band_blocks_{ik}_{i}'] = np.array([self.d_band_blocks[ik][isym][i] for isym in range(self.Nsym)])
        for i in range(len(self.D_wann_block_indices)):
            dic[f'D_wann_blocks_{i}'] = np.array([[self.D_wann_blocks[ik][isym][i] for isym in range(self.Nsym)]
                                            for ik in range(self.NKirr)])
        return dic


    # def to_npz(self, f_npz):
    #     dic = self.as_dict()
    #     print(f"saving to {f_npz} : ")
    #     np.savez_compressed(f_npz, **dic)
    #     return self

    # def from_npz(self, f_npz):
    #     dic = np.load(f_npz)
    #     self.from_dict(dic)
    #     return self


    def from_dict(self, dic):
        t0 = time()
        super().from_dict(dic)
        # for k in self.npz_tags:
        #     self.__setattr__(k, dic[k])
        t01 = time()
        self.d_band_block_indices = [dic[f'd_band_block_indices_{ik}'] for ik in range(self.NKirr)]
        self.d_band_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
        self.D_wann_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
        d_band_num_blocks = [self.d_band_block_indices[ik].shape[0] for ik in range(self.NKirr)]
        D_wann_num_blocks = self.D_wann_block_indices.shape[0]


        d_band_blocks_tmp = [[dic[f"d_band_blocks_{ik}_{i}"] for i in range(nblock)]
                             for ik, nblock in enumerate(d_band_num_blocks)]
        D_wann_blocks_tmp = [dic[f"D_wann_blocks_{i}"]
                             for i in range(D_wann_num_blocks)]

        for ik in range(self.NKirr):
            for isym in range(self.Nsym):
                self.d_band_blocks[ik][isym] = [np.ascontiguousarray(d_band_blocks_tmp[ik][i][isym])
                                                for i in range(d_band_num_blocks[ik])]
                self.D_wann_blocks[ik][isym] = [np.ascontiguousarray(D_wann_blocks_tmp[i][ik, isym])
                                                for i in range(D_wann_num_blocks)]
        t1 = time()
        print(f"time for read_npz dmn {t1 - t0}\n init {t01 - t0} \n d_blocks {t1 - t01}")
        return self


    @property
    def NK(self):
        return self._NK

    @property
    def NB(self):
        if hasattr(self, '_NB'):
            return self._NB
        else:
            return 0

    @cached_property
    def isym_little(self):
        return [np.where(self.kptirr2kpt[ik] == self.kptirr[ik])[0] for ik in range(self.NKirr)]

    @cached_property
    def kpt2kptirr_sym(self):
        return np.array([np.where(self.kptirr2kpt[self.kpt2kptirr[ik], :] == ik)[0][0] for ik in range(self.NK)])

    def from_w90_file(self, seedname="wannier90", eigenvalues=None):
        """
        eigenvalues np.array(shape=(NK,NB))
        The eigenvalues used to determine the degeneracy of the bandsm and the corresponding blocks
        of matrices which are non-zero

        Parameters
        ----------
        seedname : str
            the prefix of the file (including relative/absolute path, but not including the extensions, like `.dmn`)
        eigenvalues : np.array(shape=(NK,NB)), optional
            The Energies used to determine the degenerecy of the bands
        """
        DeprecationWarning("w90 format for dmn is deprecated is deprecated, use dmn.npz instead")
        fl = open(seedname + ".dmn", "r")
        self.comment = fl.readline().strip()
        self._NB, self.Nsym, self.NKirr, self._NK = readints(fl, 4)
        self.time_reversals = np.zeros(self.Nsym, dtype=bool)  # w90 file does not have time reversal information
        self.kpt2kptirr = readints(fl, self.NK) - 1
        self.kptirr = readints(fl, self.NKirr) - 1
        self.kptirr2kpt = np.array([readints(fl, self.Nsym) for _ in range(self.NKirr)]) - 1
        assert np.all(self.kptirr2kpt.flatten() >= 0), "kptirr2kpt has negative values"
        assert np.all(self.kptirr2kpt.flatten() < self.NK), "kptirr2kpt has values larger than NK"
        assert (set(self.kptirr2kpt.flatten()) == set(range(self.NK))), "kptirr2kpt does not cover all kpoints"
        # find an symmetry that brings the irreducible kpoint from self.kpt2kptirr into the reducible kpoint in question



        # read the rest of lines and convert to conplex array
        data = [l.strip("() \n").split(",") for l in fl.readlines()]
        data = np.array([x for x in data if len(x) == 2], dtype=float)
        data = data[:, 0] + 1j * data[:, 1]
        print("number of numbers in the dmn file :", data.shape)
        print("of those > 1e-8 :", np.sum(np.abs(data) > 1e-8))
        print("of those > 1e-5 :", np.sum(np.abs(data) > 1e-5))

        num_wann = np.sqrt(data.shape[0] // self.Nsym // self.NKirr - self.NB**2)
        assert abs(num_wann - int(num_wann)) < 1e-8, f"num_wann is not an integer : {num_wann}"
        self.num_wann = int(num_wann)
        assert data.shape[0] == (self.num_wann**2 + self.NB**2) * self.Nsym * self.NKirr, \
            f"wrong number of elements in dmn file found {data.shape[0]} expected {(self.num_wann**2 + self.NB**2) * self.Nsym * self.NKirr}"
        n1 = self.num_wann**2 * self.Nsym * self.NKirr
        # in fortran the order of indices is reversed. therefor transpose
        D_wann = data[:n1].reshape(self.NKirr, self.Nsym, self.num_wann, self.num_wann
                                          ).transpose(0, 1, 3, 2)
        d_band = data[n1:].reshape(self.NKirr, self.Nsym, self.NB, self.NB).transpose(0, 1, 3, 2)

        # arranging d_band in the block form
        if eigenvalues is not None:
            print("DMN: eigenvalues are used to determine the block structure")
            self.d_band_block_indices = [get_block_indices(eigenvalues[ik], thresh=1e-2, cyclic=False) for ik in self.kptirr]
        else:
            print("DMN: eigenvalues are NOT provided, the bands are considered as one block")
            self.d_band_block_indices = [[(0, self.NB)] for _ in range(self.NKirr)]
        self.d_band_block_indices = [np.array(self.d_band_block_indices[ik]) for ik in range(self.NKirr)]
        # np.ascontinousarray is used to speedup with Numba
        self.d_band_blocks = [[[np.ascontiguousarray(d_band[ik, isym, start:end, start:end])
                                for start, end in self.d_band_block_indices[ik]]
                               for isym in range(self.Nsym)] for ik in range(self.NKirr)]

        # arranging D_wann in the block form
        self.wann_block_indices = []
        # determine the block indices from the D_wann, excluding areas with only zeros
        start = 0
        thresh = 1e-5
        while start < self.num_wann:
            for end in range(start + 1, self.num_wann):
                if np.all(abs(D_wann[:, :, start:end, end:]) < thresh) and np.all(abs(D_wann[:, :, end:, start:end]) < thresh):
                    self.wann_block_indices.append((start, end))
                    start = end
                    break
            else:
                self.wann_block_indices.append((start, self.num_wann))
                break
        # arange blocks
        self.D_wann_block_indices = np.array(self.wann_block_indices)
        # np.ascontinousarray is used to speedup with Numba
        self.D_wann_blocks = [[[np.ascontiguousarray(D_wann[ik, isym, start:end, start:end]) for start, end in self.D_wann_block_indices]
                               for isym in range(self.Nsym)] for ik in range(self.NKirr)]
        self.clear_inverse()
        return self


    @lru_cache
    def d_band_diagonal(self, ikirr, isym):
        if ikirr is None:
            return np.array([self.d_band_diagonal(ikirr, isym) for ikirr in range(self.NKirr)])
        if isym is None:
            return np.array([self.d_band_diagonal(ikirr, isym) for isym in range(self.Nsym)])

        result = np.zeros(self.NB, dtype=complex)
        for (start, end), block in zip(self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr][isym]):
            result[start:end] = block.diagonal()
        return result

    def d_band_full_matrix(self, ikirr=None, isym=None):
        """
        Returns the full matrix of the ab initio bands transformation matrix

        Note: this funcion is used only for w90 format, which is deprecated. TODO: remove it
        """
        if ikirr is None:
            return np.array([self.d_band_full_matrix(ikirr, isym) for ikirr in range(self.NKirr)])
        if isym is None:
            return np.array([self.d_band_full_matrix(ikirr, isym) for isym in range(self.Nsym)])

        result = np.zeros((self.NB, self.NB), dtype=complex)
        for (start, end), block in zip(self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr][isym]):
            result[start:end, start:end] = block
        return result

    def D_wann_full_matrix(self, ikirr=None, isym=None):
        """
        Returns the full matrix of the Wannier function transformation matrix

        Note: this funcion is used only for w90 format, which is deprecated. TODO: remove it

        """
        if ikirr is None:
            return np.array([self.D_wann_full_matrix(ikirr, isym) for ikirr in range(self.NKirr)])
        if isym is None:
            return np.array([self.D_wann_full_matrix(ikirr, isym) for isym in range(self.Nsym)])

        result = np.zeros((self.num_wann, self.num_wann), dtype=complex)
        for (start, end), block in zip(self.D_wann_block_indices, self.D_wann_blocks[ikirr][isym]):
            result[start:end, start:end] = block
        return result


    def to_w90_file(self, seedname):
        if np.any(self.time_reversals):
            raise ValueError("time reversal information is not supported in wannier90 files")
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
        mat_fun_list = []
        if self.num_wann > 0:
            mat_fun_list.append(self.D_wann_full_matrix)
        if self.NB > 0:
            mat_fun_list.append(self.d_band_full_matrix)

        for M in mat_fun_list:
            for ik in range(self.NKirr):
                for isym in range(self.Nsym):
                    f.write("\n".join("({:17.12e},{:17.12e})".format(x.real, x.imag) for x in M(ik, isym).flatten(order='F')) + "\n\n")


    def get_disentangled(self, v_matrix_dagger, v_matrix):
        """
        Here we will loose the block-diagonal structure of the d_band matrix.
        It is ok, w90 anyway does not use it. This function is only used to finish
        the maximal localization procedure with Wannier90
        """
        NBnew = v_matrix.shape[2]
        d_band_block_indices_new = [np.array([[0, NBnew]]) for _ in range(self.NKirr)]
        d_band_blocks_new = []
        for ikirr, ik in enumerate(self.kptirr):
            d_band_blocks_new.append([])
            for isym in range(self.Nsym):
                ik2 = self.kptirr2kpt[ikirr, isym]
                result = np.zeros((NBnew, NBnew), dtype=complex)
                # print (f"ikirr = {ikirr}, isym = {isym}")
                # print (f"d_band_block_indices[ikirr] = {self.d_band_block_indices[ikirr]}")
                for (start, end), block in zip(self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr][isym]):
                    result[:, :] += v_matrix_dagger[ik2][:, start:end] @ block @ v_matrix[ik][start:end, :]
                # result = v_matrix_dagger[ik2] @ self.d_band_full_matrix(ikirr=ikirr, isym=isym) @ v_matrix[ik]
                assert result.shape == (NBnew, NBnew)
                d_band_blocks_new[ikirr].append([result.copy()])
                # d_band_new[ikirr, isym] = v_matrix_dagger[ik2] @ self.d_band[ikirr, isym] @ v_matrix[ik]
        other = deepcopy(self)
        other.d_band_block_indices = d_band_block_indices_new
        other.d_band_blocks = d_band_blocks_new
        other._NB = NBnew
        return other

    def set_identiy(self, num_wann, num_bands, nkpt):
        """
        set the object to contain only the  transformation matrices
        """
        self.comment = "only identity"
        self._NB, self.Nsym, self.NKirr, self.NK = num_bands, 1, nkpt, nkpt
        self.num_wann = num_wann
        self.kpt2kptirr = np.arange(self.NK)
        self.kptirr = self.kpt2kptirr
        self.kptirr2kpt = np.array([self.kptirr, self.Nsym])
        self.kpt2kptirr_sym = np.zeros(self.NK, dtype=int)
        self.d_band_block_indices = [np.array([(i, i + 1) for i in range(self.NB)])] * self.NKirr
        self.D_wann_block_indices = np.array([(i, i + 1) for i in range(self.num_wann)])
        self.d_band_blocks = [[[np.eye(end - start) for start, end in self.d_band_block_indicespik]
                              for isym in range(self.Nsym)] for ik in range(self.NKirr)]
        self.D_wann_blocks = [[[np.eye(end - start) for start, end in self.D_wann_block_indices]
                               for isym in range(self.Nsym)] for ik in range(self.NKirr)]
        self.clear_inverse()

    def select_bands(self, win_index_irr):
        self.d_band = [D[:, wi, :][:, :, wi] for D, wi in zip(self.d_band, win_index_irr)]

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
        d_indices = self.d_band_block_indices[ikirr]
        D_indices = self.D_wann_block_indices
        # forward = not forward
        U1 = np.zeros(U.shape, dtype=complex)
        Uloc = U.copy()
        if forward:
            if self.time_reversals[isym]:
                Uloc = Uloc.conj()
            Uloc = rotate_block_matrix(Uloc,
                                       lblocks=self.d_band_blocks[ikirr][isym],
                                       lindices=d_indices,
                                       rblocks=self.D_wann_blocks_inverse[ikirr][isym],
                                       rindices=D_indices,
                                    #    inv_left=False, inv_right=True,
                                       result=U1)

            # return d @ U @ D.conj().T
        else:
            Uloc = rotate_block_matrix(Uloc,
                                     lblocks=self.d_band_blocks_inverse[ikirr][isym],
                                     lindices=d_indices,
                                     rblocks=self.D_wann_blocks[ikirr][isym],
                                     rindices=D_indices,
                                     result=U1)
            if self.time_reversals[isym]:
                Uloc = Uloc.conj()

        return Uloc


    def clear_inverse(self, d=True, D=True):
        if d:
            if hasattr(self, 'd_band_blocks_inverse'):
                del self.d_band_blocks_inverse
        if D:
            if hasattr(self, 'D_wann_blocks_inverse'):
                del self.D_wann_blocks_inverse


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
        check that D_wann is a group

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

    def apply_window(self, selected_bands):
        if selected_bands is None:
            return
        print(f"applying window to select {sum(selected_bands)} bands from {self.NB}\n", selected_bands)
        for ikirr in range(self.NKirr):
            self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr] = self.select_window(self.d_band_blocks[ikirr], self.d_band_block_indices[ikirr], selected_bands)
        for i, block_ind in enumerate(self.d_band_block_indices):
            if i == 0:
                self._NB = block_ind[-1, -1]
            assert block_ind[0, 0] == 0
            assert np.all(block_ind[1:, 0] == block_ind[:-1, 1])
            assert block_ind[-1, -1] == self.NB
        print(f"new NB = {self.NB}")



    def select_window(self, d_band_blocks_ik, d_band_block_indices_ik, selected_bands):
        if selected_bands is None:
            return d_band_blocks_ik, d_band_block_indices_ik

        new_block_indices = []
        new_blocks = [[] for _ in range(self.Nsym)]
        st = 0
        for iblock, (start, end) in enumerate(d_band_block_indices_ik):
            select = selected_bands[start:end]
            nsel = np.sum(select)
            if nsel > 0:
                new_block_indices.append((st, st + nsel))
                st += nsel
                for isym in range(self.Nsym):
                    new_blocks[isym].append(
                        np.ascontiguousarray(d_band_blocks_ik[isym][iblock][:, select][select, :]))
        return np.array(new_block_indices), new_blocks

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

    def check_amn(self, amn, warning_precision=1e-5, ignore_upper_bands=None, ignore_lower_bands=None):
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
        assert amn.shape == (self.NK, self.NB, self.num_wann), f"amn.shape = {amn.shape} != (NK={self.NK}, NB={self.NB}, num_wann={self.num_wann}) "
        if ignore_lower_bands is not None:
            assert abs(ignore_lower_bands) < self.NB
            ignore_lower_bands = abs(int(ignore_lower_bands))
        else:
            ignore_lower_bands = 0
        if ignore_upper_bands is not None:
            assert abs(ignore_upper_bands) < self.NB - ignore_lower_bands
            ignore_upper_bands = -abs(int(ignore_upper_bands))

        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                ik = self.kptirr2kpt[ikirr, isym]
                a1 = amn[ik]
                a2 = amn[self.kptirr[ikirr]]
                a1p = self.rotate_U(a1, ikirr, isym, forward=False)
                a1 = a1[ignore_lower_bands:ignore_upper_bands]
                a1p = a1p[ignore_lower_bands:ignore_upper_bands]
                a2 = a2[ignore_lower_bands:ignore_upper_bands]
                diff = a2 - a1p
                diff = np.max(abs(diff))
                maxerr = max(maxerr, np.linalg.norm(diff))
                if diff > warning_precision:
                    print(f"ikirr={ikirr}, isym={isym} : {diff}")
                    for aaa in zip(a1, a1p, a2, a1p - a2, a1p / a2):
                        string = ""
                        for a in aaa:
                            _abs = ", ".join(f"{np.abs(_):.4f}" for _ in a)
                            _angle = ", ".join(f"{np.angle(_) / np.pi * 180:7.2f}" for _ in a)
                            string += f"[{_abs}] [{_angle}]   |    "
                        print(string)
        return maxerr

    def symmetrize_amn(self, amn):
        """
        Symmetrize the amn

        Parameters
        ----------
        amn : AMN object of np.ndarray(complex, shape=(NK, NB, NW))
            the amn

        Returns
        -------
        AMN
            the symmetrized amn
        """
        if isinstance(amn, AMN):
            amn = amn.data
        assert amn.shape == (self.NK, self.NB, self.num_wann)

        amn_sym_irr = np.zeros((self.NKirr, self.NB, self.num_wann), dtype=complex)
        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                amn_sym_irr[ikirr] += self.rotate_U(amn[self.kptirr2kpt[ikirr, isym]], ikirr, isym, forward=False)
        amn_sym_irr /= self.Nsym
        lfound = np.zeros(self.NK, dtype=bool)
        amn_sym = np.zeros((self.NK, self.NB, self.num_wann), dtype=complex)
        for ikirr in range(self.NKirr):
            ik = self.kptirr[ikirr]
            amn_sym[ik] = amn_sym_irr[ikirr]
            lfound[ik] = True
            for isym in range(self.Nsym):
                ik = self.kptirr2kpt[ikirr, isym]
                if not lfound[ik]:
                    amn_sym[ik] = self.rotate_U(amn_sym_irr[ikirr], ikirr, isym, forward=True)
                    lfound[ik] = True
        return amn_sym

    def get_random_amn(self):
        " generate a random amn file that is comaptible with the symmetries of the Wanier functions in the DMN object"
        shape = (self.NK, self.NB, self.num_wann)
        amn = np.random.random(shape) + 1j * np.random.random(shape)
        return AMN(data=self.symmetrize_amn(amn))

    def set_D_wann(self, D_wann):
        """
        set the D_wann matrix

        Parameters
        ----------
        D_wann : np.array(complex, shape=(NKirr, Nsym, num_wann, num_wann))
                the Wannier function transformation matrix (conjugate transpose)
        atommap : np.array(int, shape=(num_points,Nsym))
            the mapping of atoms under the symmetry operations
        T : np.array(int, shape=(num_points,Nsym))
            the translation lattice vector that brings the original atom in position atommap[ip,isym] to the transform of atom ip

        Notes
        -----
        the parameters can be given as lists, in that case the lengths of lists must be equal
          also updates the num_wann attribute
        """
        self.clear_inverse(d=False, D=True)
        if not isinstance(D_wann, list):
            D_wann = [D_wann]
        print("D.shape", [D.shape for D in D_wann])
        self.D_wann_block_indices = []
        num_wann = 0
        self.D_wann_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
        for D in D_wann:
            assert D.shape[0] == self.NKirr
            assert D.shape[1] == self.Nsym
            assert D.shape[2] == D.shape[3]
            self.D_wann_block_indices.append((num_wann, num_wann + D.shape[2]))
            num_wann += D.shape[2]
            for ik in range(self.NKirr):
                for isym in range(self.Nsym):
                    self.D_wann_blocks[ik][isym].append(D[ik, isym])
        self.D_wann_block_indices = np.array(self.D_wann_block_indices)
        self.num_wann = num_wann
        print("num_wann", num_wann)
        print("D_wann_block_indices", self.D_wann_block_indices)




    #
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
