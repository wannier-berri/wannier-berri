# # this module is obsolete, and will be removed in the future
# from functools import cached_property, lru_cache
# from time import time
# import numpy as np
# from copy import deepcopy
# from ..io import SavableNPZ
# from .utility import get_inverse_block, rotate_block_matrix


# class DMN(SavableNPZ):




#     # def as_dict(self):
#     #     dic = super().as_dict()
#     #     for ik in range(self.NKirr):
#     #         dic[f'd_band_block_indices_{ik}'] = self.d_band_block_indices[ik]
#     #         for i in range(len(self.d_band_block_indices[ik])):
#     #             dic[f'd_band_blocks_{ik}_{i}'] = np.array([self.d_band_blocks[ik][isym][i] for isym in range(self.Nsym)])
#     #     for i in range(len(self.D_wann_block_indices)):
#     #         dic[f'D_wann_blocks_{i}'] = np.array([[self.D_wann_blocks[ik][isym][i] for isym in range(self.Nsym)]
#     #                                         for ik in range(self.NKirr)])
#     #     return dic




#     # def from_dict(self, dic):
#     #     t0 = time()
#     #     super().from_dict(dic)
#     #     # for k in self.npz_tags:
#     #     #     self.__setattr__(k, dic[k])
#     #     t01 = time()
#     #     self.d_band_block_indices = [dic[f'd_band_block_indices_{ik}'] for ik in range(self.NKirr)]
#     #     self.d_band_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
#     #     self.D_wann_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
#     #     d_band_num_blocks = [self.d_band_block_indices[ik].shape[0] for ik in range(self.NKirr)]
#     #     D_wann_num_blocks = self.D_wann_block_indices.shape[0]


#     #     d_band_blocks_tmp = [[dic[f"d_band_blocks_{ik}_{i}"] for i in range(nblock)]
#     #                          for ik, nblock in enumerate(d_band_num_blocks)]
#     #     D_wann_blocks_tmp = [dic[f"D_wann_blocks_{i}"]
#     #                          for i in range(D_wann_num_blocks)]

#     #     for ik in range(self.NKirr):
#     #         for isym in range(self.Nsym):
#     #             self.d_band_blocks[ik][isym] = [np.ascontiguousarray(d_band_blocks_tmp[ik][i][isym])
#     #                                             for i in range(d_band_num_blocks[ik])]
#     #             self.D_wann_blocks[ik][isym] = [np.ascontiguousarray(D_wann_blocks_tmp[i][ik, isym])
#     #                                             for i in range(D_wann_num_blocks)]
#     #     t1 = time()
#     #     print(f"time for read_npz dmn {t1 - t0}\n init {t01 - t0} \n d_blocks {t1 - t01}")
#     #     return self



#     # def from_w90_file(self, seedname="wannier90", eigenvalues=None):
#     #     """
#     #     eigenvalues np.array(shape=(NK,NB))
#     #     The eigenvalues used to determine the degeneracy of the bandsm and the corresponding blocks
#     #     of matrices which are non-zero

#     #     Parameters
#     #     ----------
#     #     seedname : str
#     #         the prefix of the file (including relative/absolute path, but not including the extensions, like `.dmn`)
#     #     eigenvalues : np.array(shape=(NK,NB)), optional
#     #         The Energies used to determine the degenerecy of the bands
#     #     """
#     #     DeprecationWarning("w90 format for dmn is deprecated is deprecated, use dmn.npz instead")
#     #     fl = open(seedname + ".dmn", "r")
#     #     self.comment = fl.readline().strip()
#     #     self._NB, self.Nsym, self.NKirr, self._NK = readints(fl, 4)
#     #     self.time_reversals = np.zeros(self.Nsym, dtype=bool)  # w90 file does not have time reversal information
#     #     self.kpt2kptirr = readints(fl, self.NK) - 1
#     #     self.kptirr = readints(fl, self.NKirr) - 1
#     #     self.kptirr2kpt = np.array([readints(fl, self.Nsym) for _ in range(self.NKirr)]) - 1
#     #     assert np.all(self.kptirr2kpt.flatten() >= 0), "kptirr2kpt has negative values"
#     #     assert np.all(self.kptirr2kpt.flatten() < self.NK), "kptirr2kpt has values larger than NK"
#     #     assert (set(self.kptirr2kpt.flatten()) == set(range(self.NK))), "kptirr2kpt does not cover all kpoints"
#     #     # find an symmetry that brings the irreducible kpoint from self.kpt2kptirr into the reducible kpoint in question



#     #     # read the rest of lines and convert to conplex array
#     #     data = [l.strip("() \n").split(",") for l in fl.readlines()]
#     #     data = np.array([x for x in data if len(x) == 2], dtype=float)
#     #     data = data[:, 0] + 1j * data[:, 1]
#     #     print("number of numbers in the dmn file :", data.shape)
#     #     print("of those > 1e-8 :", np.sum(np.abs(data) > 1e-8))
#     #     print("of those > 1e-5 :", np.sum(np.abs(data) > 1e-5))

#     #     num_wann = np.sqrt(data.shape[0] // self.Nsym // self.NKirr - self.NB**2)
#     #     assert abs(num_wann - int(num_wann)) < 1e-8, f"num_wann is not an integer : {num_wann}"
#     #     self.num_wann = int(num_wann)
#     #     assert data.shape[0] == (self.num_wann**2 + self.NB**2) * self.Nsym * self.NKirr, \
#     #         f"wrong number of elements in dmn file found {data.shape[0]} expected {(self.num_wann**2 + self.NB**2) * self.Nsym * self.NKirr}"
#     #     n1 = self.num_wann**2 * self.Nsym * self.NKirr
#     #     # in fortran the order of indices is reversed. therefor transpose
#     #     D_wann = data[:n1].reshape(self.NKirr, self.Nsym, self.num_wann, self.num_wann
#     #                                       ).transpose(0, 1, 3, 2)
#     #     d_band = data[n1:].reshape(self.NKirr, self.Nsym, self.NB, self.NB).transpose(0, 1, 3, 2)

#     #     # arranging d_band in the block form
#     #     if eigenvalues is not None:
#     #         print("DMN: eigenvalues are used to determine the block structure")
#     #         self.d_band_block_indices = [get_block_indices(eigenvalues[ik], thresh=1e-2, cyclic=False) for ik in self.kptirr]
#     #     else:
#     #         print("DMN: eigenvalues are NOT provided, the bands are considered as one block")
#     #         self.d_band_block_indices = [[(0, self.NB)] for _ in range(self.NKirr)]
#     #     self.d_band_block_indices = [np.array(self.d_band_block_indices[ik]) for ik in range(self.NKirr)]
#     #     # np.ascontinousarray is used to speedup with Numba
#     #     self.d_band_blocks = [[[np.ascontiguousarray(d_band[ik, isym, start:end, start:end])
#     #                             for start, end in self.d_band_block_indices[ik]]
#     #                            for isym in range(self.Nsym)] for ik in range(self.NKirr)]

#     #     # arranging D_wann in the block form
#     #     self.wann_block_indices = []
#     #     # determine the block indices from the D_wann, excluding areas with only zeros
#     #     start = 0
#     #     thresh = 1e-5
#     #     while start < self.num_wann:
#     #         for end in range(start + 1, self.num_wann):
#     #             if np.all(abs(D_wann[:, :, start:end, end:]) < thresh) and np.all(abs(D_wann[:, :, end:, start:end]) < thresh):
#     #                 self.wann_block_indices.append((start, end))
#     #                 start = end
#     #                 break
#     #         else:
#     #             self.wann_block_indices.append((start, self.num_wann))
#     #             break
#     #     # arange blocks
#     #     self.D_wann_block_indices = np.array(self.wann_block_indices)
#     #     # np.ascontinousarray is used to speedup with Numba
#     #     self.D_wann_blocks = [[[np.ascontiguousarray(D_wann[ik, isym, start:end, start:end]) for start, end in self.D_wann_block_indices]
#     #                            for isym in range(self.Nsym)] for ik in range(self.NKirr)]
#     #     self.clear_inverse()
#     #     return self


#     # def to_npz(self, f_npz):
#     #     dic = self.as_dict()
#     #     print(f"saving to {f_npz} : ")
#     #     np.savez_compressed(f_npz, **dic)
#     #     return self

#     # def from_npz(self, f_npz):
#     #     dic = np.load(f_npz)
#     #     self.from_dict(dic)
#     #     return self


#     # def d_band_full_matrix(self, ikirr=None, isym=None):
#     #     """
#     #     Returns the full matrix of the ab initio bands transformation matrix

#     #     Note: this funcion is used only for w90 format, which is deprecated. TODO: remove it
#     #     """
#     #     if ikirr is None:
#     #         return np.array([self.d_band_full_matrix(ikirr, isym) for ikirr in range(self.NKirr)])
#     #     if isym is None:
#     #         return np.array([self.d_band_full_matrix(ikirr, isym) for isym in range(self.Nsym)])

#     #     result = np.zeros((self.NB, self.NB), dtype=complex)
#     #     for (start, end), block in zip(self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr][isym]):
#     #         result[start:end, start:end] = block
#     #     return result

#     # def D_wann_full_matrix(self, ikirr=None, isym=None):
#     #     """
#     #     Returns the full matrix of the Wannier function transformation matrix

#     #     Note: this funcion is used only for w90 format, which is deprecated. TODO: remove it

#     #     """
#     #     if ikirr is None:
#     #         return np.array([self.D_wann_full_matrix(ikirr, isym) for ikirr in range(self.NKirr)])
#     #     if isym is None:
#     #         return np.array([self.D_wann_full_matrix(ikirr, isym) for isym in range(self.Nsym)])

#     #     result = np.zeros((self.num_wann, self.num_wann), dtype=complex)
#     #     for (start, end), block in zip(self.D_wann_block_indices, self.D_wann_blocks[ikirr][isym]):
#     #         result[start:end, start:end] = block
#     #     return result


#     # def to_w90_file(self, seedname):
#     #     if np.any(self.time_reversals):
#     #         raise ValueError("time reversal information is not supported in wannier90 files")
#     #     f = open(seedname + ".dmn", "w")
#     #     print(f"writing {seedname}.dmn:  comment = {self.comment}")
#     #     f.write(f"{self.comment}\n")
#     #     f.write(f"{self.NB} {self.Nsym} {self.NKirr} {self.NK}\n\n")
#     #     f.write(writeints(self.kpt2kptirr + 1) + "\n")
#     #     f.write(writeints(self.kptirr + 1) + "\n")
#     #     for i in range(self.NKirr):
#     #         f.write(writeints(self.kptirr2kpt[i] + 1) + "\n")
#     #         # " ".join(str(x + 1) for x in self.kptirr2kpt[i]) + "\n")
#     #     # f.write("\n".join(" ".join(str(x + 1) for x in l) for l in self.kptirr2kpt) + "\n")
#     #     mat_fun_list = []
#     #     if self.num_wann > 0:
#     #         mat_fun_list.append(self.D_wann_full_matrix)
#     #     if self.NB > 0:
#     #         mat_fun_list.append(self.d_band_full_matrix)

#     #     for M in mat_fun_list:
#     #         for ik in range(self.NKirr):
#     #             for isym in range(self.Nsym):
#     #                 f.write("\n".join("({:17.12e},{:17.12e})".format(x.real, x.imag) for x in M(ik, isym).flatten(order='F')) + "\n\n")


#     # def get_disentangled(self, v_matrix_dagger, v_matrix):
#     #     """
#     #     Here we will loose the block-diagonal structure of the d_band matrix.
#     #     It is ok, w90 anyway does not use it. This function is only used to finish
#     #     the maximal localization procedure with Wannier90
#     #     """
#     #     NBnew = v_matrix.shape[2]
#     #     d_band_block_indices_new = [np.array([[0, NBnew]]) for _ in range(self.NKirr)]
#     #     d_band_blocks_new = []
#     #     for ikirr, ik in enumerate(self.kptirr):
#     #         d_band_blocks_new.append([])
#     #         for isym in range(self.Nsym):
#     #             ik2 = self.kptirr2kpt[ikirr, isym]
#     #             result = np.zeros((NBnew, NBnew), dtype=complex)
#     #             # print (f"ikirr = {ikirr}, isym = {isym}")
#     #             # print (f"d_band_block_indices[ikirr] = {self.d_band_block_indices[ikirr]}")
#     #             for (start, end), block in zip(self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr][isym]):
#     #                 result[:, :] += v_matrix_dagger[ik2][:, start:end] @ block @ v_matrix[ik][start:end, :]
#     #             # result = v_matrix_dagger[ik2] @ self.d_band_full_matrix(ikirr=ikirr, isym=isym) @ v_matrix[ik]
#     #             assert result.shape == (NBnew, NBnew)
#     #             d_band_blocks_new[ikirr].append([result.copy()])
#     #             # d_band_new[ikirr, isym] = v_matrix_dagger[ik2] @ self.d_band[ikirr, isym] @ v_matrix[ik]
#     #     other = deepcopy(self)
#     #     other.d_band_block_indices = d_band_block_indices_new
#     #     other.d_band_blocks = d_band_blocks_new
#     #     other._NB = NBnew
#     #     return other

#     # def set_identiy(self, num_wann, num_bands, nkpt):
#     #     """
#     #     set the object to contain only the  transformation matrices
#     #     """
#     #     self.comment = "only identity"
#     #     self._NB, self.Nsym, self.NKirr, self.NK = num_bands, 1, nkpt, nkpt
#     #     self.num_wann = num_wann
#     #     self.kpt2kptirr = np.arange(self.NK)
#     #     self.kptirr = self.kpt2kptirr
#     #     self.kptirr2kpt = np.array([self.kptirr, self.Nsym])
#     #     self.kpt2kptirr_sym = np.zeros(self.NK, dtype=int)
#     #     self.d_band_block_indices = [np.array([(i, i + 1) for i in range(self.NB)])] * self.NKirr
#     #     self.D_wann_block_indices = np.array([(i, i + 1) for i in range(self.num_wann)])
#     #     self.d_band_blocks = [[[np.eye(end - start) for start, end in self.d_band_block_indicespik]
#     #                           for isym in range(self.Nsym)] for ik in range(self.NKirr)]
#     #     self.D_wann_blocks = [[[np.eye(end - start) for start, end in self.D_wann_block_indices]
#     #                            for isym in range(self.Nsym)] for ik in range(self.NKirr)]
#     #     self.clear_inverse()

#     # def select_bands(self, win_index_irr):
#     #     self.d_band = [D[:, wi, :][:, :, wi] for D, wi in zip(self.d_band, win_index_irr)]

#     # def write(self):
#     #     print(self.comment)
#     #     print(self.NB, self.Nsym, self.NKirr, self.NK, self.num_wann)
#     #     for i in range(self.NKirr):
#     #         for j in range(self.Nsym):
#     #             print()
#     #             for M in self.D_band[i][j], self.d_wann[i][j]:
#     #                 print("\n".join(" ".join(("X" if abs(x)**2 > 0.1 else ".") for x in m) for m in M) + "\n")



#     # def check_unitary(self):
#     #     """
#     #     Check that the transformation matrices are unitary

#     #     Returns
#     #     -------
#     #     float
#     #         the maximum error for the bands
#     #     float
#     #         the maximum error for the Wannier functions
#     #     """
#     #     maxerr_band = 0
#     #     maxerr_wann = 0
#     #     for ik in range(self.NK):
#     #         ikirr = self.kpt2kptirr[ik]
#     #         for isym in range(self.Nsym):
#     #             d = self.d_band[ikirr, isym]
#     #             w = self.D_wann[ikirr, isym]
#     #             maxerr_band = max(maxerr_band, np.linalg.norm(d @ d.T.conj() - np.eye(self.NB)))
#     #             maxerr_wann = max(maxerr_wann, np.linalg.norm(w @ w.T.conj() - np.eye(self.num_wann)))
#     #     return maxerr_band, maxerr_wann

#     def check_group(self, matrices="wann"):
#         """
#         check that D_wann is a group

#         Parameters
#         ----------
#         matrices : str
#             the type of matrices to be checked, either "wann" or "band"

#         Returns
#         -------
#         float
#             the maximum error
#         """
#         if matrices == "wann":
#             check_matrices = self.D_wann
#         elif matrices == "band":
#             check_matrices = self.d_band
#         maxerr = 0
#         for ikirr in range(self.NKirr):
#             Dw = [check_matrices[ikirr, isym] for isym in self.isym_little[ikirr]]
#             print(f'ikirr={ikirr} : {len(Dw)} matrices')
#             for i1, d1 in enumerate(Dw):
#                 for i2, d2 in enumerate(Dw):
#                     d = d1 @ d2
#                     err = [np.linalg.norm(d - _d)**2 for _d in Dw]
#                     j = np.argmin(err)
#                     print(f"({i1}) * ({i2}) -> ({j})" + (f"err={err[j]}" if err[j] > 1e-10 else ""))
#                     maxerr = max(maxerr, err[j])
#         return maxerr
