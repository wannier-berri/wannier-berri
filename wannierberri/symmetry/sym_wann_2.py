import os
import sys
import warnings
import numpy as np
from ..system.system import num_cart_dim
from collections import defaultdict
import copy


def do_rotate_vector(key):
    return True
    # if key.startswith('dV_soc_wann_'):
    #     return True
    # if key == 'overlap_up_down':
    #     return False
    # return True


class SymWann:
    """
    Symmetrize wannier matrices in real space: Ham_R, AA_R, BB_R, SS_R,...

    Parameters
    ----------
    iRvec: array
        List of R vectors.
    symmetrizer: SymmetrizerSAWF
        Symmetrizer object for both left and right. (if they are the same)
    symmetrizer_left: SymmetrizerSAWF
        Symmetrizer object for the left side. (if None it is set to symmetrizer)
    symmetrizer_right: SymmetrizerSAWF
        Symmetrizer object for the right side. (if None it is set to symmetrizer_left)
    silent: bool
        If True, suppresses output to the console.
    use_symmetries_index: list of int
        List of symmetry indices to use for symmetrization. If None, all symmetries will be used.

    Returns
    -------
    dict(str, np.array(nRvec, num_wann, num_wann, ...), dtype=complex)
        Symmetrized matrices.
    np.array((num_wann, 3), dtype=int)
        New R vectors.

    """

    def __init__(
            self,
            iRvec,
            symmetrizer=None,
            symmetrizer_left=None,
            symmetrizer_right=None,
            silent=False,
            use_symmetries_index=None,
    ):

        self.silent = silent


        if symmetrizer_left is None:
            symmetrizer_left = symmetrizer
        if symmetrizer_right is None:
            symmetrizer_right = symmetrizer_left

        self.iRvec = [tuple(R) for R in iRvec]
        self.iRvec_index = {r: i for i, r in enumerate(self.iRvec)}
        self.nRvec = len(self.iRvec)
        self.num_wann = symmetrizer_left.num_wann
        self.spacegroup = symmetrizer_left.spacegroup
        assert self.spacegroup.equals(symmetrizer_right.spacegroup, mod1=False), "Left and right symmetrizers must have the same spacegroup"
        self.lattice = self.spacegroup.lattice

        if use_symmetries_index is None:
            self.use_symmetries_index = list(range(len(self.spacegroup.symmetries)))
        else:
            self.use_symmetries_index = use_symmetries_index

        self.symmetrizer_left = symmetrizer_left
        self.symmetrizer_right = symmetrizer_right
        self.num_blocks_left = len(symmetrizer_left.D_wann_block_indices)
        self.num_blocks_right = len(symmetrizer_right.D_wann_block_indices)
        self.num_orb_list_left = [symmetrizer_left.rot_orb_list[i][0][0].shape[0] for i in range(self.num_blocks_left)]
        self.num_orb_list_right = [symmetrizer_right.rot_orb_list[i][0][0].shape[0] for i in range(self.num_blocks_right)]
        self.num_points_list_left = [symmetrizer_left.atommap_list[i].shape[0] for i in range(self.num_blocks_left)]
        self.num_points_list_right = [symmetrizer_right.atommap_list[i].shape[0] for i in range(self.num_blocks_right)]
        self.num_points_tot_left = sum(self.num_points_list_left)
        self.num_points_tot_right = sum(self.num_points_list_right)
        points_index_left = np.cumsum([0] + self.num_points_list_left)
        points_index_right = np.cumsum([0] + self.num_points_list_right)
        self.points_index_start_left = points_index_left[:-1]
        self.points_index_end_left = points_index_left[1:]
        self.points_index_start_right = points_index_right[:-1]
        self.points_index_end_right = points_index_right[1:]
        self.possible_matrix_list = ['Ham', 'AA', 'SS', 'BB', 'CC', 'AA', 'BB', 'CC', 'OO', 'GG',
                                'SS', 'SA', 'SHA', 'SR', 'SH', 'SHR', 'overlap_up_down', 'dV_soc_wann_0_0', 'dV_soc_wann_0_1', 'dV_soc_wann_1_1']
        self.tested_matrix_list = ['Ham', 'AA', 'SS', 'BB', 'CC', 'AA', 'BB', 'CC',
                              'SS', 'SH', 'SA', 'SHA', 'overlap_up_down', 'dV_soc_wann_0_0', 'dV_soc_wann_0_1', 'dV_soc_wann_1_1']


        # Now the I-odd vectors have "-1" here (in contrast to the old confusing notation)
        self.parity_I = {
            'overlap_up_down': 1,
            'dV_soc_wann_0_0': 1,
            'dV_soc_wann_0_1': 1,
            'dV_soc_wann_1_1': 1,
            'Ham': 1,
            'AA': -1,
            'BB': -1,
            'CC': 1,
            'SS': 1,
            'OO': 1,
            'GG': 1,
            'SH': 1,
            'SA': -1,
            'SHA': -1,
            'SR': -1,
            'SHR': -1,
        }  #
        self.parity_TR = {
            'overlap_up_down': 1,
            'dV_soc_wann_0_0': -1,
            'dV_soc_wann_0_1': -1,
            'dV_soc_wann_1_1': -1,
            'Ham': 1,
            'AA': 1,
            'BB': 1,
            'CC': -1,
            'SS': -1,
            'OO': -1,
            'GG': 1,
            'SH': -1,
            'SA': -1,
            'SHA': -1,
            'SR': -1,
            'SHR': -1,
        }

    @property
    def logfile(self):
        if self.silent:
            return open(os.devnull, 'w')
        else:
            return sys.stdout


    def index_R(self, R):
        try:
            return self.iRvec_index[tuple(R)]
        except KeyError:
            return None

    def get_atom_R_map(self, iRvec, isym, block1, block2):
        """
        Get the R vector mapping for a specific symmetry operation and atom pair.

        Parameters
        ----------
        iRvec : list of tuples
            The list of R vectors to consider.
        isym : int
            The index of the symmetry operation.
        block1, block2 : int
            The block indices for the two atoms.

        Returns
        -------
        np.ndarray
            The R vector mapping for the specified symmetry operation and atom pair.
        """
        R_list = np.array(iRvec, dtype=int)
        R_map = R_list @ self.spacegroup.symmetries[isym].rotation.T
        T1 = self.symmetrizer_left.T_list[block1][:, isym]
        T2 = self.symmetrizer_right.T_list[block2][:, isym]
        return R_map[:, None, None, :] + T1[None, :, None, :] - T2[None, None, :, :]

    def find_irreducible_Rab(self, block1, block2):
        """
        Finds which Rvectors can be chosen as an irreducible set for each pair of atoms (a,b)
        where a is in block1 and b is in block2.

        Parameters
        ----------
        block1, block2 : int
            Block indices

        Return
        --------
        dict
         { (a,b):set([index of Rvecotr, if it is irreducible])}
        """
        logfile = self.logfile
        logfile.write("searching irreducible Rvectors for pairs of a,b\n")

        np1 = self.num_points_list_left[block1]
        np2 = self.num_points_list_right[block2]
        map1 = self.symmetrizer_left.atommap_list[block1]
        map2 = self.symmetrizer_right.atommap_list[block2]
        irreducible = np.ones((self.nRvec, np1, np2), dtype=bool)
        logfile.write(f"np1 = {np1}, np2 = {np2}\n")

        R_list = np.array(self.iRvec, dtype=int)
        logfile.write(f"R_list = {R_list}\n")

        for isym in self.use_symmetries_index:
            # T : np.ndarray(shape=(num_points, nsym, 3), dtype=int)
            # A matrix that contains the translation needed to bring the transformed point back to the home unit cell.
            atom_R_map = self.get_atom_R_map(self.iRvec, isym, block1, block2)
            for a, a1 in enumerate(map1[:, isym]):
                for b, b1 in enumerate(map2[:, isym]):
                    logfile.write(f"a = {a}, b = {b}, a1 = {a1}, b1 = {b1}, (a1, b1) >= (a, b) = {(a1, b1) >= (a, b)}\n")
                    if (a1, b1) >= (a, b):
                        for iR in range(self.nRvec):
                            if irreducible[iR, a, b]:
                                iR1 = self.index_R(atom_R_map[iR, a, b])
                                if iR1 is not None and (a1, b1, iR1) > (a, b, iR):
                                    irreducible[iR1, a1, b1] = False
            logfile.write(f"irreducible = {irreducible}\n")

        logfile.write(
            f"Found {np.sum(irreducible)} sets of (R,a,b) out of the total {self.nRvec * np1 * np2} ({self.nRvec}*{np1}*{np2})")
        dic = {(a, b): set([iR for iR in range(self.nRvec) if irreducible[iR, a, b]])
               for a in range(np1) for b in range(np2)}
        res = {k: v for k, v in dic.items() if len(v) > 0}
        return res


    def symmetrize(self, XX_R,
            cutoff=-1,
            cutoff_dict=None,):
        """
        Symmetrize wannier matrices in real space: Ham_R, AA_R, BB_R, SS_R,...
        and find the new R vectors

        Parameters
        ----------
        XX_R: dict {str: np.array(nRvec, num_wann, num_wann, ...), dtype=complex}
            Matrices to be symmetrized.
        cutoff: float
            Cutoff for small matrix elements in XX_R.   
        cutoff_dict: dict {str: float}
            Cutoff for small matrix elements in XX_R for each matrix type.

        Returns
        --------
        dict {str: np.array(nRvec, num_wann, num_wann, ...), dtype=complex}
            Symmetrized matrices.
        np.array((num_wann, 3), dtype=int)
            New R vectors.
        np.array((num_wann, 3), dtype=float)
            Wannier centers in cartesian coordinates.
        """

        unknown = set(XX_R.keys()) - set(self.possible_matrix_list)
        if unknown:
            raise NotImplementedError(f"symmetrization of matrices {unknown} is not implemented yet")
        unknown = set(XX_R.keys()) - set(self.tested_matrix_list)
        if unknown:
            warnings.warn(f"symmetrization of matrices {unknown} is not tested. use on your own risk")

        if cutoff_dict is None:
            cutoff_dict = {}
        for k in XX_R:
            if k not in cutoff_dict:
                cutoff_dict[k] = cutoff


        # ========================================================
        # symmetrize existing R vectors and find additional R vectors
        # ========================================================
        logfile = self.logfile
        logfile.write('##########################')
        logfile.write('Symmetrizing Started')
        full_matrix_dict_list = {}
        full_iRvec_list = {}
        full_iRvec_set = set()
        for block1 in range(self.num_blocks_left):
            ws1, we1 = self.symmetrizer_left.D_wann_block_indices[block1]
            norb1 = self.num_orb_list_left[block1]
            np1 = self.num_points_list_left[block1]
            for block2 in range(self.num_blocks_right):
                logfile.write(f"Symmetrizing blocks {block1} and {block2}\n")
                ws2, we2 = self.symmetrizer_right.D_wann_block_indices[block2]
                norb2 = self.num_orb_list_right[block2]
                np2 = self.num_points_list_right[block2]
                iRab_irred = self.find_irreducible_Rab(block1=block1, block2=block2)
                logfile.write(f"iRab_irred = {iRab_irred}\n")
                matrix_dict_list = {}
                for k, v1 in XX_R.items():
                    v = np.copy(v1)[:, ws1:we1, ws2:we2]
                    # transforms a matrix X[iR, m,n,...] into a nested dictionary like
                    # {(a,b): {iR: np.array(num_w_a, num_w_b,...)}}
                    matrix_dict_list[k] = _matrix_to_dict(v, np1=np1, norb1=norb1, np2=np2, norb2=norb2,
                                                          cutoff=cutoff_dict[k])

                matrix_dict_list_res, iRvec_ab_all = self.average_XX_block(iRab_new=iRab_irred,
                                                                        matrix_dict_in=matrix_dict_list,
                                                                        iRvec_origin=self.iRvec, mode="sum",
                                                                        block1=block1, block2=block2)

                logfile.write(f"iRvec_ab_all = {iRvec_ab_all}\n")
                logfile.write(f"iRab_irred = {iRab_irred}\n")
                for k, val in matrix_dict_list_res.items():
                    logfile.write(f"matrix_dict_list_res[{k}]  = \n")
                    for ab, X in val.items():
                        logfile.write(f"  ({ab}):\n {X}\n")


                iRvec_new_set = set.union(*iRvec_ab_all.values())
                iRvec_new_set.add((0, 0, 0))
                iRvec_new = list(iRvec_new_set)
                iRvec_new_index = {r: i for i, r in enumerate(iRvec_new)}
                iRab_new = {k: set([iRvec_new_index[irvec] for irvec in v]) for k, v in iRvec_ab_all.items()}
                matrix_dict_list_res, iRab_all_2 = self.average_XX_block(iRab_new=iRab_new,
                                                                         matrix_dict_in=matrix_dict_list_res,
                                                                         iRvec_origin=iRvec_new, mode="single",
                                                                         block1=block1, block2=block2)

                full_matrix_dict_list[(block1, block2)] = matrix_dict_list_res
                full_iRvec_list[(block1, block2)] = iRvec_new
                full_iRvec_set = set.union(full_iRvec_set, iRvec_new_set)

        iRvec_new = list(full_iRvec_set)
        logfile.write(f'\n\niRvec_new = {np.array(iRvec_new)}\n')
        nRvec_new = len(iRvec_new)
        iRvec_new_index = {r: i for i, r in enumerate(iRvec_new)}

        return_dic = {k: np.zeros((nRvec_new, self.num_wann, self.num_wann) + (3,) * num_cart_dim(k), dtype=complex)
                      for k in XX_R}
        for block1 in range(self.num_blocks_left):
            ws1, we1 = self.symmetrizer_left.D_wann_block_indices[block1]
            for block2 in range(self.num_blocks_right):
                ws2, we2 = self.symmetrizer_right.D_wann_block_indices[block2]
                norb1 = self.num_orb_list_left[block1]
                norb2 = self.num_orb_list_right[block2]
                iRvec_block = full_iRvec_list[(block1, block2)]
                iRvec_map = [iRvec_new_index[r] for i, r in enumerate(iRvec_block)]
                for k in return_dic:
                    logfile.write(f"Symmetrizing blocks {block1} and {block2} for matrix {k}\n")
                    logfile.write(f"ws1 = {ws1}, we1 = {we1}, ws2 = {ws2}, we2 = {we2}\n")
                    logfile.write(f"full_matrix_dict_list[(block1, block2)][k] = {full_matrix_dict_list[(block1, block2)][k][(0, 0)].keys()}\n")
                    for (a, b), X in full_matrix_dict_list[(block1, block2)][k].items():
                        ws1a = ws1 + a * norb1
                        we1a = ws1a + norb1
                        ws2b = ws2 + b * norb2
                        we2b = ws2b + norb2
                        for iR, XX_L in X.items():
                            # print (f"iR = {iR}, blocks: {block1, block2} \n   iRvec_map[iR] = {iRvec_map[iR]} ws1a = {ws1a}, we1a = {we1a}, ws2b = {ws2b}, we2b = {we2b}, XX_L.shap={XX_L.shape}")
                            return_dic[k][iRvec_map[iR], ws1a:we1a, ws2b:we2b] += XX_L

        logfile.write('Symmetrizing Finished\n')
        return return_dic, np.array(iRvec_new)

    # def symmetrize_inplace_no_change_iRvec(self, XX_R_dict, iRvec, cutoff=-1, cutoff_dict=None):
    #     XX_R_dict_new, iRvec_new = self.symmetrize(XX_R=XX_R_dict,
    #                                                cutoff=cutoff, cutoff_dict=cutoff_dict)
    #     iRvec_old = [tuple([int(x) for x in r]) for r in iRvec]
    #     iRvec_new = [tuple([int(x) for x in r]) for r in iRvec_new]
    #     assert len(iRvec_new) == len(iRvec), ("Number of R-vectors changed during symmetrization, this should not happen\n" +
    #                                             f" old ({len(iRvec_old)}): \n{iRvec_old}, \n new ({len(iRvec_new)}):\n {iRvec_new}\n"+
    #                                             f"extra vectors : {set(iRvec_new) - set(iRvec_old)}"+
    #                                             f"missing vectors : {set(iRvec_old) - set(iRvec_new)}")
    #     reorder = [self.index_R(r) for r in iRvec_new]
    #     assert np.all(iRvec[reorder] == iRvec_new), f"iRvec reordering failed: {iRvec[reorder]} != {iRvec_new}"
    #     for k in XX_R_dict:
    #         XX_R_copy = XX_R_dict[k].copy()
    #         XX_R_dict[k][reorder] = XX_R_dict_new[k][:]
    #         print(f"symmetrized matrix {k}, max change = {np.max(np.abs(XX_R_dict[k] - XX_R_copy))}")
    #     return XX_R_dict

    def average_XX_block(self, iRab_new, matrix_dict_in, iRvec_origin, mode, block1, block2):
        """
        Averages matrices over symmetry operations for a given pair of blocks.

        Parameters
        ----------
        iRab_new : dict
            A dictionary mapping (atom_a, atom_b) pairs to lists of new R-vectors (That need to be evaluated)

        matrix_dict_in : dict
            A dictionary containing the input matrices to be averaged.

        iRvec_origin : list
            A list of original R-vectors.

        mode : str
            The averaging mode, either "sum" or "single". In sum mode all R-vectors that map to the same R-vector are summed.
            In sungle mode only one of them is taken.

        block1 : int
            The index of the first block.

        block2 : int
            The index of the second block.

        Return
        --------
            (matrix_dict_list_res, iRab_all)
            matrix_dict_list_res : dict
                {"Ham":{ (a,b):{iR:mat} }, "AA":{...}, ...} , where iR is the index of R-vector in the old set of R-vectors
        """
        assert mode in ["sum", "single"]
        iRab_new = copy.deepcopy(iRab_new)
        iRvec_origin_array = np.array(iRvec_origin, dtype=int)

        matrix_dict_list_res = {k: defaultdict(lambda: defaultdict(lambda: 0)) for k in matrix_dict_in}

        iRab_all = defaultdict(lambda: set())
        logfile = self.logfile

        for isym in self.use_symmetries_index:
            symop = self.spacegroup.symmetries[isym]
            # T is the translation needed to return to the home unit cell after rotation
            T1 = self.symmetrizer_left.T_list[block1][:, isym]
            T2 = self.symmetrizer_right.T_list[block2][:, isym]
            atommap1 = self.symmetrizer_left.atommap_list[block1][:, isym]
            atommap2 = self.symmetrizer_right.atommap_list[block2][:, isym]
            logfile.write(f"symmetry operation  {isym + 1}/{len(self.spacegroup.symmetries)}\n")
            R_map = iRvec_origin_array @ symop.rotation.T
            R_map_round = np.rint(R_map).astype(int)
            assert np.allclose(R_map, R_map_round), f"R_map not integer: {R_map}"
            R_map = R_map_round
            atom_R_map = (R_map[:, None, None, :] + T1[None, :, None, :] - T2[None, None, :, :])
            # atom_R_map[iR, a, b] gives the new R vector to wich the original iR vector is mapped for atoms a and b
            for (atom_a, atom_b), iR_new_list in iRab_new.items():
                atom_a_map = atommap1[atom_a]
                atom_b_map = atommap2[atom_b]
                exclude_set = set()
                for iR in iR_new_list:
                    new_Rvec = tuple(atom_R_map[iR, atom_a, atom_b])
                    iRab_all[(atom_a_map, atom_b_map)].add(new_Rvec)
                    if new_Rvec in self.iRvec:
                        new_Rvec_index = self.index_R(new_Rvec)
                        for X in matrix_dict_list_res:
                            if new_Rvec_index in matrix_dict_in[X][(atom_a_map, atom_b_map)]:
                                if mode == "single":
                                    exclude_set.add(iR)
                                # we rotate from the XXL from the "new" R-vector and atom pairs back to the original,
                                # therefore the forward=False, and we use atom_a = atom_a, atom_b = atom_b
                                # (earlier it was atom_a_map, atom_b_map which was wrong)
                                XX_L = matrix_dict_in[X][(atom_a_map, atom_b_map)][new_Rvec_index]
                                XX_L_rotated = self._rotate_XX_L_backwards(XX_L, X, isym, block1=block1, block2=block2,
                                                                 atom_a=atom_a, atom_b=atom_b)
                                matrix_dict_list_res[X][(atom_a, atom_b)][iR] += XX_L_rotated
                # in single mode we need to determine it only once
                if mode == "single":
                    iR_new_list -= exclude_set

        if mode == "single":
            for (atom_a, atom_b), iR_new_list in iRab_new.items():
                assert len(iR_new_list) == 0, f"for atoms ({atom_a},{atom_b}) some R vectors were not set : {iR_new_list}" + ", ".join(
                    str(iRvec_origin[ir]) for ir in iR_new_list)

        if mode == "sum":
            for x in matrix_dict_list_res.values():
                for d in x.values():
                    for k, v in d.items():
                        v /= len(self.use_symmetries_index)
        return matrix_dict_list_res, iRab_all

    def _rotate_XX_L_backwards(self, XX_L: np.ndarray, X: str, isym, block1, block2, atom_a, atom_b):
        """
        Rotate the matrix XX_L BACKWARD (i.e. the symmetry operation inverse) 

        Parameters
        ----------
        XX_L : np.ndarray
            Matrix to be rotated
        X : str
            Matrix type, e.g. "Ham", "AA", "BB", "SS", "CC", "OO", "GG", "SH", "SA", "SHA", "SR", "SHR"
        isym : int
            Index of symmetry operation
        block1, block2 : int
            Block indices

        Returns
        -------
        np.ndarray
            Rotated matrix
        """
        if do_rotate_vector(X):
            n_cart = num_cart_dim(X)  # number of cartesian indices
            symop = self.spacegroup.symmetries[isym]
            rot_mat_loc = symop.rotation_cart  # (this is the inverse rotation, but we rotate row-vectors, not column-vectors, therefore double transpose cancels out)
            for _ in range(n_cart):
                # every np.tensordot rotates the first dimension and puts it last. So, repeateing this procedure
                # n_cart times puts dimensions on the right place
                XX_L = np.tensordot(XX_L, rot_mat_loc, axes=((-n_cart,), (0,)))
            if symop.inversion:
                XX_L *= self.parity_I[X] * (-1)**n_cart
        result = _rotate_matrix(X=XX_L,
                                L=self.symmetrizer_left.rot_orb_dagger_list[block1][atom_a, isym],
                                R=self.symmetrizer_right.rot_orb_list[block2][atom_b, isym])
        if do_rotate_vector(X):
            if symop.time_reversal:
                result = result.conj() * self.parity_TR[X]
        return result


def _rotate_matrix(X, L, R):
    """
    Rotate a matrix X[m,n,...] with L[m,n] and R[m,n] matrices
    comptes L.dot(X).dot(R) where X can have additional dimensions in the end, which are not touched
    assumed to be a faster version of np.einsum("ij,jk...,kl->il...", L, X, R)
    """
    _ = np.tensordot(L, X, axes=((1,), (0,)))
    _ = np.tensordot(R, _, axes=((0,), (1,)))
    return _.swapaxes(0, 1)


def _matrix_to_dict(mat, np1, norb1, np2, norb2, cutoff=1e-10):
    """transforms a matrix X[iR, m,n,...] into a nested dictionary like
        {(a,b): {iR: np.array(num_w_a, num_w_b,...)}}
    """
    result = defaultdict(lambda: {})
    for a in range(np1):
        s1 = a * norb1
        e1 = s1 + norb1
        for b in range(np2):
            s2 = b * norb2
            e2 = s2 + norb2
            result_ab = {}
            for iR, X in enumerate(mat[:, s1:e1, s2:e2]):
                if np.any(abs(X) > cutoff):
                    result_ab[iR] = X
            if len(result_ab) > 0:
                result[(a, b)] = result_ab
    return result
