import os
import sys
import warnings
import numpy as np
from ..system.system import num_cart_dim
from collections import defaultdict
import copy
from ..utility import cached_einsum


class SymWann:
    """
    Symmetrize wannier matrices in real space: Ham_R, AA_R, BB_R, SS_R,...

    Parameters
    ----------
    num_wann: int
        Number of wannier functions.
    lattice: array
        Unit cell lattice constant.
    positions: array
        Positions of each atom.
    atom_name: list
        Name of each atom.
    projections: list
        Should be the same with projections card in relative Wannier90.win.

        eg: ``['Te: s','Te:p']``

        If there is hybrid orbital, grouping the other orbitals.

        eg: ``['Fe':sp3d2;t2g]`` Please don't use ``['Fe':sp3d2;dxz,dyz,dxy]``

            ``['X':sp;p2]`` Please don't use ``['X':sp;pz,py]``
    iRvec: array
        List of R vectors.
    XX_R: dic
        Matrix before symmetrization.
    soc: bool
        Spin orbital coupling.
    magmom: 2D array
        Magnetic moment of each atom.
    wannier_centers_cart: np.array(num_wann, 3)
        Wannier centers in cartesian coordinates.
    logile: file
        Log file. 

    Returns
    -------
    dict(str, np.array(nRvec, num_wann, num_wann, ...), dtype=complex)
        Symmetrized matrices.
    np.array((num_wann, 3), dtype=int)
        New R vectors.

    """

    def __init__(
            self,
            symmetrizer,
            iRvec,
            wannier_centers_cart=None,
            silent=False,
    ):

        self.silent = silent
        self.wannier_centers_cart = wannier_centers_cart
        self.iRvec = [tuple(R) for R in iRvec]
        self.iRvec_index = {r: i for i, r in enumerate(self.iRvec)}
        self.nRvec = len(self.iRvec)
        self.num_wann = symmetrizer.num_wann
        self.spacegroup = symmetrizer.spacegroup
        self.lattice = self.spacegroup.lattice

        self.symmetrizer = symmetrizer
        self.num_blocks = len(symmetrizer.D_wann_block_indices)
        self.num_orb_list = [symmetrizer.rot_orb_list[i][0][0].shape[0] for i in range(self.num_blocks)]
        self.num_points_list = [symmetrizer.atommap_list[i].shape[0] for i in range(self.num_blocks)]
        self.num_points_tot = sum(self.num_points_list)
        points_index = np.cumsum([0] + self.num_points_list)
        self.points_index_start = points_index[:-1]
        self.points_index_end = points_index[1:]

        self.possible_matrix_list = ['Ham', 'AA', 'SS', 'BB', 'CC', 'AA', 'BB', 'CC', 'OO', 'GG',
                                'SS', 'SA', 'SHA', 'SR', 'SH', 'SHR']
        self.tested_matrix_list = ['Ham', 'AA', 'SS', 'BB', 'CC', 'AA', 'BB', 'CC',
                              'SS', 'SH', 'SA', 'SHA']


        # Now the I-odd vectors have "-1" here (in contrast to the old confusing notation)
        # TODO: change it
        self.parity_I = {
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

        np1 = self.num_points_list[block1]
        np2 = self.num_points_list[block2]
        map1 = self.symmetrizer.atommap_list[block1]
        map2 = self.symmetrizer.atommap_list[block2]
        irreducible = np.ones((self.nRvec, np1, np2), dtype=bool)

        R_list = np.array(self.iRvec, dtype=int)
        logfile.write(f"R_list = {R_list}\n")
        R_map = [np.dot(R_list, np.transpose(symop.rotation)) for symop in self.spacegroup.symmetries]

        for isym in range(self.symmetrizer.Nsym):
            T1 = self.symmetrizer.T_list[block1][:, isym]
            T2 = self.symmetrizer.T_list[block2][:, isym]
            logfile.write(f"symmetry operation  {isym + 1}/{len(self.spacegroup.symmetries)}\n")
            logfile.write(f"T1 = {T1}\n")
            logfile.write(f"T2 = {T2}\n")

            atom_R_map = R_map[isym][:, None, None, :] + T1[None, :, None, :] - T2[None, None, :, :]
            logfile.write(f"R_map = {R_map[isym]}\n")
            logfile.write(f"np1 = {np1}, np2 = {np2}\n")
            for a in range(np1):
                a1 = map1[a, isym]
                for b in range(np2):
                    b1 = map2[b, isym]
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
            cutoff_dict=None):
        """
        Symmetrize wannier matrices in real space: Ham_R, AA_R, BB_R, SS_R,...
        and find the new R vectors

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
        for block1 in range(self.num_blocks):
            ws1, we1 = self.symmetrizer.D_wann_block_indices[block1]
            norb1 = self.num_orb_list[block1]
            np1 = self.num_points_list[block1]
            for block2 in range(self.num_blocks):
                logfile.write(f"Symmetrizing blocks {block1} and {block2}\n")
                ws2, we2 = self.symmetrizer.D_wann_block_indices[block2]
                norb2 = self.num_orb_list[block2]
                np2 = self.num_points_list[block2]
                iRab_irred = self.find_irreducible_Rab(block1=block1, block2=block2)
                matrix_dict_list = {}
                for k, v1 in XX_R.items():
                    v = np.copy(v1)[:, ws1:we1, ws2:we2]
                    matrix_dict_list[k] = _matrix_to_dict(v, np1=np1, norb1=norb1, np2=np2, norb2=norb2,
                                                          cutoff=cutoff_dict[k])
                matrix_dict_list_res, iRvec_ab_all = self.average_XX_block(iRab_new=iRab_irred,
                                                                        matrix_dict_in=matrix_dict_list,
                                                                        iRvec_new=self.iRvec, mode="sum",
                                                                        block1=block1, block2=block2)

                iRvec_new_set = set.union(*iRvec_ab_all.values())
                iRvec_new_set.add((0, 0, 0))
                iRvec_new = list(iRvec_new_set)
                nRvec_new = len(iRvec_new)
                iRvec_new_index = {r: i for i, r in enumerate(iRvec_new)}
                iRab_new = {k: set([iRvec_new_index[irvec] for irvec in v]) for k, v in iRvec_ab_all.items()}
                matrix_dict_list_res, iRab_all_2 = self.average_XX_block(iRab_new=iRab_new,
                                                                         matrix_dict_in=matrix_dict_list_res,
                                                                         iRvec_new=iRvec_new, mode="single",
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
        for block1 in range(self.num_blocks):
            ws1, we1 = self.symmetrizer.D_wann_block_indices[block1]
            for block2 in range(self.num_blocks):
                ws2, we2 = self.symmetrizer.D_wann_block_indices[block2]
                norb1 = self.num_orb_list[block1]
                norb2 = self.num_orb_list[block2]
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

        logfile.write(f"wcc before symmetrization = \n {self.wannier_centers_cart}\n")
        wcc = self.symmetrizer.symmetrize_WCC(self.wannier_centers_cart)
        logfile.write(f"wcc after symmetrization = \n {wcc}\n")
        logfile.write('Symmetrizing WCC Finished\n')
        return return_dic, np.array(iRvec_new), wcc


    def average_XX_block(self, iRab_new, matrix_dict_in, iRvec_new, mode, block1, block2):
        """
        Return
        --------
            (matrix_dict_list_res, iRab_all)
            matrix_dict_list_res : dict
                {"Ham":{ (a,b):{iR:mat} }, "AA":{...}, ...} , where iR is the index of R-vector in the old set of R-vectors
        """
        assert mode in ["sum", "single"]
        iRab_new = copy.deepcopy(iRab_new)
        iRvec_new_array = np.array(iRvec_new, dtype=int)

        matrix_dict_list_res = {k: defaultdict(lambda: defaultdict(lambda: 0)) for k in matrix_dict_in}

        iRab_all = defaultdict(lambda: set())
        logfile = self.logfile

        for isym, symop in enumerate(self.spacegroup.symmetries):
            T1 = self.symmetrizer.T_list[block1][:, isym]
            T2 = self.symmetrizer.T_list[block2][:, isym]
            atommap1 = self.symmetrizer.atommap_list[block1][:, isym]
            atommap2 = self.symmetrizer.atommap_list[block2][:, isym]
            logfile.write(f"symmetry operation  {isym + 1}/{len(self.spacegroup.symmetries)}")
            R_map = iRvec_new_array @ np.transpose(symop.rotation)
            atom_R_map = (R_map[:, None, None, :] + T1[None, :, None, :] - T2[None, None, :, :])
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
                                # X_L: only rotation wannier centres from L to L' before rotating orbitals.
                                XX_L = matrix_dict_in[X][(atom_a_map, atom_b_map)][
                                    new_Rvec_index]
                                matrix_dict_list_res[X][(atom_a, atom_b)][iR] += self._rotate_XX_L(
                                    XX_L, X, isym, block1=block1, block2=block2, atom_a=atom_a_map, atom_b=atom_b_map)
                # in single mode we need to determine it only once
                if mode == "single":
                    iR_new_list -= exclude_set

        if mode == "single":
            for (atom_a, atom_b), iR_new_list in iRab_new.items():
                assert len(
                    iR_new_list) == 0, f"for atoms ({atom_a},{atom_b}) some R vectors were not set : {iR_new_list}" + ", ".join(
                    str(iRvec_new[ir]) for ir in iR_new_list)

        if mode == "sum":
            for x in matrix_dict_list_res.values():
                for d in x.values():
                    for k, v in d.items():
                        v /= self.spacegroup.size
        return matrix_dict_list_res, iRab_all


    def _rotate_XX_L(self, XX_L: np.ndarray, X: str, isym, block1, block2, atom_a, atom_b):
        """
        H_ab_sym = P_dagger_a dot H_ab dot P_b
        H_ab_sym_T = ul dot H_ab_sym.conj() dot ur

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

        n_cart = num_cart_dim(X)  # number of cartesian indices
        symop = self.spacegroup.symmetries[isym]
        for _ in range(n_cart):
            # every np.tensordot rotates the first dimension and puts it last. So, repeateing this procedure
            # n_cart times puts dimensions on the right place
            XX_L = np.tensordot(XX_L, symop.rotation_cart, axes=((-n_cart,), (0,)))
        if symop.inversion:
            XX_L *= self.parity_I[X] * (-1)**n_cart
        result = _rotate_matrix(XX_L, self.symmetrizer.rot_orb_dagger_list[block1][atom_a, isym], self.symmetrizer.rot_orb_list[block2][atom_b, isym])
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


def test_rotate_matrix():
    for num_wann in 1, 2, 5, 7:
        for num_cart in 0, 1, 2, 3:
            shape_LR = (num_wann, num_wann)
            shape_X = (num_wann,) * 2 + (3,) * num_cart
            L = np.random.rand(*shape_LR) + 1j * np.random.rand(*shape_LR)
            R = np.random.rand(*shape_LR) + 1j * np.random.rand(*shape_LR)
            X = np.random.rand(*shape_X) + 1j * np.random.rand(*shape_X)
            Y = _rotate_matrix(X, L, R)
            assert Y.shape == X.shape
            Z = cached_einsum("ij,jk...,kl->il...", L, X, R)
            assert np.allclose(Y, Z), f"for num_wann={num_wann}, num_cart={num_cart}, the difference is {np.max(np.abs(Y - Z))} Y.shape={Y.shape} X.shape = {X.shape}\nX={X}\nY={Y}\nZ={Z}"


def _matrix_to_dict(mat, np1, norb1, np2, norb2, cutoff=1e-10):
    """transforms a matrix X[iR, m,n,...] into a dictionary like
        {(a,b): {iR: np.array(num_w_a.num_w_b,...)}}
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
