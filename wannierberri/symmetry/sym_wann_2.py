import copy
from collections import defaultdict
from ..system.system import num_cart_dim
from ..utility import cached_einsum
import numpy as np
import logging
logger = logging.getLogger(__name__)


parity_I = {
    'overlap_up_down': 1,
    'dV_soc': 1,
    'Ham': 1,
    'AA': -1,
    'BB': -1,
    'CC': 1,
    'SS': 1,
    'OO': 1,
    'GG': 1,
    'FF': 1,
    'SH': 1,
    'SA': -1,
    'SHA': -1,
    'SR': -1,
    'SHR': -1,
}  #
parity_TR = {
    'overlap_up_down': 1,
    'dV_soc': -1,
    'Ham': 1,
    'AA': 1,
    'BB': 1,
    'CC': -1,
    'SS': -1,
    'OO': -1,
    'GG': 1,
    'FF': 1,
    'SH': -1,
    'SA': -1,
    'SHA': -1,
    'SR': -1,
    'SHR': -1,
}


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
        logger.info(f"num_blocks_left = {self.num_blocks_left}, num_blocks_right = {self.num_blocks_right}")
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
                                'SS', 'SA', 'SHA', 'SR', 'SH', 'SHR', 'overlap_up_down', 'dV_soc', 'FF']
        self.tested_matrix_list = ['Ham', 'AA', 'SS', 'BB', 'CC', 'AA', 'BB', 'CC',
                              'SS', 'SH', 'SA', 'SHA', 'overlap_up_down', 'dV_soc']


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
        logger.debug("searching irreducible Rvectors for pairs of atoms in blocks %s and %s", block1, block2)
        np1 = self.num_points_list_left[block1]
        np2 = self.num_points_list_right[block2]
        nR = self.nRvec
        map1 = self.symmetrizer_left.atommap_list[block1]
        map2 = self.symmetrizer_right.atommap_list[block2]

        # dense lookup grid: R-vector -> index (or -1)
        R_arr = np.array(self.iRvec, dtype=int)
        Rmin = R_arr.min(axis=0)
        grid_shape = R_arr.max(axis=0) - Rmin + 1
        grid = np.full(grid_shape, -1, dtype=np.int64)
        grid[tuple((R_arr - Rmin).T)] = np.arange(nR)

        # lexicographic key of the source (a, b, iR), shape (nR, np1, np2)
        iR_src = np.arange(nR)[:, None, None]
        a_src = np.arange(np1)[None, :, None]
        b_src = np.arange(np2)[None, None, :]
        key_src = (a_src * np2 + b_src) * nR + iR_src

        irreducible = np.ones((nR, np1, np2), dtype=bool)

        for isym in self.use_symmetries_index:
            atom_R_map = np.rint(self.get_atom_R_map(self.iRvec, isym, block1, block2)).astype(int)
            shifted = atom_R_map - Rmin                                  # (nR,np1,np2,3)
            inside = np.all((shifted >= 0) & (shifted < grid_shape), axis=-1)
            iR1 = np.full((nR, np1, np2), -1, dtype=np.int64)
            iR1[inside] = grid[tuple(shifted[inside].T)]

            a1 = np.broadcast_to(map1[:, isym][None, :, None], iR1.shape)
            b1 = np.broadcast_to(map2[:, isym][None, None, :], iR1.shape)
            key_dst = (a1 * np2 + b1) * nR + iR1

            mask = (iR1 >= 0) & (key_dst > key_src) & irreducible
            irreducible[iR1[mask], a1[mask], b1[mask]] = False

        logger.debug("Found %s sets of (R,a,b) out of the total %s", irreducible.sum(), nR * np1 * np2)
        res = {}
        for a in range(np1):
            for b in range(np2):
                s = np.flatnonzero(irreducible[:, a, b])
                if s.size:
                    res[(a, b)] = set(s.tolist())
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
            logger.warning("symmetrization of matrices %s is not tested. use on your own risk", unknown)

        if cutoff_dict is None:
            cutoff_dict = {}
        for k in XX_R:
            if k not in cutoff_dict:
                cutoff_dict[k] = cutoff


        # ========================================================
        # symmetrize existing R vectors and find additional R vectors
        # ========================================================
        logger.debug('Symmetrizing Started')
        full_matrix_dict_list = {}
        full_iRvec_list = {}
        full_iRvec_set = set()
        for block1 in range(self.num_blocks_left):
            ws1, we1 = self.symmetrizer_left.D_wann_block_indices[block1]
            norb1 = self.num_orb_list_left[block1]
            np1 = self.num_points_list_left[block1]
            for block2 in range(self.num_blocks_right):
                ws2, we2 = self.symmetrizer_right.D_wann_block_indices[block2]
                norb2 = self.num_orb_list_right[block2]
                np2 = self.num_points_list_right[block2]
                iRab_irred = self.find_irreducible_Rab(block1=block1, block2=block2)
                logger.debug("iRab_irred = %s", iRab_irred)
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

                logger.debug("iRvec_ab_all = %s", iRvec_ab_all)
                logger.debug("iRab_irred = %s", iRab_irred)
                for k, val in matrix_dict_list_res.items():
                    logger.debug("matrix_dict_list_res[%s]  = \n", k)
                    for ab, X in val.items():
                        logger.debug("  (%s):\n %s\n", ab, X)


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
        logger.debug('\n\niRvec_new = %s\n', np.array(iRvec_new))
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
                    for (a, b), X in full_matrix_dict_list[(block1, block2)][k].items():
                        ws1a = ws1 + a * norb1
                        we1a = ws1a + norb1
                        ws2b = ws2 + b * norb2
                        we2b = ws2b + norb2
                        for iR, XX_L in X.items():
                            return_dic[k][iRvec_map[iR], ws1a:we1a, ws2b:we2b] += XX_L

        logger.debug('Symmetrizing Finished\n')
        return return_dic, np.array(iRvec_new)

    def average_XX_block(self, iRab_new, matrix_dict_in, iRvec_origin, mode, block1, block2):
        assert mode in ["sum", "single"]
        iRab_new = copy.deepcopy(iRab_new)
        iRvec_origin_array = np.array(iRvec_origin, dtype=int)
        self._ensure_grid()

        matrix_dict_list_res = {k: defaultdict(lambda: defaultdict(lambda: 0)) for k in matrix_dict_in}
        iRab_all = defaultdict(set)

        for isym in self.use_symmetries_index:
            symop = self.spacegroup.symmetries[isym]
            # T is the translation needed to return to the home unit cell after rotation
            T1 = self.symmetrizer_left.T_list[block1][:, isym]
            T2 = self.symmetrizer_right.T_list[block2][:, isym]
            atommap1 = self.symmetrizer_left.atommap_list[block1][:, isym]
            atommap2 = self.symmetrizer_right.atommap_list[block2][:, isym]
            logger.debug("symmetry operation  %s/%s", isym + 1, len(self.spacegroup.symmetries))

            R_map = iRvec_origin_array @ symop.rotation.T  # rotation is an integer matrix

            for (atom_a, atom_b), iR_new_list in iRab_new.items():
                if not iR_new_list:                               # single mode: pair already finished
                    continue
                key_map = (atommap1[atom_a], atommap2[atom_b])

                # new R-vectors for all remaining iR of this pair at once
                iR_arr = np.array(sorted(iR_new_list), dtype=np.int64)
                new_R = R_map[iR_arr] + (T1[atom_a] - T2[atom_b])
                iRab_all[key_map].update(map(tuple, new_R.tolist()))

                idx = self.lookup_R(new_R)                        # -1 where not in self.iRvec
                ok = idx >= 0
                if not ok.any():
                    continue
                iR_ok = iR_arr[ok].tolist()
                idx_ok = idx[ok].tolist()

                hit = set()
                for X, mats in matrix_dict_in.items():
                    d = mats.get(key_map)  # .get: no empty entries are created
                    if not d:
                        continue
                    sel = [(iR, i) for iR, i in zip(iR_ok, idx_ok) if i in d]
                    if not sel:
                        continue
                    iRs, ids = zip(*sel)
                    # rotate from the "new" R-vector / atom pair back to the original one,
                    # all matrices of this (pair, X, isym) in a single call
                    rotated = self._rotate_XX_L_backwards_batch(
                        np.stack([d[i] for i in ids]), X, isym,
                        block1=block1, block2=block2, atom_a=atom_a, atom_b=atom_b)
                    res_XX = matrix_dict_list_res[X][(atom_a, atom_b)]
                    for iR, rot in zip(iRs, rotated):
                        res_XX[iR] += rot
                    if mode == "single":
                        hit.update(iRs)

                # in single mode every iR needs to be determined only once
                if mode == "single":
                    iR_new_list -= hit

        if mode == "single":
            for (atom_a, atom_b), iR_new_list in iRab_new.items():
                assert len(iR_new_list) == 0, (
                    f"for atoms ({atom_a},{atom_b}) {len(iR_new_list)} R vectors were not set : " +
                    ", ".join(str(iRvec_origin[ir]) for ir in iR_new_list))

        if mode == "sum":
            nsym = len(self.use_symmetries_index)
            for x in matrix_dict_list_res.values():
                for d in x.values():
                    for k, v in d.items():
                        v /= nsym
        return matrix_dict_list_res, iRab_all

    def _ensure_grid(self):
        if getattr(self, "_grid", None) is not None:
            return
        R_arr = np.array(self.iRvec, dtype=int)
        self._grid_min = R_arr.min(axis=0)
        self._grid_shape = R_arr.max(axis=0) - self._grid_min + 1
        grid = np.full(self._grid_shape, -1, dtype=np.int64)
        grid[tuple((R_arr - self._grid_min).T)] = np.arange(len(R_arr))
        self._grid = grid

    def lookup_R(self, R_arr):
        """Index in self.iRvec for each row of R_arr (n,3); -1 if not present."""
        self._ensure_grid()
        shifted = R_arr - self._grid_min
        inside = np.all((shifted >= 0) & (shifted < self._grid_shape), axis=1)
        idx = np.full(len(R_arr), -1, dtype=np.int64)
        idx[inside] = self._grid[tuple(shifted[inside].T)]
        return idx

    def _rotate_XX_L_backwards_batch(self, XX, X, isym, block1, block2, atom_a, atom_b):
        """Same as _rotate_XX_L_backwards, but XX has an extra leading batch axis: (p, m, n, ...)."""
        n_cart = num_cart_dim(X)
        symop = self.spacegroup.symmetries[isym]
        rot_mat_loc = symop.rotation_cart
        for _ in range(n_cart):
            XX = np.tensordot(XX, rot_mat_loc, axes=((-n_cart,), (0,)))
        if symop.inversion:
            XX = XX * (parity_I[X] * (-1) ** n_cart)
        result = cached_einsum("ij,pjk...,kl->pil...",
                            self.symmetrizer_left.rot_orb_dagger_list[block1][atom_a, isym],
                            XX,
                            self.symmetrizer_right.rot_orb_list[block2][atom_b, isym])
        if symop.time_reversal:
            result = result.conj() * parity_TR[X]
        return result



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
