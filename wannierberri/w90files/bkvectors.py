import numpy as np

from .utility import get_mp_grid
from ..io import sparselist_to_dict
from .w90file import W90_file, check_shape


class BKVectors(W90_file):

    npz_tags = ["bk_grid", "wk", "kpt_grid",
                "kptirr", "mp_grid", "recip_lattice"]
    npz_keys_dict_int = ["neighbours", "G"]
    extension = "bkvec"


    def __init__(self, recip_lattice, mp_grid,
                 wk, bk_grid, G, neighbours,
                 kpt_grid, kptirr=None):
        self.recip_lattice = recip_lattice
        self.mp_grid = np.array(mp_grid)
        self.G = sparselist_to_dict(G)
        self.NNB = len(wk)
        self.kpt_grid = np.array(kpt_grid)
        self.NK = len(kpt_grid)
        if kptirr is None:
            kptirr = range(self.NK)
        assert set(self.G.keys()) == set(kptirr), \
            f"G keys {self.G.keys()} do not match kptirr {kptirr}"
        self.kptirr = np.array(list(kptirr), dtype=int)
        self.neighbours = sparselist_to_dict(neighbours)
        assert check_shape(self.neighbours) == (self.NNB,)
        assert check_shape(self.G) == (self.NNB, 3)
        self.bk_grid = np.array(bk_grid, dtype=int)
        assert self.bk_grid.shape == (self.NNB, 3), \
            f"MMN bk_grid should have shape (NNB, 3), got {self.bk_grid.shape}"
        self.bk_cart = bk_grid.dot(recip_lattice / self.mp_grid[:, None])
        self.bk_red = self.bk_grid / self.mp_grid[None, :]
        assert self.bk_cart.shape == (self.NNB, 3), \
            f"MMN bk_cart should have shape (NNB, 3), got {self.bk_cart.shape}"
        self.wk = np.array(wk, dtype=float)
        assert self.wk.shape == (self.NNB,), \
            f"MMN wk should have shape (NNB,), got {self.wk.shape}"
        self.NNB = len(self.wk)

    def select_bands(self, selected_bands, **kwargs):
        # this class has no information on the bands, so nothing to do
        pass


    @classmethod
    def from_kpoints(cls,
                     recip_lattice,
                     mp_grid,
                     kpoints_red,
                     kmesh_tol=1e-7,
                     bk_complete_tol=1e-5,
                     search_supercell=2,
                     kptirr=None):
        """
        Create BKVectors from given k-points in the Monkhorst-Pack grid

        Parameters
        ----------
        recip_lattice : np.ndarray(shape=(3, 3), dtype=float)
            the reciprocal lattice vectors
        mp_grid : np.ndarray(shape=(3,), dtype=int)
            the Monkhorst-Pack grid
        kpoints_red : np.ndarray(shape=(NK, 3), dtype=float)
            the k-points in reduced coordinates (between 0 and 1)
        kmesh_tol : float
            the tolerance to distinguish the shells by the length of the reciprocal lattice vectors (in inverse Angstrom)
        bk_complete_tol : float
            the tolerance for the completeness of the shells
        search_supercell : int
            the number of supercells to search for the shells (in each direction)

        Returns
        -------
        BKVectors
            the BKVectors object
        """
        wk, bk_cart, bk_grid = cls.find_bk_vectors(
            recip_lattice, mp_grid,
            kmesh_tol=kmesh_tol,
            bk_complete_tol=bk_complete_tol,
            search_supercell=search_supercell)
        G, neighbours = cls.find_G_and_neighbours(kpoints_red, bk_grid, mp_grid, kptirr=kptirr)
        kpt_grid = np.rint(kpoints_red * np.array(mp_grid)[None, :]).astype(int)

        return cls(recip_lattice=recip_lattice,
                   kpt_grid=kpt_grid,
                   mp_grid=mp_grid, wk=wk, bk_grid=bk_grid, G=G, neighbours=neighbours, kptirr=kptirr)

    @classmethod
    def from_nnkp(cls, filename, kmesh_tol=1e-5,
                bk_complete_tol=1e-5,
                kptirr=None,
                params=None):
        """Create BKVectors from a NNKP file

        Parameters
        ----------
        filename : str
            the path to the NNKP file

        Returns
        -------
        BKVectors
            the BKVectors object
        """
        cls = BKVectors
        from wannier90io import parse_nnkp_raw
        nnkp = parse_nnkp_raw(open(filename).read())
        nnkpts = np.array([b for b in nnkp["nnkpts"] if b[0] == 1])

        kpoints_red = np.array(nnkp["kpoints"]["kpoints"])
        bk_red = kpoints_red[nnkpts[:, 1] - 1] + nnkpts[:, 2:5] - kpoints_red[0, None, :]
        if params is not None and "recip_lattice" in params:
            recip_lattice = params["recip_lattice"]
        else:
            recip_lattice = nnkp["reciprocal_lattice"]
            recip_lattice = np.array([recip_lattice[i] for i in ["b1", "b2", "b3"]])
        mp_grid = get_mp_grid(kpoints_red)
        bk_grid = np.round(bk_red * np.array(mp_grid)[None, :]).astype(int)
        bk_red = bk_grid / np.array(mp_grid)[None, :]  # this is done to avoid numerical issues
        # due to limited precision of kpoints in the nnkp file
        bk_cart = bk_red @ recip_lattice

        shell_klatt, shell_kcart = cls.k_to_shells(bk_grid, bk_cart, kmesh_tol=kmesh_tol)
        wk, bk_cart, bk_grid = cls.get_shell_weights(shell_kcart=shell_kcart,
                                                    shell_klatt=shell_klatt,
                                                    bk_complete_tol=bk_complete_tol)
        G, neighbours = cls.find_G_and_neighbours(kpoints_red, bk_grid, mp_grid, kptirr=kptirr)
        kpt_grid = np.rint(kpoints_red * np.array(mp_grid)[None, :]).astype(int)

        return cls(recip_lattice=recip_lattice,
                   kpt_grid=kpt_grid,
                   mp_grid=mp_grid, wk=wk, bk_grid=bk_grid, G=G, neighbours=neighbours, kptirr=kptirr)


    def reorder_bk_vectors(self, ik, neighbours, G, data):
        bk_grid_new = self.kpt_grid[neighbours] - self.kpt_grid[ik, None, :] + G * self.mp_grid[None, :]
        bk_grid_new_tuples = [tuple(bl) for bl in bk_grid_new]
        bk_grid_tuples = [tuple(bl) for bl in self.bk_grid]
        srt = [bk_grid_new_tuples.index(bl) for bl in bk_grid_tuples]
        assert len(srt) == self.NNB
        data[:] = data[srt, :]
        G[:] = G[srt]
        neighbours[:] = neighbours[srt]
        assert np.all(neighbours == self.neighbours[ik])
        assert np.all(G == self.G[ik])
        return srt

    def reorder_mmn(self, bkvec, mmn):
        srt_ref = None
        for ik in mmn.data.keys():
            srt = self.reorder_bk_vectors(ik, bkvec.neighbours[ik], bkvec.G[ik], mmn.data[ik])
            if srt_ref is None:
                srt_ref = srt
            else:
                assert np.all(srt == srt_ref), f"Different reorderings for ik={ik}: {srt} != {srt_ref}"
        bkvec.bk_grid = bkvec.bk_grid[srt_ref]
        bkvec.bk_cart = bkvec.bk_cart[srt_ref]
        bkvec.wk = bkvec.wk[srt_ref]

    @property
    def kpt_red(self):
        return self.kpt_grid / self.mp_grid[None, :]

    @classmethod
    def find_G_and_neighbours(cls, kpoints_red, bk_grid, mp_grid, kptirr=None):
        if kptirr is None:
            kptirr = range(kpoints_red.shape[0])
        k_latt_int = np.rint(kpoints_red * mp_grid).astype(int)
        NNB = bk_grid.shape[0]
        NK = kpoints_red.shape[0]
        # bk_red = bk_grid / mp_grid[None, :]
        G = {ik: np.zeros((NNB, 3), dtype=int) for ik in kptirr}
        neighbours = {ik: np.zeros(NNB, dtype=int) for ik in kptirr}
        for kirr in kptirr:
            for ib in range(NNB):
                k_latt_int_nb = k_latt_int[kirr] + bk_grid[ib]
                for ik2 in range(NK):
                    g = k_latt_int_nb - k_latt_int[ik2]
                    if np.all(g % mp_grid == 0):
                        neighbours[kirr][ib] = ik2
                        G[kirr][ib] = g // mp_grid
                        break
                else:
                    raise RuntimeError(
                        f"Could not find a neighbour for k-point {kirr} with k-lattice {k_latt_int[kirr]} and "
                        f"bk-lattice {bk_grid[ib]} in the Monkhorst-Pack grid {mp_grid}. "
                        f"Check the parameters of `find_bk_vectors`."
                    )
        return G, neighbours


    @classmethod
    def find_bk_vectors(cls, recip_lattice, mp_grid, kmesh_tol=1e-7, bk_complete_tol=1e-5, search_supercell=2):
        """
        Find the bk vectors for the finite-difference scheme

        Parameters
        ----------
        recip_lattice : np.ndarray(shape=(3, 3), dtype=float)
            the reciprocal lattice vectors
        mp_grid : np.ndarray(shape=(3,), dtype=int)
            the Monkhorst-Pack grid
        kmesh_tol : float
            the tolerance to distinguish the shells by the length of the reciprocal lattice vectors (in inverse Angstrom)
        bk_complete_tol : float
            the tolerance for the completeness of the shells
        search_supercell : int
            the number of supercells to search for the shells (in each direction)

        Returns
        -------
        wk : np.ndarray(shape=(NNB,), dtype=float)
            the weights of the bk vectors
        bk_cart : np.ndarray(shape=(NNB, 3), dtype=float)
            the bk vectors in cartesian coordinates
        bk_grid : np.ndarray(shape=(NNB, 3), dtype=int)
            the bk vectors in the basis of the reciprocal lattice divided by the Monkhorst-Pack grid
        """
        mp_grid = np.array(mp_grid, dtype=int)
        basis = recip_lattice / mp_grid[:, None]
        search_limit = search_supercell * np.array(mp_grid)
        k_latt = np.array([(i, j, k) for i in range(-search_limit[0], search_limit[0] + 1)
                        for j in range(-search_limit[1], search_limit[1] + 1)
                        for k in range(-search_limit[2], search_limit[2] + 1)])
        k_cart = k_latt @ basis
        shell_klatt, shell_kcart = cls.k_to_shells(k_latt, k_cart, kmesh_tol=kmesh_tol)
        num_shells = len(shell_kcart)
        del k_cart, k_latt

        shell_list_cart = []
        shell_list_latt = []
        projector_list_cart = []
        for i_shell in range(num_shells):
            shell_new_cart = shell_kcart[i_shell]
            shell_new_latt = shell_klatt[i_shell]
            # print (f"Trying shell {i_shell} with k_cart:\n{shell_new_cart}\nand k_latt:\n{shell_new_latt}")
            if is_parallel_shell(projector_list_cart, shell_new_latt, tol=kmesh_tol):
                # print(f"Skipping shell {i_shell} with k_cart {shell_new_cart} because it is parallel to previously selected shells {shell_list_cart}")
                continue
            shell_list_cart_tmp = shell_list_cart + [shell_new_cart]
            shell_list_latt_tmp = shell_list_latt + [shell_new_latt]
            wkbk = cls.get_shell_weights(shell_list_latt_tmp, shell_list_cart_tmp, bk_complete_tol=bk_complete_tol,
                                         msg_if_fail=True)
            if isinstance(wkbk, str):
                if wkbk == "zero singular value":
                    continue
                elif wkbk == "incomplete shells":
                    projector_list_cart.append(cls.get_projector_shell_cart(shell_new_cart))
                    shell_list_cart.append(shell_new_cart)
                    shell_list_latt.append(shell_new_latt)
                else:
                    raise RuntimeError(f"Unexpected error in get_shell_weights: {wkbk}")
            else:
                wk, bk_cart, bk_grid = wkbk
                return wk, bk_cart, bk_grid


        raise RuntimeError(
            f"Could not find a complete set of bk vectors up to {num_shells} shells. ")

    @classmethod
    def get_shell_weights(cls, shell_klatt, shell_kcart, bk_complete_tol=1e-5, msg_if_fail=False):
        """
        get the weights of the shells of bk vectors for the finite-difference scheme

        Parameters
        ----------
        shell_klatt : list of np.ndarray(shape=(NNB, 3), dtype=int)
            the reciprocal lattice vectors of the shells (in the basis of the reciprocal lattice divided by the Monkhorst-Pack grid)
        shell_kcart : list of np.ndarray(shape=(NNB, 3), dtype=float)
            the reciprocal lattice vectors of the shells (in cartesian coordinates)
        bk_complete_tol : float
            the tolerance for the check of completeness of the shells

        Returns
        -------
        weight : np.ndarray(shape=(NNB,), dtype=float)
            the weights of the bk vectors
        bk_cart : np.ndarray(shape=(NNB, 3), dtype=float)
            the cartesian coordinates of the bk vectors
        bk_grid : np.ndarray(shape=(NNB, 3), dtype=int)
            the reciprocal lattice vectors of the bk vectors (in the basis of the reciprocal lattice divided by the Monkhorst-Pack grid)

        """
        shell_mat = np.array([kcart.T.dot(kcart) for kcart in shell_kcart])
        shell_mat_line = shell_mat.reshape(-1, 9)
        u, s, v = np.linalg.svd(shell_mat_line, full_matrices=False)
        print(f"SVD values for shell matrix:\n {s}")
        if np.any(s < 1e-10) or np.any(s > 1e10):
            msg = (f"Warning: shell matrix is close to singular, with SVD values:\n {s}. "
                   f"This may indicate that the shells are not linearly independent, and the weights may be unreliable.")
            if msg_if_fail:
                return "zero singular value"
            else:
                raise RuntimeError(msg)

        s = 1. / s
        weight_shell = np.eye(3).reshape(1, -1).dot(v.T.dot(np.diag(s)).dot(u.T)).reshape(-1)
        check_eye = sum(w * m for w, m in zip(weight_shell, shell_mat))
        tol = np.linalg.norm(check_eye - np.eye(3))
        if tol > bk_complete_tol:
            if msg_if_fail:
                return "incomplete shells"
            else:
                raise RuntimeError(
                    f"Error while determining shell weights. the following matrix :\n {check_eye} \n"
                    f"failed to be identity by an error of {tol}. Further debug information :  \n"
                    f"shell_mat={shell_mat}\n"
                    f"weight_shell={weight_shell}\n"
                    f"shell_klatt={shell_klatt}\n"
                    f"shell_kcart={shell_kcart}\n")

        print(f"Shells found with weights {weight_shell} and tolerance {tol}")
        bk_grid = []
        bk_cart = []
        wk = []
        for w, sh_klatt, sh_kcart in zip(weight_shell, shell_klatt, shell_kcart):
            for kl, kc in zip(sh_klatt, sh_kcart):
                bk_grid.append(kl)
                bk_cart.append(kc)
                wk.append(w)

        wk = np.array(wk)
        bk_cart = np.array(bk_cart)
        bk_grid = np.array(bk_grid, dtype=int)
        return wk, bk_cart, bk_grid

    @classmethod
    def get_projector_shell_cart(cls, shell_kcart):
        # SVD approach (most robust for dependent vectors)
        U, s, Vh = np.linalg.svd(shell_kcart.T, full_matrices=False)

        # Keep only significant singular vectors (filter small singular values)
        threshold = 1e-10
        rank = np.sum(s > threshold)
        U_reduced = U[:, :rank]

        # Now projector projects onto the xy plane (2D subspace)
        projector = U_reduced @ U_reduced.T
        # print (f"Projector before subtracting identity:\n{projector}")
        projector -= np.eye(3)  # We want to subtract the identity, to check if a new vector is in the span
        return projector

    @property
    def real_lattice(self):
        return 2 * np.pi * np.linalg.inv(self.recip_lattice).T


    @classmethod
    def k_to_shells(cls, k_latt, k_cart, kmesh_tol=1e-7):
        k_length = np.linalg.norm(k_cart, axis=1)
        select_nonzero = k_length > kmesh_tol
        k_latt = k_latt[select_nonzero]
        k_cart = k_cart[select_nonzero]
        k_length = k_length[select_nonzero]
        srt = np.argsort(k_length)  # skip the zero vector
        k_latt = k_latt[srt]
        k_cart = k_cart[srt]
        k_length = k_length[srt]
        brd = [0] + list(np.where(k_length[1:] - k_length[:-1] > kmesh_tol)[0] + 1) + [len(k_cart)]

        shell_kcart = [k_cart[b1:b2] for b1, b2 in zip(brd, brd[1:])]
        shell_klatt = [k_latt[b1:b2] for b1, b2 in zip(brd, brd[1:])]
        return shell_klatt, shell_kcart


def is_parallel_shell(projector_list_cart, shell_new_cart, tol=1e-5):
    """Check if two sets of shells are parallel to each other

    Parameters
    ----------
    projector_list_cart : list of np.ndarray(shape=(3, 3), dtype=float)
        the list of projectors for the existing shells
    shell_new_cart : np.ndarray(shape=(NNB, 3), dtype=float)
        the cartesian coordinates of the new shell to check
    tol : float
        the tolerance to consider two vectors as parallel
    """
    for projector in projector_list_cart:
        # Project the new shell vectors onto the subspace spanned by the existing shell
        projected = shell_new_cart @ projector.T
        # If all projected vectors are close to zero, they are in the span (i.e., parallel)
        if np.all(np.linalg.norm(projected, axis=1) < tol):
            return True
    return False
