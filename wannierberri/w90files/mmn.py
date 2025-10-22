from collections import defaultdict
import functools
from itertools import islice
import multiprocessing
from time import time
import numpy as np

from .utility import convert, grid_from_kpoints
from .w90file import W90_file, auto_kptirr, check_shape
from ..io import sparselist_to_dict
from ..utility import cached_einsum


class MMN(W90_file):
    """
    class to store overlaps between Bloch functions at neighbouring k-points
    the MMN file of wannier90

    MMN.data[ik, ib, m, n] = <u_{m,k}|u_{n,k+b}>

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extension  `.mmn`)

    Attributes
    ----------
    data : np.ndarray(shape=(NK, NNB, NB, NB), dtype=complex)
        the overlap matrix elements between the Wavefunctions at neighbouring k-points
    neighbours : np.ndarray(shape=(NK, NNB), dtype=int)
        the indices of the neighbouring k-points
    G : np.ndarray(shape=(NK, NNB, 3), dtype=int)
        the reciprocal lattice vectors connecting the k-points
    """

    npz_tags = ["NK", "bk_cart", "bk_latt", "wk"]
    npz_keys_dict_int = ["data", "neighbours", "G", "bk_reorder"]
    extension = "mmn"

    def __init__(self, data, neighbours, G, bk_latt, bk_cart, wk, bk_reorder=None, NK=None):
        super().__init__(data=data, NK=NK)
        G = sparselist_to_dict(G)
        neighbours = sparselist_to_dict(neighbours)
        shape = check_shape(self.data)
        assert len(shape) == 3, f"MMN data should have 4 dimensions, got {len(shape)}"
        assert shape[1] == shape[2], f"MMN data should have NB x NB shape, got {shape[1]} x {shape[2]}"
        self.NNB = shape[0]
        if bk_reorder is None:
            bk_reorder = {ik: np.arange(self.NNB) for ik in G.keys()}
        bk_reorder = sparselist_to_dict(bk_reorder)
        self.NB = shape[1]
        self.neighbours = neighbours
        assert check_shape(self.neighbours) == (self.NNB,)
        self.G = G
        assert check_shape(self.G) == (self.NNB, 3)
        self.bk_latt = np.array(bk_latt, dtype=int)
        assert self.bk_latt.shape == (self.NNB, 3), \
            f"MMN bk_latt should have shape (NNB, 3), got {self.bk_latt.shape}"
        self.bk_cart = np.array(bk_cart, dtype=float)
        assert self.bk_cart.shape == (self.NNB, 3), \
            f"MMN bk_cart should have shape (NNB, 3), got {self.bk_cart.shape}"
        self.wk = np.array(wk, dtype=float)
        assert self.wk.shape == (self.NNB,), \
            f"MMN wk should have shape (NNB,), got {self.wk.shape}"
        if bk_reorder is None:
            bk_reorder = sparselist_to_dict([np.arange(self.NNB, dtype=int) for _ in range(self.NK)])
        self.bk_reorder = bk_reorder

    @classmethod
    def from_w90_file(cls, seedname, kpt_latt, recip_lattice, npar=multiprocessing.cpu_count(), selected_kpoints=None):
        t0 = time()
        f_mmn_in = open(seedname + ".mmn", "r")
        f_mmn_in.readline()
        NB, NK, NNB = np.array(f_mmn_in.readline().split(), dtype=int)
        if selected_kpoints is None:
            selected_kpoints = np.arange(NK)
        block = 1 + NB * NB
        data = []
        headstring = []
        mult = 4

        # TODO: FIXME: npar = 0 does not work
        if npar > 0:
            pool = multiprocessing.Pool(npar)
        # TODO : do trext conversion only for selected kpoints
        for j in range(0, NNB * NK, npar * mult):
            x = list(islice(f_mmn_in, int(block * npar * mult)))
            if len(x) == 0:
                break
            headstring += x[::block]
            y = [x[i * block + 1:(i + 1) * block] for i in range(npar * mult) if (i + 1) * block <= len(x)]
            if npar > 0:
                data += pool.map(convert, y)
            else:
                data += [convert(z) for z in y]

        if npar > 0:
            pool.close()
            pool.join()
        f_mmn_in.close()
        t1 = time()
        data = [d[:, 0] + 1j * d[:, 1] for d in data]
        data = np.array(data).reshape(NK, NNB, NB, NB).transpose((0, 1, 3, 2))
        data = {ik: data[ik] for ik in selected_kpoints}
        headstring = np.array([s.split() for s in headstring], dtype=int).reshape(NK, NNB, 5)
        assert np.all(headstring[:, :, 0] - 1 == np.arange(NK)[:, None])
        neighbours = headstring[:, :, 1] - 1
        G = headstring[:, :, 2:]
        t2 = time()
        print(f"Time for MMN.__init__() : {t2 - t0} , read : {t1 - t0} , headstring {t2 - t1}")
        bk_cart, bk_latt, wk, bk_reorder = MMN.get_bk(
            G=G, neighbours=neighbours, NNB=NNB, NK=NK,
            kpt_latt=kpt_latt, recip_lattice=recip_lattice)
        for ik in selected_kpoints:
            srt = bk_reorder[ik]
            data[ik][:] = data[ik][srt, :]
            G[ik][:] = G[ik][srt]
            neighbours[ik][:] = neighbours[ik][srt]

        return MMN(data=data,
                   neighbours=neighbours,
                   G=G,
                   bk_latt=bk_latt,
                   bk_cart=bk_cart,
                   wk=wk,
                   bk_reorder=bk_reorder,
                   NK=NK)

    def to_w90_file(self, seedname):
        f_mmn_out = open(seedname + ".mmn", "w")
        f_mmn_out.write("MMN file\n")
        f_mmn_out.write(f"{self.NB} {self.NK} {self.NNB}\n")
        for ik in range(self.NK):
            for ib in range(self.NNB):
                f_mmn_out.write(f"{ik + 1} {self.neighbours[ik, ib] + 1} {' '.join(map(str, self.G[ik][ib]))}\n")
                for m in range(self.NB):
                    for n in range(self.NB):
                        f_mmn_out.write(f"{self.data[ik][ib, n, m].real} {self.data[ik][ib, n, m].imag}\n")
        f_mmn_out.close()

    def select_bands(self, selected_bands):
        return super().select_bands(selected_bands, dimensions=(1, 2))

    # TODO : combine with find_bk_vectors
    @staticmethod
    def get_bk(G, neighbours, NNB, NK, kpt_latt, recip_lattice, kmesh_tol=1e-7, bk_complete_tol=1e-5):
        mp_grid = np.array(grid_from_kpoints(kpt_latt))
        bk_latt_all = np.array(
            np.round(
                [
                    (kpt_latt[nbrs] - kpt_latt + G) * mp_grid[None, :]
                    for nbrs, G in zip(neighbours.T, G.transpose(1, 0, 2))
                ]).transpose(1, 0, 2),
            dtype=int)

        bk_latt = bk_latt_all[0]

        ## Reorder the bk_latt vectors to match the order of the first k-point
        bk_latt_tuples_0 = [tuple(b) for b in bk_latt]
        bk_reorder = []
        for ik in range(NK):
            bk_latt_tuples = [tuple(b) for b in bk_latt_all[ik]]
            srt = [bk_latt_tuples.index(bk) for bk in bk_latt_tuples_0]
            assert len(srt) == NNB, f"Reordering failed for k-point {ik}. Expected {NNB} neighbours, got {len(srt)}"
            assert np.all(bk_latt == bk_latt_all[ik, srt]), \
                f"Reordering failed for k-point {ik}. Expected {bk_latt}, got {bk_latt_all[ik, srt]}"
            bk_reorder.append(srt)
        bk_reorder = np.array(bk_reorder, dtype=int)

        bk_cart = bk_latt.dot(recip_lattice / mp_grid[:, None])
        bk_length = np.linalg.norm(bk_cart, axis=1)
        srt = np.argsort(bk_length)
        srt_inv = np.argsort(srt)
        bk_length_srt = bk_length[srt]
        brd = [
            0,
        ] + list(np.where(bk_length_srt[1:] - bk_length_srt[:-1] > kmesh_tol)[0] + 1) + [
            NNB,
        ]
        shell_mat = []
        shell_mat = np.array([bk_cart[srt[b1:b2]].T.dot(bk_cart[srt[b1:b2]]) for b1, b2 in zip(brd, brd[1:])])
        shell_mat_line = shell_mat.reshape(-1, 9)
        u, s, v = np.linalg.svd(shell_mat_line, full_matrices=False)
        s = 1. / s
        weight_shell = np.eye(3).reshape(1, -1).dot(v.T.dot(np.diag(s)).dot(u.T)).reshape(-1)
        check_eye = sum(w * m for w, m in zip(weight_shell, shell_mat))
        tol = np.linalg.norm(check_eye - np.eye(3))
        if tol > bk_complete_tol:
            raise RuntimeError(
                f"Error while determining shell weights. the following matrix :\n {check_eye} \n"
                f"failed to be identity by an error of {tol}. Further debug information :  \n"
                f"bk_latt={bk_latt} \n bk_cart={bk_cart} \n"
                f"bk_cart_length={bk_length}\n shell_mat={shell_mat}\n"
                f"weight_shell={weight_shell}, srt={srt}\n")
        weight = np.array([w for w, b1, b2 in zip(weight_shell, brd, brd[1:]) for i in range(b1, b2)])
        wk = weight[srt_inv]

        return bk_cart, bk_latt, wk, bk_reorder

    def reorder_bk(self,
                   bk_reorder=None,
                   bk_latt_new=None,
                    ):
        """
        Reorder the bk vectors according to the given order

        Parameters
        ----------
        bk_reorder : list of int or None
            the new order of the bk vectors. If None, the order is taken from bk_latt_new
        bk_latt_new : list of tuple or None
            the new list of bk vectors in the basis of the reciprocal lattice divided by the Monkhorst-Pack grid.
            If None, the order is taken from self.bk_latt
        """
        assert (bk_reorder is not None) != (bk_latt_new is not None), \
            "Either bk_reorder or bk_latt_new should be provided, but not both."
        if bk_reorder is None:
            bk_latt_new = np.array(bk_latt_new, dtype=int)
            assert bk_latt_new.shape == (self.NNB, 3), \
                f"bk_latt_new should have shape (NNB, 3), got {bk_latt_new.shape}"
            bk_reorder = [np.where((self.bk_latt == b).all(axis=1))[0][0] for b in bk_latt_new]
        print(f"Reordering bk vectors with {bk_reorder} ")
        self.bk_latt = self.bk_latt[bk_reorder]
        if bk_latt_new is not None:
            assert np.all(self.bk_latt == bk_latt_new), \
                f"Reordered bk_latt {self.bk_latt} does not match the provided bk_latt_new {bk_latt_new}"
        self.bk_cart = self.bk_cart[bk_reorder]
        self.wk = self.wk[bk_reorder]

        for ik, d in self.data.items():
            self.data[ik] = d[bk_reorder, :]
        for ik, g in self.G.items():
            self.G[ik] = g[bk_reorder]
        for ik, n in self.neighbours.items():
            self.neighbours[ik] = n[bk_reorder]
        for ik in self.bk_reorder.keys():
            self.bk_reorder[ik] = self.bk_reorder[ik][bk_reorder]

    def equals(self, other, tolerance=1e-8, check_reorder=True):
        iseq, message = super().equals(other, tolerance)
        if not iseq:
            return iseq, message
        if self.NNB != other.NNB:
            return False, f"the number of neighbouring bands is not equal: {self.NNB} and {other.NNB} correspondingly"
        if not np.all(self.bk_latt == other.bk_latt):
            return False, f"the bk_latt vectors are not equal: {self.bk_latt} and {other.bk_latt} correspondingly"
        if not np.allclose(self.bk_cart, other.bk_cart):
            return False, f"the bk_cart vectors are not equal: {self.bk_cart} and {other.bk_cart} correspondingly"
        if not np.allclose(self.wk, other.wk):
            return False, f"the wk valuesare not equal: {self.wk} and {other.wk} correspondingly"
        if check_reorder:
            for ik in self.bk_reorder.keys():
                if not np.all(self.bk_reorder[ik] == other.bk_reorder[ik]):
                    return False, f"the bk_reorder vectors are not equal for k-point {ik}: {self.bk_reorder[ik]} and {other.bk_reorder[ik]} correspondingly"
        return True, ""


    @classmethod
    def from_bandstructure(cls, bandstructure,
                           normalize=False,
                           verbose=False,
                           param_search_bk={},
                           selected_kpoints=None,
                           kptirr=None,
                        #    kpt_from_kptirr_isym=None,
                        #    kpt2kptirr=None,
                           NK=None,
                        #    kpt_latt_grid=None,
                           symmetrizer=None,
                           irreducible=False,
                           irred_bk_only=True,
                           include_paw=True,
                           include_pseudo=True,
                           ):
        """
        Create an AMN object from a BandStructure object
        So far only delta-localised s-orbitals are implemented

        Parameters
        ----------
        bandstructure : BandStructure
            the band structure object
        normalize : bool
            if True, the wavefunctions are normalised
        return_object : bool
            if True, return an MMN object, otherwise return the data as a numpy array
        param_search_bk : dict
            additional parameters for `:func:find_bk_vectors`

        Returns
        -------
        MMN or np.ndarray
            the MMN object ( if `return_object` is True ) or the data as a numpy array ( if `return_object` is False )
        """
        if irreducible:
            assert symmetrizer is not None, "Symmetrizer should be provided if irreducible is True"
        if irreducible:
            if kptirr is None:
                kptirr = symmetrizer.kptirr
            kpt2kptirr = symmetrizer.kpt2kptirr
            kpt_from_kptirr_isym = symmetrizer.kpt_from_kptirr_isym
            kpt_latt_grid = symmetrizer.kpoints_all
        else:
            kpt_latt_grid = np.array([kp.k for kp in bandstructure.kpoints])
        mp_grid = np.array(grid_from_kpoints(kpt_latt_grid))
        k_latt_int = np.rint(kpt_latt_grid * mp_grid[None, :]).astype(int)

        NK, selected_kpoints, kptirr = auto_kptirr(
            bandstructure, selected_kpoints=selected_kpoints, kptirr=kptirr, NK=NK)
        print(f"NK= {NK}, selected_kpoints = {selected_kpoints}, kptirr = {kptirr}")

        # NK = kpt_latt_grid.shape[0]
        identity_operation = bandstructure.spacegroup.get_identity_operation()

        spinor = bandstructure.spinor
        nspinor = 2 if spinor else 1

        if verbose:
            print("Creating mmn. ")

        if selected_kpoints is None:
            selected_kpoints = np.arange(NK)

        wk, bk_cart, bk_latt = find_bk_vectors(
            recip_lattice=bandstructure.RecLattice,
            mp_grid=mp_grid,
            **param_search_bk
        )


        NNB = len(wk)
        NB = bandstructure.num_bands

        bk_red = bk_latt / mp_grid[None, :]

        G = {ik: np.zeros((NNB, 3), dtype=int) for ik in kptirr}
        neighbours = {ik: np.zeros(NNB, dtype=int) for ik in kptirr}
        for kirr in kptirr:
            for ib in range(NNB):
                k_latt_int_nb = k_latt_int[kirr] + bk_latt[ib]
                for ik2 in range(NK):
                    g = k_latt_int_nb - k_latt_int[ik2]
                    if np.all(g % mp_grid == 0):
                        neighbours[kirr][ib] = ik2
                        G[kirr][ib] = g // mp_grid
                        break
                else:
                    raise RuntimeError(
                        f"Could not find a neighbour for k-point {kirr} with k-lattice {k_latt_int[kirr]} and "
                        f"bk-lattice {bk_latt[ib]} in the Monkhorst-Pack grid {mp_grid}. "
                        f"Check the parameters of `find_bk_vectors`."
                    )

        # now get the neighbour kpoint' wavefunctions, if those points do not belong to the irreducible k-points
        if hasattr(bandstructure, "kpoints_paw"):
            use_paw = True
            kpoints_in = bandstructure.kpoints_paw
        else:
            use_paw = False
            kpoints_in = bandstructure.kpoints
        kpoints_sel = [kpoints_in[ik] for ik in selected_kpoints]
        kpoints_dict_all = {ik: kpoints_sel[ik] for ik in kptirr}  # a dictionary to store kpoints that are not in the original bandstructure

        if symmetrizer is not None and irred_bk_only:
            bkirr, bk2bkirr, bk_from_bk_irr_isym, bk_map = symmetrizer.get_bk_mapping(bk_latt, neighbours)
        else:
            bkirr = [np.arange(NNB) for _ in kptirr]
            bk2bkirr = np.array([np.arange(NNB) for _ in kptirr])
            bk_from_bk_irr_isym = np.zeros((len(kptirr), NNB), dtype=int)
            bk_map = None


        for ikirr, kirr in enumerate(kptirr):
            for ib, ik2 in enumerate(neighbours[kirr]):
                if ib in bkirr[ikirr]:
                    ik2 = int(ik2)
                    if ik2 not in kpoints_dict_all:
                        # if use_paw:
                        #     raise RuntimeError("PAW k-points should be provided for all k-points in the Monkhorst-Pack grid")``
                        isym = kpt_from_kptirr_isym[ik2]
                        # print(f"isym = {isym}, ik2={ik2}, {kpt_from_kptirr_isym=}")
                        ik_origin = kpt2kptirr[ik2]
                        kp_origin = kpoints_sel[ik_origin]
                        symop = bandstructure.spacegroup.symmetries[isym]
                        kp2 = kp_origin.get_transformed_copy(symmetry_operation=symop,
                                                        k_new=kpt_latt_grid[ik2])
                        kpoints_dict_all[ik2] = kp2



        data = defaultdict(lambda: np.zeros((NNB, NB, NB), dtype=complex))
        if use_paw:
            wavefunc_all = Grid_PAW_all(kpoints_dict_all=kpoints_dict_all,
                                        product=functools.partial(bandstructure.overlap_paw.product, include_paw=include_paw, include_pseudo=include_pseudo),
                                        identity_operation=identity_operation)
        else:
            wavefunc_all = Grid_ig_all(kpoints_dict_all, G, NB, nspinor, normalize=normalize)

        # but are needed for the finite-difference scheme (obtained by symmetry)
        for ikirr, kirr in enumerate(kptirr):
            bra = wavefunc_all.get_WF(kirr, conj=True)
            # overlaps = wavefunc_all.product(bra, bra)
            # overlaps_err = np.max(abs((overlaps - np.eye(NB))))
            # print(f"Calculating overlaps for k-point {ikirr} orthonorm_error = {overlaps_err}")
            # print(f"Calculating overlaps for k-point {ikirr}({kirr} on the grid) the little group  is {symmetrizer.isym_little[ikirr]}, {bk_from_bk_irr_isym[ikirr]=}")
            ib_is_set = np.zeros(NNB, dtype=bool)
            for ib in bkirr[ikirr]:  # only calculate for irreducible bk points
                assert not ib_is_set[ib], f"bk index {ib} for k-point {kirr} is already set."
                ik2 = int(neighbours[kirr][ib])
                ket = wavefunc_all.get_WF(ik2, conj=False, G=G[kirr][ib])
                data[kirr][ib, :, :] = wavefunc_all.product(bra, ket, bk=bk_red[ib])
                ib_is_set[ib] = True

            for ib, ik2 in enumerate(neighbours[kirr]):
                if ib not in bkirr[ikirr]:
                    assert not ib_is_set[ib], f"bk index {ib} for k-point {kirr} is already set."
                    ib_origin = bk2bkirr[ikirr][ib]
                    ikb_origin = int(neighbours[kirr][ib_origin])
                    isym = bk_from_bk_irr_isym[ikirr][ib]
                    assert symmetrizer.kptirr2kpt[ikirr, isym] == kirr
                    # print(f"  Symmetry operation {isym} is used to get the overlaps for k-point {ikirr} and bk index {ib} from bk index {ib_origin} (ikikirr={ikikirr})")
                    # print(f"calling symmetrizer.transform_Mmn_kb with isym={isym}, ikirr={ikikirr}, ib={ib_origin}, ikb={ikb_origin}")
                    data[kirr][ib, :, :] = symmetrizer.transform_Mmn_kb(M=data[kirr][ib_origin],
                                                                        isym=isym, ikirr=ikirr, ib=ib_origin,
                                                                        ikb=ikb_origin,
                                                                        bk_latt_map=bk_map, bk_cart=bk_cart)
                    ib_is_set[ib] = True
        return MMN(
            data=data,
            NK=NK,
            neighbours=neighbours,
            G=G,
            bk_latt=bk_latt,
            bk_cart=bk_cart,
            wk=wk,
            bk_reorder=None
        )


class Grid_PAW_all:

    def __init__(self, kpoints_dict_all, identity_operation, product):
        self.kpoints_dict_all = kpoints_dict_all
        self.identity_operation = identity_operation
        self.product = product

    def get_WF(self, ik, G=0, conj=False):
        kp = self.kpoints_dict_all[ik]
        kp = kp.get_transformed_copy(symmetry_operation=self.identity_operation, k_new=kp.k + G)
        return kp


class Grid_ig_all:

    def __init__(self, kpoints_dict_all, G, NB, nspinor, normalize=False):
        self.kpoints_dict_all = kpoints_dict_all
        ig_list = [kp.ig for kp in kpoints_dict_all.values()]

        igmin_k = np.array([ig[:, :3].min(axis=0) for ig in ig_list])
        igmax_k = np.array([ig[:, :3].max(axis=0) for ig in ig_list])

        del ig_list
        self.NB = NB
        self.nspinor = nspinor

        Gloc = np.array([g for g in G.values()])

        self.igmin_glob = igmin_k.min(axis=0) - Gloc.max(axis=(0, 1))
        self.igmax_glob = igmax_k.max(axis=0) - Gloc.min(axis=(0, 1))

        self.ig_grid = self.igmax_glob - self.igmin_glob + 1
        # print(f"ig_grid = {ig_grid}, igmin_glob = {igmin_glob}, igmax_glob = {igmax_glob}")
        self.normalize = normalize
        if normalize:
            self.norm = {ik2: np.linalg.norm(kp.WF.reshape(NB, -1), axis=1)
                         for ik2, kp in kpoints_dict_all.items()}

    def get_WF(self, ik, conj=False, G=0):
        kp1 = self.kpoints_dict_all[ik]
        bra = np.zeros((self.NB, self.nspinor) + tuple(self.ig_grid), dtype=complex)
        for ig, g in enumerate(kp1.ig):
            _g = g[:3] - self.igmin_glob - G
            assert np.all(_g >= 0) and np.all(_g < self.ig_grid), \
                f"g {_g} out of bounds for ig_grid {self.ig_grid} at ik1={ik}, ig={ig}"
            for ispinor in range(self.nspinor):
                bra[:, ispinor, _g[0], _g[1], _g[2]] = kp1.WF[:, ig, ispinor]
        if conj:
            bra = bra.conj()
        if self.normalize:
            bra[:] = bra / self.norm[ik][:, None, None, None, None]
        return bra

    def product(self, bra, ket, bk=None):
        """
        Calculate the product <bra|ket> 
        bk is not used for the grid representation, but is kept for compatibility with the PAW representation
        """
        return cached_einsum('asijk,bsijk->ab', bra, ket)


def find_bk_vectors(recip_lattice, mp_grid, kmesh_tol=1e-7, bk_complete_tol=1e-5, search_supercell=2):
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
    bk_latt : np.ndarray(shape=(NNB, 3), dtype=int)
        the bk vectors in the basis of the reciprocal lattice divided by the Monkhorst-Pack grid
    """
    mp_grid = np.array(mp_grid, dtype=int)
    basis = recip_lattice / mp_grid[:, None]
    search_limit = search_supercell * np.array(mp_grid)
    k_latt = np.array([(i, j, k) for i in range(-search_limit[0], search_limit[0] + 1)
                       for j in range(-search_limit[1], search_limit[1] + 1)
                       for k in range(-search_limit[2], search_limit[2] + 1)])
    k_cart = k_latt @ basis
    k_length = np.linalg.norm(k_cart, axis=1)
    srt = np.argsort(k_length)[1:]  # skip the zero vector
    k_latt = k_latt[srt]
    k_cart = k_cart[srt]
    k_length = k_length[srt]
    brd = [0] + list(np.where(k_length[1:] - k_length[:-1] > kmesh_tol)[0] + 1) + [len(k_cart)]

    shell_kcart = [k_cart[b1:b2] for b1, b2 in zip(brd, brd[1:])]
    shell_klatt = [k_latt[b1:b2] for b1, b2 in zip(brd, brd[1:])]
    num_shells = len(shell_kcart)
    del brd, k_length, k_cart, k_latt

    shells_selected = []
    k_cart_selected = np.zeros((0, 3), dtype=float)
    matrix_rank = 0
    for i_shell in range(num_shells):
        k_cart_selected_try = np.vstack([k_cart_selected, shell_kcart[i_shell]])
        matrix_rank_try = np.linalg.matrix_rank(k_cart_selected_try)
        if matrix_rank_try > matrix_rank:
            # print(f"Adding shell {i_shell} with length {k_length_shell[i_shell]} and k_cart {shell_kcart[i_shell]}")
            shells_selected.append(i_shell)
            k_cart_selected = k_cart_selected_try
            matrix_rank = matrix_rank_try
        else:
            # print(f"Skipping shell {i_shell} with length {k_length_shell[i_shell]} and k_cart {shell_kcart[i_shell]}")
            continue
        if matrix_rank == 3:
            break

    shell_kcart = [shell_kcart[i] for i in shells_selected]
    shell_klatt = [shell_klatt[i] for i in shells_selected]
    return get_shell_weights(shell_klatt, shell_kcart, bk_complete_tol=bk_complete_tol)


def get_shell_weights(shell_klatt, shell_kcart, bk_complete_tol=1e-5):
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
    bk_latt : np.ndarray(shape=(NNB, 3), dtype=int)
        the reciprocal lattice vectors of the bk vectors (in the basis of the reciprocal lattice divided by the Monkhorst-Pack grid)

    """
    shell_mat = np.array([kcart.T.dot(kcart) for kcart in shell_kcart])
    shell_mat_line = shell_mat.reshape(-1, 9)
    u, s, v = np.linalg.svd(shell_mat_line, full_matrices=False)
    s = 1. / s
    weight_shell = np.eye(3).reshape(1, -1).dot(v.T.dot(np.diag(s)).dot(u.T)).reshape(-1)
    check_eye = sum(w * m for w, m in zip(weight_shell, shell_mat))
    tol = np.linalg.norm(check_eye - np.eye(3))
    if tol > bk_complete_tol:
        raise RuntimeError(
            f"Error while determining shell weights. the following matrix :\n {check_eye} \n"
            f"failed to be identity by an error of {tol}. Further debug information :  \n"
            f"shell_mat={shell_mat}\n"
            f"weight_shell={weight_shell}\n"
            f"shell_klatt={shell_klatt}\n"
            f"shell_kcart={shell_kcart}\n")

    print(f"Shells found with weights {weight_shell} and tolerance {tol}")
    bk_latt = []
    bk_cart = []
    wk = []
    for w, sh_klatt, sh_kcart in zip(weight_shell, shell_klatt, shell_kcart):
        for kl, kc in zip(sh_klatt, sh_kcart):
            bk_latt.append(kl)
            bk_cart.append(kc)
            wk.append(w)

    wk = np.array(wk)
    bk_cart = np.array(bk_cart)
    bk_latt = np.array(bk_latt, dtype=int)
    return wk, bk_cart, bk_latt
