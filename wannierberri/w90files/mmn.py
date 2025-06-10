from itertools import islice
import multiprocessing
from time import time
import numpy as np
from .utility import convert, get_mp_grid
from .w90file import W90_file


class MMN(W90_file):
    """
    class to store overlaps between Bloch functions at neighbouring k-points
    the MMN file of wannier90

    MMN.data[ik, ib, m, n] = <u_{m,k}|u_{n,k+b}>

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extension  `.mmn`)
    npar : int
        the number of parallel processes to be used for reading the file

    Attributes
    ----------
    data : np.ndarray(shape=(NK, NNB, NB, NB), dtype=complex)
        the overlap matrix elements between the Wavefunctions at neighbouring k-points
    neighbours : np.ndarray(shape=(NK, NNB), dtype=int)
        the indices of the neighbouring k-points
    G : np.ndarray(shape=(NK, NNB, 3), dtype=int)
        the reciprocal lattice vectors connecting the k-points
    """

    @property
    def n_neighb(self):
        """
        number of nearest neighbours indices
        """
        return 1

    def __init__(self, seedname="wannier90", npar=multiprocessing.cpu_count(), **kwargs):
        self.npz_tags = ["data", "neighbours", "G"]
        super().__init__(seedname=seedname, ext="mmn", npar=npar, **kwargs)

    def from_w90_file(self, seedname, npar):
        t0 = time()
        f_mmn_in = open(seedname + ".mmn", "r")
        f_mmn_in.readline()
        NB, NK, NNB = np.array(f_mmn_in.readline().split(), dtype=int)
        block = 1 + NB * NB
        data = []
        headstring = []
        mult = 4

        # FIXME: npar = 0 does not work
        if npar > 0:
            pool = multiprocessing.Pool(npar)
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
        self.data = np.array(data).reshape(NK, NNB, NB, NB).transpose((0, 1, 3, 2))
        headstring = np.array([s.split() for s in headstring], dtype=int).reshape(NK, NNB, 5)
        assert np.all(headstring[:, :, 0] - 1 == np.arange(NK)[:, None])
        self.neighbours = headstring[:, :, 1] - 1
        self.G = headstring[:, :, 2:]
        t2 = time()
        print(f"Time for MMN.__init__() : {t2 - t0} , read : {t1 - t0} , headstring {t2 - t1}")
        return self

    def to_w90_file(self, seedname):
        f_mmn_out = open(seedname + ".mmn", "w")
        f_mmn_out.write("MMN file\n")
        f_mmn_out.write(f"{self.NB} {self.NK} {self.NNB}\n")
        for ik in range(self.NK):
            for ib in range(self.NNB):
                f_mmn_out.write(f"{ik + 1} {self.neighbours[ik, ib] + 1} {' '.join(map(str, self.G[ik, ib]))}\n")
                for m in range(self.NB):
                    for n in range(self.NB):
                        f_mmn_out.write(f"{self.data[ik, ib, n, m].real} {self.data[ik, ib, n, m].imag}\n")
        f_mmn_out.close()

    def select_bands(self, selected_bands):
        if selected_bands is not None:
            self.data = self.data[:, :, selected_bands, :][:, :, :, selected_bands]

    # def get_disentangled(self, v_left, v_right):
    #     """
    #     Reduce number of bands

    #     Parameters
    #     ----------
    #     v_matrix : np.ndarray(NB,num_wann)
    #         the matrix of column vectors defining the Wannier gauge
    #     v_matrix_dagger : np.ndarray(num_wann,NB)
    #         the Hermitian conjugate of `v_matrix`

    #     Returns
    #     -------
    #     `~wannierberri.system.w90_files.MMN`
    #         the disentangled MMN object
    #     """
    #     print(f"v shape {v_left.shape}  {v_right.shape}")
    #     data = np.zeros((self.NK, self.NNB, v_right.shape[2], v_right.shape[2]), dtype=complex)
    #     print(f"shape of data {data.shape} , old {self.data.shape}")
    #     for ik in range(self.NK):
    #         for ib, iknb in enumerate(self.neighbours[ik]):
    #             data[ik, ib] = v_left[ik] @ self.data[ik, ib] @ v_right[iknb]
    #     return self.__class__(data=data)

    
    def set_bk(self, kpt_latt, mp_grid, recip_lattice, kmesh_tol=1e-7, bk_complete_tol=1e-5):
        try:
            self.bk_cart
            self.bk_latt
            self.bk_reorder
            self.wk
            return
        except AttributeError:
            bk_latt_all = np.array(
                np.round(
                    [
                        (kpt_latt[nbrs] - kpt_latt + G) * mp_grid[None, :]
                        for nbrs, G in zip(self.neighbours.T, self.G.transpose(1, 0, 2))
                    ]).transpose(1, 0, 2),
                dtype=int)
            
            self.bk_latt = bk_latt_all[0]

            ## Reorder the bk_latt vectors to match the order of the first k-point
            bk_latt_tuples_0 = [tuple(b) for b in self.bk_latt]
            self.bk_reorder = []
            for ik in range(self.NK):
                bk_latt_tuples = [tuple(b) for b in bk_latt_all[ik]]
                srt = [bk_latt_tuples.index(bk) for bk in bk_latt_tuples_0]
                assert len(srt) == self.NNB, f"Reordering failed for k-point {ik}. Expected {self.NNB} neighbours, got {len(srt)}"
                assert np.all(self.bk_latt == bk_latt_all[ik, srt]), \
                    f"Reordering failed for k-point {ik}. Expected {self.bk_latt}, got {bk_latt_all[ik, srt]}"
                self.bk_reorder.append(srt)
                self.data[ik,:] = self.data[ik,srt, :]
                self.G[ik,:] = self.G[ik,srt]
                self.neighbours[ik,:] = self.neighbours[ik,srt]
            self.bk_reorder = np.array(self.bk_reorder, dtype=int)

            self.bk_cart = self.bk_latt.dot(recip_lattice / mp_grid[:, None])
            bk_length = np.linalg.norm(self.bk_cart, axis=1)
            srt = np.argsort(bk_length)
            srt_inv = np.argsort(srt)
            bk_length_srt = bk_length[srt]
            brd = [
                0,
            ] + list(np.where(bk_length_srt[1:] - bk_length_srt[:-1] > kmesh_tol)[0] + 1) + [
                self.NNB,
            ]
            shell_mat = []
            shell_mat = np.array([self.bk_cart[srt[b1:b2]].T.dot(self.bk_cart[srt[b1:b2]]) for b1, b2 in zip(brd, brd[1:])])
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
                    f"bk_latt={self.bk_latt} \n bk_cart={self.bk_cart} \n"
                    f"bk_cart_length={bk_length}\n shell_mat={shell_mat}\n"
                    f"weight_shell={weight_shell}, srt={srt}\n")
            weight = np.array([w for w, b1, b2 in zip(weight_shell, brd, brd[1:]) for i in range(b1, b2)])
            self.wk = weight[srt_inv]
            print (f"the weights of the bk vectors are {self.wk} ")
            

    def set_bk_chk(self, chk, **argv):
        self.set_bk(kpt_latt=chk.kpt_latt, mp_grid=chk.mp_grid, recip_lattice=chk.recip_lattice, **argv)



def mmn_from_bandstructure(bandstructure,
                           normalize=True, return_object=True, verbose=False,
                           param_search_bk={}):
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
    spinor = bandstructure.spinor

    if verbose:
        print("Creating mmn. ")
    data = []

    kpt_latt = np.array([kp.k for kp in bandstructure.kpoints])
    NK = kpt_latt.shape[0]
    mp_grid = get_mp_grid(kpt_latt)
    rec_latt = bandstructure.RecLattice
    
    wk, bk_cart, bk_latt = find_bk_vectors(
        recip_lattice=bandstructure.RecLattice, 
        mp_grid=mp_grid,
        **param_search_bk
        )
    
    NNB = len(wk)
    NB = bandstructure.NB
    
    k_latt_int = np.rint(kpt_latt * mp_grid[:, None]).astype(int)

    G = np.zeros( (NK, NNB, 3), dtype=int)
    ik_nb = np.zeros((NK, NNB), dtype=int)
    for ik in range(NK):
        for ib in range(NNB):
            k_latt_int_nb = k_latt_int[ik] + bk_latt[ib]
            for ik_nb in range(NK):
                g = k_latt_int_nb - k_latt_int[ik_nb]
                if np.all( g % mp_grid == 0):
                    ik_nb[ik, ib] = ik_nb
                    G[ik, ib] = g // mp_grid
                    break
            else:
                raise RuntimeError(
                    f"Could not find a neighbour for k-point {ik} with k-lattice {k_latt_int[ik]} and "
                    f"bk-lattice {bk_latt[ib]} in the Monkhorst-Pack grid {mp_grid}. "
                    f"Check the parameters of `find_bk_vectors`."
                )
            

    igmin_k = [kp.ig[:3, :].min(axis=1) for kp in bandstructure.kpoints]
    igmax_k = [kp.ig[:3, :].max(axis=1) for kp in bandstructure.kpoints]

    
    igmin_glob = np.array(igmin_k).min(axis=0) -G.max(axis=0)
    igmax_glob = np.array(igmax_k).max(axis=0) -G.min(axis=0)

    ig_grid = igmax_glob - igmin_glob + 1


    bra = np.zeros((NB, ) + tuple(ig_grid), dtype=complex)
    ket = np.zeros((NB, ) + tuple(ig_grid), dtype=complex)

    data = np.zeros((NK, NNB, NB, NB), dtype=complex)

    for ik1, kp1 in enumerate(bandstructure.kpoints):
        for ig, g in enumerate(kp1.ig.T):
            g_loc = g - igmin_glob
            assert np.all(g_loc >= 0) and np.all(g_loc < ig_grid), \
                f"g_loc {g_loc} out of bounds for ig_grid {ig_grid} at ik1={ik1}, ig={ig}"
            bra[ :, g_loc[0], g_loc[1], g_loc[2] ] = kp1.WF[:, ig].conj()
        for inb, ik2 in enumerate (ik_nb[ik1]):
            kp2 = bandstructure.kpoints[ik2]
            for ig, g in enumerate(kp2.ig.T):
                g_loc = g - igmin_glob - G[ik1, inb]
                assert np.all(g_loc >= 0) and np.all(g_loc < ig_grid), \
                    f"g_loc {g_loc} out of bounds for ig_grid {ig_grid} at ik1={ik1}, inb={inb}, ik2={ik2}"
                ket[:, g_loc[0], g_loc[1], g_loc[2]] = kp2.WF[:, ig]
            data[ik1, inb] = np.einsum('aijk,bijk->ab', bra, ket, optimize='greedy')
        
        
    if return_object:
        return MMN().from_dict(data=data, neighbours=ik_nb, G=G,)
    else:
        return data



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
    k_latt = np.array([(i,j,k) for i in range(-search_limit[0], search_limit[0] + 1)
                       for j in range(-search_limit[1], search_limit[1] + 1)
                       for k in range(-search_limit[2], search_limit[2] + 1)])
    k_cart = k_latt @ basis
    k_length = np.linalg.norm(k_cart, axis=1)
    srt = np.argsort(k_length)[1:]  # skip the zero vector
    k_latt = k_latt[srt]
    k_cart = k_cart[srt]
    k_length = k_length[srt]
    brd = [0] + list(np.where(k_length[1:] - k_length[:-1] > kmesh_tol)[0] + 1) + [len(k_cart)]

    k_length_shell = np.array([k_length[b1:b2].mean() for b1, b2 in zip(brd, brd[1:])])
    shell_kcart = [k_cart[b1:b2] for b1, b2 in zip(brd, brd[1:])] 
    shell_klatt = [k_latt[b1:b2] for b1, b2 in zip(brd, brd[1:])]
    num_shells = len(shell_kcart)
    del brd, k_length, k_cart, k_latt

    shells_selected = []
    k_cart_selected = np.zeros((0, 3), dtype=float)
    matrix_rank = 0
    for i_shell in range(num_shells):
        k_cart_selected_try = np.vstack( [k_cart_selected, shell_kcart[i_shell]])
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

    print (f"Shells found with weights {weight_shell} and tolerance {tol}")
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
            
    