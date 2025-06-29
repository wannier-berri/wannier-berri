from functools import cached_property
from time import time
import warnings
import numpy as np
from .utility import readstr
from ..io import FortranFileR, SavableNPZ
from ..utility import alpha_A, beta_A


class CheckPoint(SavableNPZ):
    """
    A class to store the data about wannierisation, written by Wannier90

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extension `.chk`)
    kmesh_tol : float
        tolerance to distinguish different/same k-points
    bk_complete_tol : float
        tolerance for the completeness relation for finite-difference scheme
    """

    npz_tags = ["mp_grid", "real_lattice", "num_wann", "num_bands", "num_kpts", "kpt_latt", "mp_grid",
                "wannier_centers_cart", "wannier_spreads"]
    npz_tags_optional = ["wannier_centers_cart", "wannier_spreads"]
    npz_keys_dict_int = ["v_matrix"]


    def __init__(self,
                real_lattice=None,
                num_wann=None,
                num_bands=None,
                num_kpts=None,
                wannier_centers_cart=None,
                wannier_spreads=None,
                v_matrix=None,
                kpt_latt=None,
                mp_grid=None,
                kmesh_tol=1e-7,
                bk_complete_tol=1e-5,
    ):
        if real_lattice is not None:
            real_lattice = np.array(real_lattice, dtype=float)
            assert real_lattice.shape == (3, 3), f"real_lattice should be of shape (3, 3), but got {real_lattice.shape}"
            self.recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
            self.real_lattice = real_lattice
        else:
            self.recip_lattice = None
            self.real_lattice = None

        if wannier_centers_cart is not None:
            wannier_centers_cart = np.array(wannier_centers_cart, dtype=float)
            if num_wann is None:
                num_wann = wannier_centers_cart.shape[0]
            else:
                assert wannier_centers_cart.shape == (num_wann, 3), f"wannier_centers should be of shape ({num_wann}, 3), but got {wannier_centers_cart.shape}"
        self.wannier_centers_cart = wannier_centers_cart

        if wannier_spreads is not None:
            if num_wann is None:
                num_wann = wannier_spreads.shape[0]
            else:
                assert len(wannier_spreads) == num_wann, f"wannier_spreads should be of shape ({num_wann},), but got {len(wannier_spreads)}"
        self.wannier_spreads = wannier_spreads

        if kpt_latt is not None:
            kpt_latt = np.array(kpt_latt, dtype=float)
            if num_kpts is None:
                num_kpts = kpt_latt.shape[0]
            assert kpt_latt.shape == (num_kpts, 3), f"kpt_latt should be of shape ({num_kpts}, 3), but got {kpt_latt.shape}"
        self.kpt_latt = kpt_latt

        if mp_grid is not None:
            mp_grid = np.array(mp_grid, dtype=int)
            assert mp_grid.shape == (3,), f"mp_grid should be of shape (3,), but got {mp_grid.shape}"
        self.mp_grid = mp_grid


        self.kmesh_tol = kmesh_tol
        self.bk_complete_tol = bk_complete_tol

        if v_matrix is not None:
            if isinstance(v_matrix, list) or isinstance(v_matrix, np.ndarray):
                v_matrix = {ik: np.array(v, dtype=complex) for ik, v in enumerate(v_matrix) if v is not None}
            assert isinstance(v_matrix, dict), f"v_matrix should be a dictionary with keys as integers and values as np.arrays, but got {type(v_matrix)}"
            self.v_matrix = v_matrix
            nkey_provided = len(v_matrix)
            if nkey_provided == 0:
                warnings.warn("v_matrix is empty, no matrix elements provided")
            else:
                V0 = list(v_matrix.values())[0]
            if num_bands is None:
                num_bands = V0.shape[0]
            else:
                assert V0.shape[0] == num_bands, f"v_matrix[ik] should be of shape ( **{num_bands}**, num_wann), but got {V0.shape}"
            if num_wann is None:
                num_wann = V0.shape[1]
            else:
                assert V0.shape[1] == num_wann, f"v_matrix[ik] should be of shape (num_bands, {num_wann}), but got {V0.shape}"

        self.num_wann = num_wann
        self.num_bands = num_bands
        self.num_kpts = num_kpts




    def from_w90_file(self, seedname, kmesh_tol=1e-7, bk_complete_tol=1e-5):

        kmesh_tol = kmesh_tol  # will be used in set_bk
        bk_complete_tol = bk_complete_tol  # will be used in set_bk
        t0 = time()
        seedname = seedname.strip()
        FIN = FortranFileR(seedname + '.chk')
        readint = lambda: FIN.read_record('i4')
        readfloat = lambda: FIN.read_record('f8')

        def readcomplex():
            a = readfloat()
            return a[::2] + 1j * a[1::2]

        print('Reading restart information from file ' + seedname + '.chk :')
        readstr(FIN)  # comment line
        num_bands = readint()[0]
        num_exclude_bands = readint()[0]
        exclude_bands = readint()
        assert len(exclude_bands) == num_exclude_bands, f"read exclude_bands are {exclude_bands}, length={len(exclude_bands)} while num_exclude_bands={num_exclude_bands}"
        real_lattice = readfloat().reshape((3, 3), order='F')
        recip_lattice = readfloat().reshape((3, 3), order='F')
        assert np.linalg.norm(real_lattice.dot(recip_lattice.T) / (2 * np.pi) - np.eye(3)) < 1e-14, f"the read real and reciprocal lattices are not consistent {self.real_lattice.dot(self.recip_lattice.T) / (2 * np.pi)}!=identiy"
        num_kpts = readint()[0]
        mp_grid = readint()
        assert len(mp_grid) == 3
        assert num_kpts == np.prod(mp_grid), f"the number of k-points is not consistent with the mesh {num_kpts}!={np.prod(mp_grid)}"
        kpt_latt = readfloat().reshape((num_kpts, 3))
        nntot = readint()[0]
        num_wann = readint()[0]
        readstr(FIN)  # checkpoint string
        have_disentangled = bool(readint()[0])
        # print(f"have_disentangled={have_disentangled}")
        if have_disentangled:
            self.omega_invariant = readfloat()[0]
            lwindow = np.array(readint().reshape((num_kpts, num_bands)), dtype=bool)
            ndimwin = readint()
            # print(f"ndimwin={ndimwin}")
            u_matrix_opt = readcomplex().reshape((num_kpts, num_wann, num_bands)).swapaxes(1, 2)
            win_min = np.array([np.where(lwin)[0].min() for lwin in lwindow])
            win_max = np.array([np.where(lwin)[0].max() for lwin in lwindow])
            for ik in range(num_kpts):
                assert win_max[ik] - win_min[ik] + 1 == ndimwin[ik], f"win_max={win_max}, win_min={win_min}, ndimwin={ndimwin} - inconsistent"
                assert np.sum(lwindow[ik]) == ndimwin[ik], f"lwindow={lwindow}, ndimwin={ndimwin} - inconsistent"
                assert np.all(lwindow[ik, win_min[ik]:win_max[ik] + 1])
                if win_min[ik] > 0:
                    assert np.all(np.logical_not(lwindow[ik, :win_min[ik]]))
                if win_max[ik] < num_bands - 1:
                    assert np.all(np.logical_not(lwindow[ik, win_max[ik] + 1:]))
        u_matrix = readcomplex().reshape((num_kpts, num_wann, num_wann)).swapaxes(1, 2)
        readcomplex().reshape((num_kpts, nntot, num_wann, num_wann)).swapaxes(2, 3)  # m_matrix
        if have_disentangled:
            v_matrix = np.zeros((num_kpts, num_bands, num_wann), dtype=complex)
            for ik in range(num_kpts):
                u = u_matrix[ik]
                u_opt = u_matrix_opt[ik]
                nd = ndimwin[ik]
                assert np.linalg.norm(u_opt[nd:]) < 1e-12, f"u_opt[nd:]={u_opt[nd:]} - not zero"
                v_matrix[ik, lwindow[ik], :] = u_opt[:nd, :].dot(u)
        else:
            v_matrix = u_matrix
        wannier_centers_cart = readfloat().reshape((num_wann, 3))
        wannier_spreads = readfloat().reshape((num_wann))
        print(f"Time to read .chk : {time() - t0}")
        self.__init__(real_lattice=real_lattice,
                      v_matrix=v_matrix,
                      wannier_centers_cart=wannier_centers_cart, wannier_spreads=wannier_spreads,
                      kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol,
                      kpt_latt=kpt_latt, mp_grid=mp_grid,
        )

        return self

    @property
    def wannierised(self):
        if not hasattr(self, "v_matrix"):
            return False
        elif self.v_matrix is None:
            return False
        else:
            return True

    def spin_order_block_to_interlace(self):
        """
        If the chk was obtain from a block ordering (like in old VASP versions), the ordering should be changed to interlace
        """
        v_matrix = np.zeros((self.num_kpts, self.num_bands, self.num_wann), dtype=complex)
        v_matrix[:, :, 0::2] = self.v_matrix[:, :, :self.num_wann // 2]
        v_matrix[:, :, 1::2] = self.v_matrix[:, :, self.num_wann // 2:]
        self.v_matrix = v_matrix

    def spin_order_interlace_to_block(self):
        """
        If the chk was obtain from an interlace ordering, one may want to change the ordering to block
        """
        v_matrix = np.zeros((self.num_kpts, self.num_bands, self.num_wann), dtype=complex)
        v_matrix[:, :, :self.num_wann // 2] = self.v_matrix[:, :, 0::2]
        v_matrix[:, :, self.num_wann // 2:] = self.v_matrix[:, :, 1::2]
        self.v_matrix = v_matrix

    @cached_property
    def kpt_latt_int(self):
        """
        Returns the k-points in the lattice basis
        """
        return np.array(np.round(self.kpt_latt * self.mp_grid[None, :]), dtype=int)


    def wannier_gauge(self, mat, ik1, ik2):
        """
        Returns the matrix elements in the Wannier gauge

        Parameters
        ----------
        mat : np.ndarray
            the matrix elements in the Hamiltonian gauge
        ik1, ik2 : int
            the indices of the k-points

        Returns
        -------
        np.ndarray
            the matrix elements in the Wannier gauge
        """
        # data should be of form NBxNBx ...   - any form later
        if len(mat.shape) == 1:
            mat = np.diag(mat)
        assert mat.shape[:2] == (self.num_bands,) * 2, f"mat.shape={mat.shape}, num_bands={self.num_bands}"
        v1 = self.v_matrix[ik1].conj().T
        v2 = self.v_matrix[ik2]
        return np.tensordot(np.tensordot(v1, mat, axes=(1, 0)), v2, axes=(1, 0)).transpose(
            (0, -1) + tuple(range(1, mat.ndim - 1)))

    def get_HH_q(self, eig,
                 kptirr,
                 weights_k,
                 ):
        """
        Returns the Hamiltonian matrix in the Wannier gauge

        Parameters
        ----------
        eig : `~wannierberri.w90files.EIG`
            the eigenvalues of the Hamiltonian

        Returns
        -------
        np.ndarray
            the Hamiltonian matrix in the Wannier gauge
        """
        assert (eig.NK, eig.NB) == (self.num_kpts, self.num_bands), f"eig file has NK={eig.NK}, NB={eig.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        HH_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann), dtype=complex)
        for ik, w in zip(kptirr, weights_k):
            HH_q[ik] = self.wannier_gauge(eig.data[ik], ik, ik) * w  # Hamiltonian gauge
        # HH_q = np.array([self.wannier_gauge(eig.data[ik], ik, ik) for ik in range(self.num_kpts)])
        return 0.5 * (HH_q + HH_q.transpose(0, 2, 1).conj())

    def get_SS_q(self, spn,
                 kptirr,
                 weights_k,
                 ):
        """
        Returns the spin matrix in the Wannier gauge

        Parameters
        ----------
        spn : `~wannierberri.w90files.SPN`
            the spin matrix  

        Returns
        -------
        np.ndarray
            the spin matrix in the Wannier gauge
        """

        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands), f"spn file has NK={spn.NK}, NB={spn.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        SS_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3), dtype=complex)
        for ik, w in zip(kptirr, weights_k):
            SS_q[ik] = self.wannier_gauge(spn.data[ik], ik, ik) * w
        return 0.5 * (SS_q + SS_q.transpose(0, 2, 1, 3).conj())

    #########
    # Oscar #
    ###########################################################################

    # Depart from the original matrix elements in the ab initio mesh
    # (Hamiltonian gauge) to obtain the corresponding matrix elements in the
    # Wannier gauge. The last constitute the basis to construct the real-space
    # matrix elements for Wannier interpolation, independently of the
    # finite-difference scheme used.

    def get_AABB_qb(self, mmn, kptirr, weights_k,
                    transl_inv=False, eig=None, phase=None, sum_b=False):
        """
        Returns the matrix elements AA or BB(if eig is not Flase) in the Wannier gauge

        Parameters
        ----------
        mmn : `~wannierberri.w90files.MMN`
            the overlap matrix elements between the Wavefunctions at neighbouring k-points
        transl_inv : bool
            if True, the band-diagonal matrix elements are calculated using the Marzari & Vanderbilt 
            translational invariant formula
        eig : `~wannierberri.w90files.EIG`
            the eigenvalues of the Hamiltonian, needed to calculate BB (if None, the matrix elements are AA)
        phase : np.ndarray(shape=(num_wann, num_wann, nnb), dtype=complex)
            the phase factors to be applied to the matrix elements (if None, no phase factors are applied)
        sum_b : bool
            if True, the matrix elements are summed over the neighbouring k-points. Otherwise, the matrix elements are stored in a 5D array of shape (num_kpts, num_wann, num_wann, nnb, 3)

        Returns
        -------
        np.ndarray(shape=(num_kpts, num_wann, num_wann, nnb, 3), dtype=complex) (if sum_b=False)
        or np.ndarray(shape=(num_kpts, num_wann, num_wann, nnb, 3), dtype=complex) (if sum_b=True)
            the q-resolved matrix elements AA or BB in the Wannier gauge
        """
        assert (not transl_inv) or eig is None, "transl_inv cannot be used for BB matrix elements"
        if sum_b:
            AA_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3), dtype=complex)
        else:
            AA_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, mmn.NNB, 3), dtype=complex)
        for ik, w in zip(kptirr, weights_k):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik][ib]
                # Matrix < u_k | u_k+b > (mmn)
                data = mmn.data[ik][ib]                   # Hamiltonian gauge
                if eig is not None:
                    data = data * eig.data[ik][:, None]  # Hamiltonian gauge (add energies)
                AAW = self.wannier_gauge(data, ik, iknb)  # Wannier gauge
                # Matrix for finite-difference schemes
                AA_q_ik_ib = 1.j * AAW[:, :, None] * mmn.wk[ib] * mmn.bk_cart[ib, None, None, :]
                # Marzari & Vanderbilt formula for band-diagonal matrix elements
                if transl_inv:
                    AA_q_ik_ib[range(self.num_wann), range(self.num_wann)] = -np.log(
                        AAW.diagonal()).imag[:, None] * mmn.wk[ib] * mmn.bk_cart[ib, None, :]
                if phase is not None:
                    AA_q_ik_ib *= phase[:, :, ib, None]
                if sum_b:
                    AA_qb[ik] += AA_q_ik_ib * w
                else:
                    AA_qb[ik, :, :, ib, :] = AA_q_ik_ib * w
        return AA_qb


    # --- A_a(q,b) matrix --- #


    def get_AA_qb(self, mmn, kptirr, weights_k,
                  transl_inv=False, phase=None, sum_b=False):
        """	
         A wrapper for get_AABB_qb with eig=None
         see :meth:`~wannierberri.w90files.CheckPoint.get_AABB_qb` for more details  
         """
        return self.get_AABB_qb(mmn, kptirr=kptirr, weights_k=weights_k,
                                transl_inv=transl_inv, phase=phase, sum_b=sum_b)

    def get_AA_q(self, mmn, kptirr, weights_k, transl_inv=False):
        """
        A wrapper for get_AA_qb with sum_b=True
        see :meth:`~wannierberri.w90files.CheckPoint.get_AA_qb` for more details
        """
        return self.get_AA_qb(mmn=mmn, kptirr=kptirr, weights_k=weights_k,
                              transl_inv=transl_inv, sum_b=True).sum(axis=3)

    def get_wannier_centers(self, mmn, spreads=False):
        """
        calculate wannier centers  with the Marzarri-Vanderbilt translational invariant formula
        and optionally the spreads

        Parameters
        ----------
        mmn : :class:`~wannierberri.w90files.MMN`
            the overlap matrix elements between the Wavefunctions at neighbouring k-points
        spreads : bool
            if True, the spreads are calculated

        Returns
        -------
        np.ndarray(shape=(num_wann, 3), dtype=float)
            the wannier centers
        np.ndarray(shape=(num_wann,), dtype=float)
            the wannier spreads (in Angstrom^2) (if spreads=True)
        """
        wcc = np.zeros((self.num_wann, 3), dtype=float)
        if spreads:
            r2 = np.zeros(self.num_wann, dtype=float)
        for ik in range(mmn.NK):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                mmn_loc = self.wannier_gauge(mmn.data[ik, ib], ik, iknb)
                mmn_loc = mmn_loc.diagonal()
                log_loc = np.angle(mmn_loc)
                wcc += -log_loc[:, None] * mmn.wk[ib] * mmn.bk_cart[ib]
                if spreads:
                    r2 += (1 - np.abs(mmn_loc) ** 2 + log_loc ** 2) * mmn.wk[ib]
        wcc /= mmn.NK
        if spreads:
            return wcc, r2 / mmn.NK - np.sum(wcc**2, axis=1)
        else:
            return wcc


    # --- B_a(q,b) matrix --- #


    def get_BB_qb(self, mmn, eig, kptirr, weights_k, phase=None, sum_b=False):
        """	
        a wrapper for get_AABB_qb to evaluate BB matrix elements. (transl_inv is disabled)
        see :meth:`~wannierberri.w90files.CheckPoint.get_AABB_qb` for more details
        """
        return self.get_AABB_qb(mmn, kptirr=kptirr, weights_k=weights_k,
                                eig=eig, phase=phase, sum_b=sum_b)


    def get_CCOOGG_qb(self, mmn, uhu, kptirr, weights_k, antisym=True, phase=None, sum_b=False):
        """
        Returns the matrix elements CC, OO or GG in the Wannier gauge

        Parameters
        ----------
        mmn : :class:`~wannierberri.w90files.MMN`
            the overlap matrix elements between the Wavefunctions at neighbouring k-points
        uhu : :class:`~wannierberri.w90files.UHU` or :class:`~wannierberri.w90files.UIU`
            the matrix elements uhu or uiu produced by pw2wannier90
        antisym : bool
            if True, the antisymmetric piece of the matrix elements is calculated. Otherwise, the full matrix is calculated
        phase : np.ndarray(shape=(num_wann, num_wann, nnb), dtype=complex)
            the phase factors to be applied to the matrix elements (if None, no phase factors are applied)
        sum_b : bool
            if True, the matrix elements are summed over the neighbouring k-points. Otherwise, the matrix elements are stored in a 6D array of shape (num_kpts, num_wann, num_wann, nnb, nnb, 3)

        Returns
        -------
        np.ndarray(shape=(num_kpts, num_wann, num_wann, nnb, nnb, 3), dtype=complex) (if sum_b=False)
        or np.ndarray(shape=(num_kpts, num_wann, num_wann, nnb, nnb, 3), dtype=complex) (if sum_b=True)
            the q-resolved matrix elements CC, OO or GG in the Wannier gauge
        """
        nd_cart = 1 if antisym else 2
        shape_NNB = () if sum_b else (mmn.NNB, mmn.NNB)
        shape = (self.num_kpts, self.num_wann, self.num_wann) + shape_NNB + (3,) * nd_cart
        CC_qb = np.zeros(shape, dtype=complex)
        if phase is not None:
            phase = np.reshape(phase, np.shape(phase)[:4] + (1,) * nd_cart)
        for ik, weight in zip(kptirr, weights_k):
            for ib1 in range(mmn.NNB):
                iknb1 = mmn.neighbours[ik][ib1]
                for ib2 in range(mmn.NNB):
                    iknb2 = mmn.neighbours[ik][ib2]
                    # Matrix < u_k+b1 | H_k | u_k+b2 > (uHu)
                    data = uhu.data[ik][ib1, ib2]                 # Hamiltonian gauge
                    CCW = self.wannier_gauge(data, iknb1, iknb2)  # Wannier gauge

                    if antisym:
                        # Matrix for finite-difference schemes (takes antisymmetric piece only)
                        CC_q_ik_ib = 1.j * CCW[:, :, None] * (
                            mmn.wk[ib1] * mmn.wk[ib2] * (
                                mmn.bk_cart[ib1, alpha_A] * mmn.bk_cart[ib2, beta_A] -
                                mmn.bk_cart[ib1, beta_A] * mmn.bk_cart[ib2, alpha_A]))[None, None, :]
                    else:
                        # Matrix for finite-difference schemes (takes symmetric piece only)
                        CC_q_ik_ib = CCW[:, :, None, None] * (
                            mmn.wk[ib1] * mmn.wk[ib2] * (
                                mmn.bk_cart[ib1, :, None] *
                                mmn.bk_cart[ib2, None, :]))[None, None, :, :]
                    if phase is not None:
                        CC_q_ik_ib *= phase[:, :, ib1, ib2]
                    if sum_b:
                        CC_qb[ik] += CC_q_ik_ib * weight
                    else:
                        CC_qb[ik, :, :, ib1, ib2] = CC_q_ik_ib * weight
        return CC_qb

    # --- C_a(q,b1,b2) matrix --- #
    def get_CC_qb(self, mmn, uhu, kptirr, weights_k, phase=None, sum_b=False):
        """
        A wrapper for get_CCOOGG_qb with antisym=True
        see :meth:`~wannierberri.w90files.CheckPoint.get_CCOOGG_qb` for more details
        """
        return self.get_CCOOGG_qb(mmn, uhu, kptirr=kptirr, weights_k=weights_k, phase=phase, sum_b=sum_b)

    # --- O_a(q,b1,b2) matrix --- #
    def get_OO_qb(self, mmn, uiu, kptirr, weights_k, phase=None, sum_b=False):
        """
        A wrapper for get_CCOOGG_qb with antisym=False
        see :meth:`~wannierberri.w90files.CheckPoint.get_CCOOGG_qb` for more details
        (actually, the same as :meth:`~wannierberri.w90files.CheckPoint.get_CC_qb`)
        """
        return self.get_CCOOGG_qb(mmn, uiu, kptirr=kptirr, weights_k=weights_k, phase=phase, sum_b=sum_b)

    # Symmetric G_bc(q,b1,b2) matrix
    def get_GG_qb(self, mmn, uiu, kptirr, weights_k, phase=None, sum_b=False):
        """
        A wrapper for get_CCOOGG_qb with antisym=False 
        see :meth:`~wannierberri.w90files.CheckPoint.get_CCOOGG_qb` for more details
        """
        return self.get_CCOOGG_qb(mmn, uiu, kptirr=kptirr, weights_k=weights_k, antisym=False, phase=phase, sum_b=sum_b)
    ###########################################################################



    def get_SH_q(self, spn, eig, kptirr, weights_k):
        SH_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3), dtype=complex)
        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands), \
            f"spn file has NK={spn.NK}, NB={spn.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        assert (eig.NK, eig.NB) == (self.num_kpts, self.num_bands), \
            f"eig file has NK={eig.NK}, NB={eig.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        for ik, weight in zip(kptirr, weights_k):
            SH_q[ik, :, :, :] = self.wannier_gauge(spn.data[ik][:, :, :] * eig.data[ik][None, :, None], ik, ik) * weight
        return SH_q

    def get_SHA_q(self, shu, mmn, kptirr, weights_k,
                  phase=None, sum_b=False):
        """
        SHA or SA (if siu is used instead of shu)
        """
        if sum_b:
            SHA_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        else:
            SHA_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, mmn.NNB, 3, 3), dtype=complex)
        assert shu.NNB == mmn.NNB, f"shu.NNB={shu.NNB}, mmn.NNB={mmn.NNB} - mismatch"
        for ik, weight in zip(kptirr, weights_k):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik][ib]
                SHAW = self.wannier_gauge(shu.data[ik][ib], ik, iknb)
                SHA_q_ik_ib = 1.j * SHAW[:, :, None, :] * mmn.wk[ib] * mmn.bk_cart[ib, None, None, :, None]

                if phase is not None:
                    SHA_q_ik_ib *= phase[:, :, ib, None, None]
                if sum_b:
                    SHA_qb[ik] += SHA_q_ik_ib * weight
                else:
                    SHA_qb[ik, :, :, ib, :, :] = SHA_q_ik_ib * weight

        return SHA_qb



    def get_SHR_q(self, spn, mmn, kptirr, weights_k,
                  eig=None, phase=None):
        """
        SHR or SR(if eig is None)
        """
        SHR_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands), f"spn file has NK={spn.NK}, NB={spn.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        assert (mmn.NK, mmn.NB) == (self.num_kpts, self.num_bands), f"mmn file has NK={mmn.NK}, NB={mmn.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        for ik, weight in zip(kptirr, weights_k):
            SH = spn.data[ik][:, :, :]
            if eig is not None:
                SH = SH * eig.data[ik][None, :, None]
            SHW = self.wannier_gauge(SH, ik, ik)
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik][ib]
                SHM = np.tensordot(SH, mmn.data[ik][ib], axes=((1,), (0,))).swapaxes(-1, -2)
                SHRW = self.wannier_gauge(SHM, ik, iknb)
                if phase is not None:
                    SHRW = SHRW * phase[:, :, ib, None]
                SHRW = SHRW - SHW
                SHR_q[ik, :, :, :, :] += 1.j * SHRW[:, :, None] * mmn.wk[ib] * mmn.bk_cart[ib, None, None, :, None] * weight
        return SHR_q




    def from_win(self, win):
        print("creating empty CheckPoint from Win file")
        mp_grid = np.array(win.data["mp_grid"])
        kpt_latt = win.get_kpoints()
        real_lattice = win.get_unit_cell_cart_ang()
        try:
            num_wann = win["num_wann"]
        except KeyError:
            num_wann = None
        try:
            num_bands = win["num_bands"]
        except KeyError:
            num_bands = None
        self.__init__(real_lattice=real_lattice,
                      num_wann=num_wann,
                      num_bands=num_bands,
                      kpt_latt=kpt_latt,
                      mp_grid=mp_grid,)
        return self

    def select_bands(self, selected_bands):
        if selected_bands is not None:
            assert not self.wannierised, "v_matrix already set, cannot select bands"
            selected_bands_bool = np.zeros(self.num_bands, dtype=bool)
            selected_bands_bool[selected_bands] = True
            assert np.any(selected_bands_bool), "No bands selected"
            self.num_bands = sum(selected_bands_bool)
        return self
