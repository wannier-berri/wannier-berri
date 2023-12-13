#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#   some parts of this file are originate                    #
# from the translation of Wannier90 code                     #
#------------------------------------------------------------#

import numpy as np
from ..__utility import FortranFileR
import multiprocessing
from ..__utility import alpha_A, beta_A
from time import time
from itertools import islice
import gc

readstr = lambda F: "".join(c.decode('ascii') for c in F.read_record('c')).strip()


class CheckPoint():

    def __init__(self, seedname, kmesh_tol=1e-7,bk_complete_tol=1e-5):
        self.kmesh_tol = kmesh_tol # will be used in set_bk
        self.bk_complete_tol = bk_complete_tol # will be used in set_bk
        t0 = time()
        seedname = seedname.strip()
        FIN = FortranFileR(seedname + '.chk')
        readint = lambda: FIN.read_record('i4')
        readfloat = lambda: FIN.read_record('f8')

        def readcomplex():
            a = readfloat()
            return a[::2] + 1j * a[1::2]

        print('Reading restart information from file ' + seedname + '.chk :')
        self.comment = readstr(FIN)
        self.num_bands = readint()[0]
        num_exclude_bands = readint()[0]
        self.exclude_bands = readint()
        assert len(self.exclude_bands) == num_exclude_bands
        self.real_lattice = readfloat().reshape((3, 3), order='F')
        self.recip_lattice = readfloat().reshape((3, 3), order='F')
        assert np.linalg.norm(self.real_lattice.dot(self.recip_lattice.T) / (2 * np.pi) - np.eye(3)) < 1e-14
        self.num_kpts = readint()[0]
        self.mp_grid = readint()
        assert len(self.mp_grid) == 3
        assert self.num_kpts == np.prod(self.mp_grid)
        self.kpt_latt = readfloat().reshape((self.num_kpts, 3))
        self.nntot = readint()[0]
        self.num_wann = readint()[0]
        self.checkpoint = readstr(FIN)
        self.have_disentangled = bool(readint()[0])
        if self.have_disentangled:
            self.omega_invariant = readfloat()[0]
            lwindow = np.array(readint().reshape((self.num_kpts, self.num_bands)), dtype=bool)
            ndimwin = readint()
            u_matrix_opt = readcomplex().reshape((self.num_kpts, self.num_wann, self.num_bands))
            self.win_min = np.array([np.where(lwin)[0].min() for lwin in lwindow])
            self.win_max = np.array([wm + nd for wm, nd in zip(self.win_min, ndimwin)])
        else:
            self.win_min = np.array([0] * self.num_kpts)
            self.win_max = np.array([self.num_wann] * self.num_kpts)

        u_matrix = readcomplex().reshape((self.num_kpts, self.num_wann, self.num_wann))
        m_matrix = readcomplex().reshape((self.num_kpts, self.nntot, self.num_wann, self.num_wann))
        if self.have_disentangled:
            self.v_matrix = [u.dot(u_opt[:, :nd]) for u, u_opt, nd in zip(u_matrix, u_matrix_opt, ndimwin)]
        else:
            self.v_matrix = [u for u in u_matrix]
        self.wannier_centers = readfloat().reshape((self.num_wann, 3))
        self.wannier_spreads = readfloat().reshape((self.num_wann))
        del u_matrix, m_matrix
        gc.collect()
        print("Time to read .chk : {}".format(time() - t0))

    def wannier_gauge(self, mat, ik1, ik2):
        # data should be of form NBxNBx ...   - any form later
        if len(mat.shape) == 1:
            mat = np.diag(mat)
        assert mat.shape[:2] == (self.num_bands, ) * 2, f"mat.shape={mat.shape}, num_bands={self.num_bands}"
        mat = mat[self.win_min[ik1]:self.win_max[ik1], self.win_min[ik2]:self.win_max[ik2]]
        v1 = self.v_matrix[ik1].conj()
        v2 = self.v_matrix[ik2].T
        return np.tensordot(
            np.tensordot(v1, mat, axes=(1, 0)), v2, axes=(1, 0)).transpose((
                0,
                -1,
            ) + tuple(range(1, mat.ndim - 1)))

    def get_HH_q(self, eig):
        assert (eig.NK, eig.NB) == (self.num_kpts, self.num_bands)
        HH_q = np.array([self.wannier_gauge(E, ik, ik) for ik, E in enumerate(eig.data)])
        return 0.5 * (HH_q + HH_q.transpose(0, 2, 1).conj())

    def get_SS_q(self, spn):
        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands)
        SS_q = np.array([self.wannier_gauge(S, ik, ik) for ik, S in enumerate(spn.data)])
        return 0.5 * (SS_q + SS_q.transpose(0, 2, 1, 3).conj())

    #########
    # Oscar #
    ###########################################################################

    # Depart from the original matrix elements in the ab initio mesh
    # (Hamiltonian gauge) to obtain the corresponding matrix elements in the
    # Wannier gauge. The last constitute the basis to construct the real-space
    # matrix elements for Wannier interpolation, independently of the
    # finite-difference scheme used.

    # --- A_a(q,b) matrix --- #
    def get_AA_qb(self, mmn, transl_inv=False):

        AA_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, mmn.NNB, 3), dtype=complex)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                ib_unique = mmn.ib_unique_map[ik, ib]
                # Matrix < u_k | u_k+b > (mmn)
                data = mmn.data[ik, ib]                   # Hamiltonian gauge
                AAW = self.wannier_gauge(data, ik, iknb)  # Wannier gauge

                # Matrix for finite-difference schemes
                AA_q_ik_ib = 1.j * AAW[:, :, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :]

                # Marzari & Vanderbilt formula for band-diagonal matrix elements
                if transl_inv:
                    AA_q_ik_ib[range(self.num_wann), range(self.num_wann)] = -np.log(
                        AAW.diagonal()).imag[:, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, :]

                AA_qb[ik, :, :, ib_unique, :] = AA_q_ik_ib

        return AA_qb

    # --- B_a(q,b) matrix --- #
    def get_BB_qb(self, mmn, eig):

        BB_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, mmn.NNB, 3), dtype=complex)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                ib_unique = mmn.ib_unique_map[ik, ib]

                # Matrix < u_k | H_k | u_k+b > (eig * mmn)
                data = mmn.data[ik, ib]                   # Hamiltonian gauge (only mmn)
                data = data * eig.data[ik, :, None]       # Hamiltonian gauge (add energies)
                BBW = self.wannier_gauge(data, ik, iknb)  # Wannier gauge

                # Matrix for finite-difference schemes
                BB_q_ik_ib = 1j * BBW[:, :, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :]
                BB_qb[ik, :, :, ib_unique, :] = BB_q_ik_ib

        return BB_qb

    # --- C_a(q,b1,b2) matrix --- #
    def get_CC_qb(self, mmn, uhu):

        CC_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, mmn.NNB, mmn.NNB, 3), dtype=complex)
        for ik in range(self.num_kpts):
            for ib1 in range(mmn.NNB):
                iknb1 = mmn.neighbours[ik, ib1]
                ib1_unique = mmn.ib_unique_map[ik, ib1]
                for ib2 in range(mmn.NNB):
                    iknb2 = mmn.neighbours[ik, ib2]
                    ib2_unique = mmn.ib_unique_map[ik, ib2]

                    # Matrix < u_k+b1 | H_k | u_k+b2 > (uHu)
                    data = uhu.data[ik, ib1, ib2]                 # Hamiltonian gauge
                    CCW = self.wannier_gauge(data, iknb1, iknb2)  # Wannier gauge

                    # Matrix for finite-difference schemes (takes antisymmetric piece only)
                    CC_q_ik_ib = 1.j * CCW[:, :, None] * (
                        mmn.wk[ik, ib1] * mmn.wk[ik, ib2] * (
                            mmn.bk_cart[ik, ib1, alpha_A] * mmn.bk_cart[ik, ib2, beta_A]
                            - mmn.bk_cart[ik, ib1, beta_A] * mmn.bk_cart[ik, ib2, alpha_A]))[None, None, :]

                    CC_qb[ik, :, :, ib1_unique, ib2_unique, :] = CC_q_ik_ib

        return CC_qb

    # --- O_a(q,b1,b2) matrix --- #
    def get_OO_qb(self, mmn, uiu):

        OO_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, mmn.NNB, mmn.NNB, 3), dtype=complex)
        for ik in range(self.num_kpts):
            for ib1 in range(mmn.NNB):
                iknb1 = mmn.neighbours[ik, ib1]
                ib1_unique = mmn.ib_unique_map[ik, ib1]
                for ib2 in range(mmn.NNB):
                    iknb2 = mmn.neighbours[ik, ib2]
                    ib2_unique = mmn.ib_unique_map[ik, ib2]

                    # Matrix < u_k+b1 | I | u_k+b2 > (uIu)
                    data = uiu.data[ik, ib1, ib2]                 # Hamiltonian gauge
                    OOW = self.wannier_gauge(data, iknb1, iknb2)  # Wannier gauge

                    # Matrix for finite-difference schemes (takes antisymmetric piece only)
                    OO_q_ik_ib = 1.j * OOW[:, :, None] * (
                        mmn.wk[ik, ib1] * mmn.wk[ik, ib2] * (
                            mmn.bk_cart[ik, ib1, alpha_A] * mmn.bk_cart[ik, ib2, beta_A]
                            - mmn.bk_cart[ik, ib1, beta_A] * mmn.bk_cart[ik, ib2, alpha_A]))[None, None, :]

                    OO_qb[ik, :, :, ib1_unique, ib2_unique, :] = OO_q_ik_ib

        return OO_qb

    # Symmetric G_bc(q,b1,b2) matrix
    def get_GG_qb(self, mmn, uiu):

        GG_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, mmn.NNB, mmn.NNB, 3, 3), dtype=complex)
        for ik in range(self.num_kpts):
            for ib1 in range(mmn.NNB):
                iknb1 = mmn.neighbours[ik, ib1]
                ib1_unique = mmn.ib_unique_map[ik, ib1]
                for ib2 in range(mmn.NNB):
                    iknb2 = mmn.neighbours[ik, ib2]
                    ib2_unique = mmn.ib_unique_map[ik, ib2]

                    # Matrix < u_k+b1 | I | u_k+b2 > (uIu)
                    data = uiu.data[ik, ib1, ib2]                 # Hamiltonian gauge
                    GGW = self.wannier_gauge(data, iknb1, iknb2)  # Wannier gauge

                    # Matrix for finite-difference schemes (takes symmetric piece only)
                    GG_q_ik_ib = GGW[:, :, None, None] * (
                        mmn.wk[ik, ib1] * mmn.wk[ik, ib2] * (
                            mmn.bk_cart[ik, ib1, :, None] * mmn.bk_cart[ik, ib2, None, :]))[None, None, :, :]

                    GG_qb[ik, :, :, ib1_unique, ib2_unique, :, :] = GG_q_ik_ib

        # G_bc is symmetric in the cartesian indices
        GG_qb = 0.5 * (GG_qb + GG_qb.swapaxes(5, 6))

        return GG_qb

    ###########################################################################

    def get_SA_q(self, siu, mmn):
        mmn.set_bk_chk(self)
        SA_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        assert siu.NNB == mmn.NNB
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                SAW = self.wannier_gauge(siu.data[ik, ib], ik, iknb)
                SA_q_ik = 1.j * SAW[:, :, None, :] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :, None]
                SA_q[ik] += SA_q_ik
        return SA_q

    def get_SHA_q(self, shu, mmn):
        mmn.set_bk_chk(self)
        SHA_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        assert shu.NNB == mmn.NNB
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                SHAW = self.wannier_gauge(shu.data[ik, ib], ik, iknb)
                SHA_q_ik = 1.j * SHAW[:, :, None, :] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :, None]
                SHA_q[ik] += SHA_q_ik
        return SHA_q

    def get_SR_q(self, spn, mmn):
        mmn.set_bk_chk(self)
        SR_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                for i in range(3):
                    SM_i = spn.data[ik, :, :, i].dot(mmn.data[ik, ib, :, :])
                    SRW = self.wannier_gauge(SM_i, ik, iknb) - self.wannier_gauge(spn.data[ik, :, :, i], ik, ik)
                    SR_q[ik, :, :, :, i] += 1.j * SRW[:, :, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :]
        return SR_q

    def get_SH_q(self, spn, eig):
        SH_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3), dtype=complex)
        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands)
        for ik in range(self.num_kpts):
            for i in range(3):
                SH_q[ik, :, :, i] = self.wannier_gauge(spn.data[ik, :, :, i] * eig.data[ik, None, :], ik, ik)
        return SH_q

    def get_SHR_q(self, spn, mmn, eig):
        mmn.set_bk_chk(self)
        SHR_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                for i in range(3):
                    SH_i = spn.data[ik, :, :, i] * eig.data[ik, None, :]
                    SHM_i = SH_i.dot(mmn.data[ik, ib])
                    SHRW = self.wannier_gauge(SHM_i, ik, iknb) - self.wannier_gauge(SH_i, ik, ik)
                    SHR_q[ik, :, :, :,
                          i] += 1.j * SHRW[:, :, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :]
        return SHR_q


class W90_data():

    @property
    def n_neighb(self):
        return 0

    @property
    def NK(self):
        return self.data.shape[0]

    @property
    def NB(self):
        return self.data.shape[1 + self.n_neighb]

    @property
    def NNB(self):
        if self.n_neighb > 0:
            return self.data.shape[1]
        else:
            return 0


def convert(A):
    return np.array([l.split() for l in A], dtype=float)


class MMN(W90_data):
    """
    MMN.data[ik, ib, m, n] = <u_{m,k}|u_{n,k+b}>
    """

    @property
    def n_neighb(self):
        return 1

    def __init__(self, seedname, npar=multiprocessing.cpu_count()):
        t0 = time()
        f_mmn_in = open(seedname + ".mmn", "r")
        f_mmn_in.readline()
        NB, NK, NNB = np.array(f_mmn_in.readline().split(), dtype=int)
        self.data = np.zeros((NK, NNB, NB, NB), dtype=complex)
        block = 1 + self.NB * self.NB
        data = []
        headstring = []
        mult = 4
        # FIXME: npar = 0 does not work
        if npar > 0:
            pool = multiprocessing.Pool(npar)
        for j in range(0, NNB * NK, npar * mult):
            x = list(islice(f_mmn_in, int(block * npar * mult)))
            if len(x) == 0: break
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
        self.data = np.array(data).reshape(self.NK, self.NNB, self.NB, self.NB).transpose((0, 1, 3, 2))
        headstring = np.array([s.split() for s in headstring], dtype=int).reshape(self.NK, self.NNB, 5)
        assert np.all(headstring[:, :, 0] - 1 == np.arange(self.NK)[:, None])
        self.neighbours = headstring[:, :, 1] - 1
        self.G = headstring[:, :, 2:]
        t2 = time()
        print("Time for MMN.__init__() : {} , read : {} , headstring {}".format(t2 - t0, t1 - t0, t2 - t1))

    def set_bk(self, kpt_latt,mp_grid,recip_lattice,kmesh_tol=1e-7, bk_complete_tol=1e-5):
        try:
            self.bk_cart
            self.wk
            self.bk_latt_unique
            self.bk_cart_unique
            self.ib_unique_map
            return
        except AttributeError:
            bk_latt = np.array(
                np.round(
                    [
                        (kpt_latt[nbrs] - kpt_latt + G) * mp_grid[None, :]
                        for nbrs, G in zip(self.neighbours.T, self.G.transpose(1, 0, 2))
                    ]).transpose(1, 0, 2),
                dtype=int)
            bk_latt_unique = np.array([b for b in set(tuple(bk) for bk in bk_latt.reshape(-1, 3))], dtype=int)
            assert len(bk_latt_unique) == self.NNB
            bk_cart_unique = bk_latt_unique.dot(recip_lattice / mp_grid[:, None])
            bk_cart_unique_length = np.linalg.norm(bk_cart_unique, axis=1)
            srt = np.argsort(bk_cart_unique_length)
            bk_latt_unique = bk_latt_unique[srt]
            bk_cart_unique = bk_cart_unique[srt]
            bk_cart_unique_length = bk_cart_unique_length[srt]
            brd = [
                0,
            ] + list(np.where(bk_cart_unique_length[1:] - bk_cart_unique_length[:-1] > kmesh_tol)[0] + 1) + [
                self.NNB,
            ]
            shell_mat = np.array([bk_cart_unique[b1:b2].T.dot(bk_cart_unique[b1:b2]) for b1, b2 in zip(brd, brd[1:])])
            shell_mat_line = shell_mat.reshape(-1, 9)
            u, s, v = np.linalg.svd(shell_mat_line, full_matrices=False)
            s = 1. / s
            weight_shell = np.eye(3).reshape(1, -1).dot(v.T.dot(np.diag(s)).dot(u.T)).reshape(-1)
            check_eye = sum(w * m for w, m in zip(weight_shell, shell_mat))
            tol = np.linalg.norm(check_eye - np.eye(3))
            if tol > bk_complete_tol:
                raise RuntimeError(
                    "Error while determining shell weights. the following matrix :\n {} \n failed to be identity by an error of {} Further debug informstion :  \n bk_latt_unique={} \n bk_cart_unique={} \n bk_cart_unique_length={}\nshell_mat={}\nweight_shell={}\n"
                    .format(
                        check_eye, tol, bk_latt_unique, bk_cart_unique, bk_cart_unique_length, shell_mat, weight_shell))
            weight = np.array([w for w, b1, b2 in zip(weight_shell, brd, brd[1:]) for i in range(b1, b2)])
            weight_dict = {tuple(bk): w for bk, w in zip(bk_latt_unique, weight)}
            bk_cart_dict = {tuple(bk): bkcart for bk, bkcart in zip(bk_latt_unique, bk_cart_unique)}
            self.bk_cart = np.array([[bk_cart_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])
            self.wk = np.array([[weight_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])

            #############
            ### Oscar ###
            ###################################################################

            # Wannier90 provides a list of nearest-neighbor vectors b for every
            # q point. For Jae-Mo's finite-difference scheme it is convenient
            # to evaluate the Fourier transform of the matrix elements in the
            # original ab-initio mesh before performing the sum over
            # nearest-neighbor vectors. This requires defining a mapping from
            # any pair {q,b} to a unique list of b vectors that is independent
            # of q.

            bk_latt = np.rint((self.bk_cart @ np.linalg.inv(recip_lattice)) * mp_grid[None, None, :]).astype(int)
            bk_latt_unique = np.unique(bk_latt.reshape(-1, 3), axis=0)
            bk_cart_unique = (bk_latt_unique / mp_grid[None, :]) @ recip_lattice
            assert bk_latt_unique.shape == (self.NNB, 3)

            ib_unique_map = np.zeros((self.NK,self.NNB), dtype=int)
            for ik in range(self.NK):
                for ib in range(self.NNB):
                    b_latt = np.rint((self.bk_cart[ik, ib, :] @ np.linalg.inv(recip_lattice)) * mp_grid).astype(int)
                    ib_unique = [tuple(b) for b in bk_latt_unique].index(tuple(b_latt))
                    assert np.allclose(bk_cart_unique[ib_unique, :], self.bk_cart[ik, ib, :])
                    ib_unique_map[ik,ib] = ib_unique

            self.bk_latt_unique = bk_latt_unique
            self.bk_cart_unique = bk_cart_unique
            self.ib_unique_map = ib_unique_map
            ###################################################################

    def set_bk_chk(self,chk):
        self.set_bk(chk.kpt_latt,chk.mp_grid,chk.recip_lattice,kmesh_tol=chk.kmesh_tol, bk_complete_tol=chk.bk_complete_tol)



class EIG(W90_data):

    def __init__(self, seedname):
        data = np.loadtxt(seedname + ".eig")
        NB = int(round(data[:, 0].max()))
        NK = int(round(data[:, 1].max()))
        data = data.reshape(NK, NB, 3)
        assert np.linalg.norm(data[:, :, 0] - 1 - np.arange(NB)[None, :]) < 1e-15
        assert np.linalg.norm(data[:, :, 1] - 1 - np.arange(NK)[:, None]) < 1e-15
        self.data = data[:, :, 2]


class SPN(W90_data):
    """
    SPN.data[ik, m, n, ipol] = <u_{m,k}|S_ipol|u_{n,k}>
    """

    def __init__(self, seedname='wannier90', formatted=False):
        print("----------\n SPN  \n---------\n")
        if formatted:
            f_spn_in = open(seedname + ".spn", 'r')
            SPNheader = f_spn_in.readline().strip()
            nbnd, NK = (int(x) for x in f_spn_in.readline().split())
        else:
            f_spn_in = FortranFileR(seedname + ".spn")
            SPNheader = (f_spn_in.read_record(dtype='c'))
            nbnd, NK = f_spn_in.read_record(dtype=np.int32)
            SPNheader = "".join(a.decode('ascii') for a in SPNheader)

        print("reading {}.spn : {}".format(seedname, SPNheader))

        indm, indn = np.tril_indices(nbnd)
        self.data = np.zeros((NK, nbnd, nbnd, 3), dtype=complex)

        for ik in range(NK):
            A = np.zeros((3, nbnd, nbnd), dtype=complex)
            if formatted:
                tmp = np.array([f_spn_in.readline().split() for i in range(3 * nbnd * (nbnd + 1) // 2)], dtype=float)
                tmp = tmp[:, 0] + 1.j * tmp[:, 1]
            else:
                tmp = f_spn_in.read_record(dtype=np.complex128)
            A[:, indn, indm] = tmp.reshape(3, nbnd * (nbnd + 1) // 2, order='F')
            check = np.einsum('ijj->', np.abs(A.imag))
            A[:, indm, indn] = A[:, indn, indm].conj()
            if check > 1e-10:
                raise RuntimeError("REAL DIAG CHECK FAILED : {0}".format(check))
            self.data[ik] = A.transpose(1, 2, 0)
        print("----------\n SPN OK  \n---------\n")


class UXU(W90_data):
    """
    Read and setup uHu or uIu object.
    pw2wannier90 writes data_pw2w90[n, m, ib1, ib2, ik] = <u_{m,k+b1}|X|u_{n,k+b2}>
    in column-major order. (X = H for UHU, X = I for UIU.)
    Here, we read to have data[ik, ib1, ib2, m, n] = <u_{m,k+b1}|X|u_{n,k+b2}>.
    """

    @property
    def n_neighb(self):
        return 2

    def __init__(self, seedname='wannier90', formatted=False, suffix='uHu'):
        print("----------\n  {0}   \n---------".format(suffix))
        print('formatted == {}'.format(formatted))
        if formatted:
            f_uXu_in = open(seedname + "." + suffix, 'r')
            header = f_uXu_in.readline().strip()
            NB, NK, NNB = (int(x) for x in f_uXu_in.readline().split())
        else:
            f_uXu_in = FortranFileR(seedname + "." + suffix)
            header = readstr(f_uXu_in)
            NB, NK, NNB = f_uXu_in.read_record('i4')

        print("reading {}.{} : <{}>".format(seedname, suffix, header))

        self.data = np.zeros((NK, NNB, NNB, NB, NB), dtype=complex)
        if formatted:
            tmp = np.array([f_uXu_in.readline().split() for i in range(NK * NNB * NNB * NB * NB)], dtype=float)
            tmp_cplx = tmp[:, 0] + 1.j * tmp[:, 1]
            self.data = tmp_cplx.reshape(NK, NNB, NNB, NB, NB).transpose(0, 2, 1, 3, 4)
        else:
            for ik in range(NK):
                for ib2 in range(NNB):
                    for ib1 in range(NNB):
                        tmp = f_uXu_in.read_record('f8').reshape((2, NB, NB), order='F').transpose(2, 1, 0)
                        self.data[ik, ib1, ib2] = tmp[:, :, 0] + 1j * tmp[:, :, 1]
        print("----------\n {0} OK  \n---------\n".format(suffix))
        f_uXu_in.close()


class UHU(UXU):
    """
    UHU.data[ik, ib1, ib2, m, n] = <u_{m,k+b1}|H(k)|u_{n,k+b2}>
    """

    def __init__(self, seedname='wannier90', formatted=False):
        super().__init__(seedname=seedname, formatted=formatted, suffix='uHu')


class UIU(UXU):
    """
    UIU.data[ik, ib1, ib2, m, n] = <u_{m,k+b1}|u_{n,k+b2}>
    """

    def __init__(self, seedname='wannier90', formatted=False):
        super().__init__(seedname=seedname, formatted=formatted, suffix='uIu')


class SXU(W90_data):
    """
    Read and setup sHu or sIu object.
    pw2wannier90 writes data_pw2w90[n, m, ipol, ib, ik] = <u_{m,k}|S_ipol * X|u_{n,k+b}>
    in column-major order. (X = H for SHU, X = I for SIU.)
    Here, we read to have data[ik, ib, m, n, ipol] = <u_{m,k}|S_ipol * X|u_{n,k+b}>.
    """

    @property
    def n_neighb(self):
        return 1

    def __init__(self, seedname='wannier90', formatted=False, suffix='sHu'):
        print("----------\n  {0}   \n---------".format(suffix))

        if formatted:
            f_sXu_in = open(seedname + "." + suffix, 'r')
            header = f_sXu_in.readline().strip()
            NB, NK, NNB = (int(x) for x in f_sXu_in.readline().split())
        else:
            f_sXu_in = FortranFileR(seedname + "." + suffix)
            header = readstr(f_sXu_in)
            NB, NK, NNB = f_sXu_in.read_record('i4')

        print("reading {}.{} : <{}>".format(seedname, suffix, header))

        self.data = np.zeros((NK, NNB, NB, NB, 3), dtype=complex)

        if formatted:
            tmp = np.array([f_sXu_in.readline().split() for i in range(NK * NNB * 3 * NB * NB)], dtype=float)
            tmp_cplx = tmp[:, 0] + 1j * tmp[:, 1]
            self.data = tmp_cplx.reshape(NK, NNB, 3, NB, NB).transpose(0, 1, 3, 4, 2)
        else:
            for ik in range(NK):
                for ib in range(NNB):
                    for ipol in range(3):
                        tmp = f_sXu_in.read_record('f8').reshape((2, NB, NB), order='F').transpose(2, 1, 0)
                        # tmp[m, n] = <u_{m,k}|S_ipol*X|u_{n,k+b}>
                        self.data[ik, ib, :, :, ipol] = tmp[:, :, 0] + 1j * tmp[:, :, 1]

        print("----------\n {0} OK  \n---------\n".format(suffix))
        f_sXu_in.close()


class SIU(SXU):
    """
    SIU.data[ik, ib, m, n, ipol] = <u_{m,k}|S_ipol|u_{n,k+b}>
    """

    def __init__(self, seedname='wannier90', formatted=False):
        super().__init__(seedname=seedname, formatted=formatted, suffix='sIu')


class SHU(SXU):
    """
    SHU.data[ik, ib, m, n, ipol] = <u_{m,k}|S_ipol*H(k)|u_{n,k+b}>
    """

    def __init__(self, seedname='wannier90', formatted=False):
        super().__init__(seedname=seedname, formatted=formatted, suffix='sHu')
