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
# ------------------------------------------------------------#

import multiprocessing
import gc
import functools
from scipy.constants import physical_constants
from time import time
from itertools import islice
from copy import copy
import numpy as np
import lazy_property
from .disentanglement import disentangle
from ..__utility import FortranFileR, alpha_A, beta_A

readstr = lambda F: "".join(c.decode('ascii') for c in F.read_record('c')).strip()


class CheckPoint:
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

    def __init__(self, seedname, kmesh_tol=1e-7, bk_complete_tol=1e-5):

        self.kmesh_tol = kmesh_tol  # will be used in set_bk
        self.bk_complete_tol = bk_complete_tol  # will be used in set_bk
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
        assert mat.shape[:2] == (self.num_bands,) * 2, f"mat.shape={mat.shape}, num_bands={self.num_bands}"
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

    def get_AA_q(self, mmn, eig=None, transl_inv=False):  # if eig is present - it is BB_q
        if transl_inv and (eig is not None):
            raise RuntimeError("transl_inv cannot be used to obtain BB")
        mmn.set_bk_chk(self)
        AA_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3), dtype=complex)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                data = mmn.data[ik, ib]
                if eig is not None:
                    data = data * eig.data[ik, :, None]
                AAW = self.wannier_gauge(data, ik, iknb)
                AA_q_ik = 1.j * AAW[:, :, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :]
                if transl_inv:
                    AA_q_ik[range(self.num_wann), range(self.num_wann)] = -np.log(
                        AAW.diagonal()).imag[:, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, :]
                AA_q[ik] += AA_q_ik
        if eig is None:
            AA_q = 0.5 * (AA_q + AA_q.transpose((0, 2, 1, 3)).conj())
        return AA_q

    def get_CC_q(self, uhu, mmn):  # if eig is present - it is BB_q
        mmn.set_bk_chk(self)
        assert uhu.NNB == mmn.NNB
        CC_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3), dtype=complex)
        for ik in range(self.num_kpts):
            for ib1 in range(mmn.NNB):
                iknb1 = mmn.neighbours[ik, ib1]
                for ib2 in range(mmn.NNB):
                    iknb2 = mmn.neighbours[ik, ib2]
                    data = uhu.data[ik, ib1, ib2]
                    CC_q[ik] += (1.j * self.wannier_gauge(data, iknb1, iknb2)[:, :, None] *
                                 (mmn.wk[ik, ib1] * mmn.wk[ik, ib2] *
                                  (mmn.bk_cart[ik, ib1, alpha_A] * mmn.bk_cart[ik, ib2, beta_A] -
                                     mmn.bk_cart[ik, ib1, beta_A] * mmn.bk_cart[ik, ib2, alpha_A])
                                  )[None, None, :]
                                 )
        CC_q = 0.5 * (CC_q + CC_q.transpose((0, 2, 1, 3)).conj())
        return CC_q

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





class CheckPoint_bare(CheckPoint):
    """
            Class to store data from Wanierisationm obtained internally
            Initialize without the v_matrix (to be written later by `~wannierberri.system.disentangle`)

            Parameters
            ----------
            win : `~wannierberri.system.w90_files.WIN`
            eig : `~wannierberri.system.w90_files.EIG`
            amn : `~wannierberri.system.w90_files.AMN`
            mmn : `~wannierberri.system.w90_files.MMN`
            """

    def __init__(self, win, eig, amn, mmn):
        self.mp_grid = np.array(win.get_param("mp_grid"))
        self.kpt_latt = win.get_kpoints()
        self.real_lattice = win.get_unit_cell_cart_ang()
        self.num_kpts = eig.NK
        self.num_wann = amn.NW
        self.num_bands = mmn.NB
        self.win_min = np.array([0] * self.num_kpts)
        self.win_max = np.array([self.num_bands] * self.num_kpts)
        self.recip_lattice = 2 * np.pi * np.linalg.inv(self.real_lattice).T


class Wannier90data:
    """A class to describe all input files of wannier90, and to construct the Wannier functions
     via disentanglement procedure"""

    # todo :  rotatre uHu and spn
    # todo : create a model from this
    # todo : symmetry

    def __init__(self, seedname="wannier90", read_chk=False,
                 kmesh_tol=1e-7, bk_complete_tol=1e-5, ):  # ,sitesym=False):
        self.seedname = copy(seedname)
        self.__files_classes = {'win': WIN,
                                'eig': EIG,
                                'mmn': MMN,
                                'amn': AMN,
                                'uiu': UIU,
                                'uhu': UHU,
                                'siu': SIU,
                                'shu': SHU,
                                'spn': SPN
                                }
        self.__files = {}
        if read_chk:
            self.chk = CheckPoint(seedname, kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol)
            self.wannierised = True
        else:
            self.chk = CheckPoint_bare(win=self.win, eig=self.eig, mmn=self.mmn, amn=self.amn)
            self.kpt_mp_grid = [tuple(k) for k in
                                np.array(np.round(self.chk.kpt_latt * np.array(self.chk.mp_grid)[None, :]),
                                         dtype=int) % self.chk.mp_grid]
            self.mmn.set_bk(mp_grid=self.chk.mp_grid, kpt_latt=self.chk.kpt_latt, recip_lattice=self.chk.recip_lattice)
            self.win_index = [np.arange(self.eig.NB)] * self.chk.num_kpts
            self.wannierised = False
        self.set_file(key='chk', val=self.chk)

        # if sitesym:
        #    self.Dmn=DMN(self.seedname,num_wann=self.chk.num_wann)
        # else:
        #    self.Dmn=DMN(None,num_wann=self.chk.num_wann,num_bands=self.chk.num_bands,nkpt=self.chk.num_kpts)

    def set_file(self, key, val=None, overwrite=False):
        if not overwrite:
            assert key not in self.__files, f"file `{key}` was already set"
        if val is None:
            val = self.__files_classes[key](self.seedname)
        self.check_conform(key, val)
        self.__files[key] = val

    def get_file(self, key):
        if key not in self.__files:
            self.set_file(key)
        return self.__files[key]

    def check_conform(self, key, this):
        for key2, other in self.__files.items():
            for attr in ['NK', 'NB', 'NW', 'NNB']:
                if hasattr(this, attr) and hasattr(other, attr):
                    a = getattr(this, attr)
                    b = getattr(other, attr)
                    if None not in (a, b):
                        assert a == b, f"files {key} and {key2} have different attribute {attr} : {a} and {b} respectively"

    @property
    def win(self):
        return self.get_file('win')

    @property
    def amn(self):
        return self.get_file('amn')

    @property
    def eig(self):
        return self.get_file('eig')

    @property
    def mmn(self):
        return self.get_file('mmn')

    @property
    def uhu(self):
        return self.get_file('uhu')

    @property
    def spn(self):
        return self.get_file('spn')

    @property
    def siu(self):
        return self.get_file('siu')

    @property
    def shu(self):
        return self.get_file('shu')

    @property
    def iter_kpts(self):
        return range(self.chk.num_kpts)

    @lazy_property.LazyProperty
    def wannier_centers(self):
        try:
            return self.chk.wannier_centers
        except AttributeError:
            return self.chk.get_AA_q(self.mmn, transl_inv=True).diagonal(axis1=1, axis2=2).sum(
                axis=0).real.T / self.chk.num_kpts

    def check_wannierised(self, msg=""):
        if not self.wannierised:
            raise RuntimeError(f"no wannieruisation was performed on the w90 input files, cannot proceed with {msg}")

    def disentangle(self, **parameters):
        disentangle(self, **parameters)

    # TODO : allow k-dependent window (can it be useful?)
    # def apply_outer_window(self,
    #                  win_min=-np.Inf,
    #                  win_max=np.Inf ):
    #     raise NotImplementedError("outer window does not work so far")
    #     "Excludes the bands from outside the outer window"
    #
    #     def win_index_nondegen(ik,thresh=DEGEN_THRESH):
    #         "define the indices of the selected bands, making sure that degenerate bands were not split"
    #         E=self.Eig[ik]
    #         ind=np.where( ( E<=win_max)*(E>=win_min) )[0]
    #         while ind[0]>0 and E[ind[0]]-E[ind[0]-1]<thresh:
    #             ind=[ind[0]-1]+ind
    #         while ind[0]<len(E) and E[ind[-1]+1]-E[ind[-1]]<thresh:
    #             ind=ind+[ind[-1]+1]
    #         return ind
    #
    #     # win_index_irr=[win_index_nondegen(ik) for ik in self.Dmn.kptirr]
    #     # self.excluded_bands=[list(set(ind)
    #     # self.Dmn.select_bands(win_index_irr)
    #     # win_index=[win_index_irr[ik] for ik in self.Dmn.kpt2kptirr]
    #     win_index=[win_index_nondegen(ik) for ik in self.iter_kpts]
    #     self._Eig=[E[ind] for E, ind in zip(self._Eig,win_index)]
    #     self._Mmn=[[self._Mmn[ik][ib][win_index[ik],:][:,win_index[ikb]] for ib,ikb in enumerate(self.mmn.neighbours[ik])] for ik in self.iter_kpts]
    #     self._Amn=[self._Amn[ik][win_index[ik],:] for ik in self.iter_kpts]

    # TODO : allow k-dependent window (can it be useful?)


class W90_file:

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
            return None


def convert(A):
    return np.array([l.split() for l in A], dtype=float)


class MMN(W90_file):
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
        self.data = np.array(data).reshape(self.NK, self.NNB, self.NB, self.NB).transpose((0, 1, 3, 2))
        headstring = np.array([s.split() for s in headstring], dtype=int).reshape(self.NK, self.NNB, 5)
        assert np.all(headstring[:, :, 0] - 1 == np.arange(self.NK)[:, None])
        self.neighbours = headstring[:, :, 1] - 1
        self.G = headstring[:, :, 2:]
        t2 = time()
        print("Time for MMN.__init__() : {} , read : {} , headstring {}".format(t2 - t0, t1 - t0, t2 - t1))

    def set_bk(self, kpt_latt, mp_grid, recip_lattice, kmesh_tol=1e-7, bk_complete_tol=1e-5):
        try:
            self.bk_cart
            self.wk
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

    def set_bk_chk(self, chk, **argv):
        self.set_bk(chk.kpt_latt, chk.mp_grid, chk.recip_lattice, **argv)


def str2arraymmn(A):
    a = np.array([l.split()[3:] for l in A], dtype=float)
    return (a[:, 0] + 1j * a[:, 1])


class AMN(W90_file):

    @property
    def NB(self):
        return self.data.shape[1]

    @property
    def NW(self):
        return self.data.shape[2]

    def __init__(self, seedname, npar=multiprocessing.cpu_count()):
        f_mmn_in = open(seedname + ".amn", "r").readlines()
        print("reading {}.amn: ".format(seedname) + f_mmn_in[0].strip())
        s = f_mmn_in[1]
        NB, NK, NW = np.array(s.split(), dtype=int)
        self.data = np.zeros((NK, NB, NW), dtype=complex)
        block = self.NW * self.NB
        allmmn = (f_mmn_in[2 + j * block:2 + (j + 1) * block] for j in range(self.NK))
        p = multiprocessing.Pool(npar)
        self.data = np.array(p.map(str2arraymmn, allmmn)).reshape((self.NK, self.NW, self.NB)).transpose(0, 2, 1)

    """
    def write(self,seedname,comment="written by WannierBerri"):
        comment=comment.strip()
        f_mmn_out=open(seedname+".amn","w")
        print ("writing {}.amn: ".format(seedname)+comment+"\n")
        f_mmn_out.write(comment+"\n")
        f_mmn_out.write("  {:3d} {:3d} {:3d}  \n".format(self.NB,self.NK,self.NW))
        for ik in range(self.NK):
            f_mmn_out.write("".join(" {:4d} {:4d} {:4d} {:17.12f} {:17.12f}\n".format(ib+1,iw+1,ik+1,self.data[ik,ib,iw].real,self.data[ik,ib,iw].imag) for iw in range(self.NW) for ib in range(self.NB)))
        f_mmn_out.close()
    """


class EIG(W90_file):

    def __init__(self, seedname):
        data = np.loadtxt(seedname + ".eig")
        NB = int(round(data[:, 0].max()))
        NK = int(round(data[:, 1].max()))
        data = data.reshape(NK, NB, 3)
        assert np.linalg.norm(data[:, :, 0] - 1 - np.arange(NB)[None, :]) < 1e-15
        assert np.linalg.norm(data[:, :, 1] - 1 - np.arange(NK)[:, None]) < 1e-15
        self.data = data[:, :, 2]


class SPN(W90_file):
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


class UXU(W90_file):
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


class SXU(W90_file):
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


def parse_win_raw(filename=None, text=None):
    try:
        import wannier90io as w90io
    except ImportError as err:
        raise ImportError(f"Failed to import `wannier90io` with error message `{err}`\n"
                          "please install it manually as \n"
                          "`pip install git+https://github.com/jimustafa/wannier90io-python.git`")
    if filename is not None:
        with open(filename) as f:
            return w90io.parse_win_raw(f.read())
    elif text is not None:
        return w90io.parse_win_raw(text)


class WIN():

    # TODO :use w90io to read win file
    def __init__(self, seedname='wannier90'):
        self.name = seedname + ".win"
        self.parsed = parse_win_raw(self.name)
        self.units_length = {'ang': 1., 'bohr': physical_constants['Bohr radius'][0] * 1e10}

    @functools.lru_cache()
    def get_param(self, param):
        return self.parsed['parameters'][param]

    @functools.lru_cache()
    def get_unit_cell_cart_ang(self):
        cell = self.parsed['unit_cell_cart']
        A = np.array([cell['a1'], cell['a2'], cell['a3']])
        return A * self.units_length[cell['units']]

    @functools.lru_cache()
    def get_kpoints(self):
        return np.array(self.parsed['kpoints']['kpoints'])


"""
class DMN:

    def __init__(self,seedname="wannier90",num_wann=0,num_bands=None,nkpt=None):
        if seedname is not None:
            self.read(seedname,num_wann)
        else:
            self.void(num_wann,num_bands,nkpt)

    def read(self,seedname="wannier90",num_wann=0):
        fl=open(seedname+".dmn","r")
        self.comment=fl.readline().strip()
        self.NB,self.Nsym,self.nkptirr,self.nkpt = read_numbers(fl,4)
        self.num_wann=num_wann
        self.kpt2kptirr              = read_numbers(fl,self.nkpt)-1
        self.kptirr                  = read_numbers(fl,self.nkptirr)-1
        self.kptirr2kpt= read_numbers(fl,(self.Nsym,self.nkptirr))-1
        # find an symmetry that brings the irreducible kpoint from self.kpt2kptirr into the reducible kpoint in question
        self.kpt2kptirr_sym           = np.array([np.where(self.kptirr2kpt[:,self.kpt2kptirr[ik]]==ik)[0][0] for ik in range(self.nkpt)])

        # read the rest of lines and comvert to conplex array
        data=[l.strip("() \n").split(",") for l in fl.readlines()]
        data=np.array([x for x in data if len(x)==2],dtype=float)
        data=data[:,0]+1j*data[:,1]
        n1=self.num_wann**2*self.Nsym*self.nkptirr
        self.D_wann_dag=data[:n1].reshape(self.nkptirr,self.Nsym,self.num_wann,self.num_wann).transpose((0,1,3,2)).conj()
        self.d_band=data[n1:].reshape(self.nkptirr,self.Nsym,self.NB,self.NB)

    def void(self,num_wann,num_bands,nkpt):
        self.comment="only identity"
        self.NB,self.Nsym,self.nkptirr,self.nkpt = num_bands,1,nkpt,nkpt
        self.num_wann=num_wann
        self.kpt2kptirr              = np.arange(self.nkpt)
        self.kptirr                  = self.kpt2kptirr
        self.kptirr2kpt= np.array([self.kptirr])
        self.kpt2kptirr_sym           = np.zeros(self.nkpt,dtype=int)
        # read the rest of lines and comvert to conplex array
        self.d_band=np.ones((self.nkptirr,self.Nsym),dtype=complex)[:,:,None,None]*np.eye(self.NB)[None,None,:,:]
        self.D_wann_dag=np.ones((self.nkptirr,self.Nsym),dtype=complex)[:,:,None,None]*np.eye(self.num_wann)[None,None,:,:]


    def select_bands(self,win_index_irr):
        self.d_band=[ D[:,wi,:][:,:,wi] for D,wi in zip(self.d_band,win_index_irr) ]

    def set_free(self,frozen_irr):
        free=np.logical_not(frozen_irr)
        self.d_band_free=[ d[:,f,:][:,:,f] for d,f in zip(self.d_band,free) ]

    def write(self):
        print (self.comment)
        print (self.NB,self.Nsym,self.nkptirr,self.nkpt,self.num_wann)
        for i in range(self.nkptirr):
            for j in range(self.Nsym):
                print()
                for M in self.D_band[i][j],self.d_wann[i][j]:
                    print("\n".join(" ".join("{}".format("X" if abs(x)**2>0.1 else ".") for x in m) for m in M)+"\n")
#                   print("\n".join(" ".join("{:4.2f}".format(abs(x)**2) for x in m) for m in M)+"\n")
"""
