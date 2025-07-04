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

from collections import defaultdict
import numpy as np
import os
import multiprocessing
import warnings

from ..fourier.rvectors import Rvectors
from ..utility import cached_einsum, real_recip_lattice, alpha_A, beta_A
from .system_R import System_R
from ..w90files import Wannier90data


needed_files = defaultdict(lambda: [])

needed_files['AA'] = ['mmn']
needed_files['BB'] = ['mmn', 'eig']
needed_files['CC'] = ['uhu', 'mmn']
needed_files['OO'] = ['uiu', 'mmn']  # mmn is needed here because it stores information on
needed_files['GG'] = ['uiu', 'mmn']  # neighboring k-points
needed_files['SS'] = ['spn']
needed_files['SH'] = ['spn', 'eig']
needed_files['SR'] = ['spn', 'mmn']
needed_files['SA'] = ['siu', 'mmn']
needed_files['SHA'] = ['shu', 'mmn']


class System_w90(System_R):
    """
    System initialized from the Wannier functions generated by `Wannier90 <http://wannier.org>`__ code.
    Reads the ``.chk``, ``.eig`` and optionally ``.mmn``, ``.spn``, ``.uHu``, ``.sIu``, and ``.sHu`` files

    Parameters
    ----------
    seedname : str
        the seedname used in Wannier90
    w90data : `~wannierberri.system.Wannier90data`
        object that contains all Wannier90 input files and chk all together. If provided, overrides the `seedname`
    transl_inv_JM : bool
        translational-invariant scheme for diagonal and off-diagonal matrix elements for all matrices. Follows method of Jae-Mo Lihm
    wannier_centers_from_chk : bool
        If True, the centers of the Wannier functions are read from the ``.chk`` file. If False, the centers are recalculated from the ``.mmn`` file.
    npar : int
        number of processes used in the constructor
    fft : str
        library used to perform the fast Fourier transform from **q** to **R**. ``fftw`` or ``numpy``. (practically does not affect performance,
        anyway mostly time of the constructor is consumed by reading the input files)
    read_npz : bool
    write_npz_list : tuple(str)
    write_npz_formatted : bool
        see `~wannierberri.system.w90_files.Wannier90data`
    overwrite_npz : bool
        see `~wannierberri.system.w90_files.Wannier90data`
    formatted : tuple(str)
        see `~wannierberri.system.w90_files.Wannier90data`
    symmetrize : bool
        if True, the R-matrices and wannier centers are symmetrized (highly recommended, False is for debugging only)
        works only if initialized from the w90data object, and that object has the symmetrizer
    transl_inv_MV : bool
        Use Eq.(31) of `Marzari&Vanderbilt PRB 56, 12847 (1997) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.56.12847>`_ for band-diagonal position matrix elements
        Note : it applies only to the `AA` matrix for R+!=[0,0,0] and only if `transl_inv_JM` is False
        Kept for legacy reasons, as it is not used recommended to use. 
    **parameters
        see `~wannierberri.system.System_R` and `~wannierberri.system.system.System` for the rest of the parameters

    Notes
    -----
    The R-matrices are evaluated in the nearest-neighbor vectors of the finite-difference scheme chosen.

    Attributes
    ----------
    seedname : str
        the seedname used in Wannier90
    _NKFFT_recommended : int
        recommended size of the FFT grid in the interpolation


    See Also
    --------
    `~wannierberri.system.system.System_R`
    """

    def __init__(
            self,
            seedname="wannier90",
            w90data=None,
            transl_inv_MV=False,
            transl_inv_JM=False,
            fftlib='fftw',
            npar=None,
            wannier_centers_from_chk=True,
            read_npz=True,
            write_npz_list=("eig", "mmn"),
            write_npz_formatted=True,
            overwrite_npz=False,
            formatted=tuple(),
            symmetrize=False,  # temporary set to False, because there is a bug when the basis at different atoms is rotated # TODO FIXME
            **parameters
    ):

        if npar is None:
            npar = multiprocessing.cpu_count()
        if transl_inv_MV:
            warnings.warn("transl_inv_MV is deprecated and will be removed in the future. "
                          "Use transl_inv_JM instead.")
        if "name" not in parameters:
            parameters["name"] = os.path.split(seedname)[-1]
        super().__init__(**parameters)

        if transl_inv_JM:
            known = ['Ham', 'AA', 'BB', 'CC', 'OO', 'GG', 'SS', 'SH', 'SA', 'SHA']
            unknown = set(self.needed_R_matrices) - set(known)
            if len(unknown) > 0:
                raise NotImplementedError(f"transl_inv_JM for {list(unknown)} is not implemented")
            # Deactivate transl_inv_MV if Jae-Mo's scheme is used
            if transl_inv_MV:
                warnings.warn("Jae-Mo's scheme does not apply Marzari & Vanderbilt formula for"
                              "the band-diagonal matrix elements of the position operator.")
                transl_inv_MV = False
        else:
            known = ['Ham', 'AA', 'BB', 'CC', 'OO', 'GG', 'SS', 'SH', 'SHR', 'SHA', 'SA', 'SR']
            unknown = set(self.needed_R_matrices) - set(known)
            if len(unknown) > 0:
                raise NotImplementedError(f"unknown matrices requested: {list(unknown)} is not implemented")

        self.seedname = seedname
        if w90data is None:
            _needed_files = set(["eig", "chk"])
            for key in self.needed_R_matrices:
                _needed_files.update(needed_files[key])
            _needed_files = list(_needed_files)
            w90data = Wannier90data().from_w90_files(
                self.seedname,
                write_npz_list=write_npz_list, read_npz=read_npz, overwrite_npz=overwrite_npz,
                readfiles=_needed_files,
                write_npz_formatted=write_npz_formatted,
                formatted=formatted)
            # w90data.set_chk(kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol, read=True)
        w90data.check_wannierised(msg="creation of System_w90")
        if w90data.irreducible:
            symmetrize = True
        chk = w90data.chk
        self.real_lattice, self.recip_lattice = real_recip_lattice(chk.real_lattice, chk.recip_lattice)
        self.set_pointgroup(spacegroup=w90data.get_spacegroup())
        self.wannier_centers_cart = chk.wannier_centers_cart

        mp_grid = w90data.mp_grid
        self._NKFFT_recommended = mp_grid
        self.rvec = Rvectors(lattice=self.real_lattice, shifts_left_red=self.wannier_centers_red)
        self.rvec.set_Rvec(mp_grid=mp_grid, ws_tolerance=self.ws_dist_tol)
        self.num_wann = w90data.num_wann

        self.rvec.set_fft_q_to_R(
            kpt_red=w90data.kpt_latt,
            numthreads=npar,
            fftlib=fftlib,
        )

        #########
        # Oscar #
        #######################################################################

        # Compute the Fourier transform of matrix elements in the original
        # ab-initio mesh (Wannier gauge) to real-space. These matrices are
        # resolved in b, i.e. in the nearest-neighbor vectors of the
        # finite-difference scheme chosen. After ws_dist is applied, phase
        # factors depending on the lattice vectors R can be added, and the sum
        # over nearest-neighbor vectors can be finally performed.


        # H(R) matrix

        kptirr, weights_k = w90data.kptirr_system

        HHq = chk.get_HH_q(w90data.eig, kptirr=kptirr, weights_k=weights_k)

        self.set_R_mat('Ham', self.rvec.q_to_R(HHq))

        if self.need_R_any('SS'):
            self.set_R_mat('SS', self.rvec.q_to_R(chk.get_SS_q(w90data.spn, kptirr=kptirr, weights_k=weights_k)))

        if wannier_centers_from_chk:
            self.wannier_centers_cart = w90data.wannier_centers_cart
        else:
            assert w90data.has_file('mmn'), "mmn file is needed to calculate the centers of the Wannier functions"
            AA_q = chk.get_AA_qb(w90data.mmn, kptirr=kptirr, weights_k=weights_k,
                                 transl_inv=True, sum_b=True, phase=None)
            AA_R0 = AA_q.sum(axis=0) / np.prod(mp_grid)
            self.wannier_centers_cart = np.diagonal(AA_R0, axis1=0, axis2=1).T

        # Wannier centers
        centers = self.wannier_centers_cart
        # Unique set of nearest-neighbor vectors (cartesian)
        if w90data.has_file('mmn'):
            bk_cart = w90data.mmn.bk_cart

            if transl_inv_JM:
                _r0 = 0.5 * (centers[:, None, :] + centers[None, :, :])
                sum_b = False
            else:
                _r0 = centers[None, :, :]
                sum_b = True

            expjphase1 = np.exp(1j * cached_einsum('ba,ija->ijb', bk_cart, _r0))
            print(f"expjphase1 {expjphase1.shape}")
            expjphase2 = expjphase1.swapaxes(0, 1).conj()[:, :, :, None] * expjphase1[:, :, None, :]


            # A_a(R,b) matrix
            if self.need_R_any('AA'):
                AA_qb = chk.get_AA_qb(w90data.mmn, kptirr=kptirr, weights_k=weights_k,
                                      transl_inv=transl_inv_MV, sum_b=sum_b, phase=expjphase1)
                AA_Rb = self.rvec.q_to_R(AA_qb)
                self.set_R_mat('AA', AA_Rb, Hermitian=True)

            # B_a(R,b) matrix
            if 'BB' in self.needed_R_matrices:
                BB_qb = chk.get_BB_qb(w90data.mmn, w90data.eig, kptirr=kptirr, weights_k=weights_k,
                                      sum_b=sum_b, phase=expjphase1)
                BB_Rb = self.rvec.q_to_R(BB_qb)
                self.set_R_mat('BB', BB_Rb)

            # C_a(R,b1,b2) matrix
            if 'CC' in self.needed_R_matrices:
                CC_qb = chk.get_CC_qb(w90data.mmn, w90data.uhu, kptirr=kptirr, weights_k=weights_k,
                                      sum_b=sum_b, phase=expjphase2)
                CC_Rb = self.rvec.q_to_R(CC_qb)
                self.set_R_mat('CC', CC_Rb, Hermitian=True)

            # O_a(R,b1,b2) matrix
            if 'OO' in self.needed_R_matrices:
                OO_qb = chk.get_OO_qb(w90data.mmn, w90data.uiu, kptirr=kptirr, weights_k=weights_k,
                                      sum_b=sum_b, phase=expjphase2)
                OO_Rb = self.rvec.q_to_R(OO_qb)
                self.set_R_mat('OO', OO_Rb, Hermitian=True)

            # G_bc(R,b1,b2) matrix
            if 'GG' in self.needed_R_matrices:
                GG_qb = chk.get_GG_qb(w90data.mmn, w90data.uiu, kptirr=kptirr, weights_k=weights_k,
                                      sum_b=sum_b, phase=expjphase2)
                GG_Rb = self.rvec.q_to_R(GG_qb)
                self.set_R_mat('GG', GG_Rb, Hermitian=True)

            #######################################################################

            if self.need_R_any('SR'):
                self.set_R_mat('SR', self.rvec.q_to_R(chk.get_SHR_q(spn=w90data.spn, mmn=w90data.mmn,
                                                                    kptirr=kptirr, weights_k=weights_k,
                                       phase=expjphase1)))
            if self.need_R_any('SH'):
                self.set_R_mat('SH', self.rvec.q_to_R(chk.get_SH_q(w90data.spn, w90data.eig,
                                                                   kptirr=kptirr, weights_k=weights_k,
                                       )))
            if self.need_R_any('SHR'):
                self.set_R_mat('SHR', self.rvec.q_to_R(
                    chk.get_SHR_q(spn=w90data.spn, mmn=w90data.mmn,
                                  kptirr=kptirr, weights_k=weights_k,
                                  eig=w90data.eig, phase=expjphase1)))

            if 'SA' in self.needed_R_matrices:
                self.set_R_mat('SA',
                            self.rvec.q_to_R(chk.get_SHA_q(w90data.siu, w90data.mmn,
                                                           kptirr=kptirr, weights_k=weights_k,
                                        sum_b=sum_b, phase=expjphase1)))
            if 'SHA' in self.needed_R_matrices:
                self.set_R_mat('SHA',
                            self.rvec.q_to_R(chk.get_SHA_q(w90data.shu, w90data.mmn,
                                                           kptirr=kptirr, weights_k=weights_k,
                                        sum_b=sum_b, phase=expjphase1)))

            del expjphase1, expjphase2

            if transl_inv_JM:
                self.recenter_JM(centers, bk_cart)


        self.do_at_end_of_init()
        self.check_AA_diag_zero(msg="after conversion of conventions with "
                           f"transl_inv_MV={transl_inv_MV}, transl_inv_JM={transl_inv_JM}",
                                set_zero=transl_inv_MV or transl_inv_JM,
                                threshold=0.1 if transl_inv_JM else 1e5)
        if symmetrize and w90data.has_file('symmetrizer'):
            self.symmetrize2(w90data.symmetrizer)

    ###########################################################################
    def recenter_JM(self, centers, bk_cart):
        """"
        Recenter the matrices in the Jae-Mo scheme
        (only in convention I)

        Parameters
        ----------
        centers : np.ndarray(shape=(num_wann, 3))
            Wannier centers in Cartesian coordinates
        bk_cart : np.ndarray(shape=(num_bk, 3))
            set of nearest-neighbor vectors (cartesian)

        Notes
        -----
        The matrices are recentered in the following way:
        - A_a(R) matrix: no recentering
        - B_a(R) matrix: recentered by the Hamiltonian
        - C_a(R) matrix: recentered by the B matrix
        - O_a(R) matrix: recentered by the A matrix
        - G_bc(R) matrix: no recentering
        - S_a(R) matrix: recentered by the S matrix
        - SH_a(R) matrix: recentered by the S matrix
        - SR_a(R) matrix: recentered by the S matrix
        - SA_a(R) matrix: recentered by the S matrix
        - SHA_a(R) matrix: recentered by the S matrix
        """
        #  Here we apply the phase factors associated with the
        # JM scheme not accounted above, and perform the sum over
        # nearest-neighbor vectors to finally obtain the real-space matrix
        # elements.

        # Optimal center in Jae-Mo's implementation
        phase = cached_einsum('ba,Ra->Rb', bk_cart, - 0.5 * self.rvec.cRvec)
        expiphase1 = np.exp(1j * phase)[:, None, None, :]
        expiphase2 = expiphase1[:, :, :, :, None] * expiphase1[:, :, :, None, :]

        def _reset_mat(key, phase, axis, Hermitian=True):
            if self.need_R_any(key):
                XX_Rb = self.get_R_mat(key)
                phase = np.reshape(phase, np.shape(phase) + (1,) * (XX_Rb.ndim - np.ndim(phase)))
                XX_R = np.sum(XX_Rb * phase, axis=axis)
                self.set_R_mat(key, XX_R, reset=True, Hermitian=Hermitian)

        _reset_mat('AA', expiphase1, 3)
        _reset_mat('BB', expiphase1, 3, Hermitian=False)
        _reset_mat('CC', expiphase2, (3, 4))
        _reset_mat('SA', expiphase1, 3, Hermitian=False)
        _reset_mat('SHA', expiphase1, 3, Hermitian=False)
        _reset_mat('OO', expiphase2, (3, 4))
        _reset_mat('GG', expiphase2, (3, 4))

        del expiphase1, expiphase2
        r0 = 0.5 * (centers[None, :, None, :] + centers[None, None, :, :] + self.rvec.cRvec[:, None, None, :])

        # --- A_a(R) matrix --- #
        if self.need_R_any('AA'):
            AA_R0 = self.get_R_mat('AA').copy()
        # --- B_a(R) matrix --- #
        if self.need_R_any('BB'):
            BB_R0 = self.get_R_mat('BB').copy()
            HH_R = self.get_R_mat('Ham')
            rc = (r0 - self.rvec.cRvec[:, None, None, :] - centers[None, None, :, :]) * HH_R[:, :, :, None]
            self.set_R_mat('BB', rc, add=True)
        # --- C_a(R) matrix --- #
        if self.need_R_any('CC'):
            assert BB_R0 is not None, 'Recentered B matrix is needed in Jae-Mo`s implementation of C'
            BB_R0_conj = self.rvec.conj_XX_R(BB_R0)
            rc = 1j * (r0[:, :, :, :, None] - centers[None, :, None, :, None]) * (BB_R0 + BB_R0_conj)[:, :, :, None, :]
            CC_R_add = rc[:, :, :, alpha_A, beta_A] - rc[:, :, :, beta_A, alpha_A]
            self.set_R_mat('CC', CC_R_add, add=True, Hermitian=True)
        if self.need_R_any('SA'):
            SS_R = self.get_R_mat('SS')
            rc = (r0[:, :, :, :, None] - self.rvec.cRvec[:, None, None, :, None] - centers[None, None, :, :, None]
                  ) * SS_R[:, :, :, None, :]
            self.set_R_mat('SA', rc, add=True)
        if self.need_R_any('SHA'):
            SH_R = self.get_R_mat('SH')
            rc = (r0[:, :, :, :, None] - self.rvec.cRvec[:, None, None, :, None] -
                  centers[None, None, :, :, None]) * SH_R[:, :, :, None, :]
            self.set_R_mat('SHA', rc, add=True)
        # --- O_a(R) matrix --- #
        if self.need_R_any('OO'):
            assert AA_R0 is not None, 'Recentered A matrix is needed in Jae-Mo`s implementation of O'
            rc = 1.j * (r0[:, :, :, :, None] - centers[None, :, None, :, None]) * AA_R0[:, :, :, None, :]
            OO_R_add = rc[:, :, :, alpha_A, beta_A] - rc[:, :, :, beta_A, alpha_A]
            self.set_R_mat('OO', OO_R_add, add=True, Hermitian=True)
        # --- G_bc(R) matrix --- #
        if self.need_R_any('GG'):
            pass
