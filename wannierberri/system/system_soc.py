import numpy as np

from ..utility import cached_einsum
from ..fourier.rvectors import Rvectors
from ..w90files.soc import SOC

from .system_R import System_R



class SystemSOC(System_R):

    """
    A system that adds spin-orbit coupling (SOC) as a perturbation, based on the pure spin-up and spin-down systems.

    """

    def __init__(self,
                 system_up,
                 system_down=None,
                 axis=(0, 0, 1),
                 ):
        assert isinstance(system_up, System_R), "system_up must be an instance of System_R"
        self.system_up = system_up
        assert not system_up.is_phonon, "SystemSOC does not support phonons"
        if system_down is None:
            self.system_down = system_up
            self.up_down_same = True
        else:
            assert isinstance(system_down, System_R), "system_down must be an instance of System_R"
            self.up_down_same = False
            self.system_down = system_down
            assert system_up.num_wann == system_down.num_wann, \
                f"Number of Wannier functions must match for up and down systems ({system_up.num_wann} != {system_down.num_wann})"
            assert np.allclose(system_up.real_lattice, system_down.real_lattice), \
                f"Real lattices of up and down systems should match {system_up.real_lattice} != {system_down.real_lattice}"
            assert np.all(system_up.periodic == system_down.periodic), \
                f"Periodicities of up and down systems should match {system_up.periodic} != {system_down.periodic}"
            assert not system_up.is_phonon, "SystemSOC does not support phonons in down system"


        self.is_phonon = False
        self.num_wann_scalar = system_up.num_wann
        self.num_wann = 2 * self.num_wann_scalar
        self.real_lattice = system_up.real_lattice
        self.periodic = system_up.periodic
        self.axis = np.array(axis, dtype=float)

        self.wannier_centers_cart = np.zeros((self.num_wann, 3), dtype=float)
        self.wannier_centers_cart[::2] = self.system_up.wannier_centers_cart
        self.wannier_centers_cart[1::2] = self.system_down.wannier_centers_cart

        self.pointgroup = system_up.pointgroup
        self.force_internal_terms_only = any(
            [self.system_up.force_internal_terms_only, self.system_down.force_internal_terms_only])
        self.soc_R = None  # to be set later
        self.rvec = None
        self._XX_R = dict()
        self.has_soc = False


    def set_soc_R(self, soc, 
                  chk_up, chk_down=None,
                  kptirr=None, weights_k=None, ws_dist_tol=1e-5,
                  theta=0, phi=0, alpha_soc=1.0):
        """
        Set the spin-orbit coupling matrix for a given k-point.

        Parameters:
        soc : wannierberri.w90files.soc.SOC
            the object containing the SOC Hamiltonian and overlap (between up and down) matrices.
        chk_up : bool
            Flag to check if the up-spin system is valid.
        chk_down : bool
            Flag to check if the down-spin system is valid.(needed for magnetic systems)
        kptirr : np.array, optional
            Irreducible k-points (only save data for these k-points).
        weights_k : np.array, optional
            Weights for the k-points (if kptirr is provided).
        ws_dist_tol : float, optional
            Tolerance for the Wiggins-Seitz distance for the R-vectors.
        theta : float, optional
            Polar angle for the spin-orbit coupling. (radians)
        phi : float, optional
            Azimuthal angle for the spin-orbit coupling. (radians)
        alpha_soc : float, optional
            Scaling factor for the spin-orbit coupling matrix.
        """
        assert isinstance(soc, SOC), "soc must be an instance of wannierberri.w90files.soc.SOC"
        nspin = soc.nspin
        print (f"nspin in SOC: {nspin}")
        if nspin == 1:
            def index_spin(s):
                return 0
            chk_list = [chk_up]
        elif nspin == 2:
            def index_spin(s):
                return s
            assert chk_down is not None, "chk_down must be provided for nspin=2 SOC"
            assert chk_up.num_kpts == chk_down.num_kpts, f"Number of k-points must match for up and down systems ({chk_up.num_kpts} != {chk_down.num_kpts})"
            assert np.all(chk_up.mp_grid == chk_down.mp_grid)
            assert np.allclose(chk_up.kpt_latt, chk_down.kpt_latt), f"k-point grids should match for up and down systems ({chk_up.kpt_latt} != {chk_down.kpt_latt})"
            chk_list = [chk_up, chk_down]

        assert (kptirr is None) == (weights_k is None), f"kptirr and weights_k must both be provided or both be None ({kptirr=}, {weights_k=})"
        overlap_q_H = soc.overlap
        h_soc = soc.data

        if overlap_q_H is None:
            eye = np.eye(chk_up.num_bands, dtype=complex)
            overlap_q_H = {i: eye for i in range(chk_up.num_kpts)}


        mp_grid = chk_up.mp_grid

        selected_bands_list = [chk.get_selected_bands() for chk in [chk_up, chk_down]]

        self.rvec = Rvectors(lattice=self.real_lattice, shifts_left_red=self.wannier_centers_red)
        self.rvec.set_Rvec(mp_grid=mp_grid, ws_tolerance=ws_dist_tol)

        self.rvec.set_fft_q_to_R(chk_up.kpt_latt, numthreads=1, fftlib='fftw')
        NK = chk_up.num_kpts
        if kptirr is None:
            kptirr = np.arange(NK)
            weights_k = np.ones(NK, dtype=float)
        else:
            raise NotImplementedError("kptirr and weights_k are not implemented yet")
        soc_q_W = np.zeros((NK, self.num_wann, self.num_wann), dtype=complex)
        ss_q_W = np.zeros((NK, self.num_wann, self.num_wann, 3), dtype=complex)

        S_ssa = SOC.get_S_vss(theta=theta, phi=phi).transpose(1, 2, 0)
        C_ss = SOC.get_C_ss(theta=theta, phi=phi)

            
        rng = np.arange(self.num_wann_scalar) * 2
        for ik, w in zip(kptirr, weights_k):
            v = [chk.v_matrix[ik] for chk in chk_list]
            vt = [v_.T.conj() for v_ in v]
            for i in range(2):
                i1 = index_spin(i)
                for j in range(2):
                    j1 = index_spin(j)
                    h_loc = h_soc[ik][i1, j1][:, :, selected_bands_list[i], :][:, :, :, selected_bands_list[j]]
                    h_loc = cached_einsum("abmn,a,b->mn", h_loc, C_ss[:,i].conj(), C_ss[:,j])
                    soc_q_W[ik, i::2, j::2] = w * (vt[i] @ h_loc @ v[j])
                    # Spin operator
                    if i == j:
                        ss_q_W[ik, i + rng, i + rng, :] = w * S_ssa[i, j, None, :]
                    elif i < j:  # the (i=0,j=1) case
                        if nspin == 2:
                            overlap_loc = overlap_q_H[ik][selected_bands_list[0], :][:, selected_bands_list[1]]
                            ss_q_W[ik, ::2, 1::2, :] = 2 * (vt[0] @ overlap_loc @ v[1])[:, :, None] * (w * S_ssa[0, 1, :]) # factor 2 because we use Hermiticity later
                        else:
                            ss_q_W[ik, rng, 1+rng, :] = (2 * w) * S_ssa[0, 1, None, :] # factor 2 because we use Hermiticity later
        soc_q_W = (soc_q_W + soc_q_W.transpose(0, 2, 1).conj()) / 2.0
        ss_q_W = (ss_q_W + ss_q_W.transpose(0, 2, 1, 3).conj()) / 2.0

        self.soc_R = self.rvec.q_to_R(soc_q_W) * alpha_soc
        self.set_R_mat('SS', self.rvec.q_to_R(ss_q_W))

        self.has_soc = True
        return self.soc_R
