from functools import cached_property
import os
import warnings
import numpy as np
from irrep.spacegroup import SpaceGroup

from ..symmetry.point_symmetry import PointGroup

from ..utility import cached_einsum
from ..fourier.rvectors import Rvectors
from ..w90files.soc import SOC

from .system_R import System_R
from .system_w90 import System_w90



class SystemSOC(System_R):

    """
    A system that adds spin-orbit coupling (SOC) as a perturbation, based on the pure spin-up and spin-down systems.

    """

    half_wann_matrices = set(["overlap_up_down", "dV_soc_wann_0_0", "dV_soc_wann_0_1", "dV_soc_wann_1_1"])

    def __init__(self,
                 system_up,
                 system_down=None,
                 silent=True,
                 cell=None
                 ):
        self.needed_R_matrices = set()
        self.silent = silent
        assert isinstance(system_up, System_R), f"system_up must be an instance of System_R, got {type(system_up)}"
        self.system_up = system_up
        assert not system_up.is_phonon, "SystemSOC does not support phonons"
        if system_down is None:
            self.system_down = system_up
            self.nspin = 1
        else:
            assert isinstance(system_down, System_R), "system_down must be an instance of System_R"
            self.nspin = 2
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

        self.wannier_centers_cart = np.zeros((self.num_wann, 3), dtype=float)
        self.wannier_centers_cart[::2] = self.system_up.wannier_centers_cart
        self.wannier_centers_cart[1::2] = self.system_down.wannier_centers_cart

        self.pointgroup = PointGroup()
        self.force_internal_terms_only = any(
            [self.system_up.force_internal_terms_only, self.system_down.force_internal_terms_only])
        self.rvec = None
        self._XX_R = dict()
        self.has_soc = False
        if cell is not None:
            self.set_cell(**cell)
        else:
            self.cell = None

    def set_cell(self, positions, typat, magmoms_on_axis):
        self.cell = dict(positions=np.array(positions),
                         typat=np.array(typat),
                         magmoms_on_axis=np.array(magmoms_on_axis))
        return self


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

        # chack number of spin channels
        nspin = soc.nspin
        assert self.nspin == nspin, f"Number of spin channels in SystemSOC ({self.nspin}) does not match that in SOC ({nspin})"
        print(f"nspin in SOC: {nspin}")
        if nspin == 1:
            chk_list = [chk_up]
        elif nspin == 2:
            assert chk_down is not None, "chk_down must be provided for nspin=2 SOC"
            assert chk_up.num_kpts == chk_down.num_kpts, f"Number of k-points must match for up and down systems ({chk_up.num_kpts} != {chk_down.num_kpts})"
            assert np.all(chk_up.mp_grid == chk_down.mp_grid)
            assert np.allclose(chk_up.kpt_latt, chk_down.kpt_latt), f"k-point grids should match for up and down systems ({chk_up.kpt_latt} != {chk_down.kpt_latt})"
            chk_list = [chk_up, chk_down]

        assert (kptirr is None) == (weights_k is None), f"kptirr and weights_k must both be provided or both be None ({kptirr=}, {weights_k=})"
        overlap_q_H = soc.overlap
        dV_soc = soc.data

        if overlap_q_H is None:
            eye = np.eye(chk_up.num_bands, dtype=complex)
            overlap_q_H = {i: eye for i in range(chk_up.num_kpts)}

        mp_grid = chk_up.mp_grid

        # selected_bands_list = [chk.get_selected_bands() for chk in [chk_up, chk_down]]

        # print(f"setting the Rvectors with wannier centers (cart): \n {self.wannier_centers_cart}\n (red): \n {self.wannier_centers_red}")
        self.rvec = Rvectors(lattice=self.real_lattice, shifts_left_red=self.wannier_centers_red)
        self.rvec.set_Rvec(mp_grid=mp_grid, ws_tolerance=ws_dist_tol)

        self.rvec.set_fft_q_to_R(chk_up.kpt_latt, numthreads=1, fftlib='fftw')
        NK = chk_up.num_kpts

        if kptirr is None:
            kptirr = np.arange(NK)
            weights_k = np.ones(NK, dtype=float)

        rng = np.arange(self.num_wann_scalar) * 2

        for i1 in range(nspin):
            # sel_i = selected_bands_list[i1]
            for j1 in range(i1, nspin):
                # sel_j = selected_bands_list[j1]
                dV_soc_wann_ik = np.zeros((NK, self.num_wann_scalar, self.num_wann_scalar, 3), dtype=complex)
                for ik, w in zip(kptirr, weights_k):
                    vt = chk_list[i1].v_matrix[ik].T.conj()
                    v = chk_list[j1].v_matrix[ik]
                    # dV_soc_wann_ik[ik] = w * cached_einsum("mi,cij,jn->mnc", vt, dV_soc[ik][i1, j1][:, sel_i, :][:, :, sel_j], v)
                    dV_soc_wann_ik[ik] = w * cached_einsum("mi,cij,jn->mnc", vt, dV_soc[ik][i1, j1][:, :, :][:, :, :], v)
                if i1 == j1:
                    assert np.allclose(dV_soc_wann_ik, dV_soc_wann_ik.transpose(0, 2, 1, 3).conj()), "The diagonal spin components of SOC should be Hermitian"
                    # dV_soc_wann_ik = (dV_soc_wann_ik + dV_soc_wann_ik.transpose(0,2,1,3).conj())/2.0
                key = f"dV_soc_wann_{i1}_{j1}"
                mat = self.rvec.q_to_R(dV_soc_wann_ik, select_left=rng + i1, select_right=rng + j1)
                self.set_R_mat(key, mat)
        if nspin == 2:
            overlap_ik = np.zeros((NK, self.num_wann_scalar, self.num_wann_scalar), dtype=complex)
            for ik, w in zip(kptirr, weights_k):
                vt = chk_list[0].v_matrix[ik].T.conj()
                v = chk_list[1].v_matrix[ik]
                # overlap_loc = overlap_q_H[ik][selected_bands_list[0], :][:, selected_bands_list[1]]
                # overlap_ik[ik] = w * (vt @ overlap_loc @ v)
                overlap_ik[ik] = w * (vt @ overlap_q_H[ik] @ v)
            overlap_Rud = self.rvec.q_to_R(overlap_ik, select_left=rng, select_right=rng + 1)
            self.set_R_mat('overlap_up_down', overlap_Rud)
        self.has_soc = True
        return self.set_soc_axis(theta=theta, phi=phi, alpha_soc=alpha_soc)


    def set_soc_axis(self, theta=0, phi=0, alpha_soc=1.0, units="radians"):
        units = units.lower()
        if units.startswith("r"):
            pass
        elif units.startswith("d"):
            theta = np.deg2rad(theta)
            phi = np.deg2rad(phi)
        else:
            raise ValueError(f"units must be 'radians' or 'degrees', got {units}, which is not recognized")
        assert self.has_soc, "SOC matrix must be set before setting the SOC axis"
        pauli_rotated = SOC.get_pauli_rotated(theta=theta, phi=phi)

        nRvec = self.rvec.nRvec
        soc_R_W = np.zeros((nRvec, self.num_wann, self.num_wann), dtype=complex)
        soc_R_W[:, ::2, ::2] = cached_einsum("rmnc,c->rmn", self.get_R_mat('dV_soc_wann_0_0'), pauli_rotated[0, 0, :])
        if self.nspin == 2:
            soc_R_W[:, 1::2, 1::2] = cached_einsum("rmnc,c->rmn", self.get_R_mat('dV_soc_wann_1_1'), pauli_rotated[1, 1, :])
            dV01 = self.get_R_mat('dV_soc_wann_0_1')
            soc_R_W[:, ::2, 1::2] = cached_einsum("rmnc,c->rmn", dV01, pauli_rotated[0, 1, :])
            # soc_R_W[:,1::2, ::2] = cached_einsum("rmnc,c->rmn", self.get_R_mat('dV_soc_wann_1_0'), pauli_rotated[1, 0, :])
            soc_R_W[:, 1::2, ::2] = cached_einsum("rmnc,c->rmn", self.rvec.conj_XX_R(dV01), pauli_rotated[1, 0, :])
        elif self.nspin == 1:
            soc_R_W[:, 1::2, 1::2] = cached_einsum("rmnc,c->rmn", self.get_R_mat('dV_soc_wann_0_0'), pauli_rotated[1, 1, :])
            soc_R_W[:, ::2, 1::2] = cached_einsum("rmnc,c->rmn", self.get_R_mat('dV_soc_wann_0_0'), pauli_rotated[0, 1, :])
            soc_R_W[:, 1::2, ::2] = cached_einsum("rmnc,c->rmn", self.get_R_mat('dV_soc_wann_0_0'), pauli_rotated[1, 0, :])
        self.set_R_mat('Ham_SOC', soc_R_W * alpha_soc, reset=True)

        # Spin operator
        rng = np.arange(self.num_wann_scalar) * 2
        iR0 = self.rvec.iR0
        SS_R_W = np.zeros((nRvec, self.num_wann, self.num_wann, 3), dtype=complex)
        SS_R_W[iR0, rng, rng, :] = pauli_rotated[None, 0, 0, None, None, :]
        SS_R_W[iR0, rng + 1, rng + 1, :] = pauli_rotated[None, 1, 1, None, None, :]
        if self.nspin == 2:
            overlap = self.get_R_mat('overlap_up_down')
            SS_R_W[:, 0::2, 1::2, :] = overlap[:, :, :, None] * pauli_rotated[None, 0, 1, None, None, :]
            overlap = self.rvec.conj_XX_R(overlap)
            SS_R_W[:, 1::2, 0::2, :] = overlap[:, :, :, None] * pauli_rotated[None, 1, 0, None, None, :]
        elif self.nspin == 1:
            SS_R_W[iR0, rng, rng + 1, :] = pauli_rotated[None, 0, 1, None, None, :]
            SS_R_W[iR0, rng + 1, rng, :] = pauli_rotated[None, 1, 0, None, None, :]
        else:
            raise ValueError(f"Invalid nspin: {self.nspin}")

        self.set_R_mat('SS', SS_R_W, reset=True)

        if self.cell is not None:
            axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            magmoms = self.cell["magmoms_on_axis"][:, None] * axis[None, :]
            print(f"using magmoms \n {magmoms}")
            mag_group = SpaceGroup.from_cell(real_lattice=self.real_lattice,
                                    positions=self.cell["positions"],
                                    typat=self.cell["typat"],
                                    spinor=True,
                                    include_TR=True,
                                    magmom=magmoms)
            self.set_pointgroup(spacegroup=mag_group)

        return self.get_R_mat('Ham_SOC'), self.get_R_mat('SS')

    @cached_property
    def essential_properties(self):
        return super().essential_properties + ['cell']

    def save_npz(self, path, extra_properties=(), exclude_properties=(), R_matrices=None, overwrite=True):
        # if not self.silent:
        print(f"Saving SystemSOC to {path}")
        super().save_npz(path, extra_properties=extra_properties, exclude_properties=exclude_properties, R_matrices=R_matrices, overwrite=overwrite)
        self.system_up.save_npz(path=os.path.join(path, "system_up"), overwrite=overwrite, exclude_properties=exclude_properties, R_matrices=R_matrices)
        if self.nspin == 2:
            self.system_down.save_npz(path=os.path.join(path, "system_down"), overwrite=overwrite, exclude_properties=exclude_properties, R_matrices=R_matrices)

    @property
    def has_soc_R(self):
        if self.nspin == 2:
            return self.has_R_mat_all(['dV_soc_wann_0_0', 'dV_soc_wann_1_1', 'dV_soc_wann_0_1', 'overlap_up_down'])
        else:
            return self.has_R_mat(['dV_soc_wann_0_0'])


    @classmethod
    def from_npz(cls, path, silent=True, load_all_XX_R=True, exclude_properties=()):
        if not silent:
            print(f"Loading SystemSOC from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"directory {path} does not exist")
        system_up = System_R().load_npz(path=os.path.join(path, "system_up"), load_all_XX_R=load_all_XX_R, exclude_properties=exclude_properties, legacy=False)
        path_down = os.path.join(path, "system_down")
        if os.path.exists(path_down):
            system_down = System_R().load_npz(path=path_down, load_all_XX_R=load_all_XX_R, exclude_properties=exclude_properties, legacy=False)
        else:
            system_down = None
        system_soc = cls(system_up=system_up, system_down=system_down, silent=silent)
        system_soc.load_npz(path, load_all_XX_R=load_all_XX_R, exclude_properties=exclude_properties, legacy=False)
        if system_soc.has_soc_R:
            system_soc.has_soc = True
        return system_soc

    def symmetrize2(self, symmetrizer=None,
                    symmetrizer_up=None,
                    symmetrizer_down=None,
                    silent=None, use_symmetries_index=None,
                    cutoff=-1, cutoff_dict=None):
        """
        Symmetrize the system according to the Symmetrizer object.

        Parameters
        ----------
        symmetrizer : :class:`wanierberri.symmetry.sawf.SymmetrizerSAWF`
            The symmetrizer object that will be used for symmetrization. (make sure it is consistent with the order of projections)
        silent : bool
            If True, do not print the symmetrization process. (set to False to see more debug information)
        use_symmetries_index : list of int
            List of symmetry indices to use for symmetrization. If None, all symmetries will be used.
        """
        from ..symmetry.sym_wann_2 import SymWann
        assert symmetrizer is not None or symmetrizer_up is not None, "Either symmetrizer or symmetrizer_up must be provided"
        if symmetrizer_up is None:
            symmetrizer_up = symmetrizer
        if symmetrizer_down is None:
            symmetrizer_down = symmetrizer_up
            warnings.warn("symmetrizer_down is not provided, using symmetrizer_up for both spin channels")
        if silent is None:
            silent = self.silent
        if not hasattr(self.system_up, 'symmetrized') or not self.system_up.symmetrized:
            self.system_up.symmetrize2(symmetrizer=symmetrizer_up, silent=silent,
                                    use_symmetries_index=use_symmetries_index,
                                    cutoff=cutoff, cutoff_dict=cutoff_dict)
            self.wannier_centers_cart[::2] = self.system_up.wannier_centers_cart
        if self.nspin == 2:
            if not hasattr(self.system_down, 'symmetrized') or not self.system_down.symmetrized:
                self.system_down.symmetrize2(symmetrizer=symmetrizer_down, silent=silent,
                                            use_symmetries_index=use_symmetries_index,
                                            cutoff=cutoff, cutoff_dict=cutoff_dict)
                self.wannier_centers_cart[1::2] = self.system_down.wannier_centers_cart


        symmetrize_wann_up_up = SymWann(
            symmetrizer_left=symmetrizer_up,
            symmetrizer_right=symmetrizer_up,
            iRvec=self.rvec.iRvec,
            silent=silent,
            use_symmetries_index=use_symmetries_index,
        )
        if self.nspin == 2:
            symmetrize_wann_down_down = SymWann(
                symmetrizer_left=symmetrizer_down,
                symmetrizer_right=symmetrizer_down,
                iRvec=self.rvec.iRvec,
                silent=silent,
                use_symmetries_index=use_symmetries_index,
            )
            symmetrize_wann_up_down = SymWann(
                symmetrizer_left=symmetrizer_up,
                symmetrizer_right=symmetrizer_down,
                iRvec=self.rvec.iRvec,
                silent=silent,
                use_symmetries_index=use_symmetries_index,
            )

        # self.check_AA_diag_zero(msg="before symmetrization", set_zero=True)
        logfile = self.logfile

        if not silent:
            logfile.write(f"Wannier Centers cart (raw):\n {self.wannier_centers_cart}\n")
            logfile.write(f"Wannier Centers red: (raw):\n {self.wannier_centers_red}\n")

        iRvec_new_dict = {}
        XX_new_dict = {}

        def symmetrize_part(sym_wann, key_list):
            _XX_dict = {k: self.get_R_mat(k) for k in key_list}
            _XX_dict_new, iRvec_new = sym_wann.symmetrize(XX_R=_XX_dict,
                                            cutoff=cutoff, cutoff_dict=cutoff_dict)
            for k, v in _XX_dict_new.items():
                XX_new_dict[k] = v
                iRvec_new_dict[k] = iRvec_new

        symmetrize_part(symmetrize_wann_up_up, ['dV_soc_wann_0_0'])
        if self.nspin == 2:
            symmetrize_part(symmetrize_wann_down_down, ['dV_soc_wann_1_1'])
            symmetrize_part(symmetrize_wann_up_down, ['dV_soc_wann_0_1', 'overlap_up_down'])

        iRvec_all = np.unique(np.vstack(list(iRvec_new_dict.values())), axis=0)
        iRvec_index = {tuple(R): i for i, R in enumerate(iRvec_all)}

        self.rvec.iRvec = iRvec_all
        self.rvec.clear_cached()

        for key in XX_new_dict:
            reorder = np.array([iRvec_index[tuple(R)] for R in iRvec_new_dict[key]])
            XX = XX_new_dict[key]
            XX_new = np.zeros((len(iRvec_all),) + XX.shape[1:], dtype=XX.dtype)
            XX_new[reorder] = XX
            self.set_R_mat(key, XX_new, reset=True)



    @classmethod
    def from_wannier90data_soc(cls, w90data, theta=0, phi=0, alpha_soc=1.0, symmetrize=True, **kwargs):
        system_up = System_w90(w90data=w90data.data_up, **kwargs)
        if w90data.nspin == 2:
            system_down = System_w90(w90data=w90data.data_down, **kwargs)
        else:
            system_down = None
        system_soc = cls(system_up=system_up, system_down=system_down, cell=w90data.cell)
        kptirr, weights_k = w90data.data_up.kptirr_system
        system_soc.set_soc_R(w90data.get_file("soc"),
                             chk_up=w90data.get_file_ud("up", "chk"),
                             chk_down=w90data.get_file_ud("down", "chk"),
                             kptirr=kptirr, weights_k=weights_k
        )
        if w90data.is_irreducible:
            symmetrize = True
        if symmetrize:
            system_soc.symmetrize2(symmetrizer_up=w90data.get_file_ud('up', 'symmetrizer'),
                           symmetrizer_down=w90data.get_file_ud('down', 'symmetrizer'))
        system_soc.set_soc_axis(theta=theta, phi=phi, alpha_soc=alpha_soc)
        return system_soc
