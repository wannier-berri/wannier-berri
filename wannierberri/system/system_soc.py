import copy
from functools import cached_property
import os
import warnings
import numpy as np

from ..symmetry.point_symmetry import PointGroup

from ..utility import cached_einsum
from ..fourier.rvectors import Rvectors
from ..w90files.soc import SOC

from .system_R import System_R


class SystemSOC(System_R):

    """
    A system that adds spin-orbit coupling (SOC) as a perturbation, based on the pure spin-up and spin-down systems.

    """

    half_wann_matrices = ['dV_soc', 'overlap_up_down']

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
        if cell is not None:
            self.set_cell(**cell)
        else:
            self.cell = None

    def swap_spin_channels(self):
        if self.nspin == 1:
            return self
        self.system_up, self.system_down = self.system_down, self.system_up
        self.wannier_centers_cart[::2], self.wannier_centers_cart[1::2] = self.wannier_centers_cart[1::2], self.wannier_centers_cart[::2].copy()
        if self.has_R_mat('overlap_up_down'):
            overlap = self.get_R_mat('overlap_up_down')
            overlap = self.rvec.conj_XX_R(overlap)
            self.set_R_mat('overlap_up_down', overlap, reset=True)
        if self.has_R_mat_any(['dV_soc_wann_0_0', 'dV_soc_wann_1_0', 'dV_soc_wann_1_1']):
            dV00 = self.get_R_mat('dV_soc_wann_0_0')
            dV11 = self.get_R_mat('dV_soc_wann_1_1')
            dV01 = self.get_R_mat('dV_soc_wann_0_1')
            dV10 = self.rvec.conj_XX_R(dV01)
            self.set_R_mat('dV_soc_wann_0_0', dV11, reset=True)
            self.set_R_mat('dV_soc_wann_1_1', dV00, reset=True)
            self.set_R_mat('dV_soc_wann_0_1', dV10, reset=True)
        self.clear_R_mat(['Ham_SOC', 'SS'])



    def set_cell(self, positions, typat, magmoms_on_axis, **kwargs):
        self.cell = dict(positions=np.array(positions),
                         typat=np.array(typat),
                         magmoms_on_axis=np.array(magmoms_on_axis))
        return self



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
        self.pauli_rotated = SOC.get_pauli_rotated(theta=theta, phi=phi)
        self.alpha_soc = alpha_soc

        if self.cell is not None:
            axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            magmoms = self.cell["magmoms_on_axis"][:, None] * axis[None, :]
            print(f"using magmoms \n {magmoms}")
            from irrep.spacegroup import SpaceGroup
            mag_group = SpaceGroup.from_cell(real_lattice=self.real_lattice,
                                    positions=self.cell["positions"],
                                    typat=self.cell["typat"],
                                    spinor=True,
                                    include_TR=True,
                                    magmom=magmoms)
            self.set_pointgroup(spacegroup=mag_group)

    @cached_property
    def essential_properties(self):
        return super().essential_properties + ['cell']

    def to_npz(self, path, extra_properties=(), exclude_properties=(), R_matrices=None, overwrite=True):
        # if not self.silent:
        print(f"Saving SystemSOC to {path}")
        super().to_npz(path, extra_properties=extra_properties, exclude_properties=exclude_properties, R_matrices=R_matrices, overwrite=overwrite)
        self.system_up.to_npz(path=os.path.join(path, "system_up"), overwrite=overwrite, exclude_properties=exclude_properties, R_matrices=R_matrices)
        if self.nspin == 2:
            self.system_down.to_npz(path=os.path.join(path, "system_down"), overwrite=overwrite, exclude_properties=exclude_properties, R_matrices=R_matrices)

    @property
    def has_soc(self):
        if self.nspin == 2:
            return self.has_R_mat_all(['dV_soc', 'overlap_up_down'])
        else:
            return self.has_R_mat_all(['dV_soc'])


    @classmethod
    def from_npz(cls, path, silent=True, exclude_properties=(), matrices=None):
        if not silent:
            print(f"Loading SystemSOC from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"directory {path} does not exist")
        system_up = System_R.from_npz(path=os.path.join(path, "system_up"), exclude_properties=exclude_properties, matrices=matrices)
        path_down = os.path.join(path, "system_down")
        if os.path.exists(path_down):
            system_down = System_R.from_npz(path=path_down, exclude_properties=exclude_properties, matrices=matrices)
        else:
            system_down = None
        system_soc = cls(system_up=system_up, system_down=system_down, silent=silent)
        system_soc.load_npz(path, exclude_properties=exclude_properties, matrices=matrices)
        return system_soc



    @classmethod
    def from_wannierdata(cls, wandata, symmetrize=True,
                         theta=0, phi=0, alpha_soc=1.0, angle_units="radians",
                         **kwargs):
        if wandata.irreducible:
            symmetrize = True
        if wandata.has_file("soc"):
            soc = wandata.get_file("soc")
        else:
            warnings.warn("SOC file not found in wandata, creating SystemSOC without SOC")
            soc = None

        system_up = System_R.from_wannierdata(wandata=wandata.data_up, symmetrize=symmetrize,
                                              soc=soc, soc_component=(0, 0), **kwargs)

        cell = wandata.cell
        if "altermagnetic_rotation_latt" in cell:
            altermag_rot = cell["altermagnetic_rotation_latt"]
            altermag_trans = cell["altermagnetic_translation_latt"]
            altermag_kmap = cell["altermagnetic_k_mapping"]
            altermag_kmap_isym = cell["altermagnetic_k_mapping_isym"]
            altermagnetic = True
            from irrep.symmetry_operation import SymmetryOperation
            symop = SymmetryOperation(rot=altermag_rot,
                                      trans=altermag_trans,
                                      Lattice=system_up.real_lattice,
                                      spinor=False, time_reversal=False)
        else:
            altermagnetic = False
            altermag_kmap = None
            altermag_kmap_isym = None

        if symmetrize:
            symmetrizer_up = wandata.data_up.symmetrizer

        if wandata.nspin == 2 and not altermagnetic:
            system_down = System_R.from_wannierdata(wandata=wandata.data_down, symmetrize=symmetrize,
                                                    soc=soc, soc_component=(1, 1), **kwargs)
        elif altermagnetic:
            system_down = copy.deepcopy(system_up)
            system_down.transform(symop)
        else:
            system_down = None
        system_soc = cls(system_up=system_up, system_down=system_down, cell=wandata.cell)

        kptirr, weights_k = wandata.data_up.kptirr_system

        

        if wandata.nspin == 2 and soc is not None:
            chk_up = wandata.data_up.chk
            v_matrix_list_up = wandata.data_up.chk.v_matrix
            if altermagnetic:
                v_matrix_list_down = [symmetrizer_up.rotate_U(v_matrix_list_up[ik1],
                                                            ikirr=ik1,
                                                            isym=isym)
                                      for ik1, isym in zip(altermag_kmap, altermag_kmap_isym)]
                symmetrizer_down = symmetrizer_up.get_transformed(symop)
            else:
                assert wandata.data_up.chk.num_kpts == wandata.data_down.chk.num_kpts, f"Number of k-points must match for up and down systems ({wandata.data_up.chk.num_kpts} != {wandata.data_down.chk.num_kpts})"
                assert np.all(wandata.data_up.chk.mp_grid == wandata.data_down.chk.mp_grid)
                assert np.allclose(wandata.data_up.chk.kpt_red, wandata.data_down.chk.kpt_red), f"k-point grids should match for up and down systems ({wandata.data_up.chk.kpt_red} != {wandata.data_down.chk.kpt_red})"
                v_matrix_list_down = wandata.data_down.chk.v_matrix
                if symmetrize:
                    symmetrizer_down = wandata.data_down.symmetrizer


            mp_grid = chk_up.mp_grid
            NK = chk_up.num_kpts

            ## up-down part
            rvec = Rvectors(lattice=system_up.real_lattice,
                            shifts_left_red=system_up.wannier_centers_red,
                            shifts_right_red=system_down.wannier_centers_red)
            rvec.set_Rvec(mp_grid=mp_grid, ws_tolerance=kwargs.get("ws_dist_tol", 1e-8))
            rvec.set_fft_q_to_R(kpt_red=chk_up.kpt_red, fftlib='fftw')

            overlap_q_H = soc.overlap
            dV_soc = soc.data
            overlap_ik = np.zeros((NK, system_up.num_wann, system_down.num_wann), dtype=complex)
            dV_soc_wann_ik = np.zeros((NK, system_up.num_wann, system_down.num_wann, 3), dtype=complex)
            for ik, w in zip(kptirr, weights_k):
                vt = v_matrix_list_up[ik].T.conj()
                v = v_matrix_list_down[ik]
                overlap_ik[ik] = w * (vt @ overlap_q_H[ik] @ v)
                dV_soc_wann_ik[ik] = w * cached_einsum("mi,cij,jn->mnc", vt, dV_soc[ik][0, 1][:, :, :][:, :, :], v)
            dV_soc_wann_R_01 = rvec.q_to_R(dV_soc_wann_ik)
            overlap_Rud = rvec.q_to_R(overlap_ik)
            system_soc.rvec = rvec
            system_soc.set_R_mat('dV_soc', dV_soc_wann_R_01)
            system_soc.set_R_mat('overlap_up_down', overlap_Rud)
            if symmetrize:
                from ..symmetry.sym_wann_2 import SymWann
                symm_wann_up_down = SymWann(
                    symmetrizer_left=symmetrizer_up,
                    symmetrizer_right=symmetrizer_down,
                    iRvec=system_soc.rvec.iRvec,
                    silent=True,
                )
                # self.check_AA_diag_zero(msg="before symmetrization", set_zero=True)

                system_soc._XX_R, iRvec_new = symm_wann_up_down.symmetrize(XX_R=system_soc._XX_R)
                rvec.iRvec = iRvec_new
                rvec.mp_grid = system_soc.rvec.mp_grid,
                rvec.clear_cached()
                system_soc.rvec = rvec
        system_soc.set_soc_axis(theta=theta, phi=phi, alpha_soc=alpha_soc, units=angle_units)
        return system_soc

    def get_system_R(self):
        from ..fourier.rvectors import merge_Rvectors
        rvectors_merged, rvectors_map_list = merge_Rvectors([self.rvec, self.system_up.rvec, self.system_down.rvec])

        system_R = System_R()
        system_R.rvec = rvectors_merged
        system_R.is_phonon = self.is_phonon
        system_R.num_wann = self.num_wann
        system_R.real_lattice = self.real_lattice
        system_R.periodic = self.periodic.copy()
        system_R.wannier_centers_cart = self.wannier_centers_cart.copy()
        system_R.pointgroup = self.pointgroup
        system_R.force_internal_terms_only = self.force_internal_terms_only
        system_R.cell = self.cell.copy() if self.cell is not None else None

        for key, value in self.system_up._XX_R.items():
            print(f"setting matrix {key} from system_up")
            matrix = np.zeros((rvectors_merged.nRvec, self.num_wann, self.num_wann) + value.shape[3:], dtype=value.dtype)
            if key == 'Ham':
                matrix[rvectors_map_list[0]] += self.get_R_mat('Ham_SOC')
            matrix[rvectors_map_list[1], ::2, ::2] += self.system_up.get_R_mat(key)
            matrix[rvectors_map_list[2], 1::2, 1::2] += self.system_down.get_R_mat(key)
            system_R.set_R_mat(key, matrix)
        SS_R = np.zeros((rvectors_merged.nRvec, self.num_wann, self.num_wann, 3), dtype=complex)
        SS_R[rvectors_map_list[0]] = self.get_R_mat('SS')
        system_R.set_R_mat('SS', SS_R)
        return system_R
