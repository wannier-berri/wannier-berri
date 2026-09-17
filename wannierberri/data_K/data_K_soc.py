from functools import cached_property
import numpy as np
from .data_K_R import Data_K_R
from ..system.system import num_cart_dim
from ..utility import cached_einsum


class Data_K_soc(Data_K_R):
    """
    Class for handling data_K with non-selfconsistent spin-orbit coupling (SOC).
    """

    def __init__(self, system, dK, grid, **parameters):
        super().__init__(system, dK=dK, grid=grid, **parameters)
        self.num_wann_scalar = system.num_wann_scalar
        self.data_K_up = Data_K_R(system=system.system_up, dK=dK, grid=grid,
                                  **parameters)
        self.nspin = system.nspin
        assert self.nspin in [1, 2], f"Unsupported number of spins: {self.nspin}"

        if system.nspin == 1:
            self.data_K_down = self.data_K_up
        elif system.nspin == 2:
            self.data_K_down = Data_K_R(system=system.system_down, dK=dK, grid=grid,
                                        **parameters)
        else:
            raise ValueError(f"Unsupported number of spins: {system.nspin}")
        self.has_soc = system.has_soc
        # if self.has_soc:
        #     soc_r_dk = self.rvec.apply_expdK(system.get_R_mat('Ham_SOC'))
        #     self.set_R_mat('soc', soc_r_dk)

    @cached_property
    def HH_K(self):
        """
        Returns the Hamiltonian in k-space, including spin-orbit coupling.
        """
        H = np.zeros((self.nk, self.num_wann, self.num_wann), dtype=complex)
        H[:, ::2, ::2] = self.data_K_up.HH_K
        H[:, 1::2, 1::2] = self.data_K_down.HH_K
        H += self.Hsoc()
        return H

    def Xbar(self, name, der=0):
        key = (name, der)
        if key not in self._bar_quantities:
            if name == "SS":
                Xbar = self.SS_W(der=der)
            elif name.startswith("S"):
                raise NotImplementedError(f"SHC-related operator {name} is not implemented for Data_K_soc., "
                                          "please, use kwargs_formula={'spin_current_type':'siple'} in the SHC calculator.")
            else:  # other not spin-related operators
                hermitian = (name in ['AA', 'OO', 'rotAA'])
                shape = (self.nk, ) + (self.num_wann, ) * 2 + (3,) * (der + num_cart_dim(name))
                Xbar = np.zeros(shape, dtype=complex)
                for i, datak in enumerate([self.data_K_up, self.data_K_down]):
                    Xbar[:, i::2, i::2] = datak.get_k_mat(name, hermitian=hermitian, der=der)
                if name == "Ham" and self.has_soc:
                    Xbar += self.Hsoc(der=der)
            Xbar = self._rotate(Xbar)
            self._bar_quantities[key] = Xbar
        return self._bar_quantities[key]


    def Hsoc(self, der=0):
        pauli_rotated = self.system.pauli_rotated
        dVsoc = [[None, None], [None, None]]
        dVsoc[0][0] = self.data_K_up.get_k_mat('dV_soc', der=der, hermitian=True)
        if self.system.nspin == 2:
            dVsoc[1][1] = self.data_K_down.get_k_mat('dV_soc', der=der, hermitian=True)
            dVsoc[0][1] = self.get_k_mat('dV_soc', der=der, hermitian=False)
            dVsoc[1][0] = dVsoc[0][1].conj().swapaxes(1, 2)
        else:
            dVsoc[1][1] = dVsoc[0][0]
            dVsoc[0][1] = dVsoc[0][0]
            dVsoc[1][0] = dVsoc[0][0]
        soc_k = np.zeros((self.nk, self.num_wann, self.num_wann) + (3,) * der, dtype=complex)
        # print(f"pauli_rotated shape = {pauli_rotated}, alpha_soc = {self.system.alpha_soc}")
        for i in range(2):
            for j in range(2):
                # print (f"i={i}, j={j}, dVsoc[{i}][{j}] shape = {dVsoc[i][j].shape}, pauli_rotated[{i},{j}] shape = {pauli_rotated[i,j,:].shape}, soc_k shape = {soc_k.shape}")
                soc_k[:, i::2, j::2] = cached_einsum("rmnc...,c->rmn...", dVsoc[i][j], pauli_rotated[i, j, :])
        # print("soc max abs = ", np.max(np.abs(soc_k)))
        return soc_k * self.system.alpha_soc

    def SS_W(self, der=0):
        # Spin operator
        rng = np.arange(0, self.num_wann, 2)
        pauli_rotated = self.system.pauli_rotated.reshape((1, 2, 2, 1, 1, 3) + (1,) * der)
        SSk = np.zeros((self.nk, self.num_wann, self.num_wann) + (3,) * (der + 1), dtype=complex)
        if der == 0:
            SSk[:, rng, rng, :] = pauli_rotated[:, 0, 0, ::, :]
            SSk[:, rng + 1, rng + 1, :] = pauli_rotated[:, 1, 1, :, :, :]
        if self.nspin == 2:
            overlap = self.get_k_mat('overlap_up_down', der=der).reshape((self.nk, self.num_wann // 2, self.num_wann // 2) + (1,) * (der + 1))
        else:
            if der == 0:
                overlap = np.eye(self.num_wann // 2, dtype=complex)[None, :, :].reshape((1, self.num_wann // 2, self.num_wann // 2) + (1,) * (der + 1))
            else:
                overlap = np.zeros((self.nk, self.num_wann // 2, self.num_wann // 2, 1) + (3,) * der)
        SSk[:, 0::2, 1::2, :] = overlap * pauli_rotated[:, 0, 1, :, :, :]
        SSk[:, 1::2, 0::2, :] = overlap.conj().swapaxes(1, 2) * pauli_rotated[:, 1, 0, :, :, :]
        return SSk



    def E_K_corners_tetra(self):
        # raise NotImplementedError("E_K_corners_tetra is not implemented for Data_K_soc. ")
        expdK_up = self.data_K_up.expdK_corners_tetra
        expdK_down = self.data_K_up.expdK_corners_tetra
        expdK_soc = self.expdK_corners_tetra if self.has_soc else None
        Ecorners = np.zeros((self.nk, 4, self.num_wann), dtype=float)
        for iv in range(4):
            Ecorners[:, iv, :] = self.E_K_corners(expdK_soc[iv], expdK_up[iv], expdK_down[iv])
        self.select_bands(Ecorners)
        Ecorners = Ecorners[self.select_K, :, :][:, :, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners

    def E_K_corners_parallel(self):
        expdK_up = self.data_K_up.expdK_corners_parallel
        expdK_down = self.data_K_down.expdK_corners_parallel
        if self.has_soc:
            expdK_soc = self.expdK_corners_parallel
        Ecorners = np.zeros((self.nk, 2, 2, 2, self.nbands), dtype=float)
        for ix in 0, 1:
            for iy in 0, 1:
                for iz in 0, 1:
                    _expdK_up = expdK_up[ix, :, 0] * expdK_up[iy, :, 1] * expdK_up[iz, :, 2]
                    _expdK_down = expdK_down[ix, :, 0] * expdK_down[iy, :, 1] * expdK_down[iz, :, 2]
                    _expdK_soc = expdK_soc[ix, :, 0] * expdK_soc[iy, :, 1] * expdK_soc[iz, :, 2]
                    Ecorners[:, ix, iy, iz, :] = self.E_K_corners(_expdK_soc, _expdK_up, _expdK_down)
        self.select_bands(Ecorners)
        Ecorners = Ecorners[self.select_K, :, :, :, :][:, :, :, :, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners


    def E_K_corners(self, expdK_soc, expdK_up, expdK_down):
        """
        Calculate the eigenvalues at the corners of the Brillouin zone.
        """
        Ham_up = self.data_K_up.rvec.R_to_k(self.data_K_up.Ham_R[:, :, :] * expdK_up[:, None, None], hermitian=True)
        dVsoc = [[None, None], [None, None]]
        dVsoc[0][0] = self.data_K_up.rvec.R_to_k(self.data_K_up.get_R_mat('dV_soc') * expdK_up[:, None, None, None], hermitian=True)
        if self.nspin == 2:
            Ham_dw = self.data_K_down.rvec.R_to_k(self.data_K_down.Ham_R[:, :, :] * expdK_down[:, None, None], hermitian=True)
            dVsoc[1][1] = self.data_K_down.rvec.R_to_k(self.data_K_down.get_R_mat('dV_soc') * expdK_down[:, None, None, None], hermitian=True)
            dVsoc[0][1] = self.rvec.R_to_k(self.get_R_mat('dV_soc') * expdK_soc[:, None, None, None], hermitian=False)
        else:
            Ham_dw = Ham_up
            dVsoc[1][1] = dVsoc[0][0]
            dVsoc[0][1] = dVsoc[0][0]
        dVsoc[1][0] = dVsoc[0][1].conj().swapaxes(1, 2)
        HH_K_full = np.zeros((self.nk, self.num_wann, self.num_wann), dtype=complex)
        HH_K_full[:, ::2, ::2] = Ham_up
        HH_K_full[:, 1::2, 1::2] = Ham_dw
        for i in range(2):
            for j in range(2):
                HH_K_full[:, i::2, j::2] += cached_einsum("rmnc...,c->rmn...", dVsoc[i][j], self.system.pauli_rotated[i, j, :]) * self.system.alpha_soc
        Ecorners = np.linalg.eigvalsh(HH_K_full)
        return Ecorners
