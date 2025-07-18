from functools import cached_property
import numpy as np
from .data_K_R import Data_K_R


class Data_K_soc(Data_K_R):
    """
    Class for handling data_K with non-selfconsistent spin-orbit coupling (SOC).
    """

    def __init__(self, system, dK, grid, **parameters):
        super().__init__(system, dK, grid, **parameters)
        self.num_wann_scalar = system.num_wann_scalar
        self.data_K_up = Data_K_R(system.system_up, dK, grid, **parameters)
        self.rvec_up = system.system_up.rvec.copy()

        self.rvec_up.set_fft_R_to_k(NK=self.NKFFT, num_wann=self.num_wann_scalar,
                          numthreads=self.npar_k if self.npar_k > 0 else 1,
                            fftlib=self.fftlib,
                            dK=dK)

        if system.up_down_same:
            self.data_K_down = self.data_K_up
            self.rvec_down = self.rvec_up
        else:
            self.data_K_down = Data_K_R(system.system_down, dK, grid, **parameters)
            self.rvec_down = system.system_down.rvec.copy()
            self.rvec_down.set_fft_R_to_k(NK=self.NKFFT, num_wann=self.num_wann_scalar,
                          numthreads=self.npar_k if self.npar_k > 0 else 1,
                            fftlib=self.fftlib,
                            dK=dK)
        soc_r_dk = self.rvec.apply_expdK(system.soc_R)
        self.set_R_mat('soc', soc_r_dk)

    @cached_property
    def HH_K(self):
        """
        Returns the Hamiltonian in k-space, including spin-orbit coupling.
        """
        H = np.zeros((self.nk, self.num_wann, self.num_wann), dtype=complex)
        H[:, ::2, ::2] = self.data_K_up.HH_K
        H[:, 1::2, 1::2] = self.data_K_down.HH_K
        H += self.rvec.R_to_k(self.get_R_mat('soc'), hermitian=True)
        return H

    def Xbar(self, name, der=0):
        key = (name, der)
        if key not in self._bar_quantities:
            if key == ("SS", 0):
                SS_K_W = np.array([self.SS_W] * self.nk)
                Xbar = self._rotate(SS_K_W[self.select_K])
            elif name == "SS":
                Xbar = np.zeros((self.nk, self.num_wann, self.num_wann, 3))
            elif name.startswith("S"):
                raise NotImplementedError(f"SHC-related operator {name} is not implemented for Data_K_soc., "
                                          "please, use kwargs_formula={'spin_current_type':'siple'} in the SHC calculator.")
            else:  # name in ["Ham", "AA", "OO"]:
                Xbar_up = self.data_K_up.Xbar(name, der)
                Xbar_down = self.data_K_down.Xbar(name, der)
                shape = Xbar_up.shape
                shape = (shape[0], self.num_wann, self.num_wann) + shape[3:]
                Xbar = np.zeros(shape, dtype=complex)
                Xbar[:, ::2, ::2] = Xbar_up
                Xbar[:, 1::2, 1::2] = Xbar_down
                if name == "Ham":
                    Xbar += self.rvec.R_to_k(self.get_R_mat('soc'), der=der, hermitian=True)
            self._bar_quantities[key] = Xbar

        return self._bar_quantities[key]

    # TODO : move to system
    @cached_property
    def SS_W(self):
        SS_W = np.zeros((self.num_wann, self.num_wann, 3), dtype=complex)
        for i in range(self.num_wann_scalar):
            SS_W[2 * i:2 * i + 2, 2 * i:2 * i + 2, :] = self.system.S_ssa
        return SS_W

    # Additional methods specific to spin-orbit coupling can be added here.

    def E_K_corners_tetra(self):
        # raise NotImplementedError("E_K_corners_tetra is not implemented for Data_K_soc. ")
        expdK_up = self.data_K_up.expdK_corners_parallel
        expdK_down = self.data_K_up.expdK_corners_parallel
        expdK = self.expdK_corners_parallel
        _Ecorners = np.zeros((self.nk, 4, self.num_wann), dtype=float)
        for iv in range(4):
            _HH_K_full = np.zeros((self.nk, self.num_wann, self.num_wann), dtype=complex)
            _Ham_R = self.data_K_up.Ham_R[:, :, :] * expdK_up[iv][:, None, None]
            _HH_K_full[:, ::2, ::2] = self.rvec.R_to_k(_Ham_R, hermitian=True)
            _Ham_R = self.data_K_down.Ham_R[:, :, :] * expdK_down[iv][:, None, None]
            _HH_K_full[:, 1::2, 1::2] = self.rvec.R_to_k(_Ham_R, hermitian=True)
            _Ham_R = self.get_R_mat('soc') * expdK[iv][:, None, None]
            _HH_K_full += self.rvec.R_to_k(_Ham_R, hermitian=True)
            _Ecorners[:, iv, :] = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K_full))
        self.select_bands(_Ecorners)
        Ecorners = np.zeros((self.nk_selected, 4, self.nb_selected), dtype=float)
        for iv in range(4):
            Ecorners[:, iv, :] = _Ecorners[:, iv, :][self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners

    def E_K_corners_parallel(self):
        expdK_up = self.data_K_up.expdK_corners_parallel
        expdK_down = self.data_K_up.expdK_corners_parallel
        expdK = self.expdK_corners_parallel
        Ecorners = np.zeros((self.nk_selected, 2, 2, 2, self.nb_selected), dtype=float)
        for ix in 0, 1:
            for iy in 0, 1:
                for iz in 0, 1:
                    _HH_K_full = np.zeros((self.nk, self.num_wann, self.num_wann), dtype=complex)
                    # block from up-spin
                    _expdK = expdK_up[ix, :, 0] * expdK_up[iy, :, 1] * expdK_up[iz, :, 2]
                    _Ham_R = self.data_K_up.Ham_R[:, :, :] * _expdK[:, None, None]
                    _HH_K_full[:, ::2, ::2] = self.data_K_up.rvec.R_to_k(_Ham_R, hermitian=True)
                    # block from down-spin
                    _expdK = expdK_down[ix, :, 0] * expdK_down[iy, :, 1] * expdK_down[iz, :, 2]
                    _Ham_R = self.data_K_down.Ham_R[:, :, :] * _expdK[:, None, None]
                    _HH_K_full[:, 1::2, 1::2] = self.data_K_down.rvec.R_to_k(_Ham_R, hermitian=True)
                    # block from SOC
                    _expdK = expdK[ix, :, 0] * expdK[iy, :, 1] * expdK[iz, :, 2]
                    _Ham_R = self.get_R_mat('soc') * _expdK[:, None, None]
                    _HH_K_full += self.rvec.R_to_k(_Ham_R, hermitian=True)

                    # calculate eigenvalues
                    E = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K_full))
                    Ecorners[:, ix, iy, iz, :] = E[self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners
