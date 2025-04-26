import numpy as np
from ..utility import alpha_A, beta_A
from .data_K import Data_K
from ..system.system_R import System_R


class Data_K_R(Data_K, System_R):
    """ The Data_K class for systems defined by R-space matrix elements (Wannier/TB)"""

    def __init__(self, system, dK, grid,
                 _FF_antisym=False,
                 _CCab_antisym=False,
                 **parameters):
        super().__init__(system, dK, grid, **parameters)
        self._FF_antisym = _FF_antisym
        self._CCab_antisym = _CCab_antisym

        self.rvec = system.rvec.copy()
        self.rvec.set_fft_R_to_k(NK=self.NKFFT, num_wann=self.num_wann,
                          numthreads=self.npar_k if self.npar_k > 0 else 1,
                            fftlib=self.fftlib,
                            dK=dK)

        self.dK = dK
        self._bar_quantities = {}
        self._covariant_quantities = {}
        self._XX_R = {}

    @property
    def HH_K(self):
        return self.rvec.R_to_k(self.Ham_R, hermitian=True)

    def get_R_mat(self, key):
        memoize_R = ['Ham', 'AA', 'OO', 'BB', 'CC', 'CCab', 'GG']
        try:
            return self._XX_R[key]
        except KeyError:
            if key == 'OO':
                res = self._OO_R()
            elif key == 'CCab':
                res = self._CCab_R()
            elif key == 'FF':
                res = self._FF_R()
            else:
                X_R = self.system.get_R_mat(key)
                res = self.rvec.apply_expdK(X_R)
            if key in memoize_R:
                self.set_R_mat(key, res)
        return res


    def _OO_R(self):
        # We do not multiply by expdK, because it is already accounted in AA_R
        OO = self.rvec.derivative(self.get_R_mat('AA'))
        return OO[:, :, :, beta_A, alpha_A] - OO[:, :, :, alpha_A, beta_A]

    def _CCab_R(self):
        if self._CCab_antisym:
            CCab = np.zeros((self.rvec.nRvec, self.num_wann, self.num_wann, 3, 3), dtype=complex)
            CCab[:, :, :, alpha_A, beta_A] = -0.5j * self.get_R_mat('CC')
            CCab[:, :, :, beta_A, alpha_A] = 0.5j * self.get_R_mat('CC')
            return CCab
        else:
            return self.rvec.apply_expdK(self.system.get_R_mat('CCab'))

    def _FF_R(self):
        if self._FF_antisym:
            return -1j * self.rvec.derivative(self.get_R_mat('AA')).swapaxes(3, 4)
        # self.cRvec_wcc[:, :, :, :, None] * self.get_R_mat('AA')[:, :, :, None, :]
        else:
            return self.rvec.apply_expdK(self.system.get_R_mat('FF'))

    def Xbar(self, name, der=0):
        key = (name, der)
        if key not in self._bar_quantities:
            self._bar_quantities[key] = self._R_to_k_H(
                self.get_R_mat(name).copy(), der=der, hermitian=(name in ['AA', 'SS', 'OO']))
        return self._bar_quantities[key]

    def _R_to_k_H(self, XX_R, der=0, hermitian=True):
        """ converts from real-space matrix elements in Wannier gauge to
            k-space quantities in k-space.
            der [=0] - defines the order of comma-derivative
            hermitian [=True] - consider the matrix hermitian
            WARNING: the input matrix is destroyed, use np.copy to preserve it"""
        return self._rotate((self.rvec.R_to_k(XX_R, hermitian=hermitian, der=der))[self.select_K])

    def E_K_corners_tetra(self):
        vertices = self.Kpoint.vertices_fullBZ
        # we omit the wcc phases here, because they do not affect the energies
        expdK = np.exp(2j * np.pi * self.rvec.iRvec.dot(vertices.T)).T
        _Ecorners = np.zeros((self.nk, 4, self.num_wann), dtype=float)
        for iv, _exp in enumerate(expdK):
            _Ham_R = self.Ham_R[:, :, :] * _exp[:, None, None]
            _HH_K = self.rvec.R_to_k(_Ham_R, hermitian=True)
            _Ecorners[:, iv, :] = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
        self.select_bands(_Ecorners)
        Ecorners = np.zeros((self.nk_selected, 4, self.nb_selected), dtype=float)
        for iv, _exp in enumerate(expdK):
            Ecorners[:, iv, :] = _Ecorners[:, iv, :][self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners

    def E_K_corners_parallel(self):
        dK2 = self.Kpoint.dK_fullBZ / 2
        # we omit the wcc phases here, because they do not affect the energies
        expdK = np.exp(2j * np.pi * self.rvec.iRvec * dK2[None, :])
        expdK = np.array([1. / expdK, expdK])
        Ecorners = np.zeros((self.nk_selected, 2, 2, 2, self.nb_selected), dtype=float)
        for ix in 0, 1:
            for iy in 0, 1:
                for iz in 0, 1:
                    _expdK = expdK[ix, :, 0] * expdK[iy, :, 1] * expdK[iz, :, 2]
                    _Ham_R = self.Ham_R[:, :, :] * _expdK[:, None, None]
                    _HH_K = self.rvec.R_to_k(_Ham_R, hermitian=True)
                    E = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
                    Ecorners[:, ix, iy, iz, :] = E[self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners
