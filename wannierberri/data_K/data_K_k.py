import numpy as np
from .data_K import Data_K


class Data_K_k(Data_K):
    """ The Data_K class for systems defined by k-dependent Hamiltonians  (kp)"""

    @property
    def HH_K(self):
        return np.array([self.system.Ham(k) for k in self.kpoints_all])

    def Xbar(self, name, der=0):
        key = (name, der)
        if name != 'Ham':
            raise ValueError(f'quantity {name} is not defined for a kp model')
        if key not in self._bar_quantities:
            if der == 0:
                raise RuntimeError("Why is `Ham` called through Xbar, are you sure?")
                X = self.HH_K
            else:
                if der == 1:
                    fun = self.system.derHam
                elif der == 2:
                    fun = self.system.der2Ham
                elif der == 3:
                    fun = self.system.der3Ham
                X = np.array([fun(k) for k in self.kpoints_all])
            self._bar_quantities[key] = self._rotate(X)[self.select_K]
        return self._bar_quantities[key]

    def E_K_corners_tetra(self):
        vertices = self.Kpoint.vertices_fullBZ
        _Ecorners = np.zeros((self.nk, 4, self.num_wann), dtype=float)
        for iv, v in enumerate(vertices):
            _HH_K = np.array([self.system.Ham(k + v) for k in self.kpoints_all])
            _Ecorners[:, iv, :] = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
        self.select_bands(_Ecorners)
        Ecorners = np.zeros((self.nk_selected, 4, self.nb_selected), dtype=float)
        for iv, v in enumerate(vertices):
            Ecorners[:, iv, :] = _Ecorners[:, iv, :][self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners

    def E_K_corners_parallel(self):
        dK = self.Kpoint.dK_fullBZ
        Ecorners = np.zeros((self.nk_selected, 2, 2, 2, self.nb_selected), dtype=float)
        for ix in 0, 1:
            for iy in 0, 1:
                for iz in 0, 1:
                    v = (np.array([ix, iy, iz]) - 0.5) * dK
                    _HH_K = np.array([self.system.Ham(k + v) for k in self.kpoints_all])
                    E = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
                    Ecorners[:, ix, iy, iz, :] = E[self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        return Ecorners
