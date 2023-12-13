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
#                                                            #
# ------------------------------------------------------------

# TODO : maybe to make some lazy_property's not so lazy to save some memory
import numpy as np
import abc
import lazy_property
from .parallel import pool
from .system.system import System
from .system.system_kp import SystemKP
from .__utility import print_my_name_start, print_my_name_end, FFT_R_to_k, alpha_A, beta_A
from .grid import TetraWeights, TetraWeightsParal, get_bands_in_range, get_bands_below_range
from . import formula
from .grid import KpointBZparallel, KpointBZtetra
from .symmetry import transform_ident, transform_odd


def _rotate_matrix(X):
    return X[1].T.conj().dot(X[0]).dot(X[1])


def get_transform_Inv(name, der=0):
    """returns the transformation of the quantity  under inversion
    raises for unknown quantities"""
    if name in ['Ham', 'CC', 'FF', 'OO', 'SS']:  # even before derivative
        p = 0
    elif name in ['T_wcc']:  # odd before derivative
        p = 1
    elif name in ['D', 'AA', 'BB', 'CCab']:
        return None
    else:
        raise ValueError(f"parity under inversion unknown for {name}")
    if (p + der) % 2 == 1:
        return transform_odd
    else:
        return transform_ident


def get_transform_TR(name, der=0):
    """returns transformation of quantity is under TR, (after a real trace is taken, if appropriate)
    False otherwise
    raises ValueError for unknown quantities"""
    if name in ['Ham', 'T_wcc']:  # even before derivative
        p = 0
    elif name in ['CC', 'FF', 'OO', 'SS']:  # odd before derivative
        p = 1
    elif name in ['D', 'AA', 'BB', 'CCab']:
        return None
    else:
        raise ValueError(f"parity under TR unknown for {name}")
    if (p + der) % 2 == 1:
        return transform_odd
    else:
        return transform_ident


class _Data_K(System, abc.ABC):
    default_parameters = {
        # Those are not used at the moment, but will be restored (TODO):
        # 'frozen_max': -np.Inf,
        # 'delta_fz':0.1,
        'Emin': -np.Inf,
        'Emax': np.Inf,
        'use_wcc_phase': False,
        'fftlib': 'fftw',
        'npar_k': 1,
        'random_gauge': False,
        'degen_thresh_random_gauge': 1e-4,
        '_FF_antisym': False,
        '_CCab_antisym': False
    }

    __doc__ = """
    class to store many data calculated on a specific FFT grid.
    The stored data can be used to evaluate many quantities.
    Is destroyed after  everything is evaluated for the FFT grid

    Parameters
    -----------
    random_gauge : bool
        applies random unitary rotations to degenerate states. Needed only for testing, to make sure that gauge covariance is preserved. Default: ``{random_gauge}``
    degen_thresh_random_gauge : float
        threshold to consider bands as degenerate for random_gauge Default: ``{degen_thresh_random_gauge}``
    fftlib :  str
        library used to perform fft : 'fftw' (defgault) or 'numpy' or 'slow'
    """.format(**default_parameters)

    # Those are not used at the moment , but will be restored (TODO):
    #    frozen_max : float
    #        position of the upper edge of the frozen window. Used in the evaluation of orbital moment. But not necessary.
    #        If not specified, attempts to read this value from system. Othewise set to  ``{frozen_max}``
    #    delta_fz:float
    #        size of smearing for B matrix with frozen window, from frozen_max-delta_fz to frozen_max. Default: ``{delta_fz}``

    def __init__(self, system, dK, grid, Kpoint=None, **parameters):
        self.system = system
        self.set_parameters(**parameters)

        self.grid = grid
        self.NKFFT = grid.FFT
        self.select_K = np.ones(self.nk, dtype=bool)
        #        self.findif = grid.findif
        self.real_lattice = system.real_lattice
        self.num_wann = self.system.num_wann
        self.Kpoint = Kpoint
        self.nkptot = self.NKFFT[0] * self.NKFFT[1] * self.NKFFT[2]

        self.poolmap = pool(self.npar_k)[0]

        self.dK = dK
        self._bar_quantities = {}
        self._covariant_quantities = {}

    def set_parameters(self, **parameters):
        for param in self.default_parameters:
            if param in parameters:
                vars(self)[param] = parameters[param]
            else:
                vars(self)[param] = self.default_parameters[param]
        for param in parameters:
            if param not in self.default_parameters:
                print(f"WARNING: parameter {param} was passed to data_K, which is not recognised")

    ###########################################
    #   Now the **_R objects are evaluated only on demand
    # - as Lazy_property (if used more than once)
    #   as property   - iif used only once
    #   let's write them explicitly, for better code readability
    ###########################

    @property
    def is_phonon(self):
        return self.system.is_phonon

    ###############################################################

    ###########
    #  TOOLS  #
    ###########

    def _rotate(self, mat):
        print_my_name_start()
        assert mat.ndim > 2
        if mat.ndim == 3:
            return np.array(self.poolmap(_rotate_matrix, zip(mat, self.UU_K)))
        else:
            for i in range(mat.shape[-1]):
                mat[..., i] = self._rotate(mat[..., i])
            return mat

    #####################
    #  Basic variables  #
    #####################

    @lazy_property.LazyProperty
    def nbands(self):
        return self.num_wann

    @lazy_property.LazyProperty
    def kpoints_all(self):
        return (self.grid.points_FFT + self.dK[None]) % 1

    @lazy_property.LazyProperty
    def nk(self):
        return np.prod(self.NKFFT)

    @lazy_property.LazyProperty
    def tetraWeights(self):
        if isinstance(self.Kpoint, KpointBZparallel):
            return TetraWeightsParal(eCenter=self.E_K, eCorners=self.E_K_corners_parallel())
        elif isinstance(self.Kpoint, KpointBZtetra):
            return TetraWeights(eCenter=self.E_K, eCorners=self.E_K_corners_tetra())
        else:
            raise RuntimeError()

    def get_bands_in_range_groups_ik(self, ik, emin, emax, degen_thresh=-1, degen_Kramers=False, sea=False,
                                     Emin=-np.Inf, Emax=np.Inf):
        bands_in_range = get_bands_in_range(
            emin, emax, self.E_K[ik], degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)
        weights = {(ib1, ib2): self.E_K[ik, ib1:ib2].mean() for ib1, ib2 in bands_in_range}
        if sea:
            bandmax = get_bands_below_range(emin, self.E_K[ik])
            if len(bands_in_range) > 0:
                bandmax = min(bandmax, bands_in_range[0][0])
            if bandmax > 0:
                weights[(0, bandmax)] = -np.Inf
        return weights

    def get_bands_in_range_groups(self, emin, emax, degen_thresh=-1, degen_Kramers=False, sea=False, Emin=-np.Inf,
                                  Emax=np.Inf):
        res = []
        for ik in range(self.nk):
            res.append(self.get_bands_in_range_groups_ik(ik, emin, emax, degen_thresh, degen_Kramers, sea, Emin=Emin,
                                                         Emax=Emax))
        return res

    ###################################################
    #  Basic variables and their standard derivatives #
    ###################################################

    def select_bands(self, energies):
        if hasattr(self, 'bands_selected'):
            return
        energies = energies.reshape((energies.shape[0], -1, energies.shape[-1]))
        select = np.any(energies > self.Emin, axis=1) * np.any(energies < self.Emax, axis=1)
        self.select_K = np.any(select, axis=1)
        self.select_B = np.any(select, axis=0)
        self.nk_selected = self.select_K.sum()
        self.nb_selected = self.select_B.sum()
        self.bands_selected = True

    @lazy_property.LazyProperty
    def E_K(self):
        print_my_name_start()
        EUU = self.poolmap(np.linalg.eigh, self.HH_K)
        E_K = self.phonon_freq_from_square(np.array([euu[0] for euu in EUU]))
        #        print ("E_K = ",E_K.min(), E_K.max(), E_K.mean())
        self.select_bands(E_K)
        self._UU = np.array([euu[1] for euu in EUU])[self.select_K, :][:, self.select_B]
        print_my_name_end()
        return E_K[self.select_K, :][:, self.select_B]

    # evaluate the energies in the corners of the parallelepiped, in order to use tetrahedron method

    def phonon_freq_from_square(self, E):
        """takes  sqrt(|E|)*sign(E) for phonons, returns input for electrons"""
        if self.is_phonon:
            e = np.sqrt(np.abs(E))
            e[E < 0] = -e[E < 0]
            return e
        else:
            return E

    @property
    @abc.abstractmethod
    def HH_K(self):
        """returns Wannier Hamiltonian for all points of the FFT grid"""

    @lazy_property.LazyProperty
    def delE_K(self):
        print_my_name_start()
        delE_K = np.einsum("klla->kla", self.Xbar('Ham', 1))
        check = np.abs(delE_K).imag.max()
        if check > 1e-10:
            raise RuntimeError(f"The band derivatives have considerable imaginary part: {check}")
        return delE_K.real

    def covariant(self, name, commader=0, gender=0, save=True):
        assert commader * gender == 0, "cannot mix comm and generalized derivatives"
        key = (name, commader, gender)
        if key not in self._covariant_quantities:
            if gender == 0:
                res = formula.Matrix_ln(
                    self.Xbar(name, commader),
                    transformTR=get_transform_TR(name, commader),
                    transformInv=get_transform_Inv(name, commader),
                )
            elif gender == 1:
                if name == 'Ham':
                    res = self.V_covariant
                else:
                    res = formula.Matrix_GenDer_ln(
                        self.covariant(name),
                        self.covariant(name, commader=1),
                        self.Dcov,
                        transformTR=get_transform_TR(name, gender),
                        transformInv=get_transform_Inv(name, gender)
                    )
            else:
                raise NotImplementedError()
            if not save:
                return res
            else:
                self._covariant_quantities[key] = res
        return self._covariant_quantities[key]

    @property
    def V_covariant(self):

        class V(formula.Matrix_ln):

            def __init__(self, matrix):
                super().__init__(matrix, transformTR=transform_odd, transformInv=transform_odd)

            def ln(self, ik, inn, out):
                return np.zeros((len(out), len(inn), 3), dtype=complex)

        return V(self.Xbar('Ham', der=1))

    @lazy_property.LazyProperty
    def Dcov(self):
        return formula.covariant.Dcov(self)

    @lazy_property.LazyProperty
    def dEig_inv(self):
        dEig_threshold = 1.e-7
        dEig = self.E_K[:, :, None] - self.E_K[:, None, :]
        select = abs(dEig) < dEig_threshold
        dEig[select] = dEig_threshold
        dEig = 1. / dEig
        dEig[select] = 0.
        return dEig

    #    defining sets of degenerate states - needed only for testing with random_gauge

    @lazy_property.LazyProperty
    def degen(self):
        A = [np.where(E[1:] - E[:-1] > self.degen_thresh_random_gauge)[0] + 1 for E in self.E_K]
        A = [[
                 0,
             ] + list(a) + [len(E)] for a, E in zip(A, self.E_K)]
        return [[(ib1, ib2) for ib1, ib2 in zip(a, a[1:]) if ib2 - ib1 > 1] for a in A]

    @lazy_property.LazyProperty
    def UU_K(self):
        print_my_name_start()
        self.E_K
        # the following is needed only for testing :
        if self.random_gauge:
            from scipy.stats import unitary_group
            cnt = 0
            s = 0
            for ik, deg in enumerate(self.true):
                for ib1, ib2 in deg:
                    self._UU[ik, :, ib1:ib2] = self._UU[ik, :, ib1:ib2].dot(unitary_group.rvs(ib2 - ib1))
                    cnt += 1
                    s += ib2 - ib1
        print_my_name_end()
        return self._UU

    @lazy_property.LazyProperty
    def D_H(self):
        return -self.Xbar('Ham', 1) * self.dEig_inv[:, :, :, None]

    @lazy_property.LazyProperty
    def A_H(self):
        '''Generalized Berry connection matrix, A^(H) as defined in eqn. (25) of 10.1103/PhysRevB.74.195118.'''
        return self.Xbar('AA') + 1j * self.D_H

    @property
    def A_H_internal(self):
        '''Generalized Berry connection matrix, A^(H) as defined in eqn. (25) of 10.1103/PhysRevB.74.195118. only internal term'''
        return 1j * self.D_H


#########################################################################################################################################


class Data_K_R(_Data_K):
    """ The Data_K class for systems defined by R-space matrix elements (Wannier/TB)"""

    def __init__(self, system, dK, grid, **parameters):
        super().__init__(system, dK, grid, **parameters)

        self.cRvec_wcc = self.system.cRvec_p_wcc

        self.fft_R_to_k = FFT_R_to_k(
            self.system.iRvec,
            self.NKFFT,
            self.num_wann,
            numthreads=self.npar_k if self.npar_k > 0 else 1,
            lib=self.fftlib)

        self.expdK = np.exp(2j * np.pi * self.system.iRvec.dot(dK))
        self.dK = dK
        self._bar_quantities = {}
        self._covariant_quantities = {}
        self._XX_R = {}

    @property
    def HH_K(self):
        return self.fft_R_to_k(self.Ham_R, hermitean=True)

    def get_R_mat(self, key):
        memoize_R = ['Ham', 'AA', 'OO', 'BB', 'CC', 'CCab']
        try:
            return self._XX_R[key]
        except KeyError:
            if key == 'OO':
                res = self._OO_R()
            elif key == 'CCab':
                res = self._CCab_R()
            elif key == 'FF':
                res = self._FF_R()
            elif key == 'T_wcc':
                res = self._T_wcc_R()
            else:
                X_R = self.system.get_R_mat(key)
                shape = [1] * X_R.ndim
                shape[2] = self.expdK.shape[0]
                res = X_R * self.expdK.reshape(shape)
            if key in memoize_R:
                self.set_R_mat(key, res)
        return res

    #  this is a bit ovberhead, but to maintain uniformity of the code let's use this
    def _T_wcc_R(self):
        nw = self.num_wann
        res = np.zeros((nw, nw, self.system.nRvec, 3), dtype=complex)
        res[np.arange(nw), np.arange(nw), self.system.iR0, :] = self.system.wannier_centers_cart_wcc_phase
        return res

    def _OO_R(self):
        # We do not multiply by expdK, because it is already accounted in AA_R
        return 1j * (
                self.cRvec_wcc[:, :, :, alpha_A] * self.get_R_mat('AA')[:, :, :, beta_A] -
                self.cRvec_wcc[:, :, :, beta_A] * self.get_R_mat('AA')[:, :, :, alpha_A])

    def _CCab_R(self):
        if self._CCab_antisym:
            CCab = np.zeros((self.num_wann, self.num_wann, self.system.nRvec, 3, 3), dtype=complex)
            CCab[:, :, :, alpha_A, beta_A] = -0.5j * self.get_R_mat('CC')
            CCab[:, :, :, beta_A, alpha_A] = 0.5j * self.get_R_mat('CC')
            return CCab
        else:
            return self.system.get_R_mat('CCab') * self.expdK[None, None, :, None, None]

    def _FF_R(self):
        if self._FF_antisym:
            return self.cRvec_wcc[:, :, :, :, None] * self.get_R_mat('AA')[:, :, :, None, :]
        else:
            return self.system.get_R_mat('FF') * self.expdK[None, None, :, None, None]

    def Xbar(self, name, der=0):
        key = (name, der)
        if key not in self._bar_quantities:
            self._bar_quantities[key] = self._R_to_k_H(
                self.get_R_mat(name).copy(), der=der, hermitean=(name in ['AA', 'SS', 'OO']))
        return self._bar_quantities[key]

    def _R_to_k_H(self, XX_R, der=0, hermitean=True):
        """ converts from real-space matrix elements in Wannier gauge to
            k-space quantities in k-space.
            der [=0] - defines the order of comma-derivative
            hermitean [=True] - consider the matrix hermitean
            WARNING: the input matrix is destroyed, use np.copy to preserve it"""

        for i in range(der):
            shape_cR = np.shape(self.cRvec_wcc)
            XX_R = 1j * XX_R.reshape((XX_R.shape) + (1,)) * self.cRvec_wcc.reshape(
                (shape_cR[0], shape_cR[1], self.system.nRvec) + (1,) * len(XX_R.shape[3:]) + (3,))
        return self._rotate((self.fft_R_to_k(XX_R, hermitean=hermitean))[self.select_K])

    def E_K_corners_tetra(self):
        vertices = self.Kpoint.vertices_fullBZ
        expdK = np.exp(2j * np.pi * self.system.iRvec.dot(
            vertices.T)).T  # we omit the wcc phases here, because they do not affect the energies
        _Ecorners = np.zeros((self.nk, 4, self.num_wann), dtype=float)
        for iv, _exp in enumerate(expdK):
            _Ham_R = self.Ham_R[:, :, :] * _exp[None, None, :]
            _HH_K = self.fft_R_to_k(_Ham_R, hermitean=True)
            _Ecorners[:, iv, :] = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
        self.select_bands(_Ecorners)
        Ecorners = np.zeros((self.nk_selected, 4, self.nb_selected), dtype=float)
        for iv, _exp in enumerate(expdK):
            Ecorners[:, iv, :] = _Ecorners[:, iv, :][self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        print_my_name_end()
        #        print ("Ecorners",Ecorners.min(),Ecorners.max(),Ecorners.mean())
        return Ecorners

    def E_K_corners_parallel(self):
        dK2 = self.Kpoint.dK_fullBZ / 2
        expdK = np.exp(
            2j * np.pi * self.system.iRvec *
            dK2[None, :])  # we omit the wcc phases here, because they do not affect the energies
        expdK = np.array([1. / expdK, expdK])
        Ecorners = np.zeros((self.nk_selected, 2, 2, 2, self.nb_selected), dtype=float)
        for ix in 0, 1:
            for iy in 0, 1:
                for iz in 0, 1:
                    _expdK = expdK[ix, :, 0] * expdK[iy, :, 1] * expdK[iz, :, 2]
                    _Ham_R = self.Ham_R[:, :, :] * _expdK[None, None, :]
                    _HH_K = self.fft_R_to_k(_Ham_R, hermitean=True)
                    E = np.array(self.poolmap(np.linalg.eigvalsh, _HH_K))
                    Ecorners[:, ix, iy, iz, :] = E[self.select_K, :][:, self.select_B]
        Ecorners = self.phonon_freq_from_square(Ecorners)
        print_my_name_end()
        #        print ("Ecorners",Ecorners.min(),Ecorners.max(),Ecorners.mean())
        return Ecorners


class Data_K_k(_Data_K):
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
        print_my_name_end()
        #        print ("Ecorners",Ecorners.min(),Ecorners.max(),Ecorners.mean())
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
        print_my_name_end()
        #        print ("Ecorners",Ecorners.min(),Ecorners.max(),Ecorners.mean())
        return Ecorners


def get_data_k(system, dK, grid, **parameters):
    if isinstance(system, SystemKP):
        return Data_K_k(system, dK, grid, **parameters)
    else:
        return Data_K_R(system, dK, grid, **parameters)
