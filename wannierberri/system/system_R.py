import numpy as np
import os
import lazy_property
from functools import cached_property
from termcolor import cprint
from collections import defaultdict
import glob
import multiprocessing
from .system import System, pauli_xyz
from .sym_wann import SymWann
from ..__utility import alpha_A, beta_A, clear_cached, one2three
from ..symmetry import Symmetry, Group, TimeReversal
from .ws_dist import ws_dist_map


class System_R(System):
    """
        The base class for describing a system. Does not have its own constructor,
        please use the child classes, e.g  :class:`System_w90` or :class:`System_tb`


        Parameters
        -----------
        berry : bool
            set ``True`` to enable evaluation of external term in  Berry connection or Berry curvature and their
            derivatives.
        spin : bool
            set ``True`` if quantities derived from spin  will be used.
        morb : bool
            set ``True`` to enable calculation of external terms in orbital moment and its derivatives.
            Requires the ``.uHu`` file.
        periodic : [bool,bool,bool]
            set ``True`` for periodic directions and ``False`` for confined (e.g. slab direction for 2D systems). If less then 3 values provided, the rest are treated as ``False`` .
        SHCryoo : bool
            set ``True`` if quantities derived from Ryoo's spin-current elements will be used. (RPS 2019)
        SHCqiao : bool
            set ``True`` if quantities derived from Qiao's approximated spin-current elements will be used. (QZYZ 2018).
        use_ws : bool
            minimal distance replica selection method :ref:`sec-replica`.  equivalent of ``use_ws_distance`` in Wannier90.
            (Note: for :class:`System_tb` the method is not employed in the constructor. use `do_ws_dist()` if needed)
        _getFF : bool
            generate the FF_R matrix based on the uIu file. May be used for only testing so far. Default : ``{_getFF}``
        use_wcc_phase: bool
            using wannier centers in Fourier transform. Correspoinding to Convention I (True), II (False) in Ref."Tight-binding formalism in the context of the PythTB package". Default: ``{use_wcc_phase}``
        npar : int
            number of nodes used for parallelization in the `__init__` method. Default: `multiprocessing.cpu_count()`

        """

    def __init__(self,
                 berry=False,
                 morb=False,
                 spin=False,
                 SHCryoo=False,
                 SHCqiao=False,
                 use_ws=True,
                 use_wcc_phase=False,
                 npar=None,
                 _getFF=False,
                 **parameters):

        super().__init__(**parameters)
        self.use_ws = use_ws
        self.needed_R_matrices = set(['Ham'])
        self.npar = multiprocessing.cpu_count() if npar is None else npar
        self.use_wcc_phase = use_wcc_phase


        if morb:
            self.needed_R_matrices.update(['AA', 'BB', 'CC'])
        if berry:
            self.needed_R_matrices.add('AA')
        if spin:
            self.needed_R_matrices.add('SS')
        if _getFF:
            self.needed_R_matrices.add('FF')
        if SHCryoo:
            self.needed_R_matrices.update(['AA', 'SS', 'SA', 'SHA', 'SR', 'SH', 'SHR'])
        if SHCqiao:
            self.needed_R_matrices.update(['AA', 'SS', 'SR', 'SH', 'SHR'])
        self._XX_R = dict()


    def set_wannier_centers(self, wannier_centers_cart=None, wannier_centers_reduced=None):
        """
            set self.wannier_centers_cart. Only one of parameters should be provided.
            If both are None: self.wannier_centers_cart is set to zero.
        """
        lcart = (wannier_centers_cart is not None)
        lred = (wannier_centers_reduced is not None)
        if lred:
            _wannier_centers_cart = wannier_centers_reduced.dot(self.real_lattice)
            if lcart:
                assert abs(wannier_centers_cart - _wannier_centers_cart).max() < 1e-8
            else:
                self.wannier_centers_cart = _wannier_centers_cart
        else:
            if lcart:
                self.wannier_centers_cart = wannier_centers_cart
            else:
                self.wannier_centers_cart = np.zeros((self.num_wann, 3))
        self.clear_cached_wcc()


    def need_R_any(self, keys):
        """returns True is any of the listed matrices is needed in to be set

        keys : str or list of str
            'AA', 'BB', 'CC', etc
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        for k in keys:
            if k in self.needed_R_matrices:
                return True

    def get_R_mat(self, key):
        try:
            return self._XX_R[key]
        except KeyError:
            raise ValueError(f"The real-space matrix elements '{key}' are not set in the system," +
                             " but are required for the current calculation. please check parameters of the System() initializer")

    def has_R_mat(self, key):
        return (key in self._XX_R)

    def has_R_mat_any(self, keys):
        for k in keys:
            if self.has_R_mat(k):
                return True

    def set_R_mat(self, key, value, diag=False, R=None, reset=False, add=False):
        """
        Set real-space matrix specified by `key`. Either diagonal, specific R or full matix.  Useful for model calculations

        Parameters
        ----------
        key : str
            'SS', 'AA' , etc
        value : array
            * `array(num_wann,...)` if `diag=True` . Sets the diagonal part ( if `R` not set, `R=[0,0,0]`)
            * `array(num_wann,num_wann,..)`  matrix for `R` (`R` should be set )
            * `array(num_wann,num_wann,nRvec,...)` full spin matrix for all R

            `...` denotes the vector/tensor cartesian dimensions of the matrix element
        R : list(int)
            list of 3 integer values specifying R. if
        reset : bool
            allows to reset matrix if it is already set
        add : bool
            add matrix to the already existing

        """
        assert value.shape[0] == self.num_wann
        if diag:
            if R is None:
                R = [0, 0, 0]
            XX = np.zeros((self.num_wann, self.num_wann) + value.shape[1:], dtype=value.dtype)
            XX[np.arange(self.num_wann), np.arange(self.num_wann)] = value
            self.set_R_mat(key, XX, R=R, reset=reset, add=add)
        elif R is not None:
            XX = np.zeros((self.num_wann, self.num_wann, self.nRvec) + value.shape[2:], dtype=value.dtype)
            XX[:, :, self.iR(R)] = value
            self.set_R_mat(key, XX, reset=reset, add=add)
        else:
            if key in self._XX_R:
                if reset:
                    self._XX_R[key] = value
                elif add:
                    self._XX_R[key] += value
                else:
                    raise RuntimeError(f"setting {key} for the second time without explicit permission. smth is wrong")
            else:
                self._XX_R[key] = value

    @property
    def Ham_R(self):
        return self.get_R_mat('Ham')

    def symmetrize(self, proj, positions, atom_name, soc=False, magmom=None, DFT_code='qe', method="new"):
        """
        Symmetrize Wannier matrices in real space: Ham_R, AA_R, BB_R, SS_R,... , as well as Wannier centers


        Parameters
        ----------
        positions: array
            Positions of each atom.
        atom_name: list
            Name of each atom.
        proj: list
            Should be the same with projections card in relative Wannier90.win.

            eg: ``['Te: s','Te:p']``

            If there is hybrid orbital, grouping the other orbitals.

            eg: ``['Fe':sp3d2;t2g]`` Plese don't use ``['Fe':sp3d2;dxz,dyz,dxy]``

                ``['X':sp;p2]`` Plese don't use ``['X':sp;pz,py]``

            Note: If in `wannier90.win` file one sets several projections in one line like ``['Fe':d;sp3]``
            the actual order (as written to the `wannier90.nnkp` file) may be different. It is ordered by the orbital number l,
            and the hybrids are assigned negative numbers (e.g. for sp3 l=-3, see
            `Wannier90 user guide <https://raw.githubusercontent.com/wannier-developers/wannier90/v3.1.0/doc/compiled_docs/user_guide.pdf>`__
            chapter 3). So, the actual order will be ``['Fe':sp3;d]``. To  avoid confusion, it is recommended to put the different groups of projectons
            as separate lines of the `wannier90.win` file. See also `here <https://github.com/wannier-developers/wannier90/issues/463>`__
        soc: bool
            Spin orbital coupling.
        magmom: 2D array
            Magnetic momens of each atoms.
        DFT_code: str
            DFT code used : ``'qe'`` or ``'vasp'`` . This is needed, because vasp and qe have different orbitals arrangement with SOC.(grouped by spin or by orbital type)
        method : str
            `new` or `old`. They give same result but `new` is faster. `old` will be eventually removed.

        Notes:
            Works only with phase convention I (`use_wcc_phase=True`)
        """

        if not self.use_wcc_phase:
            raise NotImplementedError("Symmetrization is implemented only for convention I")

        symmetrize_wann = SymWann(
            num_wann=self.num_wann,
            lattice=self.real_lattice,
            positions=positions,
            atom_name=atom_name,
            projections=proj,
            iRvec=self.iRvec,
            XX_R=self._XX_R,
            soc=soc,
            wannier_centers_cart=self.wannier_centers_cart,
            magmom=magmom,
            use_wcc_phase=self.use_wcc_phase,
            DFT_code=DFT_code)

        print("Wannier Centers cart (raw):\n", self.wannier_centers_cart)
        print("Wannier Centers red: (raw):\n", self.wannier_centers_reduced)
        (self._XX_R, self.iRvec), self.wannier_centers_cart = symmetrize_wann.symmetrize(method=method)

        if self.has_R_mat('AA'):
            A_diag = self.get_R_mat('AA')[:, :, self.iR0].diagonal()
            if self.use_wcc_phase:
                A_diag_max = abs(A_diag).max()
                if A_diag_max > 1e-5:
                    print(f"WARNING : the maximal value of diagonal position matrix elements is {A_diag_max}. This may signal a problem")
                self.get_R_mat('AA')[np.arange(self.num_wann), np.arange(self.num_wann), self.iR0, :] = 0
        print("Wannier Centers cart (symmetrized):\n", self.wannier_centers_cart)
        print("Wannier Centers red: (symmetrized):\n", self.wannier_centers_reduced)
        self.clear_cached_R()
        self.clear_cached_wcc()
        self.symmetrize_info = dict(proj=proj, positions=positions, atom_name=atom_name, soc=soc, magmom=magmom,
                                    DFT_code=DFT_code)

    def check_periodic(self):
        exclude = np.zeros(self.nRvec, dtype=bool)
        for i, per in enumerate(self.periodic):
            if not per:
                sel = (self.iRvec[:, i] != 0)
                if np.any(sel):
                    print(
                        """WARNING : you declared your system as non-periodic along direction {i}, but there are {nrexcl} of total {nr} R-vectors with R[{i}]!=0.
        They will be excluded, please make sure you know what you are doing """.format(
                            i=i, nrexcl=sum(sel), nr=self.nRvec))
                    exclude[sel] = True
        if np.any(exclude):
            notexclude = np.logical_not(exclude)
            self.iRvec = self.iRvec[notexclude]
            for X in ['Ham', 'AA', 'BB', 'CC', 'SS', 'FF']:
                if X in self._XX_R:
                    self.set_R_mat(X, self.get_X_mat(X)[:, :, notexclude], reset=True)

    def set_spin(self, spins, axis=[0, 0, 1], **kwargs):
        """
        Set spins along axis in  SS(R=0).  Useful for model calculations.
        Note : The spin matrix is purely diagonal, so that <up | sigma_x | down> = 0
        For more cversatility use :func:`~wannierberri.system.System.set_R_mat`
        :func:`~wannierberri.system.System.set_spin_pairs`, :func:`~wannierberri.system.System.set_spin_from_code`

        Parameters
        ----------
        spin : one on the following
            1D `array(num_wann)` of `+1` or `-1` spins are along `axis`
        axis : array(3)
            spin quantization axis (if spin is a 1D array)
        **kwargs :
            optional arguments 'R', 'reset', 'add' see :func:`~wannierberri.system.System.set_R_mat`
        """
        spins = np.array(spins)
        if max(abs(spins) - 1) > 1e-3:
            print("WARNING : some of your spins are not +1 or -1, are you sure you want it like this?")
        axis = np.array(axis) / np.linalg.norm(axis)
        value = np.array([s * axis for s in spins], dtype=complex)
        self.set_R_mat(key='SS', value=value, diag=True, **kwargs)

    def set_spin_pairs(self, pairs):
        """set SS_R, assuming that each Wannier function is an eigenstate of Sz,
        Parameters
        ----------
        pairs : list of tuple
            list of pair of indices of bands ``[(up1,down1), (up2,down2), ..]``

        Notes:
        -------
        * For abinitio calculations this is a rough approximation, that may be used on own risk.
        See also :func:`~wannierberri.system.System.set_spin_from_code`
        """
        assert all(len(p) == 2 for p in pairs)
        all_states = np.array(sum((list(p) for p in pairs), []))
        assert np.all(all_states >= 0) and (np.all(all_states < self.num_wann)), (
            f"indices of states should be 0<=i<num_wann-{self.num_wann}, found {pairs}")
        assert len(set(all_states)) == len(all_states), "some states appear more then once in pairs"
        if len(pairs) < self.num_wann / 2:
            print(f"WARNING : number of spin pairs {len(pairs)} is less then num_wann/2 = {self.num_wann / 2}. " +
                  "For other states spin properties will be set to zero. are yoiu sure ?")
        SS_R0 = np.zeros((self.num_wann, self.num_wann, 3), dtype=complex)
        for i, j in pairs:
            dist = np.linalg.norm(self.wannier_centers_cart[i] - self.wannier_centers_cart[j])
            if dist > 1e-3:
                print(f"WARNING: setting spin pair for Wannier function {i} and {j}, distance between them {dist}")
            SS_R0[i, i] = pauli_xyz[0, 0]
            SS_R0[i, j] = pauli_xyz[0, 1]
            SS_R0[j, i] = pauli_xyz[1, 0]
            SS_R0[j, j] = pauli_xyz[1, 1]
            self.set_R_mat(key='SS', value=SS_R0, diag=False, R=[0, 0, 0], reset=True)

    def set_spin_from_code(self, DFT_code="qe"):
        """set SS_R, assuming that each Wannier function is an eigenstate of Sz,
         according to the ordering of the ab-initio code

        Parameters
        ----------
        DFT_code: str
            DFT code used :
                *  ``'qe'`` : if bands are grouped by orbital type, in each pair first comes spin-up,then spin-down
                *  ``'vasp'`` : if bands are grouped by spin : first come all spin-up, then all spin-down


        Notes:
        -------
        * This is a rough approximation, that may be used on own risk
        * The pure-spin character may be broken by maximal localization. Recommended to use `num_iter=0` in Wannier90
        * if your DFT code has a different name, but uses the same spin ordering as `qe` or `vasp` - set `DFT_code='qe'` or `DFT_code='vasp'` correspondingly
        * if your DFT code has a different spin ordering, use   :func:`~wannierberri.system.System.set_spin_pairs`

        """
        assert self.num_wann % 2 == 0, f"odd number of Wannier functions {self.num_wann} cannot be grouped into spin pairs"
        nw2 = self.num_wann // 2
        if DFT_code.lower() == 'vasp':
            pairs = [(i, i + nw2) for i in range(nw2)]
        elif DFT_code.lower() in ['qe', 'quantum_espresso', 'espresso']:
            pairs = [(2 * i, 2 * i + 1) for i in range(nw2)]
        self.set_spin_pairs(pairs)

    def getXX_only_wannier_centers(self, getSS=False):
        """return AA_R, BB_R, CC_R containing only the diagonal matrix elements, evaluated from
        the wannier_centers_cart variable (tight-binding approximation).
        In practice, it is useless because corresponding terms vanish with use_wcc_phase = True.
        but for testing may be used
        Used with pythtb, tbmodels, and also fplo, ASE until proper evaluation of matrix elements is implemented for them.
        """

        iR0 = self.iR0
        if 'AA' in self.needed_R_matrices:
            self.set_R_mat('AA', np.zeros((self.num_wann, self.num_wann, self.nRvec0, 3), dtype=complex))
            if not self.use_wcc_phase:
                for i in range(self.num_wann):
                    self.get_R_mat('AA')[i, i, iR0, :] = self.wannier_centers_cart[i]

        if 'BB' in self.needed_R_matrices:
            self.set_R_mat('BB', np.zeros((self.num_wann, self.num_wann, self.nRvec0, 3), dtype=complex))
            if not self.use_wcc_phase:
                for i in range(self.num_wann):
                    self.get_R_mat('BB')[i, i, iR0, :] = self.get_R_mat('AA')[i, i, iR0, :] * self.get_R_mat('Ham')[
                        i, i, iR0]

        if 'CC' in self.needed_R_matrices:
            self.set_R_mat('CC', np.zeros((self.num_wann, self.num_wann, self.nRvec0, 3), dtype=complex))

        if 'SS' in self.needed_R_matrices and getSS:
            raise NotImplementedError()

    def do_at_end_of_init(self, convert_convention=True):
        self.set_symmetry()
        self.check_periodic()
        if convert_convention:
            self.convention_II_to_I()
        print("Number of wannier functions:", self.num_wann)
        print("Number of R points:", self.nRvec)
        print("Recommended size of FFT grid", self.NKFFT_recommended)

    def do_ws_dist(self, mp_grid, wannier_centers_cart=None):
        """
        Perform the minimal-distance replica selection method
        As a side effect - it sets the variable _NKFFT_recommended to mp_grid

        Parameters:
        -----------
        wannier_centers_cart : array(float)
            Wannier centers used (if None -- use those already stored in the system)
        mp_grid : [nk1,nk2,nk3] or int
            size of Monkhorst-Pack frid used in ab initio calculation.
        """
        try:
            mp_grid = one2three(mp_grid)
            assert mp_grid is not None
        except AssertionError:
            raise ValueError(f"mp_greid should be one integer, of three integers. found {mp_grid}")
        self._NKFFT_recommended = mp_grid
        if wannier_centers_cart is None:
            wannier_centers_cart = self.wannier_centers_cart
        ws_map = ws_dist_map(
            self.iRvec, wannier_centers_cart, mp_grid, self.real_lattice, npar=self.npar)
        for key, val in self._XX_R.items():
            print("using ws_dist for {}".format(key))
            self.set_R_mat(key, ws_map(val), reset=True)
        self.iRvec = np.array(ws_map._iRvec_ordered, dtype=int)

    def to_tb_file(self, tb_file=None):
        if tb_file is None:
            tb_file = self.seedname + "_fromchk_tb.dat"
        f = open(tb_file, "w")
        f.write("written by wannier-berri form the chk file\n")
        cprint(f"writing TB file {tb_file}", 'green', attrs=['bold'])
        np.savetxt(f, self.real_lattice)
        f.write("{}\n".format(self.num_wann))
        f.write("{}\n".format(self.nRvec))
        for i in range(0, self.nRvec, 15):
            a = self.Ndegen[i:min(i + 15, self.nRvec)]
            f.write("  ".join("{:2d}".format(x) for x in a) + "\n")
        for iR in range(self.nRvec):
            f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.iRvec[iR])))
            f.write(
                "".join(
                    "{0:3d} {1:3d} {2:15.8e} {3:15.8e}\n".format(
                        m + 1, n + 1, self.Ham_R[m, n, iR].real * self.Ndegen[iR], self.Ham_R[m, n, iR].imag *
                        self.Ndegen[iR]) for n in range(self.num_wann) for m in range(self.num_wann)))
        if self.has_R_mat('AA'):
            for iR in range(self.nRvec):
                f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.iRvec[iR])))
                f.write(
                    "".join(
                        "{0:3d} {1:3d} ".format(m + 1, n + 1) + " ".join(
                            "{:15.8e} {:15.8e}".format(a.real, a.imag)
                            for a in self.get_R_mat('AA')[m, n, iR] * self.Ndegen[iR]) + "\n" for n in
                        range(self.num_wann)
                        for m in range(self.num_wann)))
        f.close()

    def _FFT_compatible(self, FFT, iRvec):
        "check if FFT is enough to fit all R-vectors"
        return np.unique(iRvec % FFT, axis=0).shape[0] == iRvec.shape[0]

    @property
    def NKFFT_recommended(self):
        "finds a minimal FFT grid on which different R-vectors do not overlap"
        if hasattr(self, '_NKFFT_recommended'):
            return self._NKFFT_recommended
        NKFFTrec = np.ones(3, dtype=int)
        for i in range(3):
            R = self.iRvec[:, i]
            if len(R[R > 0]) > 0:
                NKFFTrec[i] += R.max()
            if len(R[R < 0]) > 0:
                NKFFTrec[i] -= R.min()
        assert self._FFT_compatible(NKFFTrec, self.iRvec)
        return NKFFTrec

    @cached_property
    def cRvec(self):
        return self.iRvec.dot(self.real_lattice)

    @cached_property
    def cRvec_p_wcc(self):
        """
        With self.use_wcc_phase=True it is R+tj-ti. With self.use_wcc_phase=False it is R. [i,j,iRvec,a] (Cartesian)
        """
        if self.use_wcc_phase:
            return self.cRvec[None, None, :, :] + self.diff_wcc_cart[:, :, None, :]
        else:
            return self.cRvec[None, None, :, :]

    def clear_cached_R(self):
        clear_cached(self, ['cRvec', 'cRvec_p_wcc'])

    @cached_property
    def diff_wcc_cart(self):
        """
        With self.use_wcc_phase=True it is tj-ti. With self.use_wcc_phase=False it is 0. [i,j,a] (Cartesian)
        """
        wannier_centers = self.wannier_centers_cart
        return wannier_centers[None, :, :] - wannier_centers[:, None, :]

    @cached_property
    def diff_wcc_red(self):
        """
        With self.use_wcc_phase=True it is tj-ti. With self.use_wcc_phase=False it is 0. [i,j,a] (Reduced)
        """
        wannier_centers = self.wannier_centers_reduced
        return wannier_centers[None, :, :] - wannier_centers[:, None, :]

    @property
    def wannier_centers_cart_wcc_phase(self):
        "returns zero array if use_wcc_phase = False"
        if self.use_wcc_phase:
            return self.wannier_centers_cart
        else:
            return np.zeros((self.num_wann, 3), dtype=float)

    def clear_cached_wcc(self):
        clear_cached(self, ['diff_wcc_cart', 'cRvec_p_wcc', 'diff_wcc_red', "wannier_centers_reduced"])

    @cached_property
    def wannier_centers_reduced(self):
        return self.wannier_centers_cart.dot(np.linalg.inv(self.real_lattice))

    def convention_II_to_I(self):
        if self.use_wcc_phase:
            R_new = {}
            if self.wannier_centers_cart is None:
                raise ValueError("use_wcc_phase = True, but the wannier centers could not be determined")
            if self.has_R_mat('AA'):
                AA_R_new = np.copy(self.get_R_mat('AA'))
                AA_R_new[np.arange(self.num_wann), np.arange(self.num_wann), self.iR0, :] -= self.wannier_centers_cart
                R_new['AA'] = AA_R_new
            if self.has_R_mat('BB'):
                print("WARNING: orbital moment does not work with wcc_phase so far")
                BB_R_new = self.get_R_mat('BB').copy() - self.get_R_mat('Ham')[:, :, :,
                                                         None] * self.wannier_centers_cart[None, :, None, :]
                R_new['BB'] = BB_R_new
            if self.has_R_mat('CC'):
                print("WARNING: orbital moment does not work with wcc_phase so far")
                norm = np.linalg.norm(self.get_R_mat('CC') - self.conj_XX_R('CC'))
                assert norm < 1e-10, f"CC_R is not Hermitian, norm={norm}"
                assert self.has_R_mat('BB'), "if you use CC_R and use_wcc_phase=True, you need also BB_R"
                T = self.wannier_centers_cart[:, None, None, :, None] * self.get_R_mat('BB')[:, :, :, None, :]
                CC_R_new = self.get_R_mat('CC').copy() + 1.j * sum(
                    s * (
                            -T[:, :, :, a, b] -  # -t_i^a * B_{ij}^b(R)
                            self.conj_XX_R(T[:, :, :, b, a]) +  # - B_{ji}^a(-R)^*  * t_j^b
                            self.wannier_centers_cart[:, None, None, a] * self.Ham_R[:, :, :, None] *
                            self.wannier_centers_cart[None, :, None, b]  # + t_i^a*H_ij(R)t_j^b
                    ) for (s, a, b) in [(+1, alpha_A, beta_A), (-1, beta_A, alpha_A)])
                norm = np.linalg.norm(CC_R_new - self.conj_XX_R(CC_R_new))
                assert norm < 1e-10, f"CC_R after applying wcc_phase is not Hermitian, norm={norm}"
                R_new['CC'] = CC_R_new
            if self.has_R_mat_any(['SA', 'SHA', 'SR', 'SH', 'SHR']):
                raise NotImplementedError("use_wcc_phase=True for spin current matrix elements not implemented")

            for X in ['AA', 'BB', 'CC']:
                if self.has_R_mat(X):
                    self.set_R_mat(X, R_new[X], reset=True)

    @property
    def iR0(self):
        return self.iRvec.tolist().index([0, 0, 0])

    def iR(self, R):
        R = np.array(np.round(R), dtype=int).tolist()
        return self.iRvec.tolist().index([0, 0, 0])

    @lazy_property.LazyProperty
    def reverseR(self):
        """indices of R vectors that has -R in irvec, and the indices of the corresponding -R vectors."""
        mapping = np.all(self.iRvec[:, None, :] + self.iRvec[None, :, :] == 0, axis=2)
        # check if some R-vectors do not have partners
        notfound = np.where(np.logical_not(mapping.any(axis=1)))[0]
        for ir in notfound:
            print(f"WARNING : R[{ir}] = {self.iRvec[ir]} does not have a -R partner")
        # check if some R-vectors have more then 1 partner
        morefound = np.where(np.sum(mapping, axis=1) > 1)[0]
        if len(morefound > 0):
            raise RuntimeError(
                f"R vectors number {morefound} have more then one negative partner : "
                f"\n{self.iRvec[morefound]} \n{np.sum(mapping, axis=1)}")
        lst_R, lst_mR = [], []
        for ir1 in range(self.nRvec):
            ir2 = np.where(mapping[ir1])[0]
            if len(ir2) == 1:
                lst_R.append(ir1)
                lst_mR.append(ir2[0])
        lst_R = np.array(lst_R)
        lst_mR = np.array(lst_mR)
        # Check whether the result is correct
        assert np.all(self.iRvec[lst_R] + self.iRvec[lst_mR] == 0)
        return lst_R, lst_mR

    def conj_XX_R(self, XX_R):
        """ reverses the R-vector and takes the hermitian conjugate """
        if isinstance(XX_R, str):
            XX_R = self.get_R_mat(XX_R)
        XX_R_new = np.zeros(XX_R.shape, dtype=complex)
        lst_R, lst_mR = self.reverseR
        XX_R_new[:, :, lst_R] = XX_R[:, :, lst_mR]
        return XX_R_new.swapaxes(0, 1).conj()

    @property
    def nRvec(self):
        return self.iRvec.shape[0]

    def check_hermitian(self, XX):
        if XX in self._XX_R:
            _X = self.get_R_mat(XX).copy()
            assert (np.max(abs(_X - self.conj_XX_R(XX))) < 1e-8), f"{XX} should obey X(-R) = X(R)^+"
        else:
            print(f"{XX} is missing,nothing to check")

    def set_structure(self, positions, atom_labels, magnetic_moments=None):
        """
        Set atomic structure of the system.

        Parameters
        ----------
        positions : (num_atom, 3) array_like of float
            Atomic positions in fractional coordinates.
        atom_labels: (num_atom,) list
            labels (integer, string, etc.) to distinguish species.
        magnetic_moments: (num_atom, 3) array_like of float (optional)
            Magnetic moment vector of each atom.
        """
        if len(positions) != len(atom_labels):
            raise ValueError("length of positions and atom_labels must be the same")
        if magnetic_moments is not None:
            if len(magnetic_moments) != len(positions):
                raise ValueError("length of positions and magnetic_moments must be the same")
            if not all([len(x) for x in magnetic_moments]):
                raise ValueError("magnetic_moments must be a list of 3d vector")
        self.positions = positions
        self.atom_labels = atom_labels
        self.magnetic_moments = magnetic_moments

    def get_spglib_cell(self):
        """Returns the atomic structure as a cell tuple in spglib format"""
        try:
            # assign integer to self.atom_labels
            atom_labels_unique = list(set(self.atom_labels))
            atom_numbers = [atom_labels_unique.index(label) for label in self.atom_labels]
            if self.magnetic_moments is None:
                return (self.real_lattice, self.positions, atom_numbers)
            else:
                return (self.real_lattice, self.positions, atom_numbers, self.magnetic_moments)
        except AttributeError:
            raise AttributeError("set_structure must be called before get_spglib_cell")

    def set_symmetry_from_structure(self):
        """
        Set the symmetry group of the :class:`System`. Requires spglib to be installed.
        :meth:`System.set_structure` must be called in advance.

        For magnetic systems, symmetries involving time reversal are not detected because
        spglib does not support time reversal symmetry for noncollinear systems.
        """
        import spglib

        spglib_symmetry = spglib.get_symmetry(self.get_spglib_cell())
        symmetry_gen = []
        for isym, W in enumerate(spglib_symmetry["rotations"]):
            # spglib gives real-space rotations in reduced coordinates. Here,
            # 1) convert to Cartesian coordinates, and
            # 2) take transpose to go to reciprocal space.
            W = spglib_symmetry["rotations"][isym]
            Wcart = self.real_lattice.T @ W @ np.linalg.inv(self.real_lattice).T
            R = Wcart.T
            try:
                TR = spglib_symmetry['time_reversals'][isym]
                tr_found = True
            except KeyError:
                TR = False
                tr_found = False
            symmetry_gen.append(Symmetry(R, TR=TR))

        if self.magnetic_moments is None:
            symmetry_gen.append(TimeReversal)
        elif not tr_found:
            print(
                "WARNING: you specified magnetic moments but spglib did not detect symmetries involving time-reversal" +
                f"proobably it is because you have an old spglib version {spglib.__version__}" +
                "We suggest upgrading to spglib>=2.0.2")
        else:
            if not all([len(x) for x in self.magnetic_moments]):
                raise ValueError("magnetic_moments must be a list of 3d vector")
            print(
                "Warning: spglib does not find symmetries including time reversal "
                "operation.\nTo include such symmetries, use set_symmetry.")

        self.symgroup = Group(symmetry_gen, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice)

    def get_sparse(self, min_values={'Ham': 1e-3}):
        ret_dic = dict(
            real_lattice=self.real_lattice,
            wannier_centers_reduced=self.wannier_centers_reduced,
            matrices={},
            use_wcc_phase=self.use_wcc_phase
        )
        if hasattr(self, 'symmetrize_info'):
            ret_dic['symmetrize_info'] = self.symmetrize_info

        def array_to_dict(A, minval):
            A_tmp = abs(A.reshape(A.shape[:3] + (-1,))).max(axis=-1)
            wh = np.argwhere(A_tmp >= minval)
            dic = defaultdict(lambda: dict())
            for w in wh:
                iR = tuple(self.iRvec[w[2]])
                dic[iR][(w[0], w[1])] = A[tuple(w)]
            return dict(dic)

        for k, v in min_values.items():
            ret_dic['matrices'][k] = array_to_dict(self.get_R_mat(k), v)
        return ret_dic

    @cached_property
    def essential_properties(self):
        return ['num_wann', 'real_lattice', 'iRvec', 'periodic',
                'use_wcc_phase', 'is_phonon', 'wannier_centers_cart', 'symgroup']

    @cached_property
    def optional_properties(self):
        return ["positions", "magnetic_moments", "atom_labels"]


    def _R_mat_npz_filename(self, key):
        return "_XX_R_" + key + ".npz"

    def save_npz(self, path, extra_properties=[], exclude_properties=[], R_matrices=None, overwrite=True):
        """
        Save system to a directory of npz files
        Parameters
        ----------
        path : str
            path to saved files. If does not exist - will be created (unless overwrite=False)
        extra_properties : list of str
            names of properties which are not essential for reconstruction, but will also be saved
        exclude_properties : list of str
            dp not save certain properties - duse on your own risk
        R_matrices : list of str
            list of the R matrices, e.g. ```['Ham','AA',...]``` to be saved. if None: all R-matrices will be saved
        overwrite : bool
            if the directory already exiists, it will be overwritten
        """

        properties = [x for x in self.essential_properties + extra_properties if x not in exclude_properties]
        if R_matrices is None:
            R_matrices = list(self._XX_R.keys())

        try:
            os.makedirs(path, exist_ok=overwrite)
        except FileExistsError:
            raise FileExistsError(f"Directorry {path} already exists. To overwrite it set overwrite=True")

        for key in properties:
            print(f"saving {key}", end="")
            fullpath = os.path.join(path, key + ".npz")
            a = getattr(self, key)
            if key in ['symgroup']:
                np.savez(fullpath, **a.as_dict(), allow_pickle=False)
            else:
                np.savez(fullpath, a, allow_pickle=False)
            print(" - Ok!")
        for key in self.optional_properties:
            if key not in properties:
                fullpath = os.path.join(path, key + ".npz")
                if hasattr(self, key):
                    a = getattr(self, key)
                    np.savez(fullpath, a, allow_pickle=False)
        for key in R_matrices:
            print(f"saving {key}", end="")
            np.savez_compressed(os.path.join(path, self._R_mat_npz_filename(key)), self.get_R_mat(key))
            print(" - Ok!")

    def load_npz(self, path, load_all_XX_R=False, exclude_properties=[]):
        """
        Save system to a directory of npz files
        Parameters
        ----------
        path : str
            path to saved files. If does not exist - will be created (unless overwrite=False)
        load_all_XX_R : list of str
            load all matrices which were saved
        exclude_properties : list of str
            dp not save certain properties - duse on your own risk
        """
        all_files = glob.glob(os.path.join(path, "*.npz"))
        all_names = [os.path.splitext(os.path.split(x)[-1])[0] for x in all_files]
        properties = [x for x in all_names if not x.startswith('_XX_R_') and x not in exclude_properties]
        for key in properties:
            print(f"loading {key}", end="")
            a = np.load(os.path.join(path, key + ".npz"), allow_pickle=False)
            if key == 'symgroup':
                val = Group(dictionary=a)
            else:
                val = a['arr_0']
            setattr(self, key, val)
            print(" - Ok!")
        if load_all_XX_R:
            R_files = glob.glob(os.path.join(path, "_XX_R_*.npz"))
            R_matrices = [os.path.splitext(os.path.split(x)[-1])[0][6:] for x in R_files]
            self.needed_R_matrices.update(R_matrices)
        for key in self.needed_R_matrices:
            print(f"loading R_matrix {key}", end="")
            a = np.load(os.path.join(path, self._R_mat_npz_filename(key)), allow_pickle=False)['arr_0']
            self.set_R_mat(key, a)
            print(" - Ok!")


def ndim_R(key):
    """
    returns the number of cartesian dimensions of a matrix by key
    """
    if key in ["Ham"]:
        return 0
    elif key in ["AA", "BB", "CC", "SS", "SH"]:
        return 1
    elif key in ["SHA", "SA", "SR", "SHR"]:
        return 2
    else:
        raise ValueError(f"unknown matrix {key}")
