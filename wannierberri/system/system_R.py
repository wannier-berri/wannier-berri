import copy
import warnings
import numpy as np
import os
from functools import cached_property
from collections import defaultdict
import glob
import multiprocessing

from ..fourier.rvectors import Rvectors
from .system import System, pauli_xyz
from ..utility import clear_cached, one2three
from ..symmetry.point_symmetry import PointSymmetry, PointGroup, TimeReversal
from ..symmetry.wyckoff_position import split_into_orbits


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
        OSD : bool
            set ``True`` to enable calculation of external terms in orbital contribution to Optical Spatial dispersion
            Requires the `uIu`` and ``.uHu`` files.
        periodic : [bool,bool,bool]
            set ``True`` for periodic directions and ``False`` for confined (e.g. slab direction for 2D systems). If less then 3 values provided, the rest are treated as ``False`` .
        SHCryoo : bool
            set ``True`` if quantities derived from Ryoo's spin-current elements will be used. (RPS 2019)
        SHCqiao : bool
            set ``True`` if quantities derived from Qiao's approximated spin-current elements will be used. (QZYZ 2018).
        _getFF : bool
            generate the FF_R matrix based on the uIu file. May be used for only testing so far. Default : ``{_getFF}``
        npar : int
            number of nodes used for parallelization in the `__init__` method. Default: `multiprocessing.cpu_count()`
        ws_dist_tol : float
            the tolerance for the Wigner-Seitz distance. Default: 1e-5

        Notes
        -----
        + The system is described by its real lattice, symmetry group, and Hamiltonian and other real-space matrices.
        + The lattice is given by the lattice vectors in the Cartesian coordinates.
        + The system can be either periodic or confined in some directions.

        Attributes
        ----------
        needed_R_matrices : set
            the set of matrices that are needed for the current calculation. The matrices are set in the constructor.
        npar : int
            number of nodes used for parallelization in the `__init__` method. Default: `multiprocessing.cpu_count()`
        _XX_R : dict(str:array)
            dictionary of real-space matrices. The keys are the names of the matrices, the values are the matrices themselves.
        wannier_centers_cart : array(float)
            the positions of the Wannier centers in the Cartesian coordinates.
        wannier_centers_red : array(float)
            the positions of the Wannier centers in the reduced coordinates.
        num_wann : int
            the number of Wannier functions.
        real_lattice : array(float, shape=(3,3))
            the lattice vectors of the model.
        NKFFT_recommended : int
            the recommended size of the FFT grid to be used in the interpolation.
        """

    def __init__(self,
                 berry=False,
                 morb=False,
                 spin=False,
                 SHCryoo=False,
                 SHCqiao=False,
                 OSD=False,
                 npar=None,
                 _getFF=False,
                 ws_dist_tol=0.05,
                 **parameters):

        super().__init__(**parameters)
        self.needed_R_matrices = {'Ham'}
        self.npar = multiprocessing.cpu_count() if npar is None else npar

        if morb:
            self.needed_R_matrices.update(['AA', 'BB', 'CC'])
        if berry:
            self.needed_R_matrices.add('AA')
        if spin:
            self.needed_R_matrices.add('SS')
        if _getFF:
            self.needed_R_matrices.add('FF')
        if SHCryoo:
            self.needed_R_matrices.update(['AA', 'SS', 'SA', 'SHA', 'SH'])
        if SHCqiao:
            self.needed_R_matrices.update(['AA', 'SS', 'SR', 'SH', 'SHR'])
        if OSD:
            self.needed_R_matrices.update(['AA', 'BB', 'CC', 'GG', 'OO'])

        if self.force_internal_terms_only:
            self.needed_R_matrices = self.needed_R_matrices.intersection(['Ham', 'SS'])

        self._XX_R = dict()
        self.ws_dist_tol = ws_dist_tol

    def set_wannier_centers(self, wannier_centers_cart=None, wannier_centers_red=None):
        """
            set self.wannier_centers_cart. Only one of parameters should be provided.
            If both are None: self.wannier_centers_cart is set to zero.
        """
        lcart = (wannier_centers_cart is not None)
        lred = (wannier_centers_red is not None)
        if lred:
            _wannier_centers_cart = wannier_centers_red.dot(self.real_lattice)
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
        return key in self._XX_R

    def has_R_mat_any(self, keys):
        for k in keys:
            if self.has_R_mat(k):
                return True

    def set_R_mat(self, key, value, diag=False, R=None, reset=False, add=False, Hermitian=False):
        """
        Set real-space matrix specified by `key`. Either diagonal, specific R or full matrix.  Useful for model calculations

        Parameters
        ----------
        key : str
            'SS', 'AA' , etc
        value : array
            * `array(num_wann,...)` if `diag=True` . Sets the diagonal part ( if `R` not set, `R=[0,0,0]`)
            * `array(num_wann,num_wann,..)`  matrix for `R` (`R` should be set )
            * `array(Rvec, num_wann,num_wann,...)` full spin matrix for all R

            `...` denotes the vector/tensor cartesian dimensions of the matrix element
        diag : bool
            set only the diagonal for a specific R-vector (if specified), or fpr R=[0,0,0]
        R : list(int)
            list of 3 integer values specifying R. if
        reset : bool
            allows to reset matrix if it is already set
        add : bool
            add matrix to the already existing
        Hermitian : bool
            force the value to be Hermitian (only if all vectors are set at once)
        """
        if diag:
            assert value.shape[0] == self.num_wann, f"the 0th dimension for 'diag=True' of value should be {self.num_wann}, found {value.shape[0]}"
            if R is None:
                R = [0, 0, 0]
            XX = np.zeros((self.num_wann, self.num_wann) + value.shape[1:], dtype=value.dtype)
            XX[self.range_wann, self.range_wann] = value
            self.set_R_mat(key, XX, R=R, reset=reset, add=add)
        elif R is not None:
            assert value.shape[0:2] == (self.num_wann, self.num_wann), f"the 0th and 1st dimensions of value for R={R}!=None should be nW={self.num_wann}, found {value.shape[0:2]}"
            XX = np.zeros((self.rvec.nRvec, self.num_wann, self.num_wann) + value.shape[2:], dtype=value.dtype)
            XX[self.rvec.iR(R), :, :] = value
            self.set_R_mat(key, XX, reset=reset, add=add)
        else:
            assert value.shape[1:3] == (self.num_wann, self.num_wann), f"for R=None  the 1st and 2nd dimensions should be nw={self.num_wann}, found {value.shape[1:3]}"
            if hasattr(self, 'rvec'):
                assert value.shape[0] == self.rvec.nRvec, f"the 0th dimension of value should be nR={self.rvec.nRvec}, found {value.shape[0]}"
            if Hermitian:
                value = 0.5 * (value + self.rvec.conj_XX_R(value))
            if key in self._XX_R:
                if reset:
                    self._XX_R[key] = value
                elif add:
                    self._XX_R[key] += value
                else:
                    raise RuntimeError(f"setting {key} for the second time without explicit permission. smth is wrong")
            else:
                self._XX_R[key] = value

    def spin_block2interlace(self, backward=False):
        """
        Convert the spin ordering from block (like in the amn file old versions of VASP) to interlace (like in the amn file of QE and new versions of VASP)
        """
        nw2 = self.num_wann // 2
        mapping = np.zeros(self.num_wann, dtype=int)

        if backward:
            mapping[:nw2] = np.arange(nw2) * 2
            mapping[nw2:] = np.arange(nw2) * 2 + 1
        else:
            mapping[::2] = np.arange(nw2)
            mapping[1::2] = np.arange(nw2) + nw2

        for key, val in self._XX_R.items():
            self._XX_R[key] = val[:, :, mapping][:, mapping, :]
        self.wannier_centers_cart = self.wannier_centers_cart[mapping]
        self.rvec.reorder(mapping)
        self.clear_cached_wcc()


    def spin_interlace2block(self, backward=False):
        """
        Convert the spin ordering from interlace (like in the amn file of QE and new versions of VASP) to block (like in the amn file old versions of VASP)
        """
        self.spin_block2interlace(backward=not backward)


    @property
    def Ham_R(self):
        return self.get_R_mat('Ham')

    def symmetrize2(self, symmetrizer, silent=True):
        """
        Symmetrize the system according to the Symmetrizer object.

        Parameters
        ----------
        symmetrizer : :class:`wanierberri.symmetry.sawf.SymmetrizerSAWF`
            The symmetrizer object that will be used for symmetrization. (make sure it is consistent with the order of projections)
        silent : bool
            If True, do not print the symmetrization process. (set to False to see more debug information)
        """
        from ..symmetry.sym_wann_2 import SymWann
        symmetrize_wann = SymWann(
            symmetrizer=symmetrizer,
            iRvec=self.rvec.iRvec,
            wannier_centers_cart=self.wannier_centers_cart,
            silent=self.silent or silent,
        )

        self.check_AA_diag_zero(msg="before symmetrization", set_zero=True)
        logfile = self.logfile

        logfile.write(f"Wannier Centers cart (raw):\n {self.wannier_centers_cart}\n")
        logfile.write(f"Wannier Centers red: (raw):\n {self.wannier_centers_red}\n")

        self._XX_R, iRvec, self.wannier_centers_cart = symmetrize_wann.symmetrize(XX_R=self._XX_R)
        self.clear_cached_wcc()
        self.rvec = Rvectors(
            lattice=self.real_lattice,
            iRvec=iRvec,
            shifts_left_red=self.wannier_centers_red,
        )
        self.set_pointgroup(spacegroup=symmetrizer.spacegroup)
        # self.clear_cached_R()
        # self.clear_cached_wcc()


    def symmetrize(self, proj, positions, atom_name, soc=False, magmom=True, silent=True,
                   reorder_back=False):
        """
        Symmetrize Wannier matrices in real space: Ham_R, AA_R, BB_R, SS_R,... , as well as Wannier centers
        Also sets the pointgroup (with method "new")


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
        store_symm_wann: bool
            Store the (temporary) SymWann object in the `sym_wann` attribute of the System object.
            Can be useful for evaluating symmetry eigenvalues of wavefunctions, etc.
        rotations: array-like (shape=(N,3,3))
            Rotations of the symmetry operations. (optional)
        translations: array-like (shape=(N,3))
            Translations of the symmetry operations. (optional)
        silent: bool
            If True, do not print the symmetrization process.
        reorder_back: bool
            If True, reorder the wannier functions back to the original order after symmetrization. (if the order happened to be changed (see note below))

        Returns
        -------
        symmetrizer : :class:`wanierberri.symmetry.sawf.SymmetrizerSAWF`
            the symmetrizer object that was used for symmetrization. Can be used for further analysis of the symmetry properties of the system.

        Notes
        -----
        * after symmetrization, the wannier functions may be reordered, to group the atoms 
        by same wyckoff positions. In this case, the symmetrizer is not returned, because it would be 
        inconsistent with the order of the projections.

        * Works only with phase convention I (`use_wcc_phase=True`) (which anyway is now the ONLY option)
        Spin ordering is assumed to be interlaced (like in the amn file of QE and new versions of VASP). If it is not, use :func:`~wannierberri.system.System.spin_block2interlace` to convert it.
        """
        from irrep.spacegroup import SpaceGroup
        from ..symmetry.sawf import SymmetrizerSAWF
        from ..symmetry.projections import Projection

        index = {key: i for i, key in enumerate(set(atom_name))}
        atom_num = np.array([index[key] for key in atom_name])

        spacegroup = SpaceGroup(cell=(self.real_lattice, positions, atom_num),
                                magmom=magmom, include_TR=True,
                                spinor=soc,)
        spacegroup.show()

        assert len(atom_name) == len(positions), "atom_name and positions should have the same length"

        proj_list = []
        new_wann_indices = []
        num_wann_loc = 0
        for proj_str in proj:
            atom, orbital = [l.strip() for l in proj_str.split(':')]
            # pos = []
            # ipos = []
            # for ia, a in enumerate(atom_name):
            #     if a == atom:
            #         pos.append(positions[ia])
            #         ipos.append(ia)
            pos = np.array([positions[i] for i, name in enumerate(atom_name) if name == atom])
            suborbit_list = split_into_orbits(pos, spacegroup=spacegroup)
            if len(suborbit_list) > 1:
                warnings.warn(f"Positions of  {atom} belong to different wyckoff positions. This case is not much tested."
                         "it is recommentded to name atoms at different wyckoff positions differently:\n"
                         "\n".join(f"{atom}{i + 1}:" + ";".join(str(pos[j]) for j in suborbit) for i, suborbit in enumerate(suborbit_list))
                )
            print(f"pos_list: {suborbit_list}")
            if ";" in orbital:
                warnings.warn("for effeciency of symmetrization, it is recommended to give orbitals separately, not combined by a ';' sign."
                              "But you need to do it consistently in wannier90 ")
            for suborbit in suborbit_list:
                pos_loc = pos[suborbit]
                proj = Projection(position_num=pos_loc, orbital=orbital, spacegroup=spacegroup,
                                  do_not_split_projections=True)
                proj_list.append(proj)
                num_wann_per_position = proj.num_wann_per_site_spinor
                print(f"orbital = {orbital}, num wann per site = {num_wann_per_position}")
                for i in suborbit:
                    for j in range(num_wann_per_position):
                        new_wann_indices.append(num_wann_loc + i * num_wann_per_position + j)
            num_wann_loc = len(new_wann_indices)

        print(f"new_wann_indices: {new_wann_indices}")
        self.reorder(new_wann_indices)
        symmetrizer = SymmetrizerSAWF().set_spacegroup(spacegroup).set_D_wann_from_projections(projections=proj_list)
        self.symmetrize2(symmetrizer, silent=silent)
        if reorder_back:
            self.reorder(np.argsort(new_wann_indices))
            if np.all(np.array(new_wann_indices) == np.arange(self.num_wann)):
                return symmetrizer
            else:
                return None
        else:
            return symmetrizer

    def reorder(self, new_wann_indices):
        """
        Reorder the wannier functions according to the new indices

        Parameters
        ----------
        new_wann_indices : list
            list of new indices for the wannier functions. The length should be equal to the number of wannier functions.
        """
        assert len(new_wann_indices) == self.num_wann, f"new_wann_indices should have length {self.num_wann}, found {len(new_wann_indices)}"
        self.wannier_centers_cart = self.wannier_centers_cart[new_wann_indices]
        for key, val in self._XX_R.items():
            self._XX_R[key] = val[:, :, new_wann_indices][:, new_wann_indices, :]
        self.rvec.reorder(new_wann_indices)
        self.clear_cached_wcc()
        self.clear_cached_R()


    def check_AA_diag_zero(self, msg="", set_zero=True):
        if self.has_R_mat('AA'):
            A_diag = self.get_R_mat('AA')[self.rvec.iR0].diagonal()
            A_diag_max = abs(A_diag).max()
            if A_diag_max > 1e-5:
                warnings.warn(
                    f"the maximal value of diagonal position matrix elements {msg} is {A_diag_max}."
                    f"This may signal a problem\n {A_diag}")
                if set_zero:
                    warnings.warn("setting AA diagonal to zero")
            if set_zero:
                self.get_R_mat('AA')[self.rvec.iR0, self.range_wann, self.range_wann, :] = 0

    def check_periodic(self):
        exclude = np.zeros(self.rvec.nRvec, dtype=bool)
        for i, per in enumerate(self.periodic):
            if not per:
                sel = (self.rvec.iRvec[:, i] != 0)
                if np.any(sel):
                    warnings.warn(f"you declared your system as non-periodic along direction {i},"
                                  f"but there are {sum(sel)} of total {self.nRvec} R-vectors with R[{i}]!=0."
                                  "They will be excluded, please make sure you know what you are doing")
                    exclude[sel] = True
        if np.any(exclude):
            notexclude = np.logical_not(exclude)
            self.iRvec = self.iRvec[notexclude]
            for X in ['Ham', 'AA', 'BB', 'CC', 'SS', 'FF']:
                if X in self._XX_R:
                    self.set_R_mat(X, self.get_X_mat(X)[:, :, notexclude], reset=True)

    def set_spin(self, spins, axis=(0, 0, 1), **kwargs):
        """
        Set spins along axis in  SS(R=0).  Useful for model calculations.
        Note : The spin matrix is purely diagonal, so that <up | sigma_x | down> = 0
        For more cversatility use :func:`~wannierberri.system.System.set_R_mat`
        :func:`~wannierberri.system.System.set_spin_pairs`, :func:`~wannierberri.system.System.set_spin_from_projections`

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
            warnings.warn("some of your spins are not +1 or -1, are you sure you want it like this?")
        axis = np.array(axis) / np.linalg.norm(axis)
        value = np.array([s * axis for s in spins], dtype=complex)
        self.set_R_mat(key='SS', value=value, diag=True, **kwargs)

    def set_spin_pairs(self, pairs):
        """set SS_R, assuming that each Wannier function is an eigenstate of Sz,

        Parameters
        ----------
        pairs : list of tuple
            list of pairs of indices of bands `[(up1,down1), (up2,down2), ..]`

        Notes
        -----
        * For abinitio calculations this is a rough approximation, that may be used on own risk.
        * See also :func:`~wannierberri.system.System.set_spin_from_projections`

        """
        assert all(len(p) == 2 for p in pairs)
        all_states = np.array(sum((list(p) for p in pairs), []))
        assert np.all(all_states >= 0) and (np.all(all_states < self.num_wann)), (
            f"indices of states should be 0<=i<num_wann-{self.num_wann}, found {pairs}")
        assert len(set(all_states)) == len(all_states), "some states appear more then once in pairs"
        if len(pairs) < self.num_wann / 2:
            warnings.warn(f"number of spin pairs {len(pairs)} is less then num_wann/2 = {self.num_wann / 2}."
                          "For other states spin properties will be set to zero. are yoiu sure ?")
        SS_R0 = np.zeros((self.num_wann, self.num_wann, 3), dtype=complex)
        for i, j in pairs:
            dist = np.linalg.norm(self.wannier_centers_cart[i] - self.wannier_centers_cart[j])
            if dist > 1e-3:
                warnings.warn(f"setting spin pair for Wannier function {i} and {j}, distance between them {dist}")
            SS_R0[i, i] = pauli_xyz[0, 0]
            SS_R0[i, j] = pauli_xyz[0, 1]
            SS_R0[j, i] = pauli_xyz[1, 0]
            SS_R0[j, j] = pauli_xyz[1, 1]
            self.set_R_mat(key='SS', value=SS_R0, diag=False, R=[0, 0, 0], reset=True)

    def set_spin_from_projections(self):
        """set SS_R, assuming that each Wannier function is an eigenstate of Sz,
         with interlaced spin ordering, which  means that the orbitals of the two spins are interlaced. (like in the amn file of QE and new versions of VASP)

        Notes
        -------
        * This is a rough approximation, that may be used on own risk
        * The pure-spin character may be broken by maximal localization. Recommended to use `num_iter=0` in Wannier90
        also see   :func:`~wannierberri.system.System.set_spin_pairs`
        """
        assert self.num_wann % 2 == 0, f"odd number of Wannier functions {self.num_wann} cannot be grouped into spin pairs"
        nw2 = self.num_wann // 2
        pairs = [(2 * i, 2 * i + 1) for i in range(nw2)]
        self.set_spin_pairs(pairs)

    def set_spacegroup(self, spacegroup):
        """
        Set the space group of the :class:`System`, which will be used for symmetrization
        R-space and k-space 
        Also sets the pointgroup

        Parameters
        ----------
        spacegroup : :class:`irrep.spacegroup.SpaceGroup`
            The space group of the system. The point group will be evaluated by the space group.
        """
        self.spacegroup = spacegroup
        self.set_pointgroup(spacegroup=spacegroup)

    def do_at_end_of_init(self):
        self.set_pointgroup()
        self.check_periodic()
        logfile = self.logfile
        logfile.write(f"Real-space lattice:\n {self.real_lattice}\n")
        logfile.write(f"Number of wannier functions: {self.num_wann}\n")
        logfile.write(f"Number of R points: {self.rvec.nRvec}\n")
        logfile.write(f"Recommended size of FFT grid {self.NKFFT_recommended}\n")


    def do_ws_dist(self, mp_grid, wannier_centers_cart=None):
        logfile = self.logfile
        try:
            mp_grid = one2three(mp_grid)
            assert mp_grid is not None
        except AssertionError:
            raise ValueError(f"mp_greid should be one integer, of three integers. found {mp_grid}")
        self._NKFFT_recommended = mp_grid
        if wannier_centers_cart is None:
            wannier_centers_cart = self.wannier_centers_cart
        iRvec_old = self.rvec.iRvec
        self.rvec = Rvectors(lattice=self.real_lattice, shifts_left_red=self.wannier_centers_red)
        self.rvec.set_Rvec(mp_grid, ws_tolerance=self.ws_dist_tol)
        for key, val in self._XX_R.items():
            logfile.write(f"using new ws_dist for {key}\n")
            self.set_R_mat(key, self.rvec.remap_XX_R(val, iRvec_old=iRvec_old), reset=True)
        self._XX_R, self.rvec = self.rvec.exclude_zeros(self._XX_R)


    def to_tb_file(self, tb_file=None, use_convention_II=True):
        """
        Write the system in the format of the wannier90_tb.dat file
        Note : it is written in phase convention II (as inb wannier90), unless use_convention_II=False
        """
        logfile = self.logfile
        if tb_file is None:
            tb_file = self.seedname + "_fromchk_tb.dat"
        f = open(tb_file, "w")
        f.write("written by wannier-berri form the chk file\n")
        logfile.write(f"writing TB file {tb_file}\n")
        np.savetxt(f, self.real_lattice)
        f.write(f"{self.num_wann}\n")
        f.write(f"{self.rvec.nRvec}\n")
        Ndegen = np.ones(self.rvec.nRvec, dtype=int)
        for i in range(0, self.rvec.nRvec, 15):
            a = Ndegen[i:min(i + 15, self.rvec.nRvec)]
            f.write("  ".join(f"{x:2d}" for x in a) + "\n")
        for iR in range(self.rvec.nRvec):
            f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.rvec.iRvec[iR])))
            _ham = self.Ham_R[iR] * Ndegen[iR]
            f.write(
                "".join(
                    f"{m + 1:3d} {n + 1:3d} {_ham[m, n].real:15.8e} {_ham[m, n].imag:15.8e}\n"
                    for n in self.range_wann for m in self.range_wann)
            )
        if self.has_R_mat('AA'):
            AA = np.copy(self.get_R_mat('AA'))
            if use_convention_II:
                AA[self.rvec.iR0, self.range_wann, self.range_wann] += self.wannier_centers_cart
            for iR in range(self.rvec.nRvec):
                f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.rvec.iRvec[iR])))
                _aa = AA[iR] * Ndegen[iR]
                f.write(
                    "".join(
                        f"{m + 1:3d} {n + 1:3d} " + " ".join(f"{a.real:15.8e} {a.imag:15.8e}" for a in _aa[m, n]) + "\n"
                        for n in self.range_wann for m in self.range_wann
                    )
                )
        f.close()

    @property
    def NKFFT_recommended(self):
        """finds a minimal FFT grid on which different R-vectors do not overlap"""
        if hasattr(self, '_NKFFT_recommended'):
            return self._NKFFT_recommended
        else:
            return self.rvec.NKFFT_recommended()

    def clear_cached_R(self):
        self.rvec.clear_cached()

    def clear_cached_wcc(self):
        clear_cached(self, ["wannier_centers_red"])
        if hasattr(self, 'rvec'):
            self.rvec.clear_cached()


    @cached_property
    def wannier_centers_red(self):
        return self.wannier_centers_cart.dot(np.linalg.inv(self.real_lattice))

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
            if not all([len(x) == 3 for x in magnetic_moments]):
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
                return self.real_lattice, self.positions, atom_numbers
            else:
                return self.real_lattice, self.positions, atom_numbers, self.magnetic_moments
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
            symmetry_gen.append(PointSymmetry(R, TR=TR))

        if self.magnetic_moments is None:
            symmetry_gen.append(TimeReversal)
        elif not tr_found:
            warnings.warn(
                "you specified magnetic moments but spglib did not detect symmetries involving time-reversal. "
                f"proobably it is because you have an old spglib version {spglib.__version__}."
                "We suggest upgrading to spglib>=2.0.2")
        else:
            if not all([len(x) for x in self.magnetic_moments]):
                raise ValueError("magnetic_moments must be a list of 3d vector")
            warnings.warn("spglib does not find symmetries including time reversal operation. "
                          "To include such symmetries, use set_symmetry.")

        self.pointgroup = PointGroup(symmetry_gen, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice)

    def get_sparse(self, min_values={'Ham': 1e-3}):
        min_values = copy.copy(min_values)
        ret_dic = dict(
            real_lattice=self.real_lattice,
            wannier_centers_red=self.wannier_centers_red,
            matrices={},
        )
        if hasattr(self, 'symmetrize_info'):
            ret_dic['symmetrize_info'] = self.symmetrize_info

        def array_to_dict(A, minval):
            A_tmp = abs(A.reshape(A.shape[:3] + (-1,))).max(axis=-1)
            wh = np.argwhere(A_tmp >= minval)
            dic = defaultdict(lambda: dict())
            for w in wh:
                iR = tuple(self.rvec.iRvec[w[0]])
                dic[iR][(w[1], w[2])] = A[tuple(w)]
            return dict(dic)

        for k, v in min_values.items():
            ret_dic['matrices'][k] = array_to_dict(self.get_R_mat(k), v)
        return ret_dic

    @cached_property
    def essential_properties(self):
        return ['num_wann', 'real_lattice', 'iRvec', 'periodic',
                'is_phonon', 'wannier_centers_cart', 'pointgroup']

    @cached_property
    def optional_properties(self):
        return ["positions", "magnetic_moments", "atom_labels"]

    def _R_mat_npz_filename(self, key):
        return "_XX_R_" + key + ".npz"

    def save_npz(self, path, extra_properties=(), exclude_properties=(), R_matrices=None, overwrite=True):
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
        logfile = self.logfile

        properties = [x for x in self.essential_properties + list(extra_properties) if x not in exclude_properties]
        if R_matrices is None:
            R_matrices = list(self._XX_R.keys())

        try:
            os.makedirs(path, exist_ok=overwrite)
        except FileExistsError:
            raise FileExistsError(f"Directorry {path} already exists. To overwrite it set overwrite=True")

        for key in properties:
            logfile.write(f"saving {key}\n")
            fullpath = os.path.join(path, key + ".npz")
            if key == 'iRvec':
                val = self.rvec.iRvec
            else:
                val = getattr(self, key)
            if key in ['pointgroup']:
                np.savez(fullpath, **val.as_dict())
            else:
                np.savez(fullpath, val)
            logfile.write(" - Ok!\n")
        for key in self.optional_properties:
            if key not in properties:
                fullpath = os.path.join(path, key + ".npz")
                if hasattr(self, key):
                    val = getattr(self, key)
                    np.savez(fullpath, val)
        for key in R_matrices:
            logfile.write(f"saving {key}")
            np.savez_compressed(os.path.join(path, self._R_mat_npz_filename(key)), self.get_R_mat(key))
            logfile.write(" - Ok!\n")

    def load_npz(self, path, load_all_XX_R=False, exclude_properties=(), legacy=False):
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
        legacy : bool
            if True, the order of indices in the XX_R matrices will be expected as in older verisons of wannierberri : [m,n,iR, ...]
            in the newer versions the order is [iR, m ,n,...]
        """
        logfile = self.logfile
        all_files = glob.glob(os.path.join(path, "*.npz"))
        all_names = [os.path.splitext(os.path.split(x)[-1])[0] for x in all_files]
        properties = [x for x in all_names if not x.startswith('_XX_R_') and x not in exclude_properties]
        assert "real_lattice" in properties, "real_lattice is required to load the system"
        properties = ["real_lattice", "wannier_centers_cart"] + properties
        keys_processed = set()
        for key in properties:
            if key in keys_processed:
                continue
            logfile.write(f"loading {key}")
            a = np.load(os.path.join(path, key + ".npz"), allow_pickle=False)

            # pointgroup was previouslly named symgroup. This is for backward compatibility
            if key == 'symgroup':
                key_loc = 'pointgroup'
            else:
                key_loc = key

            if key_loc == 'pointgroup':
                val = PointGroup(dictionary=a)
            else:
                val = a['arr_0']

            if key == "iRvec":
                self.rvec = Rvectors(lattice=self.real_lattice,
                                     iRvec=val,
                                     shifts_left_red=self.wannier_centers_red
                                     )
            else:
                setattr(self, key_loc, val)
            logfile.write(" - Ok!\n")
            keys_processed.add(key)

        if load_all_XX_R:
            R_files = glob.glob(os.path.join(path, "_XX_R_*.npz"))
            R_matrices = [os.path.splitext(os.path.split(x)[-1])[0][6:] for x in R_files]
            self.needed_R_matrices.update(R_matrices)
        for key in self.needed_R_matrices:
            logfile.write(f"loading R_matrix {key}")
            a = np.load(os.path.join(path, self._R_mat_npz_filename(key)), allow_pickle=False)['arr_0']
            if legacy:
                a = np.transpose(a, (2, 0, 1) + tuple(range(3, a.ndim)))
            self.set_R_mat(key, a)
            logfile.write(" - Ok!\n")
        return self
